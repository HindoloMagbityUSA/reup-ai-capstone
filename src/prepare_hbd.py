#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
from pathlib import Path

from PIL import Image


EXPECTED_CLASSES = [
    "wall", "curtain_wall", "floor", "ceiling", "roof", "column",
    "beam", "lift", "window", "door", "opening",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a component-crop manifest from HBD VIA, COCO, or YOLO annotations."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Extracted HBD folder.")
    parser.add_argument("--output", type=Path, required=True, help="Manifest CSV to create.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-area", type=float, default=0.01, help="Minimum crop area as image fraction.")
    parser.add_argument("--padding", type=float, default=0.08, help="Padding added around each component crop.")
    return parser.parse_args()


def normalize_name(name: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    aliases = {"curtainwall": "curtain_wall", "elevator": "lift", "pillar": "column"}
    return aliases.get(value, value)


def normalize_split(value: str | None, key: str, seed: int) -> str:
    if value:
        value = value.lower()
        if "train" in value:
            return "train"
        if "val" in value or "valid" in value:
            return "val"
        if "test" in value:
            return "test"
    number = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if number < 0.70 else "val" if number < 0.85 else "test"


def padded_box(box, width: int, height: int, padding: float):
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    return (
        max(0.0, x1 - pad_x), max(0.0, y1 - pad_y),
        min(float(width), x2 + pad_x), min(float(height), y2 + pad_y),
    )


def make_row(image_path: Path, name: str, split: str, box, width: int, height: int, source_id: str, args):
    name = normalize_name(name)
    if name not in EXPECTED_CLASSES:
        return None
    x1, y1, x2, y2 = padded_box(box, width, height, args.padding)
    if x2 <= x1 or y2 <= y1 or ((x2 - x1) * (y2 - y1)) / (width * height) < args.minimum_area:
        return None
    return {
        "image_path": str(image_path.resolve()),
        "label_id": EXPECTED_CLASSES.index(name),
        "label_name": name,
        "split": split,
        "included": "true",
        "source_dataset": "HBD",
        "source_image_id": source_id,
        "crop_x1": round(x1, 2), "crop_y1": round(y1, 2),
        "crop_x2": round(x2, 2), "crop_y2": round(y2, 2),
    }


def find_image(root: Path, file_name: str, index: dict[str, Path]) -> Path | None:
    direct = root / file_name
    if direct.exists():
        return direct
    return index.get(Path(file_name).name)


def read_coco(root: Path, args, image_index: dict[str, Path]):
    rows = []
    candidates = []
    for path in root.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if all(key in data for key in ("images", "annotations", "categories")):
            candidates.append((path, data))
    for annotation_path, data in candidates:
        categories = {item["id"]: item["name"] for item in data["categories"]}
        images = {item["id"]: item for item in data["images"]}
        hint = annotation_path.as_posix()
        for item in data["annotations"]:
            image = images.get(item.get("image_id"))
            if not image or "bbox" not in item:
                continue
            image_path = find_image(root, image["file_name"], image_index)
            if image_path is None:
                continue
            width = int(image.get("width") or 0)
            height = int(image.get("height") or 0)
            if not width or not height:
                with Image.open(image_path) as opened:
                    width, height = opened.size
            x, y, box_width, box_height = map(float, item["bbox"])
            key = f"{annotation_path}:{image['id']}"
            split = normalize_split(image.get("split") or hint, key, args.seed)
            row = make_row(
                image_path, categories.get(item.get("category_id"), ""), split,
                (x, y, x + box_width, y + box_height), width, height, str(image["id"]), args,
            )
            if row:
                rows.append(row)
    return rows


def read_via(root: Path, args, image_index: dict[str, Path]):
    """Read the native VGG Image Annotator (VIA) JSON files supplied with HBD."""
    rows = []
    for annotation_path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        for record_key, image in data.items():
            if not isinstance(image, dict) or "filename" not in image or "regions" not in image:
                continue
            file_name = image["filename"]
            image_path = annotation_path.parent / file_name
            if not image_path.exists():
                image_path = find_image(root, file_name, image_index)
            if image_path is None:
                continue
            with Image.open(image_path) as opened:
                width, height = opened.size
                if opened.getexif().get(274, 1) in {5, 6, 7, 8}:
                    width, height = height, width
            regions = image.get("regions") or []
            if isinstance(regions, dict):
                regions = regions.values()
            split = normalize_split(annotation_path.as_posix(), str(record_key), args.seed)
            for region_number, region in enumerate(regions, 1):
                if not isinstance(region, dict):
                    continue
                shape = region.get("shape_attributes") or {}
                attributes = region.get("region_attributes") or {}
                xs = shape.get("all_points_x") or []
                ys = shape.get("all_points_y") or []
                if len(xs) < 2 or len(xs) != len(ys):
                    continue
                class_name = (
                    attributes.get("ClassName")
                    or attributes.get("classname")
                    or attributes.get("class")
                    or ""
                )
                row = make_row(
                    image_path, class_name, split,
                    (min(xs), min(ys), max(xs), max(ys)), width, height,
                    f"{annotation_path.relative_to(root)}:{record_key}:{region_number}", args,
                )
                if row:
                    rows.append(row)
    return rows


def load_yaml(path: Path):
    try:
        import yaml
    except ImportError:
        # Minimal fallback for the simple data.yaml files produced by YOLO/Roboflow.
        data = {}
        section = None
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip() or ":" not in line:
                continue
            indented = line[0].isspace()
            key, value = (part.strip() for part in line.split(":", 1))
            if indented and section == "names":
                data["names"][int(key)] = value.strip("'\"")
                continue
            section = key if not value else None
            if key == "names" and not value:
                data["names"] = {}
            elif value.startswith("["):
                data[key] = ast.literal_eval(value)
            else:
                data[key] = value.strip("'\"")
        return data
    else:
        return yaml.safe_load(path.read_text(encoding="utf-8"))


def yolo_box(values: list[float], width: int, height: int):
    if len(values) == 4:
        cx, cy, box_width, box_height = values
        return ((cx - box_width / 2) * width, (cy - box_height / 2) * height,
                (cx + box_width / 2) * width, (cy + box_height / 2) * height)
    xs = values[0::2]
    ys = values[1::2]
    return min(xs) * width, min(ys) * height, max(xs) * width, max(ys) * height


def read_yolo(root: Path, args):
    yaml_paths = list(root.rglob("data.yaml")) + list(root.rglob("data.yml"))
    if not yaml_paths:
        return []
    config_path = yaml_paths[0]
    config = load_yaml(config_path)
    names_value = config.get("names", [])
    names = (
        [names_value[key] for key in sorted(names_value, key=lambda item: int(item))]
        if isinstance(names_value, dict) else list(names_value)
    )
    base = config_path.parent
    if config.get("path"):
        candidate = Path(config["path"])
        base = candidate if candidate.is_absolute() else config_path.parent / candidate
    rows = []
    for raw_split in ("train", "val", "valid", "test"):
        entries = config.get(raw_split)
        if not entries:
            continue
        for entry in entries if isinstance(entries, list) else [entries]:
            image_dir = Path(entry)
            image_dir = image_dir if image_dir.is_absolute() else base / image_dir
            image_paths = [image_dir] if image_dir.is_file() else sorted(image_dir.rglob("*"))
            for image_path in image_paths:
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                label_path = Path(str(image_path).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"))
                label_path = label_path.with_suffix(".txt")
                if not label_path.exists():
                    continue
                with Image.open(image_path) as opened:
                    width, height = opened.size
                for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    class_id = int(float(parts[0]))
                    values = [float(value) for value in parts[1:]]
                    if class_id >= len(names) or (len(values) != 4 and len(values) < 6):
                        continue
                    row = make_row(
                        image_path, names[class_id], normalize_split(raw_split, str(image_path), args.seed),
                        yolo_box(values, width, height), width, height,
                        f"{image_path.stem}:{line_number}", args,
                    )
                    if row:
                        rows.append(row)
    return rows


def main():
    args = parse_args()
    root = args.dataset.expanduser().resolve()
    image_index = {
        path.name: path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    rows = read_via(root, args, image_index)
    annotation_format = "VIA polygon JSON"
    if not rows:
        rows = read_coco(root, args, image_index)
        annotation_format = "COCO"
    if not rows:
        rows = read_yolo(root, args)
        annotation_format = "YOLO"
    if not rows:
        raise SystemExit(
            "No supported HBD annotations found. Expected VIA/COCO JSON or YOLO data.yaml plus labels."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Detected {annotation_format} annotations.")
    print(f"Wrote {len(rows):,} component crops to {args.output}")


if __name__ == "__main__":
    main()
