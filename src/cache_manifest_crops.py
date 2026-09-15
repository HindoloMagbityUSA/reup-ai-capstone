#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps


CROP_FIELDS = ("crop_x1", "crop_y1", "crop_x2", "crop_y2")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cache manifest crops as compact images so repeated training epochs are faster."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-size", type=int, default=320)
    parser.add_argument("--quality", type=int, default=90)
    return parser.parse_args()


def crop_name(row: dict[str, str], row_number: int) -> str:
    identity = "|".join([
        row.get("source_id", ""), row.get("source_image_id", ""),
        row.get("image_path", ""), *(row.get(key, "") for key in CROP_FIELDS),
        str(row_number),
    ])
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20] + ".jpg"


def main():
    args = parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise SystemExit(f"No rows found in {args.manifest}")

    grouped: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for row_number, row in enumerate(rows):
        grouped[row["image_path"]].append((row_number, row))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    for source_path, source_rows in grouped.items():
        pending = []
        for row_number, row in source_rows:
            destination = (
                args.output_dir / row["split"] / row["label_name"]
                / crop_name(row, row_number)
            )
            pending.append((row_number, row, destination))
        if all(destination.exists() for _, _, destination in pending):
            for _, row, destination in pending:
                row["image_path"] = str(destination.resolve())
                for key in CROP_FIELDS:
                    row[key] = ""
                completed += 1
            continue
        with Image.open(source_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            for row_number, row, destination in pending:
                if not destination.exists():
                    box = tuple(float(row[key]) for key in CROP_FIELDS)
                    crop = image.crop(box)
                    crop.thumbnail((args.max_size, args.max_size), Image.Resampling.BILINEAR)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    crop.save(destination, format="JPEG", quality=args.quality)
                row["image_path"] = str(destination.resolve())
                for key in CROP_FIELDS:
                    row[key] = ""
                completed += 1

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Cached {completed:,} crops from {len(grouped):,} source images.")
    print(f"Wrote {args.output_manifest}")


if __name__ == "__main__":
    main()
