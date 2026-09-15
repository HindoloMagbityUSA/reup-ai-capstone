#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from reup_ai import CLASS_NAMES

SPLITS = ("train", "val", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Audit and deduplicate the ReUP image dataset.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    issues = []
    hash_to_rows = defaultdict(list)
    counts_before = defaultdict(Counter)
    dimensions = []

    for split in SPLITS:
        image_dir = args.dataset / "images" / split
        label_dir = args.dataset / "labels" / split
        for image_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in EXTENSIONS):
            label_path = label_dir / f"{image_path.stem}.txt"
            try:
                tokens = label_path.read_text(encoding="utf-8").strip().split()
                label_id = int(tokens[0])
                if label_id not in range(len(CLASS_NAMES)):
                    raise ValueError(f"invalid class id {label_id}")
            except Exception as exc:
                issues.append({"path": str(image_path.resolve()), "issue": f"label: {exc}"})
                continue

            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    width, height = image.size
            except Exception as exc:
                issues.append({"path": str(image_path.resolve()), "issue": f"image: {exc}"})
                continue

            digest = sha256(image_path)
            row = {
                "split": split,
                "image_path": str(image_path.resolve()),
                "label_path": str(label_path.resolve()),
                "label_id": label_id,
                "label_name": CLASS_NAMES[label_id],
                "width": width,
                "height": height,
                "sha256": digest,
                "included": "true",
                "exclusion_reason": "",
            }
            rows.append(row)
            hash_to_rows[digest].append(row)
            counts_before[split][CLASS_NAMES[label_id]] += 1
            dimensions.append((width, height))

    duplicate_groups = []
    for digest, group in hash_to_rows.items():
        if len(group) < 2:
            continue
        # Preserve the first item according to train -> val -> test and deterministic filename order.
        group.sort(key=lambda r: (SPLITS.index(r["split"]), r["image_path"]))
        kept = group[0]
        for duplicate in group[1:]:
            duplicate["included"] = "false"
            duplicate["exclusion_reason"] = f"exact_duplicate_of:{kept['image_path']}"
        duplicate_groups.append({
            "sha256": digest,
            "kept": kept["image_path"],
            "excluded": [item["image_path"] for item in group[1:]],
            "cross_split": len({item["split"] for item in group}) > 1,
        })

    counts_after = defaultdict(Counter)
    for row in rows:
        if row["included"] == "true":
            counts_after[row["split"]][row["label_name"]] += 1

    fieldnames = list(rows[0].keys())
    manifest = args.output_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    audit = {
        "dataset_root": str(args.dataset.resolve()),
        "classes": CLASS_NAMES,
        "records_before_deduplication": len(rows),
        "records_after_deduplication": sum(row["included"] == "true" for row in rows),
        "counts_before": {split: dict(counts_before[split]) for split in SPLITS},
        "counts_after": {split: dict(counts_after[split]) for split in SPLITS},
        "exact_duplicate_groups": len(duplicate_groups),
        "cross_split_duplicate_groups": sum(group["cross_split"] for group in duplicate_groups),
        "duplicates": duplicate_groups,
        "unreadable_or_invalid_records": issues,
        "image_dimensions": {
            "min_width": min(width for width, _ in dimensions),
            "max_width": max(width for width, _ in dimensions),
            "min_height": min(height for _, height in dimensions),
            "max_height": max(height for _, height in dimensions),
        },
    }
    (args.output_dir / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

