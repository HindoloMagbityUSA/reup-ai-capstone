#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


SOURCE_COLUMNS = {
    "NoDamage": ("NoDamage",),
    "Crack": ("Crack",),
    "Efflorescence": ("Efflorescence",),
    "Spalling": ("Spalling", "Spallation"),
    "BarsExposed": ("BarsExposed",),
    "Rust": ("Rust",),
}
DAMAGE_CLASSES = ["Crack", "Efflorescence", "Spalling", "BarsExposed", "Rust"]
TARGET_CLASSES = ["no_visible_damage", "visible_damage_detected"]


def parse_args():
    parser = argparse.ArgumentParser(description="Create a binary ReUP condition manifest from dacl1k.")
    parser.add_argument(
        "--cache-dir", type=Path, required=True,
        help="Folder containing the extracted dacl1k download (for example data).",
    )
    parser.add_argument("--output", type=Path, required=True, help="Manifest CSV to create.")
    parser.add_argument(
        "--download", action="store_true",
        help="Download and extract dacl1k through the official Building Inspection Toolkit.",
    )
    return parser.parse_args()


def source_value(source: dict, canonical_name: str) -> float:
    for column in SOURCE_COLUMNS[canonical_name]:
        value = source.get(column)
        if value not in (None, ""):
            return float(value)
    return 0.0


def make_row(source: dict, split: str, image_path: Path):
    cues = [name for name in DAMAGE_CLASSES if source_value(source, name) > 0]
    no_damage = source_value(source, "NoDamage") > 0
    if cues:
        label_id = 1
    elif no_damage:
        label_id = 0
    else:
        return None
    return {
        "image_path": str(image_path.resolve()),
        "label_id": label_id,
        "label_name": TARGET_CLASSES[label_id],
        "split": split,
        "included": "true",
        "source_dataset": "dacl1k",
        "source_image_id": str(source.get("img_name", image_path.name)),
        "damage_cues": "|".join(cues),
    }


def read_native_csv(cache_dir: Path):
    candidates = sorted(cache_dir.rglob("annotations*.csv"))
    for annotation_path in candidates:
        with annotation_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            if not {"img_path", "img_name", "split_type", "NoDamage"}.issubset(columns):
                continue
            rows = []
            skipped = 0
            for source in reader:
                raw_path = Path(source["img_path"])
                image_path = cache_dir / raw_path
                if not image_path.is_file():
                    image_path = annotation_path.parent / "v1" / source["img_name"]
                row = make_row(source, source["split_type"], image_path)
                if row:
                    rows.append(row)
                else:
                    skipped += 1
            return rows, skipped
    return [], 0


def read_with_bikit(cache_dir: Path):
    try:
        from bikit.datasets import BikitDataset
    except ImportError as error:
        raise SystemExit(
            "No native dacl1k annotation CSV was found and Building Inspection Toolkit is not installed."
        ) from error

    rows = []
    skipped = 0
    for split in ("train", "val", "test"):
        dataset = BikitDataset(
            name="dacl1k", split=split, cache_dir=str(cache_dir), return_type="np"
        )
        for _, record in dataset.df.iterrows():
            source = record.to_dict()
            image_path = cache_dir / str(source["img_path"])
            row = make_row(source, split, image_path)
            if row:
                rows.append(row)
            else:
                skipped += 1
    return rows, skipped


def main():
    args = parse_args()
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    if args.download:
        try:
            from bikit.utils import download_dataset
        except ImportError as error:
            raise SystemExit(
                "Building Inspection Toolkit is required for automatic download. "
                "Alternatively, extract dacl1k.zip under --cache-dir and run without --download."
            ) from error
        download_dataset("dacl1k", cache_dir=str(cache_dir), rm_zip_or_rar=False)

    rows, skipped = read_native_csv(cache_dir)
    if not rows:
        rows, skipped = read_with_bikit(cache_dir)
    if not rows:
        raise SystemExit(
            f"No dacl1k records were found under {cache_dir}. Run again with --download or check the extraction path."
        )
    missing_images = [row["image_path"] for row in rows if not Path(row["image_path"]).is_file()]
    if missing_images:
        preview = "\n".join(missing_images[:5])
        raise SystemExit(
            f"dacl1k metadata loaded, but {len(missing_images)} image files are missing. "
            f"Check --cache-dir. First missing paths:\n{preview}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter((row["split"], row["label_name"]) for row in rows)
    print(f"Wrote {len(rows):,} rows to {args.output}; skipped {skipped:,} unlabeled rows.")
    for key, count in sorted(counts.items()):
        print(f"{key[0]:>5} | {key[1]:>24} | {count:>6,}")


if __name__ == "__main__":
    main()
