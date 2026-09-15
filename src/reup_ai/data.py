from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    def __init__(self, manifest: Path, split: str, transform=None):
        self.transform = transform
        with manifest.open(newline="", encoding="utf-8") as handle:
            self.rows = [
                row for row in csv.DictReader(handle)
                if row["split"] == split and row["included"] == "true"
            ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image_path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if all(row.get(key) for key in ("crop_x1", "crop_y1", "crop_x2", "crop_y2")):
                crop = tuple(float(row[key]) for key in ("crop_x1", "crop_y1", "crop_x2", "crop_y2"))
                image = image.crop(crop)
        if self.transform:
            image = self.transform(image)
        return image, int(row["label_id"]), row["image_path"]


def read_manifest_class_names(manifest: Path) -> list[str]:
    """Return class names ordered by label_id from a ReUP manifest."""
    names: dict[int, str] = {}
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("included", "true").lower() != "true":
                continue
            label_id = int(row["label_id"])
            label_name = row.get("label_name") or row.get("class_name")
            if not label_name:
                raise ValueError("Manifest must include label_name or class_name.")
            existing = names.get(label_id)
            if existing is not None and existing != label_name:
                raise ValueError(f"label_id {label_id} maps to both {existing!r} and {label_name!r}.")
            names[label_id] = label_name
    if not names:
        raise ValueError(f"No included rows found in {manifest}.")
    expected = list(range(max(names) + 1))
    if sorted(names) != expected:
        raise ValueError(f"label_id values must be contiguous from 0; found {sorted(names)}.")
    return [names[index] for index in expected]
