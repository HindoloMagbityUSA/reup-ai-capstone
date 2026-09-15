#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Create dataset and prediction visualizations.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def class_distribution(manifest, output_dir):
    counts = defaultdict(Counter)
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["included"] == "true":
                counts[row["split"]][row["label_name"]] += 1
    classes = sorted({name for values in counts.values() for name in values})
    splits = ["train", "val", "test"]
    x = list(range(len(classes)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, split in enumerate(splits):
        positions = [value + (index - 1) * width for value in x]
        values = [counts[split][name] for name in classes]
        bars = ax.bar(positions, values, width=width, label=split.title())
        ax.bar_label(bars, padding=2, fontsize=8)
    ax.set_xticks(x, [name.replace("_", " ").title() for name in classes])
    ax.set(ylabel="Unique valid images", title="Deduplicated dataset distribution by material and split")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=180)
    plt.close(fig)


def dataset_sample_grid(manifest, output_dir):
    by_class = defaultdict(list)
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["included"] == "true":
                by_class[row["label_name"]].append(row["image_path"])
    rng = random.Random(42)
    classes = sorted(by_class)
    fig, axes = plt.subplots(len(classes), 6, figsize=(13, 9))
    for row_index, class_name in enumerate(classes):
        for column_index, path in enumerate(rng.sample(by_class[class_name], 6)):
            axis = axes[row_index, column_index]
            with Image.open(path) as image:
                axis.imshow(image.convert("RGB"))
            if column_index == 0:
                axis.set_ylabel(class_name.replace("_", " ").title(), fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle("Representative dataset images by source label", fontsize=14, y=0.98)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.02, top=0.94, wspace=0.04, hspace=0.08)
    fig.savefig(output_dir / "dataset_sample_grid.png", dpi=180)
    plt.close(fig)


def prediction_gallery(predictions, output_dir):
    records = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    correct = []
    for class_name in sorted({record["actual"] for record in records}):
        candidates = [record for record in records if record["actual"] == class_name and record["correct"]]
        candidates.sort(key=lambda item: abs(item["confidence"] - 0.9))
        correct.append(candidates[0])
    errors = sorted((record for record in records if not record["correct"]), key=lambda item: item["confidence"], reverse=True)[:4]
    selected = correct + errors
    fig, axes = plt.subplots(2, 4, figsize=(13, 8.2))
    for axis, record in zip(axes.flat, selected):
        with Image.open(record["image_path"]) as image:
            axis.imshow(image.convert("RGB"))
        status = "Correct" if record["correct"] else "Error"
        axis.set_title(
            f"{status}: {record['predicted'].replace('_', ' ').title()}\n"
            f"True: {record['actual'].replace('_', ' ').title()} | {record['confidence']:.1%}",
            fontsize=9,
        )
        axis.axis("off")
    fig.suptitle("Representative held-out predictions", fontsize=14, y=0.98)
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.03, top=0.90, wspace=0.12, hspace=0.34)
    fig.savefig(output_dir / "prediction_gallery.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    class_distribution(args.manifest, args.output_dir)
    dataset_sample_grid(args.manifest, args.output_dir)
    prediction_gallery(args.predictions, args.output_dir)


if __name__ == "__main__":
    main()
