#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch


SOURCE_DIR = Path(__file__).resolve().parent
ROOT = SOURCE_DIR.parent


def main():
    model_path = ROOT / "model" / "best_model.pt"
    metrics_path = ROOT / "results" / "metrics.json"
    checkpoint = torch.load(model_path, map_location="cpu")
    assert checkpoint["class_names"] == ["brick_and_mortar", "concrete", "steel", "timber"]
    assert checkpoint["temperature"] > 0

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["split_counts"] == {"train": 3168, "val": 677, "test": 678}
    assert metrics["test"]["accuracy"] >= 0.95
    assert metrics["test"]["macro_f1"] >= 0.95

    sample = ROOT / "demo_images" / "timber.jpg"
    completed = subprocess.run(
        [sys.executable, str(SOURCE_DIR / "predict.py"), str(sample), "--model", str(model_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["verification_required"] is True
    assert result["model_scope"] == "material_type_only"
    assert result["material_suggestion"] in checkpoint["class_names"]
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
