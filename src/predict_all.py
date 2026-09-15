#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from reup_ai.modeling import predict_classifier


CODE_ROOT = Path(__file__).resolve().parent
ROOT = CODE_ROOT.parent if (CODE_ROOT.parent / "model").is_dir() else CODE_ROOT
DEFAULT_MODELS = {
    "material_type": ROOT / "model" / "best_model.pt",
    "component_type": ROOT / "model" / "component" / "best_model.pt",
    "visible_condition": ROOT / "model" / "condition" / "best_model.pt",
}
DEFAULT_THRESHOLDS = {
    "material_type": 0.65,
    "component_type": 0.60,
    "visible_condition": 0.65,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Suggest three ReUP listing fields from one image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--material-model", type=Path, default=DEFAULT_MODELS["material_type"])
    parser.add_argument("--component-model", type=Path, default=DEFAULT_MODELS["component_type"])
    parser.add_argument("--condition-model", type=Path, default=DEFAULT_MODELS["visible_condition"])
    return parser.parse_args()


def run_all(image_path: Path, model_paths: dict[str, Path] | None = None):
    paths = model_paths or DEFAULT_MODELS
    result = {
        "image": str(image_path),
        "verification_required": True,
        "structural_or_safety_assessment": False,
        "fields": {},
    }
    for task, model_path in paths.items():
        if not model_path.exists():
            result["fields"][task] = {
                "status": "model_not_trained",
                "suggestion": "unable_to_determine",
                "verification_required": True,
            }
            continue
        prediction = predict_classifier(image_path, model_path)
        threshold = DEFAULT_THRESHOLDS[task]
        accepted = prediction["confidence"] >= threshold
        result["fields"][task] = {
            "status": "suggested" if accepted else "low_confidence",
            "suggestion": prediction["suggestion"] if accepted else "unable_to_determine",
            "raw_prediction": prediction["suggestion"],
            "confidence": prediction["confidence"],
            "confidence_threshold": threshold,
            "probabilities": prediction["probabilities"],
            "verification_required": True,
        }
    return result


def main():
    args = parse_args()
    paths = {
        "material_type": args.material_model,
        "component_type": args.component_model,
        "visible_condition": args.condition_model,
    }
    print(json.dumps(run_all(args.image, paths), indent=2))


if __name__ == "__main__":
    main()
