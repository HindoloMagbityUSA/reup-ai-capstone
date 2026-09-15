#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from predict_all import run_all


ROOT = Path(__file__).resolve().parent


def main():
    package_root = ROOT.parent if (ROOT.parent / "demo_images").is_dir() else ROOT.parent / "ReUP_AI_Demo_Package"
    image = package_root / "demo_images" / "concrete.jpg"
    result = run_all(image)
    assert result["verification_required"] is True
    assert result["structural_or_safety_assessment"] is False
    assert set(result["fields"]) == {"material_type", "component_type", "visible_condition"}
    assert result["fields"]["material_type"]["status"] in {"suggested", "low_confidence"}
    assert result["fields"]["component_type"]["status"] in {"suggested", "low_confidence"}
    assert result["fields"]["visible_condition"]["status"] in {"suggested", "low_confidence"}
    print(json.dumps(result, indent=2))

    component_example = package_root / "demo_images" / "component_door.jpg"
    component_result = run_all(component_example)
    assert component_result["fields"]["component_type"]["status"] == "suggested"
    assert component_result["fields"]["component_type"]["suggestion"] == "door"
    print(json.dumps(component_result, indent=2))

    for filename, expected in (
        ("condition_no_visible_damage.jpg", "no_visible_damage"),
        ("condition_visible_damage.jpg", "visible_damage_detected"),
    ):
        condition_result = run_all(package_root / "demo_images" / filename)
        assert condition_result["fields"]["visible_condition"]["status"] == "suggested"
        assert condition_result["fields"]["visible_condition"]["suggestion"] == expected
        print(json.dumps(condition_result, indent=2))


if __name__ == "__main__":
    main()
