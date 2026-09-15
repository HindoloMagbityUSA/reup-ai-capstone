#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from fastapi.testclient import TestClient

from api_app import app


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "backend_validation.json"


def main() -> None:
    client = TestClient(app)
    record = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "backend_version": app.version,
        "scope": "local capstone demonstration backend",
        "checks": {},
        "demo_image_smoke_test": [],
    }

    health = client.get("/health")
    record["checks"]["health_ready"] = (
        health.status_code == 200 and health.json().get("status") == "ready"
    )
    record["checks"]["three_models_loaded"] = all(
        value["loaded"] for value in health.json()["models"].values()
    )
    record["checks"]["factor_catalog_loaded"] = health.json()["factor_count"] == 2
    record["assessment_boundary"] = health.json()["assessment_boundary"]

    concrete_analysis = None
    for image_path in sorted((ROOT / "demo_images").glob("*.jpg")):
        start = perf_counter()
        with image_path.open("rb") as stream:
            response = client.post(
                "/api/v1/analyze-image",
                files={"image": (image_path.name, stream, "image/jpeg")},
            )
        elapsed = perf_counter() - start
        if response.status_code != 200:
            raise RuntimeError(f"Inference failed for {image_path.name}: {response.text}")
        body = response.json()
        fields = body["fields"]
        record["demo_image_smoke_test"].append(
            {
                "image": image_path.name,
                "elapsed_seconds": round(elapsed, 4),
                "suggestions": {
                    task: {
                        "status": field["status"],
                        "suggestion": field["suggestion"],
                        "raw_prediction": field["raw_prediction"],
                        "confidence": field["confidence"],
                    }
                    for task, field in fields.items()
                },
            }
        )
        if image_path.name == "concrete.jpg":
            concrete_analysis = body

    record["checks"]["all_demo_images_processed"] = (
        len(record["demo_image_smoke_test"]) == 9
    )
    if concrete_analysis is None:
        raise RuntimeError("Concrete demonstration analysis was not produced.")

    assessment = client.post(
        "/api/v1/assessments/a1-a3",
        json={
            "analysis_id": concrete_analysis["analysis_id"],
            "verified_fields": {
                "material_type": "concrete",
                "component_type": "column",
                "visible_condition": "no_visible_damage",
                "verified_by_human": True,
            },
            "quantity": 5,
            "unit": "m3",
            "factor_id": "demo-concrete-a1a3-001",
        },
    )
    if assessment.status_code != 200:
        raise RuntimeError(assessment.text)
    assessment_body = assessment.json()
    record["concrete_assessment_scenario"] = assessment_body
    record["checks"]["concrete_formula_reconciled"] = (
        assessment_body["estimated_potential_avoided_a1_a3_kgco2e"] == 1440.0
    )

    unverified = client.post(
        "/api/v1/assessments/a1-a3",
        json={
            "verified_fields": {
                "material_type": "concrete",
                "component_type": "column",
                "visible_condition": "no_visible_damage",
                "verified_by_human": False,
            },
            "quantity": 5,
            "unit": "m3",
            "factor_id": "demo-concrete-a1a3-001",
        },
    )
    record["checks"]["unverified_calculation_blocked"] = unverified.status_code == 422

    mismatch = client.post(
        "/api/v1/assessments/a1-a3",
        json={
            "verified_fields": {
                "material_type": "concrete",
                "component_type": "column",
                "visible_condition": "no_visible_damage",
                "verified_by_human": True,
            },
            "quantity": 5,
            "unit": "kg",
            "factor_id": "demo-concrete-a1a3-001",
        },
    )
    record["checks"]["unit_mismatch_blocked"] = mismatch.status_code == 422
    record["all_checks_passed"] = all(record["checks"].values())

    OUTPUT.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Validation record written to {OUTPUT}")
    print(json.dumps(record["checks"], indent=2))


if __name__ == "__main__":
    main()
