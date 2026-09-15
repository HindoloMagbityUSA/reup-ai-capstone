#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from api_app import app


ROOT = Path(__file__).resolve().parent.parent
client = TestClient(app)


def main() -> None:
    health = client.get("/health")
    assert health.status_code == 200, health.text
    assert health.json()["status"] == "ready"

    factors = client.get("/api/v1/factors").json()["factors"]
    assert len(factors) == 2
    demo_page = client.get("/demo")
    assert demo_page.status_code == 200
    assert "ReUP Explainable AI Listing Assistant" in demo_page.text

    image_path = ROOT / "demo_images" / "concrete.jpg"
    with image_path.open("rb") as stream:
        analysis = client.post(
            "/api/v1/analyze-image",
            files={"image": (image_path.name, stream, "image/jpeg")},
        )
    assert analysis.status_code == 200, analysis.text
    analysis_body = analysis.json()
    assert set(analysis_body["fields"]) == {
        "material_type", "component_type", "visible_condition"
    }

    assessment = client.post(
        "/api/v1/assessments/a1-a3",
        json={
            "analysis_id": analysis_body["analysis_id"],
            "verified_fields": {
                "material_type": "concrete",
                "component_type": analysis_body["fields"]["component_type"]["raw_prediction"],
                "visible_condition": analysis_body["fields"]["visible_condition"]["raw_prediction"],
                "verified_by_human": True,
            },
            "quantity": 5,
            "unit": "m3",
            "factor_id": "demo-concrete-a1a3-001",
        },
    )
    assert assessment.status_code == 200, assessment.text
    body = assessment.json()
    assert body["estimated_potential_avoided_a1_a3_kgco2e"] == 1440.0

    rejected = client.post(
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
    assert rejected.status_code == 422

    unit_mismatch = client.post(
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
    assert unit_mismatch.status_code == 422

    invalid_image = client.post(
        "/api/v1/analyze-image",
        files={"image": ("not-an-image.jpg", b"not an image", "image/jpeg")},
    )
    assert invalid_image.status_code == 400

    print(
        "Backend tests passed: health, demo page, factors, three-model inference, "
        "A1-A3 calculation, verification gate, unit validation, and image validation."
    )


if __name__ == "__main__":
    main()
