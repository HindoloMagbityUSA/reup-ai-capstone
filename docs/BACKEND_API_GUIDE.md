# ReUP Capstone Local Backend

## Purpose

The local backend demonstrates the implemented capstone framework without requiring access to ReUP's production backend. It provides:

- Three-model image analysis for material type, component type, and visible condition.
- Field-level confidence, confidence thresholds, alternatives, and mandatory human verification.
- A small governed demonstration catalog of A1-A3 factors.
- A deterministic A1-A3 product-stage calculation using verified inputs.
- A browser interface for the live demonstration and REST endpoints for future frontend integration.

## Start the demonstration

1. Double-click `SETUP_REUP_BACKEND.command` once if the local environment has not been created.
2. Double-click `START_REUP_BACKEND.command`.
3. The browser opens `http://127.0.0.1:8000/demo`.
4. Keep the terminal window open while presenting.
5. Press `Control+C` in the terminal to stop the backend.

The automatically generated API documentation is available at `http://127.0.0.1:8000/docs`.

## Recommended Tuesday demonstration

1. Upload `demo_images/concrete.jpg`.
2. Explain that three separately trained classifiers process the same image.
3. Point out each suggestion, confidence value, and threshold.
4. If component type is below threshold, show that the backend returns `Unable to determine` and requires manual selection.
5. Verify `Concrete`, select the appropriate component, verify the visible-condition cue, and check the verification box.
6. Select the concrete A1-A3 factor and enter `5` cubic metres.
7. Calculate the result. The expected screening estimate is `1,440 kg CO2e` because `5 × 288 = 1,440`.
8. Explain that the estimate represents potential avoided equivalent-new A1-A3 production impact and is not confirmed carbon saved.

## REST endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Confirms that all three models and the factor catalog are ready. |
| GET | `/api/v1/models/metrics` | Returns the held-out source-dataset evaluation metrics. |
| GET | `/api/v1/factors` | Returns approved demonstration A1-A3 factors. |
| POST | `/api/v1/analyze-image` | Accepts a JPEG or PNG and returns three correctable AI suggestions. |
| POST | `/api/v1/assessments/a1-a3` | Calculates verified quantity multiplied by a compatible A1-A3 factor. |
| GET | `/demo` | Opens the local demonstration interface. |
| GET | `/docs` | Opens interactive OpenAPI documentation. |

## Future ReUP frontend integration

The frontend can upload a file as multipart form data to `/api/v1/analyze-image`, display the returned fields for editing, and then submit the verified values, quantity, unit, and factor identifier to `/api/v1/assessments/a1-a3`.

Before production use, the backend requires authenticated ReUP identities, restricted CORS origins, persistent storage, HTTPS, production logging, rate limits, deployment monitoring, a broader governed factor catalog, and external validation on representative ReUP photographs.

## Demonstration factor boundary

The included concrete and steel factors are explicit examples from ReUP's February 2026 lifecycle reference document. They are marked as prototype reference factors and must not be presented as certified product-specific EPD values. A production implementation would retrieve an approved factor matched to material, component, unit, geography, source, and validity period.
