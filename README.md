# ReUP — Explainable AI for Construction Material Reuse

### Turning material images into verified information and transparent environmental estimates.

Reusable construction materials can retain value long after their first use. Yet companies often lack the information needed to assess them in time: what the material is, which component it belongs to, what condition is visible, and how it compares environmentally with an equivalent new product.

This capstone develops an **explainable, human-in-the-loop AI framework for ReUP** to address that information gap. It connects computer vision with human verification and a transparent environmental calculation, helping users move from an uploaded image to a structured, traceable screening assessment.

## The problem

Material reuse decisions depend on trustworthy information. In practice, that information is often scattered across photographs, incomplete inventories, and inconsistent records. Identifying a potentially reusable component is only the first step; users also need a credible basis for understanding its environmental implications.

An image prediction alone cannot provide that basis. Models can mistake surface appearance for material identity, confuse components, or express confidence in an incorrect result. Environmental comparisons introduce further questions about quantities, units, factor sources, and lifecycle boundaries.

**The project addresses both challenges: extracting useful preliminary information and making the resulting assessment understandable and open to review.**

## The solution

The framework gives AI, people, and environmental data distinct roles in one workflow:

1. **Upload a material image.** Three trained vision models suggest material type, component type, and visible condition.
2. **Review the evidence.** Confidence scores, ranked alternatives, and abstention help communicate uncertainty. Grad-CAM visualizations support inspection of model attention through the project's explanation tools.
3. **Verify the information.** A person reviews and corrects the suggested fields before proceeding.
4. **Select a compatible environmental factor.** The verified material and explicit quantity are paired with a governed factor in the matching unit.
5. **Calculate a traceable estimate.** A deterministic formula produces an equivalent-new-product A1–A3 impact, with the inputs and assumptions available for review.

```mermaid
flowchart LR
    A[Material image] --> B[AI suggestions]
    B --> C[Human review and correction]
    C --> D[Verified fields and quantity]
    D --> E[Compatible environmental factor]
    E --> F[Traceable A1–A3 estimate]
```

### A transparent calculation

A1–A3 covers the product stages of raw-material supply, transport to manufacturing, and manufacturing. Within that boundary, the demonstration uses:

**Equivalent-new-product impact = verified quantity × compatible A1–A3 factor**

For example, the backend validation reproduced:

**5 m³ of concrete × 288 kg CO₂e/m³ = 1,440 kg CO₂e**

This is a screening estimate using a demonstration factor. It provides a product-stage comparison baseline; actual net savings from reuse also depend on factors such as recovery, transport, processing, and the product being displaced.

## What the project demonstrates

- **Computer vision assistance:** three MobileNetV3-Small transfer-learning models for material, component, and visible-condition suggestions.
- **Human control:** correctable predictions and mandatory verification before assessment.
- **Explainability:** confidence information, ranked alternatives, abstention, and Grad-CAM analysis tools.
- **Governed calculation:** explicit environmental factors, quantity inputs, unit checks, and reproducible arithmetic.
- **An integrated prototype:** a FastAPI backend and local listing-assistant interface, with a separate Streamlit interface for exploring AI suggestions.

## Evaluation results

The models were evaluated on held-out tests from their respective source datasets.

| Task | Test samples | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Material type | 678 images | 97.49% | 97.49% |
| Component type | 3,269 HBD instances | 61.85% | 55.19% |
| Visible condition | 219 dacl1k images | 91.78% | 87.20% |

Material and visible-condition recognition showed stronger performance than component recognition. That variation is central to the design: suggestions remain subject to human review, and uncertain outputs can be withheld.

Controlled backend validation reproduced the calculation above and rejected unverified records, incompatible units, malformed images, and low-confidence component output. Detailed evidence is available in [combined model metrics](results/ai_layer_metrics.json), [backend validation results](results/backend_validation.json), and the [dataset and model card](docs/DATASET_AND_MODEL_CARD.md).

These results describe source-dataset performance. They do not establish accuracy on real ReUP marketplace photographs or demonstrate reduced user effort in deployment.

## Scope and next steps

This repository contains the original capstone prototype. Its material classifier covers **brick-and-mortar, concrete, steel, and timber**. Visible-condition suggestions describe image cues; they do not certify structural safety or suitability for reuse.

The contribution is a traceable decision-support workflow connecting uncertain AI observations to verified inputs and reproducible environmental estimates. Product-specific carbon claims require approved, applicable Environmental Product Declaration (EPD) factors and a defensible comparison baseline.

The next step is a limited ReUP pilot with representative marketplace images, approved EPD factors, and measurement of user corrections and review effort.

## Setup and usage

### One-time setup

Use Python 3.11 or 3.12. Clone the repository and open the entire `reup-ai-capstone` folder in VS Code, rather than only `src`, `data`, or `model`.

```bash
git clone https://github.com/HindoloMagbityUSA/reup-ai-capstone.git
cd reup-ai-capstone
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`. The commands below use macOS/Linux shell syntax; in PowerShell, set `$env:PYTHONPATH = "src"` once, then run each command without its `PYTHONPATH=src` prefix.

For the backend alone, install `requirements-backend.txt` instead. On macOS, double-click `SETUP_REUP_BACKEND.command` to set up that smaller environment.

### Run the local capstone backend

With the environment activated:

```bash
PYTHONPATH=src python -m uvicorn api_app:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/demo** for the listing assistant or **http://127.0.0.1:8000/docs** for interactive API documentation. On macOS, double-click `START_REUP_BACKEND.command` as an alternative.

The backend runs independently of ReUP's existing backend. Its interface supports image analysis, mandatory human verification, governed factor selection, and the deterministic A1–A3 estimate. See the [backend guide](docs/BACKEND_API_GUIDE.md) for the endpoint contract and demonstration sequence.

### Run the Streamlit interface

Install the full `requirements.txt` dependencies, activate the environment, and run:

```bash
PYTHONPATH=src streamlit run src/demo_app.py
```

On macOS, you can also double-click `START_REUP_DEMO.command`. If macOS prompts on first launch, Control-click the file and choose **Open**.

The interface accepts an image, displays the three AI suggestions, allows corrections, and requires an explicit verification checkbox. An unavailable checkpoint or insufficient confidence produces `Unable to determine` for the relevant field.

### Run combined command-line inference

```bash
PYTHONPATH=src python src/predict_all.py demo_images/concrete.jpg
```

## Reproduce training

Install the full `requirements.txt` dependencies and restore the source datasets using the [dataset instructions](data/README.md). The training commands below regenerate manifests for your local dataset paths. They write checkpoints and evaluation outputs into the specified model folders; use a separate working copy if you want to retain the bundled trained models.

### Train component recognition with HBD

After following the HBD instructions in [dataset instructions](data/README.md):

```bash
PYTHONPATH=src python src/prepare_hbd.py \
  --dataset data/HBD \
  --output data/component_manifest.csv

# Optional but strongly recommended on a CPU-only computer:
PYTHONPATH=src python src/cache_manifest_crops.py \
  --manifest data/component_manifest.csv \
  --output-manifest data/component_manifest_cached.csv \
  --output-dir data/component_crops

PYTHONPATH=src python src/train_task.py \
  --task component_type \
  --manifest data/component_manifest_cached.csv \
  --output-dir model/component
```

#If you skip crop caching, use `data/component_manifest.csv` instead of `data/component_manifest_cached.csv` in the training command.

### Train visible-condition recognition with dacl1k

Restore the dacl1k archive into `data/dacl1k` before running the following commands. Its native `annotations_v1.csv` file supplies the image labels. To regenerate the manifest and retrain the model:

```bash
PYTHONPATH=src python src/prepare_dacl1k.py \
  --dataset data/dacl1k \
  --output data/condition_manifest.csv

PYTHONPATH=src python src/train_task.py \
  --task visible_condition \
  --manifest data/condition_manifest.csv \
  --output-dir model/condition
```

Each training run saves a checkpoint, metrics, per-image predictions, learning curves, and a confusion matrix in its output folder.

## Verify the package

With the environment activated, run:

```bash
PYTHONPATH=src python src/test_component.py
PYTHONPATH=src python src/test_multimodel.py
PYTHONPATH=src python src/test_backend.py
```

The backend test exercises the demo page, three-model inference, environmental calculation, human-verification gate, unit validation, and invalid-image handling.

## Explore the repository

| Location | Contents |
|---|---|
| [`src/`](src/) | Inference, backend, interfaces, training, explanation tools, and tests |
| [`model/`](model/) | Three trained checkpoints and task-specific evaluation outputs |
| [`results/`](results/) | Metrics, validation records, and evaluation figures |
| [`demo_images/`](demo_images/) | Images for exploring the prototype |
| [`data/`](data/) | Dataset manifests, demonstration factors, and dataset instructions |
| [`docs/`](docs/) | API guide, model documentation, and dataset attribution |

The trained models and demo assets are included. Large training datasets and local Python environments are excluded. To reproduce training, follow the [dataset instructions](data/README.md), restore the source datasets, and regenerate manifests as needed for your local paths. See the [backend guide](docs/BACKEND_API_GUIDE.md) for the API workflow and the [dataset notices](docs/DATASET_LICENSES_COMPONENT_AND_CONDITION.md) for source terms.

## About the capstone

**An Explainable AI Framework for Environmental Impact Assessment of Reusable Construction Materials**

Hindolo Magbity · CS 687 Capstone · City University of Seattle · 2026
