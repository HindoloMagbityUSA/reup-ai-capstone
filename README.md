# ReUP AI Image-Information Extraction Layer

This package supports three human-verified listing suggestions from one uploaded image:

1. Material type
2. Component type
3. Visible condition cue

All three trained checkpoints, evaluation outputs, dataset manifests, and demonstration images are included. The large source training datasets are not included in this GitHub copy; restore them locally before running training or dataset-dependent evaluations.

The package also contains a standalone local FastAPI backend for the Tuesday demonstration. It does not require ReUP's current backend and exposes a REST contract that can be connected to ReUP's frontend later.

## Open this folder in VS Code

Open the entire cloned `reup-ai-capstone` folder. Do not open only `src`, `data`, or `model`.

## Folder structure

```text
reup-ai-capstone/
├── README.md
├── requirements.txt
├── requirements-backend.txt
├── SETUP_REUP_BACKEND.command
├── START_REUP_BACKEND.command
├── START_REUP_DEMO.command
├── data/
│   ├── Material-dataset/
│   ├── HBD/
│   ├── dacl1k/
│   ├── component_manifest.csv
│   ├── condition_manifest.csv
│   └── a1_a3_factors.json
├── demo_images/
├── docs/
├── model/
│   ├── best_model.pt
│   ├── component/best_model.pt
│   └── condition/best_model.pt
├── results/
└── src/
    ├── api_app.py
    ├── demo_app.py
    ├── test_backend.py
    ├── predict_all.py
    ├── prepare_hbd.py
    ├── prepare_dacl1k.py
    ├── train_task.py
    └── reup_ai/
```

## One-time setup

In the VS Code terminal:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Python 3.11 or 3.12 is required. On Windows, activate with `.venv\Scripts\activate`.

For the backend-only demonstration on macOS, double-click `SETUP_REUP_BACKEND.command` instead. It installs the smaller `requirements-backend.txt` environment.

## Run the local capstone backend

On macOS, double-click `START_REUP_BACKEND.command`. The terminal starts the backend and opens:

```text
http://127.0.0.1:8000/demo
```

The interface demonstrates image analysis, mandatory human verification, governed A1-A3 factor selection, and the deterministic product-stage estimate. Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

Terminal alternative:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m uvicorn api_app:app --host 0.0.0.0 --port 8000
```

See `docs/BACKEND_API_GUIDE.md` for the endpoint contract and recommended demonstration sequence.

## Run the live interface

On macOS, double-click `START_REUP_DEMO.command`. The first launch may ask macOS for permission to open the file. If that happens, Control-click it, choose **Open**, and confirm.

Alternatively, run it from the VS Code terminal:

```bash
source .venv/bin/activate
PYTHONPATH=src streamlit run src/demo_app.py
```

The interface accepts an image, displays the three AI suggestions, lets the presenter correct them, and requires an explicit verification checkbox. If a checkpoint is unavailable or confidence is too low, the corresponding field displays `Unable to determine`.

## Run combined command-line inference

```bash
PYTHONPATH=src python src/predict_all.py demo_images/concrete.jpg
```

## Train component recognition with HBD

After following the HBD instructions in `data/README.md`:

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

## Train visible-condition recognition with dacl1k

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

## Current held-out test results

| Model output | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Material type | 97.5% | 97.5% | 97.5% |
| Component type (HBD) | 61.9% | 70.9% | 55.2% |
| Visible condition (dacl1k) | 91.8% | 86.0% | 87.2% |

The complete material evaluation is in `results/metrics.json`. Detailed component and condition results are in their respective `model` subfolders. A concise combined record is in `results/ai_layer_metrics.json`. These values describe held-out images from the source datasets; they do not guarantee the same performance on ReUP user photographs.

## Verify the package

```bash
PYTHONPATH=src python src/test_component.py
PYTHONPATH=src python src/test_multimodel.py
```

## Model boundaries

The AI outputs are suggestions only. Material and component labels may be wrong when an image is unclear or outside the training distribution. The condition model checks for visible cracking, spalling, efflorescence, rust, and exposed reinforcement, but dacl1k primarily contains reinforced-concrete bridge imagery. It does not establish reuse suitability or provide a structural or safety assessment. Human verification remains mandatory before the three values populate the listing form.

## Repository packaging

This repository is a separate copy of the stable ReUP_AI_Demo_Package 2 capstone. Application code, model files, and evaluation results were copied without changes. The original local demo is preserved. Local Python environments, caches, macOS metadata, and the approximately 8 GB source training datasets are excluded. The dataset directories shown above describe the layout after restoring datasets; they are not bundled here. Dataset manifests retain their original paths and may require regeneration for training on another computer. No new redistribution license is asserted for third-party datasets or model assets.
