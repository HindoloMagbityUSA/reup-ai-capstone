# Dataset downloads and placement

Keep the original datasets inside this `data` folder. The original local working package includes them. This GitHub copy omits raw datasets because of their size and licensing terms; restore them before reproducing training.

## Material type

Download `Material-dataset.zip`, not `FF-dataset.zip`, from:

https://zenodo.org/records/18067669

- Approximate size: 204.2 MB
- MD5: `24f043c118e1ac63dbb8faf58597d2f6`

Extract it as `data/Material-dataset`.

## Component type — HBD

Use the official HBD download linked from:

https://github.com/EnochYing/Image2BIM/tree/main/HBD

Extract it as `data/HBD`. The preparation script accepts:

- HBD's native VGG Image Annotator (VIA) polygon JSON files;
- COCO annotations containing `images`, `annotations`, and `categories`; or
- YOLO annotations containing `data.yaml`, image folders, and matching label folders.

HBD contains the component classes wall, curtain wall, floor, ceiling, roof, column, beam, lift, window, door, and opening.

## Visible condition — dacl1k

Download `dacl1k.zip` from the official dataset source and extract it as `data/dacl1k`. The preparation script reads the native `annotations_v1.csv` file directly:

```bash
PYTHONPATH=src python src/prepare_dacl1k.py \
  --dataset data/dacl1k \
  --output data/condition_manifest.csv
```

The source archive is approximately 714 MB and contains 1,474 real bridge-inspection images. Its labels are No Damage, Crack, Efflorescence, Spalling, Bars Exposed, and Rust.

Important: the software repository is MIT-licensed, but its dacl1k dataset metadata currently states `license: TODO`. Do not publicly redistribute the raw dacl1k images without permission. Preserve all downloaded notices and cite the dataset paper.

## Final data folder

```text
data/
├── README.md
├── Material-dataset/
├── HBD/
└── dacl1k/
```

Do not rename or rearrange files inside the extracted source datasets.
