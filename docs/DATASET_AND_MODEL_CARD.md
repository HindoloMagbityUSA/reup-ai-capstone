# ReUP material-type image classifier

## Intended use

This model suggests one of four material types from a ReUP listing image: brick and mortar, concrete, steel, or timber. It returns calibrated probabilities and a confidence score. The listing owner must verify or correct the suggestion before it can be used by any environmental-factor lookup or A1–A3 calculation.

The model does not identify component type, assess condition, estimate quantity, select an EPD, or calculate environmental impact.

## Dataset

- Source: *Dataset for the article: From image to fire safety: A BIM-driven fire risk assessment method for existing buildings*
- Authors: Yangze Liang, Yuhui Xia, Mina Merzouk, and Zhao Xu
- Zenodo record and DOI: https://doi.org/10.5281/zenodo.18067669
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Downloaded file: `Material-dataset.zip`
- Published checksum: `md5:24f043c118e1ac63dbb8faf58597d2f6`
- Source description: 4,532 images in predefined 70/15/15 train, validation, and test splits.

The reproducible audit found 4,529 records with non-empty valid labels. It excluded six exact duplicate images, including three cross-split duplicate groups. The final manifest contains 4,523 unique usable images: 3,168 train, 677 validation, and 678 test.

## Model and training

- Architecture: MobileNetV3-Small initialized from official ImageNet weights
- Input: 192 × 192 RGB image
- Phase 1: frozen convolutional feature extractor; trained classification head
- Phase 2: fine-tuned the final three feature blocks and classification head
- Optimization: AdamW with cosine learning-rate decay
- Regularization: random crop, horizontal flip, rotation, color jitter, weight decay, and label smoothing
- Reproducibility seed: 42
- Selection: lowest validation loss; the test split was not used for training or checkpoint selection
- Calibration: scalar temperature fitted on validation logits

## Held-out evaluation

The deduplicated held-out test set contains 678 images. The selected model achieved:

- Accuracy: 97.49%
- Balanced accuracy: 97.49%
- Macro precision: 97.48%
- Macro recall: 97.49%
- Macro F1: 97.49%
- Expected calibration error: 1.23%

Class F1 scores were 99.36% for brick and mortar, 96.36% for concrete, 95.55% for steel, and 98.67% for timber. Steel was the most difficult class and was most often confused with concrete.

## Governance and limitations

This is a bounded proof-of-concept classifier, not an autonomous material-verification system. Visual inspection shows that the source is dominated by close-up material textures and installed building surfaces, which may differ substantially from user-generated ReUP listing photos of loose or salvaged components. The source's `Brick&Mortar` class also contains stone-masonry surfaces. That label ambiguity is material to environmental-factor selection because masonry products are not interchangeable. Performance therefore needs external validation on representative ReUP images and a domain-expert taxonomy review before deployment.

The confidence score is calibrated on the source validation distribution and should not be treated as a guarantee for out-of-distribution images. Images outside the four known classes can still receive a high softmax probability. The interface consequently marks every prediction as `verification_required: true`; it must not silently convert an AI suggestion into a verified material record.

Grad-CAM heatmaps are local visual explanations showing influential image regions. They help inspect model attention but do not prove causal reasoning or material correctness.

The reported held-out performance measures generalization within this dataset and its source-label definitions. It should not be interpreted as 97.49% accuracy on future ReUP listings.
