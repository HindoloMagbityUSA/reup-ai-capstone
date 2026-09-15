# Dataset license and attribution

The training data are from:

Liang, Y., Xia, Y., Merzouk, M., & Xu, Z. (2025). *Dataset for the article: From image to fire safety: A BIM-driven fire risk assessment method for existing buildings* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.18067669

Zenodo identifies the record as licensed under the [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/). The dataset may be shared and adapted, including commercially, provided appropriate attribution is given, the license is linked, and changes are indicated.

Changes made for this ReUP experiment:

- validated image readability and labels;
- excluded three records with empty labels;
- excluded six exact duplicate images, including three cross-split duplicates;
- resized/cropped images and applied training augmentation;
- trained a derived MobileNetV3-Small classifier and generated aggregate evaluation outputs.

The original archive was not modified. Its published MD5 checksum is `24f043c118e1ac63dbb8faf58597d2f6`.
