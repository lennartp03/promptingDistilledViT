# Prompt Engineering in knowledge distilled Vision Transformers (ViTs)

This repository is part of a bachelors thesis examining the usability of prompt engineering in distilled transformer models in CV.
The prompt engineering method used are adapter modules as proposed with Convpass. The models used are ViT and DeiT (sizes: B / S / T).

### Specific Versions

To ensure compatibility, the specific versions required are:

- `tfds-nightly`: 4.4.0.dev202201080107
- `torch`: 2.2.1
- `torchvision`: 0.17.1
- `tensorflow`: 2.15.0
- `tensorflow-addons`: 0.23.0
- `avalanche-lib`: 0.5.0
- `tqdm`: 4.66.2
- `timm`: 0.9.16

The python version used is 3.10.12. Please install or update this package accordingly.
(NOTE: `tensorflow-addons` might be deprecated and has a planned end of life in May 2024. Please check for updates on the package.)

### Acknowledgements

The core implementation of the Convpass modules is used by [Convolutional Bypasses Are Better Vision Transformer Adapters](https://github.com/JieShibo/PETL-ViT/blob/main/convpass/vtab/convpass.py). Own modifications in regard to the project have been made though.
The process of converting VTAB datasets from tensorflow_datasets library to a pytorch-acceptable representation was used by [Visual Prompt Tuning](https://github.com/KMnP/vpt/tree/main/src/data).
