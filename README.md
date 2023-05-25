# Renderers are Good Zero-Shot Representation Learners: Exploring Diffusion Latents for Metric Learning
Code for paper ["Renderers are Good Zero-Shot Representation Learners: Exploring Diffusion
Latents for Metric Learning"](https://michaelwilliamtang.github.io/blog/papers/DiffusionRetriever.pdf) and joint work with David Shustin.

## Code
Notebooks for training and retrieval evaluation (+ reproduction of key figures) are available at ``train.ipynb`` and ``run.ipynb``, respectively. Training code relies on our adaptation ``SimCLR/`` of a
[PyTorch implementation of SimCLR](https://github.com/sthalles/SimCLR) to take in multiple embeddings from different views instead of transformations of single images.

## Data
For convenience, we uploaded the precomputed EfficientNet and Shap-E embeddings for the dataset of 300 scenes (20 images per scene) to Google Drive. Our precomputed database of embeddings can be found [here](https://drive.google.com/file/d/1l6iSZibWuvclYNO8mfWd4Z-l1MVX7cs2/view?usp=sharing).

To replicate these embeddings, the original ShapeNet SRN Cars dataset can be found [here](https://drive.google.com/drive/folders/1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR)
(maintained by the authors of PixelNeRF). The code for EfficientNet is available in torchvision, while the code for Shap-E is available [here](https://github.com/openai/shap-e)
(maintained by OpenAI), where [this notebook](https://github.com/openai/shap-e/blob/main/shap_e/examples/sample_image_to_3d.ipynb) is particularly helpful.
