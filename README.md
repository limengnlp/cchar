# cchar
This project is for *Learning Part-whole Hierarchies from the Sequence of Handwriting*.

- **Data**
  - Please download CChar dataset from [google drive](https://drive.google.com/drive/folders/1TbChyaGb5_XIAyeK70mkOajRZVZ8vNVO).
  - It includes 
    - $images$: 73,086 images for training visual feature extractors;
    - $annotations$: annotations for image classification and sequence generation.
    - $feats$: VGG/MAE/ViT visual feature files for sequence generation.

- **Experiment**:
  - Visual feature extraction:
    - MAE(ViT-based) models and visual features are based on [official code](https://github.com/facebookresearch/mae).
  - Sequence generation:
    - Place different feature zip files under `./cchar_seq/data/feats` and unzip them. For example, `/cchar_seq/data/feats/tinyvgg_256`.
    - Place different json files (random splits for train/val/test) under `./cchar_seq/data` for experiment 3. 

- Versions:
  - python>=3.7
  - pytorch=1.13.1