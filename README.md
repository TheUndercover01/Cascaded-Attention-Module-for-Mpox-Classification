# Mpox Classification with Cascaded Attention Module

## Overview
This repository contains the implementation of a cascaded attention module for Mpox skin-lesion classification, developed during a research internship at the **CANDLE Research Lab, IIT Roorkee** (supervisor: Prof. Sparsh Mittal). The module replaces the self-attention in blocks 1-6 of the fourth stage of an EfficientViT-L1 backbone and is trained with a class-weighted focal loss.

Project page with the architecture diagram and result figures: https://theundercover01.github.io/ayushdeshmukh/projects/cascaded-attention-module/

## Key Results (MCSI, 10-fold cross-validation)
- **Accuracy**: 97.5% +/- 0.77 (mean and sample standard deviation over 10 trials); precision 0.9789, recall 0.9711, F1 0.9742.
- **Comparison**: higher accuracy than six baselines trained on the same data: EfficientViT-L1 (97.25%), CoAtNet-1 (96.75%), DeiT3-medium (95.75%), ViT-B/16 (95.75%), MobileNetV3-L (93.75%) and ResNet101 (89.25%). The best baseline is only 0.25 points behind, and the baselines' fold-to-fold standard deviations are 1.8-5.4 points.
- **Parameters**: 13% fewer than EfficientViT-L1 (45.8M vs 52.7M, timm stock 1000-class head, counted with random weights).
- **FLOPs**: 5.5x fewer in the attention modules it replaces (0.86 -> 0.16 GFLOPs per 224x224 image). Whole-network saving is about 7% (10.5 -> 9.8 GFLOPs). Counted with PyTorch's `FlopCounterMode` (convolutions and matrix multiplies, two FLOPs per multiply-add).

The accuracy, precision, recall and F1 above come from the author's internship runs and are not saved in this repository. The baseline accuracies come from the per-model result files of the follow-up repository (A-Cascaded-Dilated-Convolution-Approach-for-Mpox-Lesion-Classification).

---

## Technical Details

### Cascaded Attention Module (`model.py`, `CascadedAttentionModule`)
- A 3x3 convolution expands the input from C to 3 x 16 channels and splits it into 3 heads of 16 channels.
- Each head runs two Q/K/V attention branches (a 3x3 then a 1x1 convolution giving Q, K and V of 8 channels each, then softmax attention over the H x W positions) and concatenates their outputs. The two branches use the same weights in all heads.
- The heads are concatenated, projected back to C channels with a 1x1 convolution, added to the input (residual) and batch-normalised (`affine=False`).
- Each head works on its own slice of the channels rather than the whole feature map, which cuts redundant computation in multi-head self-attention.

### Model Architecture
- **Backbone**: EfficientViT-L1 (`efficientvit_l1.r224_in1k` from timm), with the module swapped in by `replace_context_modules` in `utils.py`.
- **Classifier head**: fully connected layers, set by `change_classifier` in `model.py`.

### Loss and Training (`loss.py`, `main.py`)
- **Loss**: focal loss (gamma = 2) with class weights.
- **Optimiser**: AdamW, learning rate 9.6e-5, weight decay 0.01, exponential LR decay (gamma = 0.95).
- **Schedule**: up to 55 epochs, batch size 16, early stopping on validation accuracy (patience 9).

---

## Repository Structure

```
.
|-- model.py                 # Cascaded Attention Module and classifier head
|-- loss.py                  # Focal loss and early stopping
|-- train_test_functions.py  # Training and evaluation loops
|-- utils.py                 # Module swap, metrics and helper functions
|-- data_splitter.py         # Fold splitting
|-- main.py                  # Main training pipeline (10-fold cross-validation)
|-- model.ipynb              # Notebook with the model code
|-- backbone.ipynb           # Notebook with the backbone code
`-- requirments              # Dependencies (file name as committed)
```

---

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU for training (recommended)

### Installation Steps
1. Clone the repository:
```
git clone https://github.com/TheUndercover01/Cascaded-Attention-Module-for-Mpox-Classification.git
cd Cascaded-Attention-Module-for-Mpox-Classification
```
2. Install the dependencies:
```
pip install -r requirments
```
3. Run the training pipeline:
```
python main.py
```

## Dataset

### MCSI (Mpox Close Skin Image) Dataset
- 4 classes: Acne, Normal, Chickenpox, Mpox.
- 400 images, class-balanced, arranged as 10 folds of 40 images for cross-validation.
- `main.py` expects the preprocessed arrays `X_train_final_multi_10_folds_40_each_equal.npy` and `y_train_final_multi_10_folds_40_each__equal.npy`. The preprocessing and the arrays are in the follow-up repository, not here.

## Experimental Setup
- Cross-validation on MCSI: 10 folds.
- Evaluation metrics: accuracy, macro precision, macro recall and macro F1.

## Citation

If you find this work useful in your research, please consider citing:
```
@research{MpoxClassification2024,
  author    = {Ayush Deshmukh, Sparsh Mittal},
  title     = {Mpox Classification with Cascaded Attention Module},
  year      = {2024},
  institute = {CANDLE Research Lab, IIT Roorkee}
}
```
## Contact

- Institution: CANDLE Research Lab, IIT Roorkee
- Email: [aad.211it014@nitk.edu.in]
- Research Supervisor: Assoc. Prof. Sparsh Mittal

## License

This project is licensed under the MIT License.

## Acknowledgements

I extend my gratitude to:

- **IIT Roorkee** for providing the infrastructure and resources.  
- **Research Collaborators** for their valuable insights.  
- **Dataset Providers** for access to critical datasets.  

**Disclaimer**: This research is intended for academic and medical research purposes only.
