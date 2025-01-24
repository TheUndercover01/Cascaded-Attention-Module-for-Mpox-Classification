# Mpox Classification with Cascaded Attention Module

## Overview
This repository contains the implementation of an advanced medical image classification model designed for precise Mpox detection. Developed at the **CANDLE Research Lab, IIT Roorkee**, this project leverages cutting-edge deep learning techniques, including a novel Cascaded Attention Module, to achieve high accuracy, optimized performance, and reduced computational complexity. The model has been rigorously tested on MSID dataset and outperforms six state-of-the-art models.

## Key Features
- **High Accuracy**: Achieved 97.5% accuracy on the MCSI dataset and 95.54% on the MSID dataset.
- **Model Efficiency**: Reduced parameters by 28%, ensuring lower latency and hardware efficiency.
- **Advanced Innovations**:
  - **Cascaded Attention Module**: Reduces redundant computations in multi-head self-attention, enhancing feature extraction and model interpretability.
  - **Focal Loss with Class-Specific Weights**: Effectively mitigates class imbalance, boosting minority class performance.

---

## Technical Details

### Cascaded Attention Module
The Cascaded Attention Module is a core innovation in this project, designed to optimize multi-head self-attention mechanisms:
- **Redundancy Reduction**: Eliminates duplicate attention across heads.
- **Computational Efficiency**: Reduces overhead by 2.2x.
- **Enhanced Feature Extraction**: Improves the model’s ability to learn meaningful features.

### Model Architecture
- **Backbone**: EfficientViT
  - Multi-scale feature representation for enhanced learning.
  - Hardware-efficient design with low latency.
- **Key Characteristics**:
  - Scalability for different datasets.
  - High accuracy with reduced computational demands.

### Loss Optimization
- **Technique**: Focal Loss with class-specific weights.
- **Benefits**:
  - Mitigates extreme class imbalance.
  - Ensures better learning for minority classes.
  - Improves overall classification metrics.

---

## Repository Structure

```
mpox-classification/
|
├── model.py                # Custom model and Cascaded Attention Module
├── loss.py                 # Loss function implementations
├── train_test_functions.py # Training and evaluation utilities
├── utils.py                # Data preprocessing, metrics, and helper functions
├── data_splitter.py        # Dataset splitting and management
├── requirements.txt        # Project dependencies
└── training.py             # Main training pipeline
```


---

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU for training (recommended)

### Installation Steps
1. Clone the repository:
```
   git clone https://github.com/yourusername/mpox-classification.git 
```
2. Navigate to the project directory:
```
   cd mpox-classification 
```
3. Navigate to the project directory:
```
   pip install -r requirements.txt 
```
4. Navigate to the project directory:
```
   python training.py
```
## Datasets

### MCSI Dataset
- Description: Primary dataset used for training and evaluation.
- Source: Medical image classification dataset.
- 4 Classes: Acne, Normal, Chickenpox, Mpox
### MSID Dataset

- Description: Secondary dataset used for validation.
- Source: Independent medical dataset for performance evaluation.
- 4 Classes: Chickenpox, Mpox, Normal, Measles

## Experimental Setup

- Cross-Validation on MCSI: 10-fold stratified sampling for robust evaluation.
- Evaluation Metrics: Accuracy, Precision, Recall, F1-Score

## Results

- **Accuracy**:
  - **MCSI Dataset**: 97.5%
  - **MSID Dataset**: 95.54%
- **Efficiency**: Achieved a **28% reduction** in model parameters compared to baseline models.
- **Comparative Performance**: Outperformed **six state-of-the-art models** in terms of accuracy and computational efficiency.

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

I extend our gratitude to:

- **IIT Roorkee** for providing the infrastructure and resources.  
- **Research Collaborators** for their valuable insights.  
- **Dataset Providers** for access to critical datasets.  

**Disclaimer**: This research is intended for academic and medical research purposes only.
