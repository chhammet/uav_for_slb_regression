
# Baseline CNN Model for SLB Disease Severity Regression

This repository contains the baseline deep learning model for predicting Southern Leaf Blight (SLB) disease severity from UAV RGB imagery.

Set to run on NC PSI's remote server sunny with NVIDIA A100 80GB GPUs

## 📄 Contents
- `notebooks/baselineCNN.ipynb`: Jupyter notebook containing the baseline model with:
  - Simple CNN architecture
  - Training & validation loss visualization
  - Evaluation metrics: RMSE, MAE, R², Spearman correlation
  - Residuals histogram
  - Auto GPU selection
  - Learning rate scheduler
  - Early stopping

### The notebook will generate:

    Training & validation loss plots
    Predicted vs actual score scatter plot
    Histogram of residuals
    Evaluation metrics summary

- `scr/baseline_analysis.py`: Data loader script using DataFrame splits.
- `scr/utils.py`: GPU auto selection and early stopping functions

- `uav_regression_env.yaml`: Conda environment file.

- `requirements.txt`: Minimal Python package requirements.

## 🚀 Quick Start

1. **Clone the repository**

git clone (https://github.com/chhammet/uav_for_slb_regression.git)
git checkout baseline


conda env create -f environment/uav_regression_env.yaml
conda activate uav_regression_env

cd notebooks
jupyter notebook baselineCNN.ipynb

This is intended as a baseline model.
Further improvements (transfer learning, hyperparameter tuning, K-Fold CV) will be developed in separate branches.

To-Do:
Improve GPU utilization

Parameter	            Setting
batch_size	            512 (try 256 first, if stable → 512)
num_workers	            8–12
pin_memory	            True
persistent_workers	    True (optional, improves large DataLoader startup)
Mixed Precision	Highly  Recommended (your A100 is built for FP16)

For questions, contact Cole Hammett

