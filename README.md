# Baseline CNN Model for SLB Disease Severity Regression

This repository contains the baseline deep learning model for predicting Southern Leaf Blight (SLB) disease severity from UAV RGB imagery.
Training using NC Plant Sciences Initiative's HPCs (NIVIDIA A100 GPUs)

---

## 📄 Contents

- `notebooks/baselineCNN.ipynb`:  
  Baseline CNN notebook including:
  - Simple CNN architecture
  - Training & validation loss visualization
  - Evaluation metrics: RMSE, MAE, R², Spearman correlation
  - Residual histogram
  - Auto GPU selection function
  - Learning rate scheduler
  - Early stopping

- `scr/baseline_dataloader.py`:  
  Custom PyTorch DataLoader using DataFrame splitting.

- `uav_regression_env.yaml`:  
  Conda environment file with dependencies.

---

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/chhammet/uav_for_slb_regression.git
git checkout main


conda env create -f environment/uav_regression_env.yaml
conda activate uav_regression_env

cd notebooks
jupyter notebook baselineCNN.ipynb

✅ Outputs
The notebook will generate:

Training & validation loss plots

Predicted vs actual score scatter plot

Histogram of residuals

Evaluation metrics summary

🟢 Notes
This is intended as a baseline model.
Further improvements (transfer learning, hyperparameter tuning, K-Fold CV) will be developed in separate branches.

For questions, contact Cole Hammett
