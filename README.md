# Breast Cancer Treatment Response Prediction (MLP)
---
## Overview
This project implements a **multilayer perceptron (MLP)** model to predict **relapse-free survival time** for breast cancer patients prior to chemotherapy treatment. The model uses a combination of **clinical data** and **MRI-derived features**.
---
## Motivation
Predicting treatment response can help clinicians make **personalized treatment decisions** and improve patient outcomes.
---
## Methods
- Data preprocessing (including data imputation) on clinical and imaging datasets
- Applied **data augmentation** to expand training samples
- Implemented **dimensionality reduction** to remove redundant features
- Hyperparameter tuning to optimise model performance
- Regularisation techniques to reduce overfitting
- Explored **SMOTER-style regression oversampling** to address imbalanced outputs
---
## Results

| Run | Data Strategy | Mean Absolute Error (MAE) |
|-----|---------------|---------------------------|
| 1   | No augmentation | 24.43 |
| 2   | Gaussian noise augmentation | 27.26 |
| 3   | SMOTER regression oversampling | 24.63 |
| 4   | SMOTER + ensure inclusion of ER, HER2, and Gene features | 25.92 |
---
**Insights:**
- Regression oversampling (SMOTER) helped address imbalance, slightly improving performance over naïve augmentation.  
- Including key biological features (ER, HER2, Gene) is important but requires careful balancing.  
- Demonstrates ability to **experiment with multiple ML strategies and analyse results critically**.
---

## How to Run
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run `main.py` to train and evaluate the model

## Notes
This project was completed as part of MSc Machine Learning in Science coursework.


