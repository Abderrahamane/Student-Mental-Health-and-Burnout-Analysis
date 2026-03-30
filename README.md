# Student Mental Health and Burnout Analysis

This project analyzes the **Student Mental Health and Burnout Dataset** (150K rows, 20 features) and predicts `burnout_level` (`Low`, `Medium`, `High`) using classical ML models.

## What was done in the notebook
- Downloaded the dataset from Kaggle.
- Performed preprocessing (category mapping, one-hot encoding for course, dropped `student_id`).
- Added engineered features (for example: mental health score, lifestyle balance, pressure ratio, total load).
- Trained and evaluated:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Checked target correlation with numeric features.

## What happened
All models achieved around **33% accuracy**, which is close to random guessing for a 3-class problem.

## Why models are only guessing
The dataset is **synthetic** (as noted by the dataset author), and feature-target relationships appear very weak.
- Correlations with `burnout_level` are near zero.
- Class predictions are close to evenly distributed.
- There is little learnable signal linking inputs to the target.

## Takeaway
For this dataset, model complexity does not help much: even stronger models (like XGBoost) cannot outperform random-like baseline behavior when the target signal is weak.

