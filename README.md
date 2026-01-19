# Customer Churn Prediction

A reproducible end-to-end pipeline for preprocessing customer data, engineering features, training a churn prediction model, and exporting a production-ready pipeline.

This repository contains modular Python scripts you can adapt to your dataset to predict which customers are likely to churn.

## Table of contents
- [Overview](#overview)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Data expectations](#data-expectations)
- [Install](#install)
- [Quickstart (train)](#quickstart-train)
- [Inference example](#inference-example)
- [What each module does](#what-each-module-does)
- [Modeling & evaluation notes](#modeling--evaluation-notes)
- [Productionization tips](#productionization-tips)
- [Next steps](#next-steps)
- [License](#license)

## Overview
This project provides:
- Data loading and cleaning utilities
- Domain-aware feature engineering (e.g., tenure bins, avg charges)
- A preprocessing pipeline (imputation, scaling, encoding)
- A baseline model training script (Random Forest by default)
- Evaluation printing (ROC AUC, PR AUC, classification metrics)
- Export of the full pipeline (preprocessing + model) via joblib

The code is intentionally modular so you can extend each part (add more derived features, swap models, add hyperparameter tuning).

## Project structure
- data_processing.py — load, inspect, and clean raw data; prepare target
- feature_engineering.py — derived features and preprocessing pipeline builder
- train_model.py — orchestrates training, evaluation, and saving the pipeline
- requirements.txt — Python dependencies
- README.md — this file

(If you add a `data/` or `models/` directory, keep them out of version control or add them to .gitignore as appropriate.)

## Requirements
Python 3.8+ recommended.

Install dependencies:
```bash
pip install -r requirements.txt
```

requirements.txt includes:
- pandas
- numpy
- scikit-learn
- joblib
Optional (recommended for production/analysis):
- xgboost
- imbalanced-learn
- shap

## Data expectations
- Input: a CSV or Parquet file with one customer per row.
- Required/expected columns (examples — adapt to your dataset):
  - customerID (optional but recommended to identify rows)
  - Churn — target (values "Yes"/"No" or 1/0)
  - tenure (numeric months)
  - MonthlyCharges (numeric)
  - TotalCharges (numeric or string that can be coerced)
  - Contract, PaymentMethod, gender, SeniorCitizen, etc.
- The pipeline will:
  - Trim whitespace for string columns
  - Coerce `TotalCharges` to numeric (non-numeric -> NaN)
  - Convert `Churn` to binary 0/1 if provided as "Yes"/"No"

If your dataset uses different column names, either rename them before running or modify the scripts accordingly.

## Install
1. Clone repository:
```bash
git clone <repo-url>
cd <repo-dir>
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quickstart (train)
Train a pipeline on your data:
```bash
python train_model.py --data path/to/customers.csv --target Churn --out models/churn_pipeline.joblib
```
This will:
- Inspect the data, clean common issues
- Build derived features
- Construct a preprocessing pipeline and a RandomForest classifier
- Fit on train set (80/20 split stratified by churn)
- Print evaluation metrics for the test set
- Save the full pipeline at `models/churn_pipeline.joblib` (or `--out` you specify)

## Inference example
Load trained pipeline and score new customers:
```python
import joblib
import pandas as pd

pipeline = joblib.load("models/churn_pipeline.joblib")
new_customers = pd.read_csv("data/new_customers.csv")  # same columns as training features
proba = pipeline.predict_proba(new_customers)[:, 1]     # churn probability
# Append results
new_customers["churn_proba"] = proba
```

Tip: Always apply the same features and column names as used during training.

## What each module does
- data_processing.py
  - load_data(path): loads CSV or Parquet
  - initial_inspect(df): prints shape, dtypes, missing values
  - clean_common_issues(df): trims string columns, coerces TotalCharges to numeric, drops duplicates
  - prepare_target(df, target_col="Churn"): converts target to 0/1
- feature_engineering.py
  - TenureBinner transformer: creates tenure bins and numeric bin code
  - build_preprocessing_pipeline(numeric_features, categorical_features): returns ColumnTransformer with imputation, scaling and one-hot encoding
  - create_derived_features(df): example derived features (avg_charge_per_month, is_new_customer)
- train_model.py
  - Orchestrates full training pipeline, evaluation and saves the pipeline with joblib

## Modeling & evaluation notes
- Baseline model: RandomForestClassifier with `class_weight="balanced"`.
- Evaluate using:
  - ROC AUC and PR AUC (average precision)
  - Precision/Recall trade-offs and threshold selection based on business cost/benefit
  - Confusion matrix and classification report at chosen threshold
- If churn is rare:
  - Use stratified splits and StratifiedKFold for CV
  - Consider class_weight, focal loss, or resampling (SMOTE) inside a pipeline
  - Prefer PR-AUC and precision@k as metrics when the positive class is rare

## Interpretability & debugging
- Use global feature importance from tree models for a high-level view.
- Use SHAP for per-customer explanations (recommended for retention teams to understand why a customer is flagged).
- Track drift: monitor feature distributions and score distributions over time; retrain as data shifts.

## Productionization tips
- Save the full preprocessing + model pipeline (joblib) so inference uses identical transforms.
- Expose as a microservice that returns probability and top contributing features (via SHAP).
- Implement monitoring and alerting:
  - Data validation on incoming batches (missing columns, unexpected types)
  - Model performance monitoring (predicted churn rate vs observed churn)
- Maintain a retraining cadence or trigger retraining when drift is detected.

## Next steps
- Provide a sample of your data (first 10–20 rows or full schema) and I can:
  - Tailor cleaning/feature engineering to your data
  - Run a quick baseline and share evaluation and top features
  - Create a Jupyter notebook with plots and SHAP visuals
- Add hyperparameter tuning (RandomizedSearchCV) or try XGBoost/LightGBM for better performance.

## License
MIT (or update as needed)

If you want, I can generate a ready-to-run notebook version of the pipeline or scaffold a Dockerfile and a simple scoring endpoint next.
