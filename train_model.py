# train_model.py
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from data_processing import load_data, clean_common_issues, prepare_target
from feature_engineering import TenureBinner, create_derived_features, build_preprocessing_pipeline

def evaluate_model(clf, X_test, y_test):
    preds_proba = clf.predict_proba(X_test)[:, 1]
    preds = (preds_proba >= 0.5).astype(int)
    print("ROC AUC:", roc_auc_score(y_test, preds_proba))
    print("Avg Precision (PR AUC):", average_precision_score(y_test, preds_proba))
    print("Classification report:\n", classification_report(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    # You can add threshold tuning based on business objective

def main(data_path: str, target_col: str = "Churn", save_path: str = "model.joblib"):
    df = load_data(data_path)
    print("Initial inspect:")
    from data_processing import initial_inspect
    initial_inspect(df)
    df = clean_common_issues(df)
    df = prepare_target(df, target_col=target_col)
    df = create_derived_features(df)
    # Select feature columns automatically (exclude id and target)
    exclude = {target_col, "customerID"}
    feature_cols = [c for c in df.columns if c not in exclude]
    # Example numeric and categorical split:
    numeric_features = df[feature_cols].select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = df[feature_cols].select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)
    preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)
    # Choose classifier — try a simple RandomForest baseline
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42)
    pipeline = Pipeline(steps=[
        ("tenure_binner", TenureBinner()),
        ("preproc", preprocessor),
        ("clf", clf)
    ])
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipeline.fit(X_train, y_train)
    evaluate_model(pipeline, X_test, y_test)
    joblib.dump(pipeline, save_path)
    print("Saved pipeline to", save_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV or parquet file")
    parser.add_argument("--target", default="Churn")
    parser.add_argument("--out", default="model.joblib")
    args = parser.parse_args()
    main(args.data, args.target, args.out)
