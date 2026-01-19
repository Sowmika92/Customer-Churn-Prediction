# feature_engineering.py
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

class TenureBinner(TransformerMixin, BaseEstimator):
    """Create tenure bins and a numeric tenure group feature."""
    def __init__(self, bins=(0, 12, 24, 48, 72, np.inf)):
        self.bins = bins
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        if "tenure" not in X.columns:
            return X
        labels = [f"t{i}" for i in range(1, len(self.bins))]
        X["tenure_bin"] = pd.cut(X["tenure"].astype(float), bins=self.bins, labels=labels, include_lowest=True)
        # Numeric encoding for tier
        X["tenure_bin_code"] = X["tenure_bin"].cat.codes
        return X

def build_preprocessing_pipeline(numeric_features, categorical_features):
    """Returns a ColumnTransformer that imputes and encodes features."""
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features)
    ], remainder="drop")
    return preprocessor

def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-specific features. Modify according to available columns."""
    df = df.copy()
    # Example: average charges per month (guard divide by zero)
    if "TotalCharges" in df.columns and "tenure" in df.columns:
        df["avg_charge_per_month"] = df["TotalCharges"] / df["tenure"].replace(0, np.nan)
        df["avg_charge_per_month"] = df["avg_charge_per_month"].fillna(df["MonthlyCharges"])
    # Example: flag for short tenure
    if "tenure" in df.columns:
        df["is_new_customer"] = (df["tenure"] <= 3).astype(int)
    return df
