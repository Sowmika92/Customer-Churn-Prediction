# data_processing.py
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Load CSV (or parquet) into a pandas DataFrame."""
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def initial_inspect(df: pd.DataFrame) -> None:
    """Quick inspection prints — adapt or expand for logging."""
    print("Shape:", df.shape)
    print("Dtypes:\n", df.dtypes.value_counts())
    print("Missing per column:\n", df.isna().sum().sort_values(ascending=False).head(20))
    print("Examples:\n", df.head(3).T)

def clean_common_issues(df: pd.DataFrame, total_charges_col: str = "TotalCharges") -> pd.DataFrame:
    """Fix common formatting issues, e.g., TotalCharges as string, extra spaces."""
    df = df.copy()
    # Trim whitespace in string columns
    obj_cols = df.select_dtypes(include=["object"]).columns
    df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())
    # Convert TotalCharges to numeric if necessary (coerce errors -> NaN)
    if total_charges_col in df.columns:
        df[total_charges_col] = pd.to_numeric(df[total_charges_col], errors="coerce")
    # Drop duplicates if any
    df = df.drop_duplicates()
    return df

def prepare_target(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """Convert target to binary 0/1 if necessary."""
    df = df.copy()
    if target_col not in df.columns:
        raise KeyError(f"Target column {target_col} not found")
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].str.lower().map({"yes": 1, "no": 0})
    df[target_col] = df[target_col].astype(int)
    return df
