from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"

TARGET_COLUMN = "Machine failure"
IDENTIFIER_COLUMNS = ["UDI", "Product ID"]
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

BASE_NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
BASE_CATEGORICAL_FEATURES = ["Type"]
ENGINEERED_FEATURES = ["Temperature Difference [K]", "Torque Speed Product"]

RANDOM_STATE = 42


def load_dataset(path=DEFAULT_DATA_PATH):
    """Load the AI4I dataset from a local CSV file."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Place the CSV there or update DEFAULT_DATA_PATH."
        )

    return pd.read_csv(data_path)


def inspect_dataset(df):
    """Return a compact set of useful data-understanding checks."""
    return {
        "shape": df.shape,
        "data_types": df.dtypes,
        "missing_values": df.isna().sum(),
        "duplicate_count": int(df.duplicated().sum()),
        "target_distribution": df[TARGET_COLUMN].value_counts(dropna=False)
        if TARGET_COLUMN in df.columns
        else None,
        "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "numerical_columns": df.select_dtypes(include=["number"]).columns.tolist(),
    }


def clean_data(df):
    """Apply conservative cleaning without inventing transformations."""
    cleaned = df.copy()

    for column in BASE_NUMERIC_FEATURES + [TARGET_COLUMN]:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if "Type" in cleaned.columns:
        cleaned["Type"] = cleaned["Type"].astype("category")

    duplicate_count = cleaned.duplicated().sum()
    if duplicate_count:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    return cleaned


def add_engineered_features(df):
    """Create simple domain-motivated features used by the model."""
    engineered = df.copy()

    if {"Process temperature [K]", "Air temperature [K]"}.issubset(engineered.columns):
        engineered["Temperature Difference [K]"] = (
            engineered["Process temperature [K]"] - engineered["Air temperature [K]"]
        )

    if {"Torque [Nm]", "Rotational speed [rpm]"}.issubset(engineered.columns):
        engineered["Torque Speed Product"] = (
            engineered["Torque [Nm]"] * engineered["Rotational speed [rpm]"]
        )

    return engineered


def get_model_feature_columns(df):
    """Return model inputs after excluding identifiers and failure-mode indicators."""
    excluded_columns = set(IDENTIFIER_COLUMNS + FAILURE_MODE_COLUMNS + [TARGET_COLUMN])
    return [column for column in df.columns if column not in excluded_columns]


def split_train_validation_test(
    df,
    target_column=TARGET_COLUMN,
    test_size=0.15,
    validation_size=0.15,
    random_state=RANDOM_STATE,
):
    """Create stratified 70/15/15 train, validation, and test splits."""
    feature_columns = get_model_feature_columns(df)
    X = df[feature_columns]
    y = df[target_column]

    X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    validation_fraction = validation_size / (1 - test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=validation_fraction,
        stratify=y_train_validation,
        random_state=random_state,
    )

    return X_train, X_validation, X_test, y_train, y_validation, y_test
