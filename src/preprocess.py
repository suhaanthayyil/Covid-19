"""
Preprocessing for Long COVID prediction pipeline.
Supports: (1) Mexico/Kaggle schema, (2) PhysioNet COVID-19 in MS schema.
Loads raw CSV, keeps confirmed positives, creates LONG_COVID target,
handles unknown codes (97/98/99), encodes, scales, and applies SMOTE on train only.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# Column name variants (Kaggle Mexico dataset sometimes has typos)
CLASSIFICATION_COLS = ["CLASIFFICATION_FINAL", "CLASIFICATION_FINAL"]
FEATURE_COLS = [
    "AGE",
    "SEX",
    "PNEUMONIA",
    "DIABETES",
    "ASTHMA",
    "HIPERTENSION",
    "OBESITY",
    "CARDIOVASCULAR",
    "RENAL_CHRONIC",
    "TOBACCO",
]
# PhysioNet COVID-19 in MS: outcome 1+= hospitalised (Long COVID proxy); features available
PHYSIONET_OUTCOME_COL = "covid19_outcome_levels_2"
PHYSIONET_FEATURE_COLS = [
    "age_in_cat",
    "sex",
    "bmi_in_cat2",
    "com_diabetes",
    "com_hypertension",
    "com_chronic_kidney_disease",
    "com_cardiovascular_disease",
    "com_lung_disease",
    "current_or_former_smoker",
    "covid19_sympt_pneumonia",
]
UNKNOWN_CODES = (97, 98, 99)


def detect_data_source(df: pd.DataFrame) -> str:
    """Return 'physionet' or 'mexico' based on column names."""
    if PHYSIONET_OUTCOME_COL in df.columns:
        return "physionet"
    if "PATIENT_TYPE" in df.columns and any(c in df.columns for c in CLASSIFICATION_COLS):
        return "mexico"
    raise ValueError(
        "Unknown dataset schema. Expected Mexico (PATIENT_TYPE, CLASIFFICATION_FINAL) "
        "or PhysioNet (covid19_outcome_levels_2)."
    )


def find_classification_column(df: pd.DataFrame) -> str:
    """Return the actual classification column name (handles CLASIFFICATION vs CLASIFICATION)."""
    for c in CLASSIFICATION_COLS:
        if c in df.columns:
            return c
    raise KeyError(f"Expected one of {CLASSIFICATION_COLS} in columns: {list(df.columns)}")


def load_raw_data(data_dir: str = "data/raw", filename: str = None) -> pd.DataFrame:
    """
    Load the COVID-19 dataset from data/raw/.
    Tries: filename, then physionet_covid_ms.csv (real research data), then Mexico-style CSVs.
    """
    if filename:
        path = os.path.join(data_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Data file not found: {path}")
    else:
        candidates = [
            "Covid Data.csv",           # Kaggle Mexico (meirnizri/covid19-dataset) — prefer for research
            "covid19-dataset.csv",
            "COVID19_data.csv",
            "mexico_covid.csv",
            "physionet_covid_ms.csv",  # PhysioNet fallback
            "covid.csv",
        ]
        path = None
        for name in candidates:
            p = os.path.join(data_dir, name)
            if os.path.isfile(p):
                path = p
                break
        if path is None:
            raise FileNotFoundError(
                f"No dataset found in {data_dir}. Tried: {candidates}. "
                "Run: python scripts/download_real_data.py (see data/README.md)."
            )
    try:
        df = pd.read_csv(path, low_memory=False)
        print(f"Loaded {len(df):,} rows from {os.path.basename(path)}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load {path}: {e}") from e


def create_target_and_filter(df: pd.DataFrame, class_col: str) -> pd.DataFrame:
    """
    Mexico schema: Keep only confirmed positive COVID cases (CLASIFFICATION_FINAL in [1,2,3]).
    Create LONG_COVID = 1 if PATIENT_TYPE == 2 (hospitalized/returned), else 0.
    """
    df = df.copy()
    if class_col not in df.columns:
        raise KeyError(f"Column {class_col} not in dataframe")
    df = df[df[class_col].isin([1, 2, 3])].copy()
    if "PATIENT_TYPE" not in df.columns:
        raise KeyError("PATIENT_TYPE column required for Long COVID label")
    df["LONG_COVID"] = (df["PATIENT_TYPE"] == 2).astype(int)
    return df


def create_target_and_filter_physionet(df: pd.DataFrame, require_confirmed: bool = False) -> pd.DataFrame:
    """
    PhysioNet COVID-19 in MS: LONG_COVID = 1 if hospitalised (outcome 1 or 2).
    Outcome: 0 = not hospitalised, 1 = hospitalised, 2 = ICU/ventilation.
    If require_confirmed=True, keep only confirmed COVID cases; else use all rows with valid outcome
    (to retain enough positives for ML when confirmed subset is very small).
    """
    df = df.copy()
    if PHYSIONET_OUTCOME_COL not in df.columns:
        raise KeyError(f"PhysioNet outcome column {PHYSIONET_OUTCOME_COL} not found")
    if require_confirmed:
        if "covid19_confirmed_case" in df.columns:
            df = df[df["covid19_confirmed_case"].astype(str).str.lower().eq("yes")].copy()
        if "covid19_diagnosis" in df.columns:
            df = df[df["covid19_diagnosis"].astype(str).str.lower().eq("confirmed")].copy()
    outcome = pd.to_numeric(df[PHYSIONET_OUTCOME_COL], errors="coerce")
    df = df[outcome.notna()].copy()
    outcome = outcome[outcome.notna()]
    df["LONG_COVID"] = (outcome >= 1).astype(int)
    return df


def replace_unknown_with_nan(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Replace 97, 98, 99 (unknown) with NaN for numeric columns."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].replace(list(UNKNOWN_CODES), np.nan)
    return df


def _encode_physionet_value(val) -> float:
    """Encode PhysioNet yes/no, male/female, overweight/not_overweight to 0/1."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in ("yes", "1", "true", "male", "overweight"):
        return 1.0
    if s in ("no", "0", "false", "female", "not_overweight"):
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def prepare_features_and_target_physionet(
    df: pd.DataFrame,
    drop_pct_missing: float = 0.50,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    PhysioNet schema: select PHYSIONET_FEATURE_COLS, encode yes/no and sex to 0/1,
    drop rows with too many missing, fill median. Return X, y.
    Uses 50% missing threshold by default to retain more samples in small datasets.
    """
    available = [c for c in PHYSIONET_FEATURE_COLS if c in df.columns]
    if not available:
        raise KeyError(f"None of {PHYSIONET_FEATURE_COLS} found in dataframe")
    if "LONG_COVID" not in df.columns:
        raise KeyError("LONG_COVID target column required")
    X = df[available].copy()
    y = df["LONG_COVID"].copy()
    for col in X.columns:
        if X[col].dtype == object or X[col].dtype.name == "category":
            X[col] = X[col].apply(_encode_physionet_value)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    thresh = max(1, int((1 - drop_pct_missing) * len(X.columns)))
    valid = X.notna().sum(axis=1) >= thresh
    X = X.loc[valid].copy()
    y = y.loc[valid].copy()
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())
    return X, y


def prepare_features_and_target(
    df: pd.DataFrame,
    feature_cols: list = None,
    drop_pct_missing: float = 0.30,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Mexico schema: select features, replace unknown codes, drop rows with > drop_pct_missing missing,
    encode SEX 0/1, fill numeric NaNs with median, return X and y.
    """
    feature_cols = feature_cols or FEATURE_COLS
    df = df.copy()
    # Ensure we only use columns that exist
    available = [c for c in feature_cols if c in df.columns]
    missing_cols = set(feature_cols) - set(available)
    if missing_cols:
        print(f"Warning: Feature columns not found (dropped): {missing_cols}")
    if "LONG_COVID" not in df.columns:
        raise KeyError("LONG_COVID target column required")
    X = df[available].copy()
    y = df["LONG_COVID"].copy()
    # Replace unknown codes
    X = replace_unknown_with_nan(X, list(X.columns))
    # Drop rows with more than drop_pct_missing missing
    thresh = int((1 - drop_pct_missing) * len(X.columns))
    valid = X.notna().sum(axis=1) >= thresh
    X = X.loc[valid].copy()
    y = y.loc[valid].copy()
    # Encode SEX if present (assume 1=Male, 2=Female or similar; map to 0/1)
    if "SEX" in X.columns:
        uniq = X["SEX"].dropna().unique()
        if set(uniq).issubset({0, 1}):
            pass
        else:
            X["SEX"] = X["SEX"].map({1: 0, 2: 1}).fillna(X["SEX"].median())
    # Fill remaining NaN with median
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].fillna(X[col].median())
    return X, y


def train_test_split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    use_smote: bool = True,
) -> tuple:
    """
    Split into 80% train / 20% test, scale with StandardScaler on train,
    optionally apply SMOTE on training set only. Returns:
    X_train, X_test, y_train, y_test, scaler
    """
    # Stratify only if every class has at least 2 samples (else split fails)
    stratify_arg = y if (y.value_counts().min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    if use_smote:
        min_count = int(y_train.value_counts().min())
        k = min(5, min_count - 1) if min_count > 1 else 0
        if k >= 1:
            smote = SMOTE(random_state=random_state, k_neighbors=k)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE: train size {len(X_train):,} (balanced)")
        else:
            print(f"SMOTE skipped (minority class has {min_count} samples; need ≥2 for SMOTE).")
    return X_train, X_test, y_train, y_test, scaler


def run_full_preprocessing(
    data_dir: str = "data/raw",
    filename: str = None,
    drop_pct_missing: float = 0.30,
    test_size: float = 0.2,
    random_state: int = 42,
    use_smote: bool = True,
) -> tuple:
    """
    End-to-end: load → filter positives → create target → features → split → scale → SMOTE.
    Returns X_train, X_test, y_train, y_test, scaler, feature_names, df_original (for EDA).
    """
    df = load_raw_data(data_dir=data_dir, filename=filename)
    class_col = find_classification_column(df)
    df = create_target_and_filter(df, class_col)
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X, y = prepare_features_and_target(
        df, feature_cols=feature_cols, drop_pct_missing=drop_pct_missing, random_state=random_state
    )
    X_train, X_test, y_train, y_test, scaler = train_test_split_and_scale(
        X, y, test_size=test_size, random_state=random_state, use_smote=use_smote
    )
    return X_train, X_test, y_train, y_test, scaler, list(X.columns), df
