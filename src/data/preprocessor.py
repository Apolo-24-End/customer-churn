import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import config
from src.features.engineer import engineer_features


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean, engineer features, encode and scale the raw DataFrame. Returns (X, y)."""
    df = df.copy()

    # Extract temporal features from signup_date before dropping it
    date_cols: list[str] = []
    if config.DATE_COL in df.columns:
        dates = pd.to_datetime(df[config.DATE_COL], errors="coerce")
        ref_date = pd.Timestamp.now().normalize()
        df["signup_month"] = dates.dt.month.fillna(6).astype(int)
        df["signup_quarter"] = dates.dt.quarter.fillna(2).astype(int)
        df["days_since_signup"] = (ref_date - dates).dt.days.clip(lower=0).fillna(0).astype(int)
        date_cols = ["signup_month", "signup_quarter", "days_since_signup"]
        joblib.dump(ref_date, config.MODELS_DIR / "signup_ref_date.joblib")

    # Drop columns not useful for modeling
    drop_cols = [config.ID_COL, config.DATE_COL]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Compute and save medians before imputation (needed at inference time)
    numerical_medians: dict = {}
    if "credit_score" in df.columns:
        cs_median = float(df["credit_score"].median())
        numerical_medians["credit_score"] = cs_median
        df["credit_score_missing"] = df["credit_score"].isnull().astype(int)
        df["credit_score"] = df["credit_score"].fillna(cs_median)

    for col in config.NUMERICAL_COLS:
        if col in df.columns:
            col_median = float(df[col].median())
            numerical_medians[col] = col_median
            if df[col].isnull().any():
                df[col] = df[col].fillna(col_median)

    # Feature engineering (must run BEFORE encoding so binary cols are still 0/1 int)
    df, engineered_cols = engineer_features(df)
    print(f"[preprocessor] Engineered {len(engineered_cols)} new features: {engineered_cols}")

    # Encode categorical columns
    encoders = {}
    for col in config.CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # Binary columns: ensure int
    for col in config.BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Separate target
    y = df.pop(config.TARGET_COL).astype(int)
    X = df

    # Scale: original numerical cols + date cols + engineered numerical cols + credit_score_missing
    base_num_cols = config.NUMERICAL_COLS + ["credit_score_missing"] + date_cols
    num_cols_present = [c for c in base_num_cols + engineered_cols if c in X.columns]
    scaler = StandardScaler()
    X[num_cols_present] = scaler.fit_transform(X[num_cols_present])

    # Persist artifacts
    joblib.dump(encoders, config.MODELS_DIR / "encoders.joblib")
    joblib.dump(scaler, config.MODELS_DIR / "scaler.joblib")
    joblib.dump(engineered_cols, config.MODELS_DIR / "engineered_cols.joblib")
    joblib.dump(list(X.columns), config.MODELS_DIR / "feature_names.joblib")
    joblib.dump(numerical_medians, config.MODELS_DIR / "numerical_medians.joblib")

    print(f"[preprocessor] X shape: {X.shape} | Churn rate: {y.mean():.3f}")
    return X, y


def split(X: pd.DataFrame, y: pd.Series):
    """Stratified train/test split."""
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )


def preprocess_single(data: dict) -> pd.DataFrame:
    """Preprocess a single customer dict for inference."""
    encoders = joblib.load(config.MODELS_DIR / "encoders.joblib")
    scaler = joblib.load(config.MODELS_DIR / "scaler.joblib")
    engineered_cols = joblib.load(config.MODELS_DIR / "engineered_cols.joblib")
    feature_names = joblib.load(config.MODELS_DIR / "feature_names.joblib")
    medians_path = config.MODELS_DIR / "numerical_medians.joblib"
    numerical_medians = joblib.load(medians_path) if medians_path.exists() else {}

    row = pd.DataFrame([data])

    # Temporal features from signup_date
    date_cols: list[str] = []
    ref_date_path = config.MODELS_DIR / "signup_ref_date.joblib"
    if config.DATE_COL in row.columns:
        ref_date = joblib.load(ref_date_path) if ref_date_path.exists() else pd.Timestamp.now()
        dates = pd.to_datetime(row[config.DATE_COL], errors="coerce")
        row["signup_month"] = dates.dt.month.fillna(6).astype(int)
        row["signup_quarter"] = dates.dt.quarter.fillna(2).astype(int)
        row["days_since_signup"] = (ref_date - dates).dt.days.clip(lower=0).fillna(0).astype(int)
        date_cols = ["signup_month", "signup_quarter", "days_since_signup"]
    elif ref_date_path.exists():
        # Training used date features — inject median defaults so the scaler doesn't fail
        row["signup_month"] = 6
        row["signup_quarter"] = 2
        row["days_since_signup"] = 0
        date_cols = ["signup_month", "signup_quarter", "days_since_signup"]

    # Drop non-feature columns if present
    row.drop(columns=[c for c in [config.ID_COL, config.DATE_COL, config.TARGET_COL]
                      if c in row.columns], inplace=True)

    # credit_score null flag
    if "credit_score" in row.columns:
        row["credit_score_missing"] = row["credit_score"].isnull().astype(int)
        row["credit_score"] = row["credit_score"].fillna(numerical_medians.get("credit_score", 0))

    # Impute remaining numerical nulls with training medians
    for col in config.NUMERICAL_COLS:
        if col in row.columns:
            row[col] = row[col].fillna(numerical_medians.get(col, 0))

    # Feature engineering (same transformations applied at training time)
    row, _ = engineer_features(row)

    # Encode categoricals
    for col, le in encoders.items():
        if col in row.columns:
            val = str(row[col].iloc[0])
            if val in le.classes_:
                row[col] = le.transform([val])[0]
            else:
                row[col] = 0

    # Binary to int
    for col in config.BINARY_COLS:
        if col in row.columns:
            row[col] = row[col].astype(int)

    # Apply target encoding if available (adds te_{col} columns)
    te_path = config.MODELS_DIR / "target_encoder.joblib"
    if te_path.exists():
        from src.features.target_encoder import MeanTargetEncoder
        te = joblib.load(te_path)
        row = te.transform(row)

    # Ensure all base training features are present (te_* cols added by target encoder are kept)
    for col in feature_names:
        if col not in row.columns:
            row[col] = 0

    # Scale: same columns as training (original numerical + date cols + engineered)
    base_num_cols = config.NUMERICAL_COLS + ["credit_score_missing"] + date_cols
    num_cols_present = [c for c in base_num_cols + engineered_cols if c in row.columns]
    row[num_cols_present] = scaler.transform(row[num_cols_present])

    return row
