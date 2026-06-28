import pandas as pd
import config


def load_raw() -> pd.DataFrame:
    """Load the raw CSV and return a validated DataFrame."""
    print(f"[loader] Reading {config.DATA_FILE} ...")
    df = pd.read_csv(config.DATA_FILE, parse_dates=[config.DATE_COL], low_memory=False)
    print(f"[loader] Shape: {df.shape}")
    _report_nulls(df)
    return df


def _report_nulls(df: pd.DataFrame) -> None:
    null_counts = df.isnull().sum()
    nulls = null_counts[null_counts > 0]
    if nulls.empty:
        print("[loader] No null values found.")
    else:
        print("[loader] Columns with nulls:")
        for col, n in nulls.items():
            print(f"         {col}: {n} ({n / len(df) * 100:.1f}%)")


def get_null_summary(df: pd.DataFrame) -> dict:
    null_counts = df.isnull().sum()
    return {
        col: {"count": int(n), "pct": round(n / len(df) * 100, 2)}
        for col, n in null_counts.items()
        if n > 0
    }
