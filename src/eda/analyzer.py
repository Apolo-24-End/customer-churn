import pandas as pd
import numpy as np
import json
import config


def run_eda(df: pd.DataFrame) -> dict:
    """Compute all EDA statistics and return as a JSON-serializable dict."""
    print("[eda] Running exploratory data analysis ...")
    results = {
        "overview": _overview(df),
        "churn_distribution": _churn_distribution(df),
        "churn_by_contract": _churn_by_category(df, "contract"),
        "churn_by_gender": _churn_by_category(df, "gender"),
        "churn_by_education": _churn_by_category(df, "education"),
        "churn_by_payment": _churn_by_category(df, "payment_method"),
        "churn_by_senior": _churn_by_category(df, "senior_citizen"),
        "numerical_stats": _numerical_stats(df),
        "age_distribution": _histogram(df, "age", bins=20),
        "income_distribution": _histogram(df, "annual_income", bins=20),
        "monthly_charges_distribution": _histogram(df, "monthlycharges", bins=20),
        "tenure_distribution": _histogram(df, "tenure", bins=20),
        "correlation_with_churn": _correlation_with_churn(df),
        "churn_by_tenure_group": _churn_by_tenure_group(df),
        "avg_monthly_charges_by_churn": _mean_by_churn(df, "monthlycharges"),
        "avg_satisfaction_by_churn": _mean_by_churn(df, "customer_satisfaction"),
        "null_summary": _null_summary(df),
    }
    _save(results)
    return results


def _overview(df: pd.DataFrame) -> dict:
    churn_rate = float(df[config.TARGET_COL].mean())
    return {
        "total_customers": len(df),
        "total_features": len(df.columns) - 1,
        "churn_count": int(df[config.TARGET_COL].sum()),
        "non_churn_count": int((df[config.TARGET_COL] == 0).sum()),
        "churn_rate": round(churn_rate * 100, 2),
    }


def _churn_distribution(df: pd.DataFrame) -> dict:
    counts = df[config.TARGET_COL].value_counts().to_dict()
    return {"labels": ["No Churn", "Churn"], "values": [int(counts.get(0, 0)), int(counts.get(1, 0))]}


def _churn_by_category(df: pd.DataFrame, col: str) -> dict:
    if col not in df.columns:
        return {}
    grouped = df.groupby(col)[config.TARGET_COL].agg(["mean", "count"]).reset_index()
    grouped.columns = [col, "churn_rate", "count"]
    grouped["churn_rate"] = (grouped["churn_rate"] * 100).round(2)
    return {
        "labels": grouped[col].astype(str).tolist(),
        "churn_rates": grouped["churn_rate"].tolist(),
        "counts": grouped["count"].tolist(),
    }


def _numerical_stats(df: pd.DataFrame) -> dict:
    num_cols = [c for c in config.NUMERICAL_COLS if c in df.columns]
    stats = df[num_cols].describe().round(2).to_dict()
    return {col: {k: float(v) for k, v in vals.items()} for col, vals in stats.items()}


def _histogram(df: pd.DataFrame, col: str, bins: int = 20) -> dict:
    if col not in df.columns:
        return {}
    counts, edges = np.histogram(df[col].dropna(), bins=bins)
    labels = [f"{edges[i]:.0f}–{edges[i+1]:.0f}" for i in range(len(edges) - 1)]
    return {"labels": labels, "values": counts.tolist()}


def _correlation_with_churn(df: pd.DataFrame) -> dict:
    num_cols = [c for c in config.NUMERICAL_COLS if c in df.columns]
    corr = df[num_cols + [config.TARGET_COL]].corr()[config.TARGET_COL].drop(config.TARGET_COL)
    corr_sorted = corr.abs().sort_values(ascending=False)
    return {
        "features": corr_sorted.index.tolist(),
        "correlations": [round(float(corr[f]), 4) for f in corr_sorted.index],
    }


def _churn_by_tenure_group(df: pd.DataFrame) -> dict:
    if "tenure" not in df.columns:
        return {}
    df = df.copy()
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 6, 12, 24, 36, 60, 999],
        labels=["0-6m", "6-12m", "1-2y", "2-3y", "3-5y", "5y+"],
    )
    grouped = df.groupby("tenure_group", observed=True)[config.TARGET_COL].mean().reset_index()
    return {
        "labels": grouped["tenure_group"].astype(str).tolist(),
        "churn_rates": (grouped[config.TARGET_COL] * 100).round(2).tolist(),
    }


def _mean_by_churn(df: pd.DataFrame, col: str) -> dict:
    if col not in df.columns:
        return {}
    means = df.groupby(config.TARGET_COL)[col].mean().round(2)
    return {"labels": ["No Churn", "Churn"], "values": [float(means.get(0, 0)), float(means.get(1, 0))]}


def _null_summary(df: pd.DataFrame) -> dict:
    null_counts = df.isnull().sum()
    nulls = null_counts[null_counts > 0]
    return {
        "columns": nulls.index.tolist(),
        "counts": nulls.values.tolist(),
        "percentages": [(n / len(df) * 100).__round__(2) for n in nulls.values],
    }


def _save(results: dict) -> None:
    out = config.OUTPUTS_DIR / "eda_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eda] Results saved to {out}")
