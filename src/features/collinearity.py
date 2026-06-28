"""
Multicollinearity analysis and intelligent feature pruning.

Usage (standalone):
    python -m src.features.collinearity

Usage (from pipeline):
    from src.features.collinearity import run_collinearity_analysis
    X_clean, dropped = run_collinearity_analysis(X, y)
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

import config

# ── tuneable thresholds ────────────────────────────────────────────────────────
CORR_THRESHOLD = 0.75   # absolute Pearson correlation between predictors
VIF_THRESHOLD  = 5.0    # VIF above this signals problematic multicollinearity
# ──────────────────────────────────────────────────────────────────────────────


# ── 1. Pearson correlation heatmap (lower triangle) ───────────────────────────

def plot_correlation_heatmap(X: pd.DataFrame, save_path: Path | None = None) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix and render a lower-triangle heatmap.
    Returns the full correlation DataFrame.
    """
    corr = X.corr(method="pearson")

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # hide upper triangle + diagonal

    n = len(corr)
    fig_size = max(10, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.4,
        linecolor="white",
        annot_kws={"size": 7},
        ax=ax,
    )
    ax.set_title("Pearson Correlation Matrix (lower triangle)", fontsize=14, pad=14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[collinearity] Heatmap saved -> {save_path}")

    plt.show()
    return corr


# ── 2. Detect pairs with |corr| > CORR_THRESHOLD ─────────────────────────────

def find_high_correlation_pairs(
    corr: pd.DataFrame, threshold: float = CORR_THRESHOLD
) -> pd.DataFrame:
    """
    Return a DataFrame of variable pairs whose absolute Pearson correlation
    exceeds *threshold*, sorted descending by |correlation|.
    """
    cols = corr.columns.tolist()
    records = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if abs(val) > threshold:
                records.append({
                    "var_a":    cols[i],
                    "var_b":    cols[j],
                    "pearson_r": round(val, 4),
                    "abs_r":    round(abs(val), 4),
                })

    pairs_df = (
        pd.DataFrame(records)
        .sort_values("abs_r", ascending=False)
        .reset_index(drop=True)
    )

    if pairs_df.empty:
        print(f"[collinearity] No pairs found with |r| > {threshold}.")
    else:
        print(f"\n[collinearity] {len(pairs_df)} pair(s) with |r| > {threshold}:")
        print(pairs_df.to_string(index=False))

    return pairs_df


# ── 3. VIF calculation ────────────────────────────────────────────────────────

def compute_vif(X: pd.DataFrame, threshold: float = VIF_THRESHOLD) -> pd.DataFrame:
    """
    Calculate VIF for every column in X (must be numeric, no NaNs).
    Returns a DataFrame of features with VIF > threshold, sorted descending.
    """
    X_vals = X.select_dtypes(include=[np.number]).dropna(axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vif_scores = [
            variance_inflation_factor(X_vals.values, i)
            for i in range(X_vals.shape[1])
        ]

    vif_df = pd.DataFrame({"feature": X_vals.columns, "VIF": vif_scores})
    vif_df["VIF"] = vif_df["VIF"].round(2)
    vif_df = vif_df.sort_values("VIF", ascending=False).reset_index(drop=True)

    high_vif = vif_df[vif_df["VIF"] > threshold].copy()

    print(f"\n[collinearity] VIF > {threshold} ({len(high_vif)} feature(s)):")
    if high_vif.empty:
        print("  None — all features within acceptable range.")
    else:
        print(high_vif.to_string(index=False))

    return high_vif


# ── 4. Intelligent pruning ───────────────────────────────────────────────────

def recommend_drops(
    X: pd.DataFrame,
    y: pd.Series,
    pairs_df: pd.DataFrame,
    auto_drop: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """
    For each high-correlation pair, keep the variable more correlated with the
    target y; recommend (or drop) the weaker one.

    Parameters
    ----------
    X         : feature DataFrame (pre-processed, numeric)
    y         : binary target Series
    pairs_df  : output of find_high_correlation_pairs()
    auto_drop : if True, actually removes recommended columns from X

    Returns
    -------
    X_out     : DataFrame with (optionally) dropped columns
    to_drop   : list of column names recommended for removal
    """
    # Correlation of every predictor with the target
    target_corr = X.corrwith(y).abs().rename("corr_with_target")

    recommendations = []
    to_drop_set: set[str] = set()

    for _, row in pairs_df.iterrows():
        a, b = row["var_a"], row["var_b"]

        # A variable already marked for dropping stays dropped
        if a in to_drop_set:
            loser = a
            winner = b
        elif b in to_drop_set:
            loser = b
            winner = a
        else:
            corr_a = target_corr.get(a, 0.0)
            corr_b = target_corr.get(b, 0.0)
            if corr_a >= corr_b:
                winner, loser = a, b
            else:
                winner, loser = b, a

        to_drop_set.add(loser)
        recommendations.append({
            "keep":               winner,
            "drop":               loser,
            "pair_pearson_r":     row["pearson_r"],
            "corr_target_keep":   round(target_corr.get(winner, np.nan), 4),
            "corr_target_drop":   round(target_corr.get(loser,  np.nan), 4),
        })

    rec_df = pd.DataFrame(recommendations)

    print(f"\n[collinearity] Pruning recommendations ({len(rec_df)} pair(s)):")
    if rec_df.empty:
        print("  Nothing to prune.")
    else:
        print(rec_df.to_string(index=False))

    to_drop = list(to_drop_set)

    if auto_drop and to_drop:
        X_out = X.drop(columns=[c for c in to_drop if c in X.columns])
        print(f"\n[collinearity] Auto-dropped {len(to_drop)} column(s): {to_drop}")
        print(f"[collinearity] X shape after pruning: {X_out.shape}")
    else:
        X_out = X.copy()
        if to_drop:
            print(f"\n[collinearity] Recommended drops (not applied): {to_drop}")

    return X_out, to_drop


# ── 5. Orchestrator ───────────────────────────────────────────────────────────

def run_collinearity_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    corr_threshold: float = CORR_THRESHOLD,
    vif_threshold:  float = VIF_THRESHOLD,
    auto_drop:      bool  = False,
    save_outputs:   bool  = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Full collinearity pipeline:
      1. Heatmap
      2. High-correlation pair detection
      3. VIF computation
      4. Intelligent pruning

    Returns (X_clean, dropped_columns).
    """
    print("=" * 60)
    print("  MULTICOLLINEARITY ANALYSIS")
    print("=" * 60)

    heatmap_path = config.OUTPUTS_DIR / "correlation_heatmap.png" if save_outputs else None
    corr = plot_correlation_heatmap(X, save_path=heatmap_path)

    pairs_df = find_high_correlation_pairs(corr, threshold=corr_threshold)

    high_vif = compute_vif(X, threshold=vif_threshold)

    X_clean, dropped = recommend_drops(X, y, pairs_df, auto_drop=auto_drop)

    if save_outputs:
        out = {
            "high_corr_pairs": pairs_df.to_dict(orient="records"),
            "high_vif_features": high_vif.to_dict(orient="records"),
            "recommended_drops": dropped,
            "thresholds": {
                "pearson_r": corr_threshold,
                "vif": vif_threshold,
            },
        }
        out_path = config.OUTPUTS_DIR / "collinearity_report.json"
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[collinearity] Report saved -> {out_path}")

    print("=" * 60)
    return X_clean, dropped


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data.loader import load_data
    from src.data.preprocessor import preprocess

    print("[collinearity] Loading data …")
    df = load_data()
    X, y = preprocess(df)

    # Use only numeric predictors for correlation / VIF
    X_num = X.select_dtypes(include=[np.number])

    X_clean, dropped = run_collinearity_analysis(
        X_num, y,
        corr_threshold=CORR_THRESHOLD,
        vif_threshold=VIF_THRESHOLD,
        auto_drop=False,      # change to True to apply drops automatically
        save_outputs=True,
    )
