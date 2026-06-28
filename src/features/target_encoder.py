"""
Mean target encoder with additive smoothing.
Fitted on training data only (post-split) to avoid label leakage.
Adds te_{col} columns — original label-encoded columns are preserved.
"""
import pandas as pd
import numpy as np


class MeanTargetEncoder:
    """
    Smoothed mean target encoder.
    smoothing controls how much we shrink toward the global mean for low-count categories.
    """

    def __init__(self, cols: list[str], smoothing: int = 10):
        self.cols = cols
        self.smoothing = smoothing
        self.global_mean_: float | None = None
        self.mappings_: dict[str, dict] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MeanTargetEncoder":
        self.global_mean_ = float(y.mean())
        for col in self.cols:
            if col not in X.columns:
                continue
            frame = pd.DataFrame({"cat": X[col], "target": y.values})
            stats = frame.groupby("cat")["target"].agg(["mean", "count"])
            smooth = (
                (stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing)
                / (stats["count"] + self.smoothing)
            )
            self.mappings_[col] = smooth.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, mapping in self.mappings_.items():
            if col in X.columns:
                X[f"te_{col}"] = X[col].map(mapping).fillna(self.global_mean_)
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
