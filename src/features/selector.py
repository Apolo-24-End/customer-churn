import pandas as pd
import numpy as np
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

import config


def select_features(X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame, y_test: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Rank features and return trimmed X_train, X_test, and selected feature names."""
    print("[selector] Computing feature importance with RandomForest ...")

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)

    importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    top_features = importance_df.head(config.TOP_N_FEATURES)["feature"].tolist()

    results = {
        "features": importance_df["feature"].tolist(),
        "importances": importance_df["importance"].round(6).tolist(),
        "top_features": top_features,
    }

    out = config.OUTPUTS_DIR / "feature_importance.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[selector] Top {config.TOP_N_FEATURES} features saved to {out}")

    return X_train[top_features], X_test[top_features], top_features
