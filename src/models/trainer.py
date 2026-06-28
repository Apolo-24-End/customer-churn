import pandas as pd
import numpy as np
import json
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import lightgbm as lgb
import xgboost as xgb

import config


def _build_models(best_lgbm_params: dict | None = None) -> dict:
    lgbm_params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 63,
        "class_weight": "balanced",
        "random_state": config.RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }
    if best_lgbm_params:
        lgbm_params.update(best_lgbm_params)
        lgbm_params.setdefault("class_weight", "balanced")
        lgbm_params["verbose"] = -1

    return {
        "lightgbm": lgb.LGBMClassifier(**lgbm_params),
        "xgboost": xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=10,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=0,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def train_all(X_train: pd.DataFrame, y_train: pd.Series,
              best_lgbm_params: dict | None = None) -> dict:
    """Train all models with SMOTE + CV. Returns {name: fitted_pipeline}."""
    print("[trainer] Training models with SMOTE + cross-validation ...")
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    models = _build_models(best_lgbm_params=best_lgbm_params)
    cv_results = {}
    fitted_pipelines = {}

    for name in config.MODELS_TO_TRAIN:
        model = models[name]
        print(f"[trainer]   -> {name} ...")

        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=config.RANDOM_STATE, sampling_strategy=0.3)),
            ("clf", model),
        ])

        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv, scoring="roc_auc", n_jobs=1,
        )
        cv_results[name] = {
            "cv_auc_mean": round(float(scores.mean()), 4),
            "cv_auc_std": round(float(scores.std()), 4),
            "cv_scores": [round(float(s), 4) for s in scores],
        }
        print(f"[trainer]     AUC: {scores.mean():.4f} ± {scores.std():.4f}")

        # Fit on full training set
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

    # Save CV results
    out = config.OUTPUTS_DIR / "cv_results.json"
    with open(out, "w") as f:
        json.dump(cv_results, f, indent=2)
    print(f"[trainer] CV results saved to {out}")

    # Persist all pipelines
    for name, pipe in fitted_pipelines.items():
        joblib.dump(pipe, config.MODELS_DIR / f"{name}.joblib")

    return fitted_pipelines, cv_results
