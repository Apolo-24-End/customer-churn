"""
Optuna-based hyperparameter tuning for LightGBM.
Uses TPE sampler + StratifiedKFold AUC as objective.
"""
import optuna
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import config

import json

optuna.logging.set_verbosity(optuna.logging.WARNING)

_PARAMS_CACHE = config.OUTPUTS_DIR / "optuna_best_params.json"


def tune_lightgbm(X_train, y_train, n_trials: int = config.OPTUNA_TRIALS) -> dict:
    """Return best LightGBM hyperparameters found by Optuna. Uses cache if available."""
    if _PARAMS_CACHE.exists():
        with open(_PARAMS_CACHE) as f:
            cached = json.load(f)
        print(f"[tuner] Loaded cached params (AUC {cached.get('best_auc', '?')}): {cached['params']}")
        return cached["params"]

    print(f"[tuner] Starting Optuna search ({n_trials} trials) ...")

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 700),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 9),
            "num_leaves": trial.suggest_int("num_leaves", 20, 120),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 2.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "class_weight": "balanced",
            "random_state": config.RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=config.RANDOM_STATE, sampling_strategy=0.3)),
            ("clf", model),
        ])
        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv, scoring="roc_auc", n_jobs=1,
        )
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"[tuner] Best AUC: {study.best_value:.4f} | params: {best}")
    with open(_PARAMS_CACHE, "w") as f:
        json.dump({"best_auc": round(study.best_value, 6), "params": best}, f, indent=2)
    return best
