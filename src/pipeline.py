"""
Main ML pipeline orchestrator.
Run this once to produce all artifacts in /models and /outputs.
"""
import joblib
import config
from src.data.loader import load_raw
from src.data.preprocessor import preprocess, split
from src.eda.analyzer import run_eda
from src.features.selector import select_features
from src.features.collinearity import run_collinearity_analysis
from src.features.target_encoder import MeanTargetEncoder
from src.models.trainer import train_all
from src.models.tuner import tune_lightgbm
from src.models.stacker import StackedModel
from src.models.evaluator import evaluate_all


def run():
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.OUTPUTS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("  CUSTOMER CHURN — ML PIPELINE")
    print("=" * 60)

    # 1. Load
    df = load_raw()

    # 2. EDA (on raw data before encoding)
    run_eda(df)

    # 3. Preprocess
    X, y = preprocess(df)

    # 4. Collinearity analysis + intelligent pruning
    X, _ = run_collinearity_analysis(X, y, auto_drop=True, save_outputs=True)

    # 5. Train/test split
    X_train, X_test, y_train, y_test = split(X, y)
    print(f"[pipeline] Train: {X_train.shape} | Test: {X_test.shape}")

    # 6. Target encoding (fitted on train only → no leakage)
    print("[pipeline] Applying mean target encoding ...")
    te = MeanTargetEncoder(cols=config.TARGET_ENCODE_COLS)
    X_train = te.fit_transform(X_train, y_train)
    X_test = te.transform(X_test)
    joblib.dump(te, config.MODELS_DIR / "target_encoder.joblib")
    print(f"[pipeline] Added {len(config.TARGET_ENCODE_COLS)} target-encoded columns: "
          f"{[f'te_{c}' for c in config.TARGET_ENCODE_COLS]}")

    # 7. Feature selection
    X_train_sel, X_test_sel, top_features = select_features(X_train, y_train, X_test, y_test)
    print(f"[pipeline] Selected {len(top_features)} features")

    # 8. Hyperparameter tuning for LightGBM
    best_lgbm_params = tune_lightgbm(X_train_sel, y_train, n_trials=config.OPTUNA_TRIALS)

    # 9. Train all models (LightGBM uses tuned params)
    fitted_pipelines, cv_results = train_all(X_train_sel, y_train, best_lgbm_params=best_lgbm_params)

    # 10. Stacking ensemble
    print("[pipeline] Building stacking ensemble ...")
    stacker = StackedModel(fitted_pipelines)
    stacker.fit(X_train_sel, y_train)
    joblib.dump(stacker, config.MODELS_DIR / "stacker.joblib")

    # 11. Evaluate all models + stacker
    all_models = {**fitted_pipelines, "stacked": stacker}
    eval_results = evaluate_all(all_models, cv_results, X_test_sel, y_test)

    print("=" * 60)
    print(f"  PIPELINE COMPLETE")
    print(f"  Best model: {eval_results['best_model']}")
    print(f"  AUC: {eval_results[eval_results['best_model']]['auc']}")
    print("=" * 60)
    return eval_results


if __name__ == "__main__":
    run()
