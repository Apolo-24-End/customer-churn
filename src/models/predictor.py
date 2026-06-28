import joblib
import json
import config

_model_cache: dict = {}  # keyed by model name; populated on first use or at startup


def _get_model():
    key = config.BEST_MODEL_NAME
    if key not in _model_cache:
        _model_cache[key] = joblib.load(config.MODELS_DIR / f"{key}.joblib")
    return _model_cache[key]


def _top_features() -> list[str]:
    if "top_features" not in _model_cache:
        path = config.OUTPUTS_DIR / "feature_importance.json"
        with open(path) as f:
            _model_cache["top_features"] = json.load(f)["top_features"]
    return _model_cache["top_features"]


def _get_optimal_threshold() -> float:
    if "optimal_threshold" not in _model_cache:
        path = config.OUTPUTS_DIR / "evaluation_results.json"
        threshold = 0.5
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            best = data.get("best_model", "")
            threshold = data.get(best, {}).get("optimal_threshold", 0.5)
        _model_cache["optimal_threshold"] = threshold
    return _model_cache["optimal_threshold"]


def warm_up() -> bool:
    """Pre-load model, feature list and optimal threshold into the in-process cache."""
    model_path = config.MODELS_DIR / f"{config.BEST_MODEL_NAME}.joblib"
    features_path = config.OUTPUTS_DIR / "feature_importance.json"
    if not model_path.exists() or not features_path.exists():
        return False
    _get_model()
    _top_features()
    _get_optimal_threshold()
    return True


def predict_single(customer_data: dict) -> dict:
    """Predict churn probability for a single customer."""
    from src.data.preprocessor import preprocess_single

    pipeline = _get_model()
    X = preprocess_single(customer_data)

    top_features = _top_features()
    missing = [f for f in top_features if f not in X.columns]
    if missing:
        raise ValueError(
            f"Preprocessing is missing {len(missing)} feature(s) expected by the model: {missing}"
        )
    X = X[top_features]

    prob = float(pipeline.predict_proba(X)[0, 1])
    threshold = _get_optimal_threshold()
    prediction = int(prob >= threshold)

    risk_level = "High" if prob >= 0.7 else "Medium" if prob >= 0.4 else "Low"

    return {
        "churn_probability": round(prob, 4),
        "churn_prediction": prediction,
        "risk_level": risk_level,
        "interpretation": (
            f"This customer has a {prob * 100:.1f}% probability of churning. "
            f"Risk level: {risk_level}."
        ),
    }


def get_decile_groups(top_n_per_decile: int = 20) -> list[dict]:
    """Score all customers, assign deciles, return groups D10→D1 with top N customers each."""
    import pandas as pd
    from src.data.loader import load_raw
    from src.data.preprocessor import preprocess

    pipeline = _get_model()
    top_features = _top_features()

    df_raw = load_raw()
    ids = df_raw[config.ID_COL].tolist()

    X, _ = preprocess(df_raw)

    te_path = config.MODELS_DIR / "target_encoder.joblib"
    if te_path.exists():
        from src.features.target_encoder import MeanTargetEncoder
        te = joblib.load(te_path)
        X = te.transform(X)

    missing = [f for f in top_features if f not in X.columns]
    if missing:
        raise ValueError(
            f"Decile scoring is missing {len(missing)} feature(s) expected by the model: {missing}"
        )
    X = X[top_features]

    probs = pipeline.predict_proba(X)[:, 1]

    full_df = pd.DataFrame({"customer_id": ids, "churn_probability": probs})
    full_df["decile"] = (
        full_df["churn_probability"]
        .rank(pct=True)
        .apply(lambda x: min(int(x * 10) + 1, 10))
        .astype(int)
    )

    groups = []
    for d in range(10, 0, -1):
        bucket = full_df[full_df["decile"] == d].sort_values(
            "churn_probability", ascending=False
        )
        customers = (
            bucket.head(top_n_per_decile)
            .assign(churn_probability=lambda df: df["churn_probability"].round(4))
            [["customer_id", "churn_probability"]]
            .to_dict(orient="records")
        )
        groups.append({
            "decile": d,
            "customer_count": len(bucket),
            "avg_probability": round(float(bucket["churn_probability"].mean()), 4),
            "min_probability": round(float(bucket["churn_probability"].min()), 4),
            "max_probability": round(float(bucket["churn_probability"].max()), 4),
            "customers": customers,
        })

    out = config.OUTPUTS_DIR / "decile_groups.json"
    with open(out, "w") as f:
        json.dump(groups, f, indent=2)

    return groups
