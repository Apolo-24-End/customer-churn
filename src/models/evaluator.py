import shutil
import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, roc_curve,
)

import config


def evaluate_all(fitted_pipelines: dict, cv_results: dict,
                 X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate all models on test set and pick the best by AUC."""
    print("[evaluator] Evaluating models on test set ...")
    results = {}
    best_name = None
    best_auc = -1

    for name, pipeline in fitted_pipelines.items():
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = pipeline.predict(X_test)

        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        # Optimal threshold by Youden's J (maximises sensitivity + specificity)
        best_idx = int(np.argmax(tpr - fpr))
        optimal_threshold = round(float(thresholds[best_idx]), 4)

        results[name] = {
            "auc": round(auc, 4),
            "f1": round(f1_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "optimal_threshold": optimal_threshold,
            "confusion_matrix": cm.tolist(),
            "roc_curve": {
                "fpr": [round(float(v), 4) for v in fpr],
                "tpr": [round(float(v), 4) for v in tpr],
            },
            "cv_auc_mean": cv_results.get(name, {}).get("cv_auc_mean"),
            "cv_auc_std": cv_results.get(name, {}).get("cv_auc_std"),
            "cv_scores": cv_results.get(name, {}).get("cv_scores", []),
        }
        print(f"[evaluator]   {name}: AUC={auc:.4f} | F1={results[name]['f1']}")

        if auc > best_auc:
            best_auc = auc
            best_name = name

    results["best_model"] = best_name
    print(f"[evaluator] Best model: {best_name} (AUC={best_auc:.4f})")

    best_pipeline = fitted_pipelines[best_name]
    dest = config.MODELS_DIR / f"{config.BEST_MODEL_NAME}.joblib"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if dest.exists():
        archive = config.MODELS_DIR / f"{config.BEST_MODEL_NAME}_{ts}.joblib"
        shutil.copy2(dest, archive)
        print(f"[evaluator] Previous model archived → {archive.name}")

    joblib.dump(best_pipeline, dest)

    registry_path = config.MODELS_DIR / "model_registry.json"
    registry = json.load(open(registry_path)) if registry_path.exists() else []
    registry.append({
        "timestamp": ts,
        "algorithm": best_name,
        "auc": round(best_auc, 4),
        "optimal_threshold": results[best_name]["optimal_threshold"],
        "file": dest.name,
    })
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    out = config.OUTPUTS_DIR / "evaluation_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[evaluator] Results saved to {out}")

    return results
