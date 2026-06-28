import json
from fastapi import APIRouter, HTTPException
import config

router = APIRouter(prefix="/model", tags=["Model"])


def _load_eval() -> dict:
    path = config.OUTPUTS_DIR / "evaluation_results.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Evaluation results not found. Run the pipeline first.")
    with open(path) as f:
        return json.load(f)


def _load_features() -> dict:
    path = config.OUTPUTS_DIR / "feature_importance.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Feature importance not found. Run the pipeline first.")
    with open(path) as f:
        return json.load(f)


@router.get("/results")
def get_all_results():
    return _load_eval()


@router.get("/best")
def get_best_model():
    data = _load_eval()
    best = data["best_model"]
    return {"best_model": best, **data[best]}


@router.get("/comparison")
def get_model_comparison():
    data = _load_eval()
    best = data["best_model"]
    all_models = config.MODELS_TO_TRAIN + ["stacked"]
    models = [m for m in all_models if m in data]
    return {
        "models": models,
        "metrics": {
            m: {k: v for k, v in data[m].items() if k not in ("roc_curve", "cv_scores")}
            for m in models
        },
        "best_model": best,
    }


@router.get("/roc-curves")
def get_roc_curves():
    data = _load_eval()
    all_models = config.MODELS_TO_TRAIN + ["stacked"]
    return {
        m: data[m]["roc_curve"]
        for m in all_models if m in data
    }


@router.get("/confusion-matrix/{model_name}")
def get_confusion_matrix(model_name: str):
    data = _load_eval()
    if model_name not in data:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")
    return {"model": model_name, "confusion_matrix": data[model_name]["confusion_matrix"]}


@router.get("/features")
def get_feature_importance():
    return _load_features()
