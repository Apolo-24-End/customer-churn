import json
from fastapi import APIRouter, HTTPException
import config

router = APIRouter(prefix="/eda", tags=["EDA"])


def _load_eda() -> dict:
    path = config.OUTPUTS_DIR / "eda_results.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="EDA results not found. Run the pipeline first.")
    with open(path) as f:
        return json.load(f)


@router.get("/overview")
def get_overview():
    return _load_eda()["overview"]


@router.get("/churn-distribution")
def get_churn_distribution():
    return _load_eda()["churn_distribution"]


@router.get("/churn-by/{category}")
def get_churn_by_category(category: str):
    data = _load_eda()
    key = f"churn_by_{category}"
    if key not in data:
        raise HTTPException(status_code=404, detail=f"Category '{category}' not found.")
    return data[key]


@router.get("/distributions/{column}")
def get_distribution(column: str):
    data = _load_eda()
    key = f"{column}_distribution"
    if key not in data:
        raise HTTPException(status_code=404, detail=f"Distribution for '{column}' not found.")
    return data[key]


@router.get("/correlation")
def get_correlation():
    return _load_eda()["correlation_with_churn"]


@router.get("/numerical-stats")
def get_numerical_stats():
    return _load_eda()["numerical_stats"]


@router.get("/churn-by-tenure")
def get_churn_by_tenure():
    return _load_eda()["churn_by_tenure_group"]


@router.get("/all")
def get_all_eda():
    return _load_eda()
