import csv
import io
import json
import logging
import threading
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["Predict"])

# ── Decile computation state ─────────────────────────────────
_decile_lock = threading.Lock()
_decile_status = "idle"    # "idle" | "computing" | "ready" | "error"
_decile_error: str | None = None


def _run_decile_computation(top_n: int) -> None:
    global _decile_status, _decile_error
    try:
        from src.models.predictor import get_decile_groups
        get_decile_groups(top_n)
        _decile_status = "ready"
        logger.info("[decile] Background computation complete — cache written.")
    except Exception as exc:
        _decile_status = "error"
        _decile_error = str(exc)
        logger.error("[decile] Background computation failed: %s", exc)


def maybe_start_decile_computation(top_n: int = 20) -> None:
    """Start background decile computation if cache is absent and not already running."""
    global _decile_status
    cached = config.OUTPUTS_DIR / "decile_groups.json"
    if cached.exists():
        _decile_status = "ready"
        return
    model_path = config.MODELS_DIR / f"{config.BEST_MODEL_NAME}.joblib"
    if not model_path.exists():
        return
    with _decile_lock:
        if _decile_status not in ("computing", "ready"):
            _decile_status = "computing"
            threading.Thread(
                target=_run_decile_computation, args=(top_n,), daemon=True
            ).start()
            logger.info("[decile] Background computation thread started.")


# ── Customer input schema ────────────────────────────────────
class CustomerInput(BaseModel):
    age: float = Field(..., ge=0, le=120)
    gender: str
    annual_income: float = Field(..., ge=0)
    education: str
    marital_status: str
    dependents: int = Field(..., ge=0, le=20)
    tenure: int = Field(..., ge=0)
    contract: str
    payment_method: str
    paperless_billing: str
    senior_citizen: int = Field(..., ge=0, le=1)
    monthlycharges: float = Field(..., ge=0)
    totalcharges: float = Field(..., ge=0)
    num_services: int = Field(..., ge=0, le=8)
    has_phone_service: int = Field(..., ge=0, le=1)
    has_internet_service: int = Field(..., ge=0, le=1)
    has_online_security: int = Field(..., ge=0, le=1)
    has_online_backup: int = Field(..., ge=0, le=1)
    has_device_protection: int = Field(..., ge=0, le=1)
    has_tech_support: int = Field(..., ge=0, le=1)
    has_streaming_tv: int = Field(..., ge=0, le=1)
    has_streaming_movies: int = Field(..., ge=0, le=1)
    customer_satisfaction: Optional[float] = Field(5.0, ge=0, le=10)
    num_complaints: Optional[float] = Field(0.0, ge=0)
    num_service_calls: Optional[int] = Field(0, ge=0)
    late_payments: Optional[int] = Field(0, ge=0)
    avg_monthly_gb: Optional[float] = Field(50.0, ge=0)
    days_since_last_interaction: Optional[int] = Field(30, ge=0)
    credit_score: Optional[float] = Field(None, ge=300, le=850)


# ── Endpoints ────────────────────────────────────────────────
@router.post("/single")
def predict_single(customer: CustomerInput):
    from src.models.predictor import predict_single as _predict
    try:
        result = _predict(customer.model_dump())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found. Run the pipeline first.")
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result


@router.get("/decile-groups")
def get_decile_groups(top_n: int = 20):
    global _decile_status, _decile_error

    # Serve from cache if available
    cached = config.OUTPUTS_DIR / "decile_groups.json"
    if cached.exists():
        with open(cached) as f:
            data = json.load(f)
        if data and "customers" in data[0]:
            _decile_status = "ready"
            return data

    # Surface computation errors
    if _decile_status == "error":
        raise HTTPException(
            status_code=500,
            detail=f"Decile computation failed: {_decile_error}",
        )

    # Already running — tell the client to retry
    if _decile_status == "computing":
        raise HTTPException(
            status_code=503,
            detail="Decile groups are being computed. Please retry in ~60 seconds.",
            headers={"Retry-After": "60"},
        )

    # No cache and not running — trigger computation and return immediately
    maybe_start_decile_computation(top_n)
    raise HTTPException(
        status_code=503,
        detail="Decile group computation started in the background. Please retry in ~60 seconds.",
        headers={"Retry-After": "60"},
    )


@router.get("/export-decile/{decile}")
def export_decile_csv(decile: int):
    predictions_path = config.OUTPUTS_DIR / "predictions_all.csv"
    if not predictions_path.exists():
        raise HTTPException(status_code=404, detail="predictions_all.csv not found. Run the pipeline first.")
    if not (1 <= decile <= 10):
        raise HTTPException(status_code=400, detail="Decile must be between 1 and 10.")

    def generate():
        with open(predictions_path, newline="") as f:
            reader = csv.DictReader(f)
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=["customer_id", "churn_probability", "decile"])
            writer.writeheader()
            yield output.getvalue()
            for row in reader:
                if int(row["decile"]) == decile:
                    output = io.StringIO()
                    writer = csv.DictWriter(output, fieldnames=["customer_id", "churn_probability", "decile"])
                    writer.writerow(row)
                    yield output.getvalue()

    filename = f"decil_{decile}_clientes.csv"
    return StreamingResponse(
        generate(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
