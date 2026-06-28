import warnings
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import config
from api.routers import eda, model, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.models.predictor import warm_up
    from api.routers.predict import maybe_start_decile_computation

    if warm_up():
        print("[API] Model and feature list loaded into memory cache.")
    else:
        warnings.warn(
            "[API] Model or feature_importance.json not found — run the pipeline first. "
            "Prediction endpoints will return 404 until artifacts are available.",
            stacklevel=1,
        )

    maybe_start_decile_computation()
    yield


app = FastAPI(title="Customer Churn Predictor", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(eda.router)
app.include_router(model.router)
app.include_router(predict.router)

app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="frontend")


@app.get("/health")
def health():
    return {"status": "ok"}
