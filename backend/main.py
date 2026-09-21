"""
Flight Delay Detective — FastAPI Backend
=========================================
Serves the trained GradientBoostingClassifier via a minimal REST API.

Endpoints
---------
GET  /health   → {"status": "ok"}
POST /predict  → delay probability for a single pre-flight input

Artifacts loaded ONCE at startup (no CSV reads, no training):
  models/best_model.joblib
  models/best_encoders.joblib

Feature encoding mirrors the training pipeline exactly:
  - Categorical columns (op_unique_carrier, origin, dest) are mapped via the
    smoothed target-encoding dict stored in best_encoders.joblib.
  - Unseen categories fall back to global_mean (also stored in the encoders).
  - Numerical columns are passed through unchanged.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from dataclasses import asdict
from backend.analyst import analyse

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
MODEL_PATH    = ROOT / "models" / "best_model.joblib"
ENCODERS_PATH = ROOT / "models" / "best_encoders.joblib"

# ── Feature column order — must match training exactly ────────────────────────
CATEGORICAL_FEATURES = ["op_unique_carrier", "origin", "dest"]
NUMERICAL_FEATURES   = [
    "month", "day_of_month", "day_of_week",
    "dep_hour", "arr_hour",
    "crs_elapsed_time", "distance",
]
FEATURE_COLS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


# ── Load artifacts at import time (once per process) ─────────────────────────
# Loading at module level means the artifacts are available whether the app is
# run via Uvicorn, via FastAPI's TestClient, or imported directly by tests.
_MODEL    = joblib.load(MODEL_PATH)
_ENCODERS = joblib.load(ENCODERS_PATH)
assert _MODEL.n_features_in_ == len(FEATURE_COLS), (
    f"Model expects {_MODEL.n_features_in_} features, "
    f"but FEATURE_COLS has {len(FEATURE_COLS)}"
)
print(f"[init] Model loaded: {type(_MODEL).__name__}")
print(f"[init] Encoders loaded: {list(_ENCODERS.keys())}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """No-op lifespan — artifacts already loaded at module import."""
    yield


app = FastAPI(
    title="Flight Delay Detective API",
    description="Predicts whether a scheduled flight will arrive >= 15 min late.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """The 10 pre-flight features — all available at booking time."""
    op_unique_carrier: str  = Field(..., examples=["AA"],  description="Carrier IATA code")
    origin:            str  = Field(..., examples=["DFW"], description="Origin airport IATA code")
    dest:              str  = Field(..., examples=["ORD"], description="Destination airport IATA code")
    month:             int  = Field(..., ge=1,  le=12,  examples=[7],    description="Month (1-12)")
    day_of_month:      int  = Field(..., ge=1,  le=31,  examples=[18],   description="Day of month (1-31)")
    day_of_week:       int  = Field(..., ge=1,  le=7,   examples=[5],    description="Day of week (1=Mon, 7=Sun)")
    dep_hour:          int  = Field(..., ge=0,  le=23,  examples=[17],   description="Scheduled departure hour (0-23)")
    arr_hour:          int  = Field(..., ge=0,  le=23,  examples=[20],   description="Scheduled arrival hour (0-23)")
    crs_elapsed_time:  float = Field(..., gt=0,         examples=[100.0], description="Scheduled flight duration (minutes)")
    distance:          float = Field(..., gt=0,         examples=[802.0], description="Route distance (miles)")

    @field_validator("op_unique_carrier", "origin", "dest", mode="before")
    @classmethod
    def uppercase_strip(cls, v: str) -> str:
        return v.strip().upper()


class PredictResponse(BaseModel):
    prediction:          str   = Field(..., description="'delayed' or 'not_delayed'")
    prob_delayed:        float = Field(..., description="P(arrival delay >= 15 min)")
    prob_not_delayed:    float = Field(..., description="P(arrival delay < 15 min)")
    model:               str   = Field(..., description="Model class used")
    input_features_used: dict  = Field(..., description="Feature values sent to the model")


# ── Encoding helper (mirrors training pipeline exactly) ───────────────────────

def _encode_and_predict(data: dict) -> tuple[int, float, float]:
    """
    1. Build a single-row DataFrame from the raw input dict.
    2. Apply target-encoding mappings (unseen categories -> global_mean).
    3. Run model.predict / predict_proba.
    Returns (label, prob_not_delayed, prob_delayed).
    """
    row = pd.DataFrame([data])[FEATURE_COLS]

    for col in CATEGORICAL_FEATURES:
        mapping     = _ENCODERS[col]["mapping"]
        global_mean = _ENCODERS[col]["global_mean"]
        row[col] = row[col].map(mapping).fillna(global_mean)

    X     = row.astype(float)
    label = int(_MODEL.predict(X)[0])
    probs = _MODEL.predict_proba(X)[0]
    return label, float(probs[0]), float(probs[1])


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Utility"])
def health() -> dict:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest) -> PredictResponse:
    """
    Predict arrival delay for a single scheduled flight.

    All 10 inputs must be known before departure (pre-flight features only).
    Returns the predicted class, both class probabilities, and the feature
    values that were sent to the model.
    """
    try:
        label, p0, p1 = _encode_and_predict(request.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictResponse(
        prediction          = "delayed" if label == 1 else "not_delayed",
        prob_delayed        = round(p1, 6),
        prob_not_delayed    = round(p0, 6),
        model               = type(_MODEL).__name__,
        input_features_used = request.model_dump(),
    )


# ── /analyse endpoint ─────────────────────────────────────────────────────────

class AnalyseRequest(BaseModel):
    inputs:     dict = Field(..., description="The 10 pre-flight feature values")
    prediction: dict = Field(..., description="The /predict response dict")


@app.post("/analyse", tags=["Analysis"])
def analyse_prediction(request: AnalyseRequest) -> dict:
    """
    Return a grounded natural-language analysis of a prediction.
    All statistics come from pre-computed EDA and model-metrics JSON files.
    Nothing is invented.
    """
    try:
        result = analyse(request.inputs, request.prediction)
        return asdict(result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
