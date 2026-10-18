"""FastAPI service for the packaged repayment-risk model."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.inference.predict import predict_frame

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models/repayment_risk_model.joblib"

app = FastAPI(
    title="Explainable telecom credit risk API",
    version="1.0.0",
    description=(
        "Returns calibrated five-day repayment-risk probabilities. "
        "The service supports prioritisation and does not implement an automated lending decision."
    ),
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cnt_ma_rech90: Annotated[float | None, Field(description="Main-account recharge count over 90 days")]
    daily_decr30: Annotated[float | None, Field(description="30-day daily decrement measure")]
    last_rech_date_ma: Annotated[float | None, Field(description="Main-account recharge recency measure")]
    sumamnt_ma_rech90: Annotated[float | None, Field(description="Main-account recharge amount over 90 days")]
    aon: Annotated[float | None, Field(description="Account/network tenure")]
    last_rech_amt_ma: Annotated[float | None, Field(description="Most recent main-account recharge amount")]
    daily_decr90: Annotated[float | None, Field(description="90-day daily decrement measure")]
    sumamnt_ma_rech30: Annotated[float | None, Field(description="Main-account recharge amount over 30 days")]
    medianamnt_ma_rech30: Annotated[float | None, Field(description="Median main-account recharge amount over 30 days")]
    medianmarechprebal90: Annotated[float | None, Field(description="Median pre-recharge balance over 90 days")]
    rental30: Annotated[float | None, Field(description="30-day balance/activity measure")]
    cnt_ma_rech30: Annotated[float | None, Field(description="Main-account recharge count over 30 days")]


class PredictionResponse(BaseModel):
    raw_delinquency_probability: float
    calibrated_delinquency_probability: float
    higher_risk_flag: bool


@lru_cache(maxsize=1)
def get_model_package() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artefact not found at {MODEL_PATH}. Run python -m src.modeling.train first."
        )
    return joblib.load(MODEL_PATH)


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "service": "explainable-telecom-credit-risk-api"}


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model": "available" if MODEL_PATH.exists() else "not_built",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        package = get_model_package()
        row = pd.DataFrame([request.model_dump()])
        result = predict_frame(row, package).iloc[0]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictionResponse(
        raw_delinquency_probability=float(result["raw_delinquency_probability"]),
        calibrated_delinquency_probability=float(result["calibrated_delinquency_probability"]),
        higher_risk_flag=bool(result["higher_risk_flag"]),
    )
