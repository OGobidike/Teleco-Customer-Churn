"""A small prediction API for the champion model.

    uvicorn churn.api:app

POST /predict takes one or more customers and returns each one's churn chance and
whether the retention offer is worth sending, by the same rule used in evaluation.
Requests that describe an impossible customer (internet add-ons with no internet,
say) are rejected with a 422 instead of being scored.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator

from churn.evaluate import RetentionOffer
from churn.serving import load_model

YesNo = Literal["Yes", "No"]
Addon = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    """One customer, as the published dataset describes them.

    Gender is not asked for: the model does not use it (see features.py).
    """

    SeniorCitizen: YesNo
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months as a customer")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Addon
    OnlineBackup: Addon
    DeviceProtection: Addon
    TechSupport: Addon
    StreamingTV: Addon
    StreamingMovies: Addon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(gt=0, le=1000, allow_inf_nan=False)
    TotalCharges: float = Field(ge=0, le=1_000_000, allow_inf_nan=False)

    @model_validator(mode="after")
    def consistent(self) -> Customer:
        addons = [
            self.OnlineSecurity,
            self.OnlineBackup,
            self.DeviceProtection,
            self.TechSupport,
            self.StreamingTV,
            self.StreamingMovies,
        ]
        no_internet = self.InternetService == "No"
        if any((a == "No internet service") != no_internet for a in addons):
            raise ValueError(
                "Add-ons must be 'No internet service' exactly when InternetService is 'No'."
            )
        if (self.MultipleLines == "No phone service") != (self.PhoneService == "No"):
            raise ValueError(
                "MultipleLines must be 'No phone service' exactly when PhoneService is 'No'."
            )
        return self


class Prediction(BaseModel):
    churn_probability: float
    send_offer: bool
    expected_gain: float = Field(description="Churn chance × value kept if saved, in dollars")


class PredictResponse(BaseModel):
    model: str
    version: str
    predictions: list[Prediction]


def models_dir() -> Path:
    return Path(os.environ.get("CHURN_MODELS_DIR", "models"))


@lru_cache(maxsize=1)
def loaded():
    return load_model(models_dir())


app = FastAPI(title="Telco churn", version="2.0.0")


@app.get("/health")
def health() -> dict:
    pipeline, meta = loaded()
    return {"status": "ok", "model": meta["model"], "version": meta["version"]}


@app.get("/model")
def model_card() -> dict:
    return loaded()[1]


@app.post("/predict", response_model=PredictResponse)
def predict(
    customers: Annotated[list[Customer], Field(min_length=1, max_length=1000)],
) -> PredictResponse:
    pipeline, meta = loaded()
    offer = RetentionOffer(**meta["offer_assumptions"])
    frame = pd.DataFrame([c.model_dump() for c in customers])
    p = pipeline.predict_proba(frame)[:, 1]
    gains = p * offer.gain_if_saved(frame["MonthlyCharges"])
    return PredictResponse(
        model=meta["model"],
        version=meta["version"],
        predictions=[
            Prediction(
                churn_probability=round(float(pi), 4),
                send_offer=bool(g > offer.offer_cost),
                expected_gain=round(float(g), 2),
            )
            for pi, g in zip(p, gains, strict=True)
        ],
    )
