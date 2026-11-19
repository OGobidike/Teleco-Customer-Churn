from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn.data import INTERNET_ADDONS, clean


def make_raw(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Synthetic customers shaped exactly like the published CSV (text where it is text)."""
    rng = np.random.default_rng(seed)
    internet = rng.choice(["DSL", "Fiber optic", "No"], n)
    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    tenure = rng.integers(0, 73, n)
    monthly = np.round(rng.uniform(18, 119, n), 2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n)
    churn_logit = (
        -1.0
        - 0.04 * tenure
        + 1.2 * (contract == "Month-to-month")
        + 0.5 * (internet == "Fiber optic")
    )
    churn = rng.random(n) < 1 / (1 + np.exp(-churn_logit))
    frame = pd.DataFrame(
        {
            "customerID": [f"{i:04d}-TEST" for i in range(n)],
            "gender": rng.choice(["Female", "Male"], n),
            "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n),
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": np.where(
                phone == "No", "No phone service", rng.choice(["Yes", "No"], n)
            ),
            "InternetService": internet,
            "Contract": contract,
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "MonthlyCharges": monthly,
            "TotalCharges": [
                " " if t == 0 else f"{t * m:.2f}" for t, m in zip(tenure, monthly, strict=True)
            ],
            "Churn": np.where(churn, "Yes", "No"),
        }
    )
    for addon in INTERNET_ADDONS:
        frame[addon] = np.where(
            internet == "No", "No internet service", rng.choice(["Yes", "No"], n)
        )
    return frame


@pytest.fixture
def raw() -> pd.DataFrame:
    return make_raw()


@pytest.fixture
def customers(raw) -> pd.DataFrame:
    return clean(raw).set_index("customerID")
