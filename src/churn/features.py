"""Feature preparation, built so it can only ever learn from training rows.

Everything that *learns* from data (the scaler's means, the encoder's categories)
lives in one scikit-learn ColumnTransformer that is fitted inside the model
Pipeline. Cross-validation and the final fit therefore refit it on training rows
only, and the test rows are just transformed. That is the fix for the 2025
notebook, which scaled and imputed all 7,043 rows before splitting them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn.data import INTERNET_ADDONS

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
ENGINEERED = ["n_addons", "avg_monthly_spend"]
# Gender is deliberately left out. Measured on the training rows it adds no signal (every
# run records the comparison under "gender_check" in results/metrics.json), and who gets a
# retention offer should not depend on it. It is still used to *check* the model: the
# evaluation reports results for women and men separately.
CATEGORICAL = [
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    *INTERNET_ADDONS,
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
RAW_FEATURES = NUMERIC + CATEGORICAL


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Row-by-row features: nothing is learned, so nothing can leak.

    n_addons: how many internet add-ons the customer pays for (0 to 6).
    avg_monthly_spend: total billed divided by months on the books. For new
    customers (tenure 0) it falls back to this month's charge.
    """
    X = X.copy()
    X["n_addons"] = (X[INTERNET_ADDONS] == "Yes").sum(axis=1)
    months = X["tenure"].where(X["tenure"] > 0)
    X["avg_monthly_spend"] = (X["TotalCharges"] / months).fillna(X["MonthlyCharges"])
    return X


def to_float32(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def preprocessor(extra_categorical: tuple[str, ...] = ()) -> Pipeline:
    """The feature pipeline. extra_categorical adds columns the model does not normally
    see; it exists so the decision to leave gender out can be measured, not assumed."""
    columns = ColumnTransformer(
        [
            ("numbers", StandardScaler(), NUMERIC + ENGINEERED),
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False),
                [*CATEGORICAL, *extra_categorical],
            ),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("engineer", FunctionTransformer(add_features, validate=False)),
            ("columns", columns),
            ("float32", FunctionTransformer(to_float32)),
        ]
    )
