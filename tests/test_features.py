import numpy as np
import pandas as pd
import pytest

from churn.data import split_xy
from churn.features import add_features, preprocessor


def test_preprocessing_learns_only_from_the_rows_it_is_fitted_on(customers):
    """The 2025 notebook's main flaw, as a test: the scaler must not see test rows."""
    X, _ = split_xy(customers)
    train, test = X.iloc[:300], X.iloc[300:].copy()
    test["MonthlyCharges"] = 10_000.0  # absurd test rows that would drag a leaky mean

    fitted = preprocessor().fit(train)
    scaler = fitted.named_steps["columns"].named_transformers_["numbers"]
    expected = add_features(train)[["tenure", "MonthlyCharges", "TotalCharges"]].mean()
    np.testing.assert_allclose(scaler.mean_[:3], expected.to_numpy())

    # Transforming the test rows reads the scaler; it never refits it.
    before = scaler.mean_.copy()
    fitted.transform(test)
    np.testing.assert_array_equal(scaler.mean_, before)


def test_no_internet_service_stays_its_own_category(customers):
    X, _ = split_xy(customers)
    fitted = preprocessor().fit(X)
    names = fitted.named_steps["columns"].get_feature_names_out()
    assert "OnlineSecurity_No internet service" in names
    out = fitted.transform(X)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_gender_is_not_an_input(customers):
    X, _ = split_xy(customers)
    names = preprocessor().fit(X).named_steps["columns"].get_feature_names_out()
    assert not any(n.startswith("gender") for n in names)


def test_an_unseen_category_does_not_crash_prediction(customers):
    X, _ = split_xy(customers)
    fitted = preprocessor().fit(X)
    odd = X.iloc[:1].copy()
    odd["PaymentMethod"] = "Carrier pigeon"
    with pytest.warns(UserWarning, match="unknown categories"):
        out = fitted.transform(odd)
    assert np.isfinite(out).all()


def test_engineered_features():
    X = pd.DataFrame(
        {
            "tenure": [0, 10],
            "MonthlyCharges": [50.0, 20.0],
            "TotalCharges": [0.0, 300.0],
            **{
                a: ["No internet service", "Yes"]
                for a in [
                    "OnlineSecurity",
                    "OnlineBackup",
                    "DeviceProtection",
                    "TechSupport",
                    "StreamingTV",
                    "StreamingMovies",
                ]
            },
        }
    )
    out = add_features(X)
    assert out["avg_monthly_spend"].tolist() == [50.0, 30.0]
    assert out["n_addons"].tolist() == [0, 6]
