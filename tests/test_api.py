import pytest
from fastapi.testclient import TestClient
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from churn.data import split_xy
from churn.evaluate import RetentionOffer
from churn.models import candidates
from churn.serving import load_model, save_model

CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.5,
    "TotalCharges": 190.0,
}


def _results() -> dict:
    return {
        "version": "test",
        "champion": "logistic",
        "data": {"sha256": "x"},
        "champion_test": {"scores": {}},
        "offer_assumptions": RetentionOffer().as_dict(),
        "versions": {},
    }


@pytest.fixture
def client(customers, tmp_path, monkeypatch):
    from churn import api

    X, y = split_xy(customers)
    pipe = next(c for c in candidates(0, include_nn=False) if c.name == "logistic").pipeline
    save_model(clone(pipe).fit(X, y), tmp_path, _results())
    monkeypatch.setenv("CHURN_MODELS_DIR", str(tmp_path))
    api.loaded.cache_clear()
    yield TestClient(api.app)
    api.loaded.cache_clear()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_scores_each_customer(client):
    other = CUSTOMER | {"Contract": "Two year", "tenure": 70, "TotalCharges": 6000.0}
    body = client.post("/predict", json=[CUSTOMER, other]).json()
    assert len(body["predictions"]) == 2
    for p in body["predictions"]:
        assert 0 <= p["churn_probability"] <= 1
        assert isinstance(p["send_offer"], bool)


def test_an_impossible_customer_is_rejected(client):
    no_internet_but_backup = CUSTOMER | {"InternetService": "No"}
    assert client.post("/predict", json=[no_internet_but_backup]).status_code == 422


def test_an_empty_batch_is_rejected_not_crashed(client):
    assert client.post("/predict", json=[]).status_code == 422


def test_an_unknown_category_is_rejected(client):
    assert client.post("/predict", json=[CUSTOMER | {"Contract": "Forever"}]).status_code == 422


def _shout(values):  # not on the trusted list
    return values


def test_a_model_file_with_unexpected_code_is_refused(customers, tmp_path):
    X, y = split_xy(customers)
    sneaky = Pipeline([("odd", FunctionTransformer(_shout)), ("model", LogisticRegression())]).fit(
        X[["tenure", "MonthlyCharges"]], y
    )
    save_model(sneaky, tmp_path, _results())
    with pytest.raises(ValueError, match="Refusing to load"):
        load_model(tmp_path)
