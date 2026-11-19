import numpy as np
import pytest
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from churn.data import load, split_xy
from churn.models import KerasMLPClassifier, candidates, tensorflow_available
from churn.train import OUTER_FOLDS, choose_champion, cross_validate, split


def _cv(pr_auc: dict[str, tuple[float, float]]) -> dict:
    return {
        name: {"summary": {"pr_auc": {"mean": mean, "se": se}}}
        for name, (mean, se) in pr_auc.items()
    }


def test_a_simpler_model_within_one_standard_error_wins():
    cv = _cv({"baseline": (0.27, 0), "logistic": (0.700, 0.012), "boosting": (0.708, 0.011)})
    assert choose_champion(cv)[0] == "logistic"


def test_a_clear_winner_wins():
    cv = _cv({"baseline": (0.27, 0), "logistic": (0.60, 0.012), "boosting": (0.708, 0.011)})
    assert choose_champion(cv)[0] == "boosting"


def test_the_baseline_is_never_champion():
    cv = _cv({"baseline": (0.99, 0), "logistic": (0.50, 0.01)})
    assert choose_champion(cv)[0] == "logistic"


def test_training_twice_gives_identical_predictions(customers):
    X, y = split_xy(customers)
    pipe = next(c for c in candidates(0, include_nn=False) if c.name == "boosting").pipeline
    a = clone(pipe).fit(X, y).predict_proba(X)[:, 1]
    b = clone(pipe).fit(X, y).predict_proba(X)[:, 1]
    np.testing.assert_array_equal(a, b)


@pytest.mark.network
def test_every_model_beats_the_baseline_on_real_data():
    # Uses training rows only (split again here), so running the tests never touches
    # the held-out test rows.
    X_all, _, y_all, _ = split(load())
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.25, stratify=y_all, random_state=0
    )
    scores = {}
    for c in candidates(42, include_nn=False):
        model = clone(c.pipeline).fit(X_train, y_train)
        scores[c.name] = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
    assert scores["baseline"] == pytest.approx(y_test.mean(), abs=1e-3)
    # Guessing scores about 0.27 here; a real model should be far above it.
    assert scores["logistic"] > scores["baseline"] + 0.25
    assert scores["boosting"] > scores["baseline"] + 0.25


@pytest.mark.nn
@pytest.mark.skipif(not tensorflow_available(), reason="needs the nn extra")
def test_the_network_stops_early_on_its_own_validation_slice():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5)).astype("float32")
    y = (X[:, 0] + rng.normal(scale=0.5, size=400) > 0).astype(int)
    model = KerasMLPClassifier(max_epochs=30, patience=3, random_state=1).fit(X, y)
    assert 1 <= model.best_epoch_ <= 30
    p = model.predict_proba(X)
    assert p.shape == (400, 2)
    np.testing.assert_allclose(p.sum(axis=1), 1, rtol=1e-6)
    again = KerasMLPClassifier(max_epochs=30, patience=3, random_state=1).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(p, again, atol=1e-6)


def test_cross_validation_never_fits_preprocessing_on_held_out_rows(customers, monkeypatch):
    """The 2025 flaw, tested on this project's own training loop: spy on every scaler fit
    made during nested cross-validation and check none of them saw a whole fold-set."""
    X, y = split_xy(customers)
    seen = []
    original = StandardScaler.fit

    def spying_fit(self, data, *args, **kwargs):
        seen.append(len(data))
        return original(self, data, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", spying_fit)
    tuned = next(c for c in candidates(0, include_nn=False) if c.name == "logistic")
    cross_validate(tuned, X, y, log=lambda *_: None)

    outer_training_rows = len(X) * (OUTER_FOLDS - 1) / OUTER_FOLDS
    assert seen, "the scaler was never fitted"
    # Every fit is on an outer fold's training rows, or on an inner fold carved from them.
    assert max(seen) <= outer_training_rows + 1
    assert max(seen) < len(X)
