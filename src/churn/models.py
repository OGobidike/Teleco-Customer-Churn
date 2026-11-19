"""The candidate models. Each one is a full Pipeline: features, then the model.

* baseline: predicts every customer's churn chance as the training churn rate.
  Anything worth shipping has to beat it.
* logistic: logistic regression, tuned over its regularization strength.
* boosting: histogram gradient boosting, tuned over a small grid.
* neural_net: the 2025 TensorFlow network (64-32-1 with dropout), rebuilt so it
  stops early on a validation slice carved out of the *training* rows, never the
  test rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from churn.features import preprocessor

# Simplest first. When a simpler model is within one standard error of the best,
# the simpler one wins. The rule is fixed here in code and applied mechanically.
SIMPLICITY_ORDER = ["baseline", "logistic", "boosting", "neural_net"]


def tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        return False
    return True


class KerasMLPClassifier(ClassifierMixin, BaseEstimator):
    """A scikit-learn wrapper around a small Keras network.

    Early stopping watches a stratified slice of the rows passed to `fit`, so
    inside a Pipeline or cross-validation it only ever sees training data.
    """

    def __init__(
        self,
        hidden=(64, 32),
        dropout=(0.3, 0.2),
        learning_rate=1e-3,
        batch_size=32,
        max_epochs=200,
        patience=15,
        validation_fraction=0.15,
        random_state=0,
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def _build(self, n_features: int):
        import keras

        layers = [keras.Input(shape=(n_features,))]
        for units, rate in zip(self.hidden, self.dropout, strict=True):
            layers += [keras.layers.Dense(units, activation="relu"), keras.layers.Dropout(rate)]
        layers.append(keras.layers.Dense(1, activation="sigmoid"))
        model = keras.Sequential(layers)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
        )
        return model

    def fit(self, X, y):
        import keras
        import tensorflow as tf

        keras.utils.set_random_seed(self.random_state)
        tf.config.experimental.enable_op_determinism()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        X_fit, X_val, y_fit, y_val = train_test_split(
            X, y, test_size=self.validation_fraction, stratify=y, random_state=self.random_state
        )
        self.model_ = self._build(X.shape[1])
        stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True
        )
        history = self.model_.fit(
            X_fit,
            y_fit,
            validation_data=(X_val, y_val),
            epochs=self.max_epochs,
            batch_size=self.batch_size,
            callbacks=[stop],
            verbose=0,
        )
        self.history_ = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        self.best_epoch_ = int(np.argmin(self.history_["val_loss"])) + 1
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = self.model_.predict(np.asarray(X, dtype=np.float32), verbose=0).ravel()
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


@dataclass
class Candidate:
    name: str
    label: str
    pipeline: Pipeline
    grid: dict = field(default_factory=dict)

    def estimator(self, seed: int, inner_folds: int = 5):
        """The pipeline, wrapped in an inner grid search when it has knobs to tune."""
        if not self.grid:
            return self.pipeline
        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
        return GridSearchCV(self.pipeline, self.grid, scoring="average_precision", cv=inner)


def candidates(seed: int, include_nn: bool = True) -> list[Candidate]:
    pipe = lambda model: Pipeline([("features", preprocessor()), ("model", model)])  # noqa: E731
    found = [
        Candidate(
            "baseline", "Baseline (guess the churn rate)", pipe(DummyClassifier(strategy="prior"))
        ),
        Candidate(
            "logistic",
            "Logistic regression",
            pipe(LogisticRegression(max_iter=5000)),
            {"model__C": [0.03, 0.1, 0.3, 1.0, 3.0]},
        ),
        Candidate(
            "boosting",
            "Gradient boosting",
            pipe(
                HistGradientBoostingClassifier(
                    max_iter=300, l2_regularization=1.0, random_state=seed
                )
            ),
            {
                "model__learning_rate": [0.03, 0.1],
                "model__max_leaf_nodes": [7, 15],
                "model__min_samples_leaf": [40, 100],
            },
        ),
    ]
    if include_nn and tensorflow_available():
        found.append(
            Candidate(
                "neural_net",
                "Neural network (TensorFlow)",
                pipe(KerasMLPClassifier(random_state=seed)),
            )
        )
    return found
