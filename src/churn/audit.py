"""An audit of my 2025 notebook: replay it, then fix one flaw at a time and measure.

The 2025 notebook (notebooks/2025-original) reported "Test Accuracy: 81.05%" from a
single unseeded run. Reading it again, I found five problems:

1. Scaling and gap-filling were learned from all 7,043 rows *before* the split, so
   the test rows helped shape the training data (leakage).
2. Blank TotalCharges were filled with the mean (about $2,283), though all 11
   belong to brand-new customers whose true total is $0. And "No internet
   service" became a gap that was then filled with a column average, turning
   customers without internet into fractional subscribers.
3. SMOTE balanced the training rows, but the model was trained on the
   unbalanced ones, so the balancing never reached it.
4. The test set was passed as validation data, so every epoch's test score was on
   screen while training choices were made.
5. Accuracy was the only score, on data where 73.5% of customers stay.

Each experiment below changes exactly one thing and runs over several seeds, so the
effect of a flaw can be told apart from run-to-run noise.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from churn.data import load_raw

REPORTED_2025_ACCURACY = 0.8105
BINARY_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
SCALED = ["tenure", "MonthlyCharges", "TotalCharges"]


def notebook_2025_features(
    raw: pd.DataFrame, fit_rows: pd.Index | None = None, fix_cleaning: bool = False
) -> pd.DataFrame:
    """The 2025 notebook's preprocessing, step for step.

    fit_rows: which rows the scaler and the gap-filling means are learned from.
    None reproduces 2025 (all rows). fix_cleaning repairs flaw 2 only.
    """
    df = raw.drop(columns="customerID").copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    fit = df.index if fit_rows is None else fit_rows
    if fix_cleaning:
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    else:
        df["TotalCharges"] = df["TotalCharges"].fillna(df.loc[fit, "TotalCharges"].mean())
    mapping = {"Yes": 1, "No": 0, "Female": 0, "Male": 1}
    if fix_cleaning:
        mapping |= {"No internet service": 0, "No phone service": 0}
    for column in BINARY_COLS:
        df[column] = df[column].map(mapping)
    df = pd.get_dummies(
        df,
        columns=["InternetService", "Contract", "PaymentMethod", "MultipleLines"],
        drop_first=True,
    )
    scaler = StandardScaler().fit(df.loc[fit, SCALED])
    df[SCALED] = scaler.transform(df[SCALED])
    df = df.fillna(df.loc[fit].mean())
    return df.astype("float32")


def notebook_2025_model(n_features: int, seed: int):
    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    model = keras.Sequential(
        [
            keras.Input(shape=(n_features,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _scores(y, p) -> dict:
    return {
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "recall": float(recall_score(y, p >= 0.5)),
        "roc_auc": float(roc_auc_score(y, p)),
    }


def _fit_predict(X_tr, y_tr, X_te, seed, epochs=20):
    model = notebook_2025_model(X_tr.shape[1], seed)
    model.fit(X_tr, y_tr, epochs=epochs, batch_size=32, verbose=0)
    return model.predict(X_te, verbose=0).ravel()


def _summary(values: list[float]) -> dict:
    """Mean, range and a 95% t-interval for the mean (a t-interval, since runs are few)."""
    from scipy import stats

    a = np.asarray(values)
    half = 0.0
    if len(a) > 1:
        half = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return {
        "mean": float(a.mean()),
        "min": float(a.min()),
        "max": float(a.max()),
        "ci95": [float(a.mean() - half), float(a.mean() + half)],
        "runs": [float(v) for v in a],
    }


def replay(raw: pd.DataFrame, seeds: list[int], log=print) -> dict:
    """The notebook as written: its split (random_state=42), its model, 20 epochs."""
    df = notebook_2025_features(raw)
    X, y = df.drop(columns="Churn"), df["Churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    runs = []
    for seed in seeds:
        p = _fit_predict(X_tr.to_numpy(), y_tr.to_numpy(), X_te.to_numpy(), seed)
        runs.append(_scores(y_te, p))
        log(f"  replay seed {seed}: accuracy {runs[-1]['accuracy']:.4f}")
    return {
        "reported_2025_accuracy": REPORTED_2025_ACCURACY,
        "majority_class_accuracy": float(1 - y_te.mean()),
        "test_rows": int(len(y_te)),
        "missing_values_filled": int(notebook_2025_raw_gaps(raw)),
        **{m: _summary([r[m] for r in runs]) for m in ("accuracy", "recall", "roc_auc")},
    }


def notebook_2025_raw_gaps(raw: pd.DataFrame) -> int:
    """How many gaps the 2025 mapping created (the notebook printed 9156)."""
    df = raw.drop(columns="customerID").copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())
    for column in BINARY_COLS:
        df[column] = df[column].map({"Yes": 1, "No": 0, "Female": 0, "Male": 1})
    df = pd.get_dummies(
        df,
        columns=["InternetService", "Contract", "PaymentMethod", "MultipleLines"],
        drop_first=True,
    )
    return int(df.isna().sum().sum())


FLAW_FIXES = ("leakage", "cleaning", "smote")


def one_flaw_at_a_time(raw: pd.DataFrame, seeds: list[int], log=print) -> dict:
    """Per seed: train the 2025 way once, then once more for each single fix.

    Every arm of a seed shares the same split and the same training seed, so the
    difference between two arms is the fix (plus training noise, which the
    interval over seeds measures).
    """
    from imblearn.over_sampling import SMOTE

    y_all = raw["Churn"].eq("Yes").astype(int)
    runs = {arm: [] for arm in ("as_2025", *FLAW_FIXES)}
    for seed in seeds:
        idx_tr, idx_te = train_test_split(
            raw.index, test_size=0.2, random_state=seed, stratify=y_all
        )
        as_2025 = notebook_2025_features(raw)
        frames = {
            "as_2025": as_2025,
            "leakage": notebook_2025_features(raw, fit_rows=idx_tr),
            "cleaning": notebook_2025_features(raw, fix_cleaning=True),
            "smote": as_2025,
        }
        for arm, frame in frames.items():
            X, y = frame.drop(columns="Churn"), frame["Churn"]
            X_tr, y_tr = X.loc[idx_tr].to_numpy(), y.loc[idx_tr].to_numpy()
            if arm == "smote":
                X_tr, y_tr = SMOTE(random_state=seed).fit_resample(X_tr, y_tr)
            p = _fit_predict(X_tr, y_tr, X.loc[idx_te].to_numpy(), seed)
            runs[arm].append(_scores(y.loc[idx_te], p))
        log(
            f"  seed {seed}: accuracy as 2025 {runs['as_2025'][-1]['accuracy']:.4f} | "
            + " | ".join(
                f"{arm} fixed {runs[arm][-1]['accuracy']:.4f} "
                f"(recall {runs['as_2025'][-1]['recall']:.3f} -> {runs[arm][-1]['recall']:.3f})"
                for arm in FLAW_FIXES
            )
        )
    report = {}
    for arm in FLAW_FIXES:
        report[arm] = {
            m: {
                "as_2025": _summary([r[m] for r in runs["as_2025"]]),
                "fixed": _summary([r[m] for r in runs[arm]]),
                "change": _summary(
                    [a[m] - b[m] for a, b in zip(runs[arm], runs["as_2025"], strict=True)]
                ),
            }
            for m in ("accuracy", "recall", "roc_auc")
        }
    return report


def _EpochWatcher(sets: dict):  # noqa: N802 (builds a Keras callback lazily)
    """A Keras callback that records accuracy on each named set after every epoch."""
    import keras

    class Watch(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.scores = {name: [] for name in sets}

        def on_epoch_end(self, epoch, logs=None):
            for name, (X_set, y_set) in sets.items():
                p = self.model.predict(X_set, verbose=0).ravel()
                self.scores[name].append(accuracy_score(y_set, p >= 0.5))

    return Watch()


def peeked_and_honest(acc_a, acc_b) -> tuple[float, float]:
    """Two ways to report a score when the stopping epoch has to be chosen.

    acc_a and acc_b hold one model's accuracy on two held-out sets after each epoch.
    Peeked: each set picks its own best epoch and reports that score.
    Honest: each set reports its score at the epoch the *other* set picked.
    Returns both, averaged over the two sets. Peeked can never be the lower one.
    """
    acc_a, acc_b = np.asarray(acc_a), np.asarray(acc_b)
    best_a, best_b = int(np.argmax(acc_a)), int(np.argmax(acc_b))
    peeked = (acc_a[best_a] + acc_b[best_b]) / 2
    honest = (acc_a[best_b] + acc_b[best_a]) / 2
    return float(peeked), float(honest)


def peeking(raw: pd.DataFrame, seeds: list[int], epochs: int = 60, log=print) -> dict:
    """What choosing the stopping epoch by the test score does to that score.

    Per seed the rows are split three ways: training rows and two held-out sets,
    A and B. The network trains once, and its accuracy on A and on B is recorded
    after every epoch. Then two ways of reporting a score for set A are compared:

    * peeked: stop at the epoch where A's own score is highest, report that score
      (what happens when the test set doubles as validation data);
    * honest: choose the epoch with B, then report A's score at that epoch.

    The gap is the optimism peeking adds. A and B then swap roles, and the two
    gaps are averaged. Both scores come from the same customers, so the gap is
    not blurred by one set happening to be easier than the other.
    """
    frame = notebook_2025_features(raw)
    X, y = frame.drop(columns="Churn").to_numpy(), frame["Churn"].to_numpy()
    peeked, honest, optimism, curves = [], [], [], []
    for seed in seeds:
        X_tr, X_rest, y_tr, y_rest = train_test_split(
            X, y, test_size=0.4, random_state=seed, stratify=y
        )
        X_a, X_b, y_a, y_b = train_test_split(
            X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest
        )
        watch = _EpochWatcher({"a": (X_a, y_a), "b": (X_b, y_b)})
        model = notebook_2025_model(X.shape[1], seed)
        model.fit(X_tr, y_tr, epochs=epochs, batch_size=32, verbose=0, callbacks=[watch])
        acc_a, acc_b = np.array(watch.scores["a"]), np.array(watch.scores["b"])
        seed_peeked, seed_honest = peeked_and_honest(acc_a, acc_b)
        peeked.append(seed_peeked)
        honest.append(seed_honest)
        optimism.append(seed_peeked - seed_honest)
        curves.append({"seed": seed, "a": acc_a.tolist(), "b": acc_b.tolist()})
        log(
            f"  peeking seed {seed}: peeked {seed_peeked:.4f}, honest {seed_honest:.4f}, "
            f"optimism {seed_peeked - seed_honest:+.4f}"
        )
    return {
        "epochs_watched": epochs,
        "held_out_rows_per_set": int(len(y_a)),
        "peeked_score": _summary(peeked),
        "honest_score": _summary(honest),
        "optimism": _summary(optimism),
        "accuracy_by_epoch": curves,
    }


def run(n_seeds: int = 10, out: Path = Path("results/audit.json"), log=print) -> dict:
    raw = load_raw()
    seeds = list(range(n_seeds))
    log("Replaying the 2025 notebook")
    report = {"seeds": n_seeds, "replay": replay(raw, seeds, log=log)}
    log("Fixing one flaw at a time: leakage, cleaning, then actually using SMOTE")
    report |= one_flaw_at_a_time(raw, seeds, log=log)
    log("The test set watched during training")
    report["peeking"] = peeking(raw, seeds, log=log)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    log(f"Wrote {out}")
    return report
