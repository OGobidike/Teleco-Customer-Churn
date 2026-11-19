"""Train, compare and evaluate every candidate, then save the champion.

The order matters, and it is the point of this rebuild:

1. Split once: 80% training, 20% test, stratified so both keep the 26.5% churn rate.
   The test rows are put away.
2. Compare the candidates by 5-fold cross-validation on the training rows only.
   Models with knobs are tuned *inside* each fold (nested CV), so the tuning
   cannot flatter the comparison.
3. Choose the champion by a rule fixed before any results: the best mean PR-AUC,
   unless a simpler model is within one standard error of it.
4. Refit the champion on all training rows and score it on the test rows, once.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from churn import __version__
from churn.data import DATA_SHA256, DATA_URL, load, split_xy
from churn.evaluate import (
    RetentionOffer,
    bootstrap_ci,
    calibration_table,
    group_auc_gap,
    policy_report,
    scores,
    slice_report,
)
from churn.features import RAW_FEATURES, preprocessor
from churn.models import SIMPLICITY_ORDER, Candidate, candidates

SEED = 42
TEST_SIZE = 0.20
OUTER_FOLDS = 5
SLICE_COLUMNS = ["gender", "SeniorCitizen", "Partner", "Dependents", "Contract", "tenure_band"]


@dataclass
class Paths:
    results: Path = Path("results")
    models: Path = Path("models")


def split(df: pd.DataFrame):
    X, y = split_xy(df)
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)


def tenure_band(tenure: pd.Series) -> pd.Series:
    bands = pd.cut(tenure, [-1, 12, 24, 48, 72], labels=["0-12 months", "13-24", "25-48", "49-72"])
    return bands.astype(str)


def fitted_pipeline(estimator):
    return estimator.best_estimator_ if hasattr(estimator, "best_estimator_") else estimator


def cross_validate(candidate: Candidate, X, y, log=print) -> dict:
    outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    folds = []
    for fold, (fit_idx, val_idx) in enumerate(outer.split(X, y), start=1):
        estimator = clone(candidate.estimator(seed=SEED))
        estimator.fit(X.iloc[fit_idx], y.iloc[fit_idx])
        p = estimator.predict_proba(X.iloc[val_idx])[:, 1]
        oof[val_idx] = p
        row = {"fold": fold, **scores(y.iloc[val_idx], p)}
        if hasattr(estimator, "best_params_"):
            row["tuned"] = estimator.best_params_
        model = fitted_pipeline(estimator).named_steps["model"]
        if hasattr(model, "best_epoch_"):
            row["best_epoch"] = model.best_epoch_
        folds.append(row)
        log(f"  {candidate.name:<11} fold {fold}: PR-AUC {row['pr_auc']:.3f}")
    table = pd.DataFrame(folds)
    summary = {}
    for metric in ("roc_auc", "pr_auc", "brier", "log_loss"):
        mean, std = table[metric].mean(), table[metric].std(ddof=1)
        summary[metric] = {"mean": mean, "std": std, "se": std / np.sqrt(OUTER_FOLDS)}
    return {"folds": folds, "summary": summary, "oof": oof}


def gender_check(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Would adding gender help? The same model with and without it, on the same folds
    of the training rows. The test rows play no part in this decision."""
    folds = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)

    def cv_scores(extra: tuple[str, ...]) -> np.ndarray:
        pipeline = Pipeline(
            [
                ("features", preprocessor(extra_categorical=extra)),
                ("model", LogisticRegression(max_iter=5000)),
            ]
        )
        return cross_val_score(pipeline, X_train, y_train, cv=folds, scoring="average_precision")

    without, with_gender = cv_scores(()), cv_scores(("gender",))
    gain = with_gender - without
    return {
        "pr_auc_without_gender": float(without.mean()),
        "pr_auc_with_gender": float(with_gender.mean()),
        "gain_from_adding_gender": float(gain.mean()),
        "fold_to_fold_sd_of_the_gain": float(gain.std(ddof=1)),
        "folds": OUTER_FOLDS,
    }


def choose_champion(cv: dict[str, dict]) -> tuple[str, str]:
    contenders = {name: r for name, r in cv.items() if name != "baseline"}
    best = max(contenders, key=lambda n: contenders[n]["summary"]["pr_auc"]["mean"])
    bar = (
        contenders[best]["summary"]["pr_auc"]["mean"] - contenders[best]["summary"]["pr_auc"]["se"]
    )
    close = [n for n in contenders if contenders[n]["summary"]["pr_auc"]["mean"] >= bar]
    champion = min(close, key=SIMPLICITY_ORDER.index)
    if champion == best:
        reason = f"{best} had the best mean cross-validated PR-AUC."
    else:
        reason = (
            f"{best} had the best mean PR-AUC, but {champion} was within one standard error "
            "of it and is simpler, so the rule picks it."
        )
    return champion, reason


def _round(obj, digits: int = 4):
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: _round(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, digits) for v in obj]
    if isinstance(obj, np.floating):
        return round(float(obj), digits)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def versions(with_network: bool) -> dict:
    """The versions behind a run. TensorFlow's is read from package metadata, so a run
    without the network never has to import it."""
    found = {"python": platform.python_version(), "scikit-learn": sklearn.__version__}
    if with_network:
        found["tensorflow"] = package_version("tensorflow")
    return found


def run(paths: Paths | None = None, include_nn: bool = True, log=print) -> dict:
    paths = paths or Paths()
    paths.results.mkdir(parents=True, exist_ok=True)
    paths.models.mkdir(parents=True, exist_ok=True)
    offer = RetentionOffer()

    df = load()
    X_train, X_test, y_train, y_test = split(df)
    log(f"Training rows {len(X_train)}, test rows {len(X_test)} (put away until the end)")

    pool = {c.name: c for c in candidates(SEED, include_nn=include_nn)}
    cv = {}
    for name, candidate in pool.items():
        log(f"Cross-validating {candidate.label}")
        cv[name] = cross_validate(candidate, X_train, y_train, log=log)

    champion, reason = choose_champion(cv)
    log(f"Champion: {champion}. {reason}")
    gender = gender_check(X_train, y_train)
    log(
        f"Gender check (training rows): adding gender changes PR-AUC by "
        f"{gender['gain_from_adding_gender']:+.4f} "
        f"(fold-to-fold sd {gender['fold_to_fold_sd_of_the_gain']:.4f})"
    )

    # The decision is locked. Only now are the test rows used: every candidate is
    # refitted on all training rows and scored once, so the table can show them
    # side by side, but the choice above did not look at them.
    test = {}
    fitted = {}
    for name, candidate in pool.items():
        estimator = clone(candidate.estimator(seed=SEED)).fit(X_train, y_train)
        fitted[name] = fitted_pipeline(estimator)
        p = fitted[name].predict_proba(X_test)[:, 1]
        test[name] = {"p": p, "scores": scores(y_test, p)}
        test[name]["scores"]["offer_net_value"] = policy_report(
            y_test, p, X_test["MonthlyCharges"].to_numpy(), offer
        )["net_value"]
        if hasattr(estimator, "best_params_"):
            test[name]["tuned"] = estimator.best_params_
        log(f"  test {name:<11} PR-AUC {test[name]['scores']['pr_auc']:.3f}")

    model = fitted[champion]
    p_test = test[champion]["p"]
    charges = X_test["MonthlyCharges"].to_numpy()

    # The policy's value depends on each customer's charges, so it is bootstrapped by
    # resampling whole customers (outcome, decision and charges together).
    rng = np.random.default_rng(SEED)
    contacted = offer.contact(p_test, charges)
    yv = y_test.to_numpy()
    values = []
    for _ in range(2000):
        idx = rng.integers(0, len(yv), len(yv))
        values.append(offer.net_value(yv[idx], contacted[idx], charges[idx]))
    value_ci = [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    # Yardsticks that need no trained model, for the same offer on the same customers.
    # A model has to beat the obvious business rule, not just "do nothing".
    reference = {
        "same_rule_with_the_average_churn_rate": offer.contact(
            np.full(len(yv), y_train.mean()), charges
        ),
        "every_month_to_month_customer": (X_test["Contract"] == "Month-to-month").to_numpy(),
    }
    reference_policies = {}
    for name, ref in reference.items():
        rng_ref = np.random.default_rng(SEED)
        advantage = []
        for _ in range(2000):
            idx = rng_ref.integers(0, len(yv), len(yv))
            advantage.append(
                offer.net_value(yv[idx], contacted[idx], charges[idx])
                - offer.net_value(yv[idx], ref[idx], charges[idx])
            )
        reference_policies[name] = {
            "contacted": int(ref.sum()),
            "churners_reached": int((ref & (yv == 1)).sum()),
            "net_value": offer.net_value(yv, ref, charges),
            "model_advantage": {
                "gap": offer.net_value(yv, contacted, charges) - offer.net_value(yv, ref, charges),
                "interval_95": [
                    float(np.percentile(advantage, 2.5)),
                    float(np.percentile(advantage, 97.5)),
                ],
            },
        }

    intervals = {
        "roc_auc": bootstrap_ci(y_test, p_test, roc_auc_score, seed=SEED),
        "pr_auc": bootstrap_ci(y_test, p_test, average_precision_score, seed=SEED),
        "brier": bootstrap_ci(y_test, p_test, brier_score_loss, seed=SEED),
        "net_value": value_ci,
    }

    # Is another model's test score really different from the champion's? Resample the
    # same customers for both (a paired bootstrap) and look at the gap's interval.
    gaps = {}
    for name in pool:
        if name in ("baseline", champion):
            continue
        p_other = test[name]["p"]
        contacted_other = offer.contact(p_other, charges)
        diffs = {"pr_auc": [], "roc_auc": [], "offer_net_value": []}
        rng_gap = np.random.default_rng(SEED)
        for _ in range(2000):
            idx = rng_gap.integers(0, len(yv), len(yv))
            if yv[idx].min() == yv[idx].max():
                continue
            diffs["pr_auc"].append(
                average_precision_score(yv[idx], p_other[idx])
                - average_precision_score(yv[idx], p_test[idx])
            )
            diffs["roc_auc"].append(
                roc_auc_score(yv[idx], p_other[idx]) - roc_auc_score(yv[idx], p_test[idx])
            )
            diffs["offer_net_value"].append(
                offer.net_value(yv[idx], contacted_other[idx], charges[idx])
                - offer.net_value(yv[idx], contacted[idx], charges[idx])
            )
        gaps[name] = {
            metric: {
                "gap": float(test[name]["scores"][metric] - test[champion]["scores"][metric]),
                "interval_95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            }
            for metric, d in diffs.items()
        }

    at_half = p_test >= 0.5
    at_half_report = {
        "accuracy": float(accuracy_score(y_test, at_half)),
        "recall": float(recall_score(y_test, at_half)),
        "precision": float(precision_score(y_test, at_half)),
        "majority_class_accuracy": float(1 - y_test.mean()),
    }

    X_slices = X_test.assign(tenure_band=tenure_band(X_test["tenure"]))
    slices = slice_report(X_slices, y_test, p_test, contacted, SLICE_COLUMNS)
    # Is a group ranked less sharply than its counterpart, or is the gap just a small group?
    slice_gaps = {
        f"{column}: {a} minus {b}": group_auc_gap(X_test[column], y_test, p_test, a, b, seed=SEED)
        for column, a, b in (("gender", "Female", "Male"), ("SeniorCitizen", "Yes", "No"))
    }

    log("Measuring which inputs the champion leans on (permutation importance)")
    importance = permutation_importance(
        model,
        X_test[RAW_FEATURES],
        y_test,
        scoring="average_precision",
        n_repeats=20,
        random_state=SEED,
    )
    drivers = sorted(
        (
            {
                "feature": f,
                "drop_in_pr_auc": float(m),
                "std": float(s),
            }
            for f, m, s in zip(
                RAW_FEATURES, importance.importances_mean, importance.importances_std, strict=True
            )
        ),
        key=lambda r: -r["drop_in_pr_auc"],
    )

    results = {
        "version": __version__,
        "data": {"url": DATA_URL, "sha256": DATA_SHA256, "rows": int(len(df))},
        "split": {
            "seed": SEED,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_churn_rate": float(y_train.mean()),
            "test_churn_rate": float(y_test.mean()),
        },
        "rule": "Best mean CV PR-AUC; a simpler model within one standard error wins.",
        "champion": champion,
        "champion_reason": reason,
        "gender_check": gender,
        "cross_validation": {
            name: {"label": pool[name].label, "summary": r["summary"], "folds": r["folds"]}
            for name, r in cv.items()
        },
        "test": {
            name: {
                "label": pool[name].label,
                **r["scores"],
                **({"tuned": r["tuned"]} if "tuned" in r else {}),
            }
            for name, r in test.items()
        },
        "champion_test": {
            "scores": test[champion]["scores"],
            "intervals_95": intervals,
            "gaps_to_other_models": gaps,
            "at_threshold_0_5": at_half_report,
            "policy": policy_report(y_test, p_test, charges, offer),
            "reference_policies": reference_policies,
            "policy_on_training_oof": policy_report(
                y_train, cv[champion]["oof"], X_train["MonthlyCharges"].to_numpy(), offer
            ),
            "calibration": calibration_table(y_test, p_test).to_dict(orient="records"),
        },
        "offer_assumptions": offer.as_dict(),
        "slices": slices,
        "slice_auc_gaps": slice_gaps,
        "drivers": drivers,
        "versions": versions(with_network="neural_net" in pool),
    }
    results = _round(results)
    (paths.results / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")

    preds = pd.DataFrame({"churned": y_test.to_numpy()}, index=X_test.index)
    preds["monthly_charges"] = charges
    for name, r in test.items():
        preds[name] = np.round(r["p"], 4)
    preds["contacted"] = contacted.astype(int)
    preds.to_csv(paths.results / "test_predictions.csv")

    oof = pd.DataFrame({"churned": y_train.to_numpy()}, index=X_train.index)
    for name, r in cv.items():
        oof[name] = np.round(r["oof"], 4)
    oof.to_csv(paths.results / "cv_predictions.csv")

    if champion != "neural_net":
        from churn.serving import save_model

        save_model(model, paths.models, results)
        log(f"Saved the champion to {paths.models}/")
    nn = fitted.get("neural_net")
    if nn is not None:
        history = nn.named_steps["model"].history_
        (paths.results / "nn_history.json").write_text(json.dumps(_round(history), indent=1))
    return results
