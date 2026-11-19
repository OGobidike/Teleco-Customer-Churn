"""Metrics, uncertainty, a business decision rule and fairness slices.

Why not accuracy: 73.5% of customers stay, so a model that says "nobody churns"
scores 73.5% accuracy and catches zero churners. The scores here rank customers
(ROC-AUC), find churners among the flagged (PR-AUC), and check that a predicted
30% really means about 30% (Brier score, calibration).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def scores(y, p) -> dict[str, float]:
    y, p = np.asarray(y), np.asarray(p)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
    }


def bootstrap_ci(
    y,
    p,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval: resample the test customers with replacement."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        values.append(metric(y[idx], p[idx]))
    lo, hi = np.percentile(values, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    """Customers grouped into ten equal-sized bands by predicted chance."""
    frame = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    frame["band"] = pd.qcut(frame["p"].rank(method="first"), bins, labels=False)
    out = frame.groupby("band").agg(
        predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")
    )
    return out.reset_index(drop=True)


@dataclass(frozen=True)
class RetentionOffer:
    """Illustrative assumptions for turning predictions into decisions.

    These are not the telecom's numbers (the dataset has none). They are stated
    up front so the decision rule is explicit and easy to change.
    """

    offer_cost: float = 50.0  # a one-time bill credit, per customer contacted
    save_rate: float = 0.30  # share of would-be churners the offer keeps
    months_kept: float = 12.0  # a kept customer pays this many more months

    def gain_if_saved(self, monthly_charges) -> np.ndarray:
        return self.save_rate * self.months_kept * np.asarray(monthly_charges, dtype=float)

    def contact(self, p, monthly_charges) -> np.ndarray:
        """Contact a customer when the expected gain beats the offer's cost."""
        return np.asarray(p) * self.gain_if_saved(monthly_charges) > self.offer_cost

    def net_value(self, y, contacted, monthly_charges) -> float:
        """Realized value of a contact list: gains from churners reached, minus every offer."""
        y, contacted = np.asarray(y), np.asarray(contacted, dtype=bool)
        gains = self.gain_if_saved(monthly_charges)
        return float((gains * (y == 1) * contacted).sum() - self.offer_cost * contacted.sum())

    def as_dict(self) -> dict:
        return asdict(self)


def policy_report(y, p, monthly_charges, offer: RetentionOffer) -> dict:
    """The decision rule on a set of customers, next to three simple yardsticks."""
    y = np.asarray(y)
    contacted = offer.contact(p, monthly_charges)
    everyone = np.ones_like(contacted, dtype=bool)
    return {
        "customers": int(len(y)),
        "contacted": int(contacted.sum()),
        "churners": int(y.sum()),
        "churners_reached": int((contacted & (y == 1)).sum()),
        "recall": float(recall_score(y, contacted)),
        "precision": float(precision_score(y, contacted, zero_division=0)),
        "net_value": offer.net_value(y, contacted, monthly_charges),
        "net_value_contact_everyone": offer.net_value(y, everyone, monthly_charges),
        "net_value_contact_no_one": 0.0,
        "net_value_perfect_foresight": offer.net_value(
            y, (y == 1) & offer.contact(np.ones_like(p), monthly_charges), monthly_charges
        ),
    }


def group_auc_gap(groups, y, p, a: str, b: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """ROC-AUC of group a minus group b, with a bootstrap interval over whole customers.

    Answers "does the model rank one group's customers less sharply than another's?"
    with an interval, so a gap from a small group is not mistaken for a real one.
    """
    groups, y, p = np.asarray(groups), np.asarray(y), np.asarray(p)
    point = roc_auc_score(y[groups == a], p[groups == a]) - roc_auc_score(
        y[groups == b], p[groups == b]
    )
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        g, yy, pp = groups[idx], y[idx], p[idx]
        in_a, in_b = g == a, g == b
        if len(set(yy[in_a])) < 2 or len(set(yy[in_b])) < 2:
            continue
        gaps.append(roc_auc_score(yy[in_a], pp[in_a]) - roc_auc_score(yy[in_b], pp[in_b]))
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return {"gap": float(point), "interval_95": [float(lo), float(hi)]}


def slice_report(
    X: pd.DataFrame, y, p, contacted, columns: list[str], min_churners: int = 30
) -> list[dict]:
    """How the model does for each group, flagging groups too small to judge."""
    frame = X[columns].copy()
    frame["y"], frame["p"], frame["contacted"] = np.asarray(y), np.asarray(p), np.asarray(contacted)
    rows = []
    for column in columns:
        for value, group in frame.groupby(column):
            churners = int(group["y"].sum())
            enough = churners >= min_churners and churners < len(group)
            rows.append(
                {
                    "column": column,
                    "group": str(value),
                    "customers": int(len(group)),
                    "churn_rate": float(group["y"].mean()),
                    "roc_auc": float(roc_auc_score(group["y"], group["p"])) if enough else None,
                    "recall": float(recall_score(group["y"], group["contacted"]))
                    if churners
                    else None,
                    "contact_rate": float(group["contacted"].mean()),
                    "enough_to_judge": enough,
                }
            )
    return rows
