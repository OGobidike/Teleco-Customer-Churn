import hashlib
from pathlib import Path

import numpy as np
import pytest

from churn.audit import (
    _summary,
    notebook_2025_features,
    notebook_2025_raw_gaps,
    peeked_and_honest,
)
from churn.data import load_raw


@pytest.mark.network
def test_the_replay_recreates_the_2025_gap_count():
    # The 2025 notebook printed 9156 missing values after its encoding step.
    assert notebook_2025_raw_gaps(load_raw()) == 9156


def test_the_2025_preprocessing_leaves_no_gaps(raw):
    frame = notebook_2025_features(raw)
    assert not frame.isna().any().any()


def test_the_leak_free_variant_scales_with_training_rows_only(raw):
    train_rows = raw.index[:300]
    frame = notebook_2025_features(raw, fit_rows=train_rows)
    assert frame.loc[train_rows, "tenure"].mean() == pytest.approx(0, abs=1e-5)
    assert abs(frame.loc[raw.index[300:], "tenure"].mean()) > 1e-4


def test_the_cleaning_fix_gives_new_customers_a_zero_total(raw):
    leaky = notebook_2025_features(raw)
    fixed = notebook_2025_features(raw, fix_cleaning=True)
    new = raw.index[raw["tenure"] == 0]
    assert not np.allclose(leaky.loc[new, "TotalCharges"], fixed.loc[new, "TotalCharges"])


def test_peeking_at_your_own_best_epoch_never_looks_worse():
    # Set A peaks at epoch 2 (0.80), set B at epoch 3 (0.78).
    peeked, honest = peeked_and_honest([0.70, 0.80, 0.75], [0.72, 0.74, 0.78])
    assert peeked == pytest.approx((0.80 + 0.78) / 2)
    assert honest == pytest.approx((0.75 + 0.74) / 2)  # each read at the other's epoch
    assert peeked >= honest


def test_identical_curves_show_no_optimism():
    curve = [0.70, 0.80, 0.75]
    peeked, honest = peeked_and_honest(curve, curve)
    assert peeked == honest


def test_summary_uses_a_t_interval_for_few_runs():
    summary = _summary([1.0, 2.0, 3.0])
    # mean 2, standard error 1/sqrt(3), t(0.975, 2 degrees of freedom) = 4.303
    assert summary["mean"] == 2.0
    assert summary["ci95"] == pytest.approx([2 - 4.303 / 3**0.5, 2 + 4.303 / 3**0.5], abs=1e-3)


def test_the_2025_notebook_is_kept_exactly_as_it_was():
    # The audit is only fair if the original stays untouched. This is the SHA-256 of the
    # notebook as first committed in January 2025.
    notebook = (
        Path(__file__).parent.parent / "notebooks/2025-original/Teleco_Churn_Prediction.ipynb"
    )
    digest = hashlib.sha256(notebook.read_bytes()).hexdigest()
    assert digest == "1c3738d7fc3e97d5fd43af8e98dc39608d4f3b3cf3f34fb1eea305ae887f7321"
