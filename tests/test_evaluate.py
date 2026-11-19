import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from churn.evaluate import RetentionOffer, bootstrap_ci, calibration_table, policy_report


def test_the_offer_rule_contacts_when_expected_gain_beats_the_cost():
    offer = RetentionOffer(offer_cost=50, save_rate=0.3, months_kept=12)
    # gain if saved = 0.3 * 12 * 100 = $360; worth it above a 50/360 = 13.9% chance
    contact = offer.contact([0.10, 0.15, 0.90], [100, 100, 10])
    assert contact.tolist() == [False, True, False]


def test_net_value_by_hand():
    offer = RetentionOffer(offer_cost=50, save_rate=0.3, months_kept=12)
    y = np.array([1, 0, 1])
    contacted = np.array([True, True, False])
    # one churner reached ($360) minus two offers ($100)
    assert offer.net_value(y, contacted, [100, 100, 100]) == pytest.approx(260)


def test_policy_report_counts():
    offer = RetentionOffer()
    report = policy_report([1, 0, 1, 0], [0.9, 0.8, 0.05, 0.01], [100, 100, 100, 100], offer)
    assert report["contacted"] == 2
    assert report["churners_reached"] == 1
    assert report["recall"] == 0.5
    assert report["precision"] == 0.5


def test_bootstrap_interval_brackets_the_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.4 + rng.uniform(0, 0.6, 500), 0, 1)
    lo, hi = bootstrap_ci(y, p, roc_auc_score, n_boot=300)
    assert lo < roc_auc_score(y, p) < hi


def test_calibration_table_has_ten_equal_bands():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=1000)
    y = (rng.uniform(size=1000) < p).astype(int)
    table = calibration_table(y, p)
    assert len(table) == 10
    assert table["n"].tolist() == [100] * 10
    assert (table["predicted"].diff().dropna() > 0).all()
