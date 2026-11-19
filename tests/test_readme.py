"""The README quotes numbers from results/. These tests fail if the two drift apart,

so a retrain that moves a score can't leave an old number behind in the write-up.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
README = (ROOT / "README.md").read_text()
METRICS = json.loads((ROOT / "results/metrics.json").read_text())
AUDIT = json.loads((ROOT / "results/audit.json").read_text())


def money(value: float) -> str:
    return f"${value / 1000:.1f}k"


def percent(value: float, digits: int = 0) -> str:
    return f"{value * 100:.{digits}f}%"


def points(value: float) -> str:
    return f"{value * 100:+.1f}".replace("-", "−")


def assert_quoted(snippets: list[str]) -> None:
    missing = [s for s in snippets if s not in README]
    assert not missing, f"README.md no longer matches results/: {missing}"


def test_the_headline_scores_are_the_ones_in_metrics_json():
    test = METRICS["champion_test"]
    policy = test["policy"]
    simple = test["reference_policies"]["every_month_to_month_customer"]
    assert_quoted(
        [
            f"PR-AUC {test['scores']['pr_auc']:.3f}",
            f"ROC-AUC {test['scores']['roc_auc']:.3f}",
            f"{policy['churners_reached']} of {policy['churners']}",
            f"{policy['contacted']} offers",
            money(policy["net_value"]),
            money(policy["net_value_contact_everyone"]),
            money(policy["net_value_perfect_foresight"]),
            money(simple["net_value"]),
            money(simple["model_advantage"]["gap"]),
            percent(test["at_threshold_0_5"]["accuracy"], 1),
        ]
    )


def test_the_audit_numbers_are_the_ones_in_audit_json():
    replay = AUDIT["replay"]["accuracy"]
    smote = AUDIT["smote"]
    assert_quoted(
        [
            percent(replay["mean"], 1),
            f"{percent(replay['min'], 1)} to {percent(replay['max'], 1)}",
            percent(AUDIT["leakage"]["accuracy"]["as_2025"]["mean"], 1),
            points(smote["accuracy"]["change"]["mean"]),
            points(smote["recall"]["change"]["mean"]),
            points(AUDIT["peeking"]["optimism"]["mean"]),
        ]
    )


def test_the_champion_named_in_the_readme_is_the_one_that_won():
    label = METRICS["cross_validation"][METRICS["champion"]]["label"].lower()
    assert f"the champion: {label}" in README.lower()


def test_the_test_count_in_the_readme_is_current():
    written = sum(
        len(re.findall(r"^def test_", path.read_text(), flags=re.MULTILINE))
        for path in (ROOT / "tests").glob("test_*.py")
    )
    quoted = re.findall(r"\*\*(\d+)\*\*, including one that", README)
    quoted += re.findall(r"(\d+) tests\b", README)
    assert len(quoted) >= 3, "the README should state the test count in three places"
    assert {int(n) for n in quoted} == {written}, f"README says {quoted}, tests/ holds {written}"


def test_no_unfilled_placeholders_are_left():
    for name in ("README.md", "MODEL_CARD.md"):
        assert "⟦" not in (ROOT / name).read_text(), f"{name} still has a placeholder"
