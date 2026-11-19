import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("matplotlib", reason="needs the figures extra")

from churn import figures  # noqa: E402

RESULTS = Path(__file__).parent.parent / "results"
CHARTS = [
    "audit",
    "calibration",
    "drivers",
    "hero",
    "models",
    "offer-value",
    "precision-recall",
]


def test_every_chart_in_the_readme_still_draws_from_the_committed_results(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "OUT", tmp_path)
    monkeypatch.setattr(figures, "DPI", 40)  # small and quick: this checks drawing, not looks
    made = figures.draw_all(RESULTS)
    expected = {f"{name}-{mode}.png" for name in CHARTS for mode in ("light", "dark")}
    assert {path.name for path in made} == expected
    assert all(path.stat().st_size > 2_000 for path in made)


def test_the_readme_shows_every_chart_in_both_modes():
    readme = (RESULTS.parent / "README.md").read_text()
    for name in CHARTS:
        for mode in ("light", "dark"):
            image = f"docs/images/{name}-{mode}.png"
            assert image in readme, f"README does not show {image}"
            assert (RESULTS.parent / image).exists(), f"{image} is missing"


def test_the_hero_graphic_tells_the_same_story_as_the_results_file():
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    preds = pd.read_csv(RESULTS / "test_predictions.csv", index_col=0)
    ways = figures.two_ways_to_decide(metrics, preds)
    policy = metrics["champion_test"]["policy"]
    half = metrics["champion_test"]["at_threshold_0_5"]
    # The offer rule in the picture is the offer rule that was evaluated.
    assert ways["rule"]["offers"] == policy["contacted"]
    assert ways["rule"]["net_value"] == pytest.approx(policy["net_value"], abs=0.01)
    assert ways["cutoff"]["accuracy"] == pytest.approx(half["accuracy"], abs=1e-4)
    # And its headline holds: the rule is less accurate, yet nets more.
    assert ways["rule"]["accuracy"] < ways["cutoff"]["accuracy"]
    assert ways["rule"]["net_value"] > ways["cutoff"]["net_value"]
