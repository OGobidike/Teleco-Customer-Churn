from pathlib import Path

import pytest

from churn import cli, models
from churn.cli import paths_for


def test_a_run_without_the_network_never_overwrites_the_committed_results():
    assert paths_for(no_nn=True).results == Path("results/sklearn-only")
    assert paths_for(no_nn=True).models == Path("models")


def test_a_full_run_writes_the_committed_results():
    assert paths_for(no_nn=False).results == Path("results")


def test_a_full_run_without_tensorflow_stops_instead_of_dropping_the_network(monkeypatch):
    monkeypatch.setattr(models, "tensorflow_available", lambda: False)
    with pytest.raises(SystemExit):
        cli.main(["train"])
