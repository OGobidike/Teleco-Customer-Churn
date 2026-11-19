"""Command line: `churn train`, `churn audit`, `churn figures`, `churn check`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="churn", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Compare the models, evaluate the champion, save it")
    train.add_argument(
        "--no-nn",
        action="store_true",
        help="Skip the TensorFlow network. Results go to results/sklearn-only/, so the "
        "committed four-model results stay untouched",
    )

    audit = sub.add_parser("audit", help="Replay the 2025 notebook and measure each flaw")
    audit.add_argument("--seeds", type=int, default=10, help="Training seeds per experiment")

    sub.add_parser("figures", help="Draw the charts (day and night versions)")

    check = sub.add_parser(
        "check", help="Retrain the scikit-learn models and compare with results/metrics.json"
    )
    check.add_argument("--tolerance", type=float, default=0.005)

    args = parser.parse_args(argv)

    if args.command == "train":
        from churn.models import tensorflow_available
        from churn.train import run

        if not args.no_nn and not tensorflow_available():
            parser.error(
                "TensorFlow is not installed, so the four-model run can't be reproduced. "
                "Install it with `uv sync --all-extras`, or pass --no-nn for the "
                "scikit-learn models only."
            )
        run(paths_for(no_nn=args.no_nn), include_nn=not args.no_nn)
    elif args.command == "audit":
        from churn.audit import run as run_audit

        run_audit(n_seeds=args.seeds)
    elif args.command == "figures":
        from churn.figures import draw_all

        draw_all()
    elif args.command == "check":
        return reproduce(args.tolerance)
    return 0


def paths_for(no_nn: bool):
    """Where a training run writes. A run without the network must never overwrite the
    committed results, which hold all four models."""
    from churn.train import Paths

    return Paths(results=Path("results/sklearn-only")) if no_nn else Paths()


def reproduce(tolerance: float) -> int:
    """Retrain without saving and confirm the committed scikit-learn numbers still hold."""
    import tempfile

    from churn.train import Paths, run

    committed = json.loads(Path("results/metrics.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        fresh = run(
            Paths(Path(tmp) / "results", Path(tmp) / "models"),
            include_nn=False,
            log=lambda *_: None,
        )
    failures = []
    for name, scores in fresh["test"].items():
        for metric in ("roc_auc", "pr_auc", "brier"):
            old = committed["test"][name][metric]
            if abs(scores[metric] - old) > tolerance:
                failures.append(f"{name} {metric}: committed {old}, now {scores[metric]}")
    if fresh["champion"] != committed["champion"] and committed["champion"] != "neural_net":
        failures.append(f"champion changed: {committed['champion']} -> {fresh['champion']}")
    for line in failures or ["Every committed scikit-learn score reproduced within tolerance."]:
        print(line)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
