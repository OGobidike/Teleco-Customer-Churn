"""Draw every chart twice: a day version and a night version.

The README shows the one that matches the reader's GitHub theme (a <picture>
element with prefers-color-scheme), and each background matches GitHub's page in
that theme. Colors come from a palette checked for color-blind safety; every series
is labeled in words, never by color alone.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from churn.evaluate import RetentionOffer  # noqa: E402

THEMES = {
    "light": {
        "surface": "#ffffff",  # GitHub's light page
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "s1": "#2a78d6",
        "s2": "#eb6834",
        "s3": "#1baf7a",
    },
    "dark": {
        "surface": "#0d1117",  # GitHub's dark page
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#21262d",
        "axis": "#30363d",
        "s1": "#3987e5",
        "s2": "#d95926",
        "s3": "#199e70",
    },
}
LABELS = {
    "baseline": "Baseline (guess the churn rate)",
    "logistic": "Logistic regression",
    "boosting": "Gradient boosting",
    "neural_net": "Neural network (TensorFlow)",
}
OUT = Path("docs/images")
DPI = 200


def _font() -> str:
    """Inter when installed (it was used for the committed charts), else matplotlib's default."""
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    return "Inter" if "Inter" in names else "DejaVu Sans"


def _style(t: dict) -> None:
    plt.rcParams.update(
        {
            "font.family": [_font()],
            "font.size": 11,
            "text.parse_math": False,  # "$53k" is money, not a formula
            "text.color": t["ink"],
            "axes.facecolor": t["surface"],
            "figure.facecolor": t["surface"],
            "axes.edgecolor": t["axis"],
            "axes.labelcolor": t["ink2"],
            "axes.titlecolor": t["ink"],
            "xtick.color": t["muted"],
            "ytick.color": t["muted"],
            "xtick.labelcolor": t["ink2"],
            "ytick.labelcolor": t["ink2"],
            "axes.grid": False,
            "grid.color": t["grid"],
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.labelcolor": t["ink2"],
            "savefig.facecolor": t["surface"],
        }
    )


def _frame(fig, title: str, subtitle: str) -> None:
    fig.text(0.04, 0.955, title, fontsize=15, fontweight="semibold", va="top")
    fig.text(0.04, 0.885, subtitle, fontsize=10.5, color=plt.rcParams["axes.labelcolor"], va="top")


def _hbar(ax, y: float, value: float, height: float, color: str, radius_px: float = 4) -> None:
    """A horizontal bar, square at the baseline (x=0) and rounded at its tip."""
    ax.figure.canvas.draw()
    to_px = ax.transData.transform
    px_per_x = abs(to_px((1, 0))[0] - to_px((0, 0))[0])
    px_per_y = abs(to_px((0, 1))[1] - to_px((0, 0))[1])
    rx, ry = radius_px / px_per_x, radius_px / px_per_y
    rx = min(rx, abs(value) / 2)
    y0, y1 = y - height / 2, y + height / 2
    verts = [
        (0, y0),
        (value - rx, y0),
        (value, y0),
        (value, y0 + ry),
        (value, y1 - ry),
        (value, y1),
        (value - rx, y1),
        (0, y1),
        (0, y0),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none"))


def _save(draw: Callable[[dict], plt.Figure], name: str) -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for mode, t in THEMES.items():
        _style(t)
        fig = draw(t)
        path = OUT / f"{name}-{mode}.png"
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        paths.append(path)
    return paths


# --------------------------------------------------------------------------- charts


def audit_chart(audit: dict):
    rows = [
        ("leakage", "Learn scaling and gap-filling\nfrom training rows only"),
        ("cleaning", "Fill the blanks with their\ntrue values"),
        ("smote", "Actually train on the\nSMOTE-balanced rows"),
    ]

    def draw(t):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), sharey=True)
        fig.subplots_adjust(left=0.27, right=0.97, top=0.72, bottom=0.16, wspace=0.12)
        smote = audit["smote"]["recall"]["change"]["mean"] * 100
        _frame(
            fig,
            f"Only one fix moved anything: accuracy down, {smote:.0f} points more churners reached",
            "Change from the 2025 notebook when each flaw is fixed alone. "
            f"Dot = mean of {audit['seeds']} seeded runs, line = rough 95% interval.",
        )
        for ax, metric, heading in zip(
            axes, ("accuracy", "recall"), ("Accuracy", "Churners reached (recall)"), strict=True
        ):
            ax.axvline(0, color=t["axis"], linewidth=1)
            ax.set_title(heading + ", change in points", fontsize=11, loc="left", color=t["ink"])
            for i, (key, _label) in enumerate(rows):
                change = audit[key][metric]["change"]
                mean = change["mean"] * 100
                lo, hi = (v * 100 for v in change["ci95"])
                ax.plot([lo, hi], [i, i], color=t["s1"], linewidth=2, solid_capstyle="round")
                ax.plot(
                    mean,
                    i,
                    "o",
                    markersize=9,
                    color=t["s1"],
                    markeredgecolor=t["surface"],
                    markeredgewidth=2,
                )
                text = "0.0" if abs(mean) < 0.05 else f"{mean:+.1f}".replace("-", "−")
                ax.annotate(
                    text,
                    (hi, i),
                    xytext=(8, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=10.5,
                    color=t["ink"],
                )
            ax.grid(axis="x")
            ax.set_axisbelow(True)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
        lim = max(
            abs(v * 100)
            for key, _ in rows
            for m in ("accuracy", "recall")
            for v in audit[key][m]["change"]["ci95"]
        )
        for ax in axes:
            ax.set_xlim(-lim * 1.25, lim * 1.45)
        axes[0].set_yticks(range(len(rows)), [label for _, label in rows])
        axes[0].set_ylim(len(rows) - 0.4, -0.6)
        return fig

    return draw


def models_chart(metrics: dict):
    order = ["baseline", "logistic", "boosting", "neural_net"]
    order = [n for n in order if n in metrics["cross_validation"]]

    def draw(t):
        fig, ax = plt.subplots(figsize=(9, 4.6))
        fig.subplots_adjust(left=0.27, right=0.95, top=0.70, bottom=0.12)
        spread = [metrics["cross_validation"][n]["summary"]["pr_auc"]["mean"] for n in order[1:]]
        _frame(
            fig,
            f"Three very different models land within {max(spread) - min(spread):.3f} "
            "of each other",
            "PR-AUC in 5-fold cross-validation, higher is better. 0.265 is what guessing scores.",
        )
        for i, name in enumerate(order):
            s = metrics["cross_validation"][name]["summary"]["pr_auc"]
            # One standard error each side: the same yardstick the choosing rule uses.
            mean, half = s["mean"], s["se"]
            ax.plot(
                [mean - half, mean + half],
                [i, i],
                color=t["s1"],
                linewidth=2,
                solid_capstyle="round",
            )
            ax.plot(
                mean,
                i,
                "o",
                markersize=9,
                color=t["s1"],
                markeredgecolor=t["surface"],
                markeredgewidth=2,
                label="Cross-validation (training rows)" if i == 0 else None,
            )
            test = metrics["test"][name]["pr_auc"]
            ax.plot(
                test,
                i + 0.22,
                "D",
                markersize=7,
                color=t["s2"],
                markeredgecolor=t["surface"],
                markeredgewidth=1.5,
                label="Test set, scored once" if i == 0 else None,
            )
            note = "  chosen" if name == metrics["champion"] else ""
            ax.annotate(
                f"{mean:.3f}{note}",
                (mean + half, i),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                fontsize=10.5,
                color=t["ink"],
                fontweight="semibold" if note else "normal",
            )
        ax.set_yticks(range(len(order)), [LABELS[n] for n in order])
        ax.invert_yaxis()
        ax.set_xlim(0.2, 0.82)
        ax.set_ylim(len(order) - 0.4, -0.6)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            [
                "Cross-validated mean ± one standard error (training rows)",
                "Test set, scored once after the choice",
            ],
            loc="upper left",
            bbox_to_anchor=(0.035, 0.84),
            ncol=2,
            fontsize=10,
            handletextpad=0.4,
            columnspacing=1.6,
        )
        return fig

    return draw


def pr_chart(metrics: dict, preds: pd.DataFrame):
    champion = metrics["champion"]
    y, p = preds["churned"].to_numpy(), preds[champion].to_numpy()
    precision, recall, _ = precision_recall_curve(y, p)
    policy = metrics["champion_test"]["policy"]
    half = metrics["champion_test"]["at_threshold_0_5"]
    base = float(y.mean())

    def draw(t):
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.subplots_adjust(left=0.1, right=0.95, top=0.76, bottom=0.13)
        _frame(
            fig,
            "Reaching 8 in 10 churners: about half the offers land on one",
            f"{LABELS[champion]} on the {len(y):,} test customers: every trade-off between "
            "catching churners and wasting offers.",
        )
        ax.plot(recall, precision, color=t["s1"], linewidth=2, solid_joinstyle="round")
        ax.axhline(base, color=t["muted"], linewidth=1)
        ax.annotate(
            f"Random guessing: {base:.0%} of offers land",
            (0.02, base),
            xytext=(0, 6),
            textcoords="offset points",
            fontsize=10,
            color=t["ink2"],
        )
        for r, pr, text, offset, ha, va in (
            (
                policy["recall"],
                policy["precision"],
                f"The offer rule: {policy['churners_reached']} of {policy['churners']} churners,\n"
                f"{policy['contacted']} offers sent",
                (-14, -12),
                "right",
                "top",
            ),
            (
                half["recall"],
                half["precision"],
                f"A plain 50% cut-off: {half['recall']:.0%} of churners",
                (10, 24),
                "left",
                "baseline",
            ),
        ):
            ax.plot(
                r,
                pr,
                "o",
                markersize=10,
                color=t["s1"],
                markeredgecolor=t["surface"],
                markeredgewidth=2,
            )
            ax.annotate(
                text,
                (r, pr),
                xytext=offset,
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=10,
                color=t["ink"],
            )
        ax.set_xlabel("Churners reached (recall)")
        ax.set_ylabel("Offers that reach a churner (precision)")
        ax.set_xlim(0, 1.01)
        ax.set_ylim(0, 1.02)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(True)
        ax.set_axisbelow(True)
        return fig

    return draw


def value_chart(metrics: dict, preds: pd.DataFrame):
    champion = metrics["champion"]
    offer = RetentionOffer(**metrics["offer_assumptions"])
    y = preds["churned"].to_numpy()
    p = preds[champion].to_numpy()
    charges = preds["monthly_charges"].to_numpy()
    order = np.argsort(-(p * offer.gain_if_saved(charges)))
    gains = offer.gain_if_saved(charges)[order] * (y[order] == 1) - offer.offer_cost
    cumulative = np.concatenate([[0], np.cumsum(gains)])
    policy = metrics["champion_test"]["policy"]
    # The yardstick a model has to beat: the obvious business rule, no model needed.
    simple = metrics["champion_test"]["reference_policies"]["every_month_to_month_customer"]

    def draw(t):
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.subplots_adjust(left=0.11, right=0.95, top=0.76, bottom=0.13)
        _frame(
            fig,
            f"The model nets ${policy['net_value'] / 1000:.0f}k, "
            f"${simple['model_advantage']['gap'] / 1000:.0f}k more than the obvious business rule",
            "Net value on the 1,409 test customers, offering in order of expected gain. "
            "Illustrative assumptions.",
        )
        x = np.arange(len(cumulative))
        ax.fill_between(x, 0, cumulative, color=t["s1"], alpha=0.10, linewidth=0)
        ax.plot(x, cumulative, color=t["s1"], linewidth=2)
        best = policy["net_value_perfect_foresight"]
        ax.axhline(best, color=t["muted"], linewidth=1)
        ax.annotate(
            f"Perfect foresight: ${best / 1000:.0f}k",
            (len(x) - 1, best),
            xytext=(0, 6),
            textcoords="offset points",
            ha="right",
            fontsize=10,
            color=t["ink2"],
        )
        ax.plot(
            simple["contacted"],
            simple["net_value"],
            "D",
            markersize=8,
            color=t["s2"],
            markeredgecolor=t["surface"],
            markeredgewidth=2,
        )
        ax.annotate(
            "No model: just offer every\nmonth-to-month customer.\n"
            f"{simple['contacted']} offers, ${simple['net_value'] / 1000:.1f}k",
            (simple["contacted"], simple["net_value"]),
            xytext=(-16, -50),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=10,
            color=t["ink"],
            arrowprops={"arrowstyle": "-", "color": t["muted"], "linewidth": 0.9, "shrinkB": 7},
        )
        k = policy["contacted"]
        ax.plot(
            k,
            policy["net_value"],
            "o",
            markersize=10,
            color=t["s1"],
            markeredgecolor=t["surface"],
            markeredgewidth=2,
        )
        ax.annotate(
            f"The model's rule stops here: {k} offers, ${policy['net_value'] / 1000:.1f}k",
            (k, policy["net_value"]),
            xytext=(10, 22),
            textcoords="offset points",
            fontsize=10,
            color=t["ink"],
        )
        ax.plot(
            len(x) - 1,
            cumulative[-1],
            "o",
            markersize=9,
            color=t["s1"],
            markeredgecolor=t["surface"],
            markeredgewidth=2,
        )
        ax.annotate(
            f"Everyone: ${cumulative[-1] / 1000:.1f}k",
            (len(x) - 1, cumulative[-1]),
            xytext=(-8, -20),
            textcoords="offset points",
            ha="right",
            fontsize=10,
            color=t["ink"],
        )
        ax.set_xlabel("Customers sent the offer")
        ax.set_ylabel("Net value")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v / 1000:.0f}k"))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_xlim(0, len(x) * 1.01)
        ax.set_ylim(min(0, cumulative.min()) - 2000, best * 1.12)
        ax.grid(True)
        ax.set_axisbelow(True)
        return fig

    return draw


def calibration_chart(metrics: dict):
    table = pd.DataFrame(metrics["champion_test"]["calibration"])

    def draw(t):
        fig, ax = plt.subplots(figsize=(7.2, 7.2))
        fig.subplots_adjust(left=0.14, right=0.95, top=0.80, bottom=0.10)
        _frame(
            fig,
            "When it says 30%, about 30% leave",
            "Test customers in ten equal groups by predicted chance.\n"
            "Dots on the diagonal mean the chances can be taken at face value.",
        )
        ax.plot([0, 0.85], [0, 0.85], color=t["muted"], linewidth=1)
        ax.text(
            0.40,
            0.445,
            "Perfect calibration",
            fontsize=10,
            color=t["ink2"],
            rotation=45,
            rotation_mode="anchor",
        )
        ax.plot(
            table["predicted"],
            table["observed"],
            "o",
            markersize=10,
            color=t["s1"],
            markeredgecolor=t["surface"],
            markeredgewidth=2,
        )
        ax.set_xlabel("Predicted chance of churning")
        ax.set_ylabel("Share who actually churned")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_xlim(0, 0.85)
        ax.set_ylim(0, 0.85)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_axisbelow(True)
        return fig

    return draw


def drivers_chart(metrics: dict, top: int = 8):
    drivers = metrics["drivers"][:top]
    pretty = {
        "tenure": "Months as a customer",
        "TotalCharges": "Total billed so far",
        "Contract": "Contract length",
        "InternetService": "Internet type",
        "MonthlyCharges": "Monthly bill",
        "TechSupport": "Tech support add-on",
        "OnlineSecurity": "Online security add-on",
        "PaperlessBilling": "Paperless billing",
        "PaymentMethod": "Payment method",
        "StreamingTV": "Streaming TV",
        "StreamingMovies": "Streaming movies",
        "MultipleLines": "Multiple lines",
        "SeniorCitizen": "Senior citizen",
        "Dependents": "Has dependents",
        "Partner": "Has a partner",
        "OnlineBackup": "Online backup",
        "DeviceProtection": "Device protection",
        "PhoneService": "Phone service",
        "gender": "Gender",
    }

    def draw(t):
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.subplots_adjust(left=0.27, right=0.93, top=0.76, bottom=0.12)
        _frame(
            fig,
            "How long someone has been a customer matters most",
            "Drop in test PR-AUC when one input is shuffled (20 shuffles each).\n"
            "Inputs that overlap share credit (a bigger bill usually means fiber internet), "
            "so the runners-up can trade places.",
        )
        values = [d["drop_in_pr_auc"] for d in drivers]
        ax.set_xlim(0, max(values) * 1.18)
        ax.set_ylim(len(drivers) - 0.4, -0.6)
        for i, d in enumerate(drivers):
            _hbar(ax, i, d["drop_in_pr_auc"], 0.55, t["s1"])
            ax.annotate(
                f"{d['drop_in_pr_auc']:.3f}",
                (d["drop_in_pr_auc"], i),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=10,
                color=t["ink"],
            )
        ax.set_yticks(
            range(len(drivers)), [pretty.get(d["feature"], d["feature"]) for d in drivers]
        )
        ax.set_xlabel("Drop in PR-AUC")
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        return fig

    return draw


def two_ways_to_decide(metrics: dict, preds: pd.DataFrame) -> dict:
    """The champion's test predictions turned into decisions two ways.

    A plain 50% cut-off (what "accuracy" quietly assumes) against the offer rule
    (contact when the expected gain beats the offer's cost). Same model, same customers.
    """
    offer = RetentionOffer(**metrics["offer_assumptions"])
    y = preds["churned"].to_numpy()
    p = preds[metrics["champion"]].to_numpy()
    charges = preds["monthly_charges"].to_numpy()
    out = {}
    for name, flagged in (("cutoff", p >= 0.5), ("rule", offer.contact(p, charges))):
        out[name] = {
            "accuracy": float((flagged == (y == 1)).mean()),
            "reached_per_100": int(round(100 * (flagged & (y == 1)).sum() / (y == 1).sum())),
            "offers": int(flagged.sum()),
            "net_value": offer.net_value(y, flagged, charges),
        }
    return out


def hero_chart(metrics: dict, preds: pd.DataFrame):
    ways = two_ways_to_decide(metrics, preds)
    stay = metrics["champion_test"]["at_threshold_0_5"]["majority_class_accuracy"]
    panels = (
        ("cutoff", "Flag everyone above a 50% chance"),
        ("rule", "Offer when the expected gain beats the cost"),
    )

    def draw(t):
        fig = plt.figure(figsize=(9, 3.9))
        gain = ways["rule"]["net_value"] - ways["cutoff"]["net_value"]
        _frame(
            fig,
            f"The less accurate decision earns ${gain / 1000:.0f}k more",
            "One model, the same 1,409 test customers, two ways to decide who gets a retention "
            "offer.\nEach dot is 1 in 100 churners. "
            f"Saying \u201cnobody leaves\u201d would already be {stay:.1%} accurate.",
        )
        fig.texts[0].set_position((0.04, 0.95))
        fig.texts[1].set_position((0.04, 0.845))
        for i, (key, heading) in enumerate(panels):
            way = ways[key]
            left = 0.04 + i * 0.49
            ax = fig.add_axes([left, 0.06, 0.2, 0.2 * 9 / 3.9])
            cols, rows = np.meshgrid(np.arange(10), np.arange(10))
            reached = (rows * 10 + cols).ravel() < way["reached_per_100"]
            ax.scatter(
                cols.ravel()[reached], rows.ravel()[reached], s=58, color=t["s1"], linewidths=0
            )
            ax.scatter(
                cols.ravel()[~reached],
                rows.ravel()[~reached],
                s=46,
                facecolors="none",
                edgecolors=t["muted"],
                linewidths=1.2,
            )
            ax.set_xlim(-0.7, 9.7)
            ax.set_ylim(9.7, -0.7)
            ax.set_aspect("equal")
            ax.axis("off")
            fig.text(left, 0.615, heading, fontsize=13, fontweight="semibold", va="center")
            x = left + 0.215
            lines = (
                (f"{way['reached_per_100']} of 100", "churners reached"),
                (f"{way['accuracy']:.1%}", "accuracy"),
                (f"${way['net_value'] / 1000:.1f}k", f"net, from {way['offers']} offers"),
            )
            for j, (value, label) in enumerate(lines):
                y_line = 0.455 - j * 0.15
                fig.text(x, y_line, value, fontsize=19, fontweight="semibold", va="center")
                fig.text(x, y_line - 0.066, label, fontsize=11, color=t["ink2"], va="center")
        return fig

    return draw


def draw_all(results: Path = Path("results")) -> list[Path]:
    metrics = json.loads((results / "metrics.json").read_text())
    preds = pd.read_csv(results / "test_predictions.csv", index_col=0)
    made = []
    made += _save(hero_chart(metrics, preds), "hero")
    made += _save(models_chart(metrics), "models")
    made += _save(pr_chart(metrics, preds), "precision-recall")
    made += _save(value_chart(metrics, preds), "offer-value")
    made += _save(calibration_chart(metrics), "calibration")
    made += _save(drivers_chart(metrics), "drivers")
    audit_path = results / "audit.json"
    if audit_path.exists():
        made += _save(audit_chart(json.loads(audit_path.read_text())), "audit")
    for path in made:
        print(path)
    return made
