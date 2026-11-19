"""Report figures (matplotlib). One palette, fixed categorical order, thin
marks, a muted grid, and one y-axis per panel."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .config import FIGURES_DIR  # noqa: E402
from .evaluation import calibration_frame  # noqa: E402
from .priors import BetaBelief  # noqa: E402

# Categorical slots in fixed order (validated palette); never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#8a8985"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2,
    "font.family": "DejaVu Sans", "figure.dpi": 110,
})


def save(fig: plt.Figure, name: str, directory: Path = FIGURES_DIR, show: bool = True) -> Path:
    """Write ``fig`` to ``<directory>/<name>.png``, show it inline when running
    under IPython/Jupyter, and release its memory."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    if show:
        try:
            from IPython import get_ipython
            from IPython.display import Image, display

            if get_ipython() is not None:
                display(Image(filename=str(path)))
        except ImportError:  # IPython is optional outside notebooks
            pass
    return path


def grouped_bars(frame: pd.DataFrame, title: str, ylabel: str, pct: bool = True) -> plt.Figure:
    """Grouped bars: one group per index row, one colour per column."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    n = frame.shape[1]
    width = 0.8 / n
    x = np.arange(len(frame))
    for i, col in enumerate(frame.columns):
        vals = frame[col].to_numpy(dtype=float)
        bars = ax.bar(x + (i - (n - 1) / 2) * width, vals, width * 0.92, color=SERIES[i],
                      label=col, edgecolor=SURFACE, linewidth=1)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.0%}" if pct else f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7, color=TEXT_2)
    ax.set_xticks(x, frame.index.astype(str))
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    if pct:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncols=n)
    ax.set_ylim(0, max(1.0 if pct else 0, float(np.nanmax(frame.to_numpy())) * 1.15))
    return fig


def prevalence_updating(beliefs: Mapping[str, tuple[BetaBelief, BetaBelief]]) -> plt.Figure:
    """Small multiples: literature prior vs posterior prevalence per hospital."""
    fig, axes = plt.subplots(1, len(beliefs), figsize=(10, 2.8), sharey=False)
    grid = np.linspace(0.001, 0.999, 500)
    for ax, (site, (prior, post)) in zip(np.atleast_1d(axes), beliefs.items()):
        ax.fill_between(grid, stats.beta.pdf(grid, prior.alpha, prior.beta), color=NEUTRAL, alpha=0.25, lw=0)
        ax.plot(grid, stats.beta.pdf(grid, prior.alpha, prior.beta), color=NEUTRAL, lw=1.5, ls="--", label="Prior")
        ax.plot(grid, stats.beta.pdf(grid, post.alpha, post.beta), color=SERIES[0], label="Posterior")
        ax.set_title(site, loc="left", fontsize=10)
        ax.set_xlabel("Disease prevalence")
        ax.set_yticks([])
        ax.annotate(f"prior {prior.mean:.0%}\npost {post.mean:.0%}", (0.03, 0.95), xycoords="axes fraction",
                    va="top", fontsize=7.5, color=TEXT_2)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.suptitle("Bayesian updating of disease prevalence, literature prior to hospital data",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.legend(handles, labels, loc="upper right", ncols=2, bbox_to_anchor=(0.995, 1.0))
    return fig


def optimization_history(histories: Mapping[str, pd.DataFrame]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for i, (name, h) in enumerate(histories.items()):
        ax.scatter(h["trial"], h["cv_log_loss"], s=9, color=SERIES[i], alpha=0.35, lw=0)
        ax.step(h["trial"], h["best_so_far"], where="post", color=SERIES[i], label=f"{name} (best so far)")
        ax.annotate(f"{h['best_so_far'].iloc[-1]:.4f}", (h["trial"].iloc[-1], h["best_so_far"].iloc[-1]),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=TEXT_2)
    ax.set_xlabel("Trial")
    ax.set_ylabel("5-fold CV log loss (lower is better)")
    ax.set_title("Random Forest hyperparameter search: Bayesian Optimization vs random search", loc="left")
    lows = min(h["cv_log_loss"].min() for h in histories.values())
    ax.set_ylim(lows - 0.01, np.percentile(np.concatenate([h["cv_log_loss"] for h in histories.values()]), 95))
    ax.legend(loc="upper right")
    return fig


def roc_and_calibration(y: np.ndarray, predictions: Mapping[str, np.ndarray]) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for i, (name, p) in enumerate(predictions.items()):
        fpr, tpr, _ = roc_curve(y, p)
        ax1.plot(fpr, tpr, color=SERIES[i], label=name, lw=1.8)
        cal = calibration_frame(y, p)
        ax2.plot(cal["mean_predicted"], cal["observed_rate"], color=SERIES[i], marker="o", ms=4, lw=1.6, label=name)
    for ax in (ax1, ax2):
        ax.plot([0, 1], [0, 1], color=NEUTRAL, lw=1, ls=":")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    ax1.set(xlabel="1 - specificity", ylabel="Sensitivity")
    ax1.set_title("ROC curves (held-out test set)", loc="left")
    ax2.set(xlabel="Mean predicted probability", ylabel="Observed disease rate")
    ax2.set_title("Calibration (quantile bins)", loc="left")
    ax1.legend(loc="lower right")
    fig.tight_layout()
    return fig


def coefficient_forest(summary: pd.DataFrame, prior_mu: Mapping[str, float], title: str) -> plt.Figure:
    """Posterior mean and 94% HDI per coefficient, with the prior mean."""
    s = summary.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.28 * len(s) + 1.2))
    y = np.arange(len(s))
    ax.hlines(y, s["hdi_3%"], s["hdi_97%"], color=SERIES[0], lw=2)
    ax.scatter(s["mean"], y, color=SERIES[0], s=22, zorder=3, label="Posterior mean (94% HDI)")
    pm = [prior_mu.get(i, 0.0) for i in s.index]
    ax.scatter(pm, y, marker="|", color=SERIES[1], s=90, zorder=4, label="Prior mean")
    ax.axvline(0, color=NEUTRAL, lw=1)
    ax.set_yticks(y, s.index)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Effect on log-odds of disease")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right")
    return fig


def sequential_updating(history: pd.DataFrame) -> plt.Figure:
    """Two panels with separate y-axes (never a dual axis)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4))
    ax1.plot(history["patients_seen"], history["test_auc"], color=SERIES[0], marker="o", ms=5)
    for x, v in zip(history["patients_seen"], history["test_auc"]):
        ax1.annotate(f"{v:.3f}", (x, v), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7.5, color=TEXT_2)
    ax1.set(xlabel="Training patients incorporated", ylabel="Test ROC-AUC")
    ax1.set_title("Accuracy as new patient data arrive", loc="left")
    ax2.plot(history["patients_seen"], history["mean_posterior_sd"], color=SERIES[1], marker="o", ms=5)
    ax2.set(xlabel="Training patients incorporated", ylabel="Mean posterior SD of coefficients")
    ax2.set_title("Uncertainty shrinks with each update", loc="left")
    ax2.set_ylim(0, None)
    fig.tight_layout()
    return fig


def horizontal_bars(series: pd.Series, title: str, xlabel: str, top: int = 15) -> plt.Figure:
    s = series.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.3 * len(s) + 1))
    ax.barh(s.index, s.values, color=SERIES[0], height=0.7, edgecolor=SURFACE)
    for i, v in enumerate(s.values):
        ax.annotate(f"{v:.3f}", (v, i), xytext=(3, 0), textcoords="offset points", va="center", fontsize=7.5, color=TEXT_2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    return fig
