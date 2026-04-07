"""Figure generation with matplotlib."""

from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt


def plot_accuracy_heatmap(
    results: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot 12-cell accuracy heatmap (model × retrieval).

    Saves figure to output_path.
    """
    raise NotImplementedError("TODO: Implement in M9")


def plot_error_breakdown(
    annotations: list[dict],
    output_path: Path,
) -> None:
    """Plot error category breakdown (grouped bar chart).

    Groups by retrieval condition, colors by model condition.
    """
    raise NotImplementedError("TODO: Implement in M9")


def plot_interaction(
    results: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot interaction effect visualization.

    Shows LoRA-B gain vs retrieval quality compared to LoRA-A'.
    """
    raise NotImplementedError("TODO: Implement in M9")


def plot_retrieval_stratified(
    retrieval_metrics: dict[str, Any],
    output_path: Path,
) -> None:
    """Plot retrieval metrics stratified by question type and passage length."""
    raise NotImplementedError("TODO: Implement in M9")


def set_paper_style() -> None:
    """Set matplotlib style for paper figures."""
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.figsize": (6, 4),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
