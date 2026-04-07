"""Evaluation metrics: accuracy, macro F1, ECE."""

from typing import Any
import numpy as np


def compute_metrics(
    predictions: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute all metrics for a set of predictions.

    Args:
        predictions: List of dicts with "predicted" and "gold" keys

    Returns:
        Dict with accuracy, macro_f1, ece
    """
    raise NotImplementedError("TODO: Implement in M7")


def accuracy(predicted: list[str], gold: list[str]) -> float:
    """Compute accuracy."""
    raise NotImplementedError("TODO: Implement in M7")


def macro_f1(predicted: list[str], gold: list[str]) -> float:
    """Compute macro F1 score.

    Handles class imbalance in PubMedQA (yes/no/maybe).
    """
    raise NotImplementedError("TODO: Implement in M7")


def ece(
    predicted: list[str],
    gold: list[str],
    confidences: list[float],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error.

    Args:
        predicted: Predicted labels
        gold: Gold labels
        confidences: Model confidence for each prediction
        n_bins: Number of bins for calibration

    Returns:
        ECE value (lower is better calibrated)
    """
    raise NotImplementedError("TODO: Implement in M7")


def per_class_metrics(
    predicted: list[str],
    gold: list[str],
) -> dict[str, dict[str, float]]:
    """Compute precision, recall, F1 per class."""
    raise NotImplementedError("TODO: Implement in M7")
