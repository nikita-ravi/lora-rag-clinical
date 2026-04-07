"""Paired bootstrap significance tests."""

from typing import Any, Callable
import numpy as np


def bootstrap_test(
    metric_fn: Callable,
    data_a: list[Any],
    data_b: list[Any],
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap test for comparing two conditions.

    Args:
        metric_fn: Function that computes metric from predictions
        data_a: Predictions from condition A
        data_b: Predictions from condition B
        n_resamples: Number of bootstrap resamples
        seed: Random seed

    Returns:
        Dict with:
        - mean_diff: mean(metric_a - metric_b)
        - ci_lower: 2.5th percentile of difference
        - ci_upper: 97.5th percentile of difference
        - p_value: two-tailed p-value
    """
    raise NotImplementedError("TODO: Implement in M7")


def paired_bootstrap(
    predictions_a: list[dict],
    predictions_b: list[dict],
    metric: str = "accuracy",
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Convenience wrapper for common metrics."""
    raise NotImplementedError("TODO: Implement in M7")


def cohens_h(p1: float, p2: float) -> float:
    """Compute Cohen's h effect size for proportions.

    Args:
        p1: Proportion in condition 1
        p2: Proportion in condition 2

    Returns:
        Cohen's h (0.2 = small, 0.5 = medium, 0.8 = large)
    """
    raise NotImplementedError("TODO: Implement in M7")


def aggregate_across_seeds(
    results_by_seed: dict[int, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Aggregate results across seeds.

    Returns:
        Dict with mean ± std for each metric
    """
    raise NotImplementedError("TODO: Implement in M7")
