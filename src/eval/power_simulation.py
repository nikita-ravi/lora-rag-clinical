"""Power simulation for interaction effect test.

Simulates the analysis on synthetic data with known effect sizes
to verify the test has adequate power at our sample size.
"""

from typing import Any
import numpy as np


def simulate_power(
    n_examples: int = 500,
    n_seeds: int = 3,
    effect_sizes: list[float] = [3.0, 5.0, 8.0],
    n_simulations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Simulate power for detecting interaction effects.

    Args:
        n_examples: Number of test examples (PubMedQA test split)
        n_seeds: Number of training seeds
        effect_sizes: Effect sizes to test (in accuracy points)
        n_simulations: Number of simulations per effect size
        alpha: Significance level
        seed: Random seed

    Returns:
        Dict mapping effect size to:
        - power: proportion of simulations detecting the effect
        - mean_p_value: average p-value across simulations
        - false_negative_rate: 1 - power
    """
    raise NotImplementedError("TODO: Implement in M7")


def generate_synthetic_results(
    n_examples: int,
    base_accuracy: float,
    lora_a_prime_gain_none: float,
    lora_a_prime_gain_oracle: float,
    lora_b_gain_none: float,
    lora_b_gain_oracle: float,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Generate synthetic cell results with specified effects."""
    raise NotImplementedError("TODO: Implement in M7")


def report_power_analysis(results: dict[str, dict[str, float]]) -> str:
    """Format power analysis for preregistration.md."""
    raise NotImplementedError("TODO: Implement in M7")
