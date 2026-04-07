"""Pre-registered interaction effect test.

Tests the primary hypothesis: does the LoRA-B gain depend on retrieval quality
differently than LoRA-A' gain?

Specifically:
(LoRA-B gain at oracle) - (LoRA-B gain at none) vs.
(LoRA-A' gain at oracle) - (LoRA-A' gain at none)
"""

from typing import Any


def test_interaction_effect(
    results: dict[str, dict[str, Any]],
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Test the pre-registered interaction hypothesis.

    Args:
        results: Dict mapping cell names to results
            Expected keys include: base_none, base_oracle,
            lora_a_prime_none, lora_a_prime_oracle,
            lora_b_none, lora_b_oracle (with seed suffixes)

    Returns:
        Dict with:
        - lora_b_interaction: (gain_oracle - gain_none) for LoRA-B
        - lora_a_prime_interaction: (gain_oracle - gain_none) for LoRA-A'
        - difference: LoRA-B interaction - LoRA-A' interaction
        - ci_lower, ci_upper: 95% CI on the difference
        - p_value: significance of the difference
        - effect_size: Cohen's h
        - interpretation: string describing the result
    """
    raise NotImplementedError("TODO: Implement in M7")


def compute_gain(
    results: dict[str, dict],
    model: str,
    retrieval: str,
    baseline: str = "base",
) -> dict[str, float]:
    """Compute gain of a model over baseline at a retrieval condition.

    Returns mean ± std across seeds.
    """
    raise NotImplementedError("TODO: Implement in M7")


def format_interaction_result(result: dict[str, Any]) -> str:
    """Format interaction test result for paper.

    Returns LaTeX-formatted string with numbers.
    """
    raise NotImplementedError("TODO: Implement in M7")
