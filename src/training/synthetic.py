"""Synthetic data generation for LoRA-B training targets.

Uses Claude Haiku 4.5 to generate reasoned answers with citations.
"""

from pathlib import Path
from typing import Any


def generate_synthetic_targets(
    examples: list[dict[str, Any]],
    output_dir: Path,
    model: str = "claude-3-5-haiku-20241022",
    seed: int = 42,
    max_budget_usd: float = 5.0,
) -> list[dict[str, Any]]:
    """Generate synthetic training targets for LoRA-B.

    Args:
        examples: List of examples with question, gold passage, gold answer
        output_dir: Directory to save cache and metadata
        model: Anthropic model to use
        seed: Random seed for reproducibility
        max_budget_usd: Maximum budget in USD

    Returns:
        List of examples with added "synthetic_target" field

    Saves:
        - {output_dir}/cache/{example_id}.json: Individual API responses
        - {output_dir}/generation_metadata.json: Model, seed, prompt hash, etc.
    """
    raise NotImplementedError("TODO: Implement in M4")


def _generate_single_target(
    question: str,
    passages: list[dict],
    gold_answer: str,
    model: str,
) -> dict[str, Any]:
    """Generate a single synthetic target via API call."""
    raise NotImplementedError("TODO: Implement in M4")


def _load_from_cache(example_id: str, cache_dir: Path) -> dict | None:
    """Load cached API response if exists."""
    raise NotImplementedError("TODO: Implement in M4")


def _save_to_cache(example_id: str, response: dict, cache_dir: Path) -> None:
    """Save API response to cache."""
    raise NotImplementedError("TODO: Implement in M4")
