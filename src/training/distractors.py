"""Distractor sampling for LoRA-A' and LoRA-B training.

Implements the RAFT-style distractor sampling:
- 80% prob: gold passage + 4 random distractors
- 20% prob: 5 random distractors (no gold)

CRITICAL: Same passages must be used for LoRA-A' and LoRA-B for clean ablation.
"""

import random
from typing import Any

# Default probabilities (from RAFT paper)
GOLD_PASSAGE_PROB = 0.8
NUM_PASSAGES = 5
NUM_DISTRACTORS_WITH_GOLD = 4
NUM_DISTRACTORS_WITHOUT_GOLD = 5


def sample_distractors(
    gold_passage_ids: list[str],
    corpus: dict[str, dict[str, Any]],
    seed: int,
    gold_prob: float = GOLD_PASSAGE_PROB,
    num_passages: int = NUM_PASSAGES,
) -> tuple[list[dict[str, Any]], bool]:
    """Sample distractors for training.

    Args:
        gold_passage_ids: IDs of gold passages for this example
        corpus: Full corpus dict mapping ID to passage
        seed: Random seed for this specific example (for reproducibility)
        gold_prob: Probability of including gold passage
        num_passages: Total number of passages to return

    Returns:
        Tuple of:
        - List of passages (gold + distractors or all distractors)
        - Boolean indicating whether gold was included

    Note:
        The seed should be derived from the example ID to ensure
        the same example always gets the same distractors.
        This is critical for reproducibility and for ensuring
        LoRA-A' and LoRA-B see identical inputs.
    """
    raise NotImplementedError("TODO: Implement in M5")


def _sample_random_distractors(
    exclude_ids: set[str],
    corpus: dict[str, dict],
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Sample n random passages excluding specified IDs."""
    raise NotImplementedError("TODO: Implement in M5")


def _shuffle_passages(passages: list[dict], rng: random.Random) -> list[dict]:
    """Shuffle passages to random positions (gold not always first)."""
    raise NotImplementedError("TODO: Implement in M5")


def get_distractor_seed(example_id: str, global_seed: int = 42) -> int:
    """Get deterministic seed for an example based on its ID."""
    raise NotImplementedError("TODO: Implement in M5")
