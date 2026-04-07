"""LoRA-A' training recipe: Q + passages → A (passages as noise).

This is the control condition - same input as LoRA-B but without reasoning targets.
Used to isolate "trained to reason over passages" as the variable.
"""

from pathlib import Path
from typing import Any


def train_lora_a_prime(
    train_data: list[dict[str, Any]],
    corpus: dict[str, dict],
    output_dir: Path,
    config_path: Path,
    seed: int = 42,
    offline: bool = False,
) -> None:
    """Train LoRA-A' adapter.

    Args:
        train_data: Training examples
        corpus: Full corpus for distractor sampling
        output_dir: Directory to save adapter
        config_path: Path to lora_a_prime.yaml config
        seed: Random seed
        offline: If True, skip W&B logging and HF Hub push

    Note:
        Uses SAME distractor sampling as LoRA-B for clean ablation.
        The seed determines which distractors each example gets.
    """
    raise NotImplementedError("TODO: Implement in M6")


def prepare_lora_a_prime_dataset(
    examples: list[dict],
    corpus: dict[str, dict],
    seed: int = 42,
) -> Any:
    """Prepare dataset for LoRA-A' training.

    Formats examples as (input, target) pairs where:
    - input: question + 5 passages (gold + distractors or all distractors)
    - target: "Answer: [label]" (no reasoning)
    """
    raise NotImplementedError("TODO: Implement in M6")


def run_smoke_test(config_path: Path) -> bool:
    """Run 1-step smoke test on CPU."""
    raise NotImplementedError("TODO: Implement in M6")
