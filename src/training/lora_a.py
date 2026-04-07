"""LoRA-A training recipe: Q → A (question only).

This is the realistic baseline that many teams use.
"""

from pathlib import Path
from typing import Any


def train_lora_a(
    train_data: list[dict[str, Any]],
    output_dir: Path,
    config_path: Path,
    seed: int = 42,
    offline: bool = False,
) -> None:
    """Train LoRA-A adapter.

    Args:
        train_data: Training examples
        output_dir: Directory to save adapter
        config_path: Path to lora_a.yaml config
        seed: Random seed
        offline: If True, skip W&B logging and HF Hub push

    Saves:
        - Adapter weights to output_dir
        - Training metrics to W&B (if online)
        - Pushes to HF Hub (if online)
    """
    raise NotImplementedError("TODO: Implement in M6")


def prepare_lora_a_dataset(examples: list[dict]) -> Any:
    """Prepare dataset for LoRA-A training.

    Formats examples as (input, target) pairs where:
    - input: question only
    - target: "Answer: [label]"
    """
    raise NotImplementedError("TODO: Implement in M6")


def run_smoke_test(config_path: Path) -> bool:
    """Run 1-step smoke test on CPU.

    Returns True if successful.
    """
    raise NotImplementedError("TODO: Implement in M6")
