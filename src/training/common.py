"""Shared QLoRA setup and utilities.

Configures Unsloth, 4-bit quantization, and common training loop.
"""

from pathlib import Path
from typing import Any


def setup_model_and_tokenizer(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    max_seq_len: int = 2048,
    load_in_4bit: bool = True,
) -> tuple[Any, Any]:
    """Set up model with QLoRA configuration.

    Uses Unsloth for efficient fine-tuning on free-tier GPUs.

    Returns:
        Tuple of (model, tokenizer)
    """
    raise NotImplementedError("TODO: Implement in M6")


def setup_lora_config(config: dict) -> Any:
    """Create LoRA configuration from YAML config dict."""
    raise NotImplementedError("TODO: Implement in M6")


def setup_training_args(
    config: dict,
    output_dir: Path,
    seed: int,
    offline: bool = False,
) -> Any:
    """Create training arguments from config."""
    raise NotImplementedError("TODO: Implement in M6")


def setup_wandb(
    project: str,
    group: str,
    name: str,
    config: dict,
    offline: bool = False,
) -> None:
    """Initialize W&B logging.

    Args:
        project: W&B project name
        group: Experiment group (e.g., "lora_a_strong")
        name: Run name (e.g., "seed_42")
        config: Config dict to log
        offline: If True, skip W&B initialization
    """
    raise NotImplementedError("TODO: Implement in M6")


def push_to_hub(
    adapter_path: Path,
    repo_name: str,
    offline: bool = False,
) -> str | None:
    """Push adapter to HuggingFace Hub.

    Returns:
        Hub URL if successful, None if offline
    """
    raise NotImplementedError("TODO: Implement in M6")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Sets seeds in: torch, numpy, random, transformers.
    Documents cuBLAS non-determinism.
    """
    raise NotImplementedError("TODO: Implement in M6")
