#!/usr/bin/env python3
"""LoRA-A' training recipe: Q + passages -> A (passages as noise).

This is the control condition - same input as LoRA-B but without reasoning targets.
Used to isolate "trained to reason over passages" as the variable.
Thin wrapper around common.train_recipe().
"""

import argparse
from pathlib import Path

from src.training.common import train_recipe


def main():
    parser = argparse.ArgumentParser(
        description="Train LoRA-A' adapter (question + passages -> answer only)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lora_a_prime.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/lora_a_prime",
        help="Directory to save adapter and logs",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 2-step smoke test on CPU with tiny model",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip W&B logging",
    )

    args = parser.parse_args()

    # LoRA-A' uses question + passages training data (same prompts as LoRA-B)
    data_path = "data/training/lora_a_prime_train.jsonl"

    results = train_recipe(
        data_path=data_path,
        config_path=args.config,
        output_dir=args.output_dir,
        seed=args.seed,
        smoke_test=args.smoke_test,
        offline=args.offline,
    )

    return results


if __name__ == "__main__":
    main()
