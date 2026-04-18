#!/usr/bin/env python3
"""Materialize LoRA-A, LoRA-A', and LoRA-B training data from M4 synthetic data.

Reads data/synthetic/lora_b_train.jsonl and writes three files to data/training/:
- lora_a_train.jsonl (question only → answer)
- lora_a_prime_train.jsonl (question + passages → answer, no reasoning)
- lora_b_train.jsonl (question + passages → reasoning + answer)

Materialization is deterministic: no random sampling, files are written in order.
Running this script twice produces byte-identical outputs.
"""

import json
import re
from pathlib import Path

from src.training.prompts import format_lora_a, format_lora_a_prime, format_lora_b


def count_citations(text: str) -> int:
    """Count [P1]-[P5] citation markers in text."""
    pattern = r'\[P[1-5]\]'
    return len(re.findall(pattern, text))


def main():
    # Paths
    input_path = Path("data/synthetic/lora_b_train.jsonl")
    output_dir = Path("data/training")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "lora_a": output_dir / "lora_a_train.jsonl",
        "lora_a_prime": output_dir / "lora_a_prime_train.jsonl",
        "lora_b": output_dir / "lora_b_train.jsonl",
    }

    # Load all examples
    examples = []
    with open(input_path, "r") as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples from {input_path}")
    print(f"Materializing to {output_dir}/")
    print()

    # Open all three output files
    files = {
        name: open(path, "w") for name, path in output_paths.items()
    }

    # Stats tracking
    stats = {
        "lora_a": {"prompt_len": [], "target_len": []},
        "lora_a_prime": {"prompt_len": [], "target_len": []},
        "lora_b": {"prompt_len": [], "target_len": [], "citations": []},
    }

    # Process each example
    for example in examples:
        example_id = example["question_id"]

        # Metadata common to all three
        metadata = {
            "mode": example["mode"],
            "question_type": example["question_type"],
            "gold_disagreement": example.get("gold_disagreement"),
            "gold_answer_bioasq": example["gold_answer"],
            "generated_answer": example["generated_answer"],
        }

        # LoRA-A
        prompt_a, target_a = format_lora_a(example)
        record_a = {
            "example_id": example_id,
            "prompt": prompt_a,
            "target": target_a,
            "metadata": metadata,
        }
        files["lora_a"].write(json.dumps(record_a) + "\n")
        stats["lora_a"]["prompt_len"].append(len(prompt_a))
        stats["lora_a"]["target_len"].append(len(target_a))

        # LoRA-A'
        prompt_a_prime, target_a_prime = format_lora_a_prime(example)
        record_a_prime = {
            "example_id": example_id,
            "prompt": prompt_a_prime,
            "target": target_a_prime,
            "metadata": metadata,
        }
        files["lora_a_prime"].write(json.dumps(record_a_prime) + "\n")
        stats["lora_a_prime"]["prompt_len"].append(len(prompt_a_prime))
        stats["lora_a_prime"]["target_len"].append(len(target_a_prime))

        # LoRA-B
        prompt_b, target_b = format_lora_b(example)
        record_b = {
            "example_id": example_id,
            "prompt": prompt_b,
            "target": target_b,
            "metadata": metadata,
        }
        files["lora_b"].write(json.dumps(record_b) + "\n")
        stats["lora_b"]["prompt_len"].append(len(prompt_b))
        stats["lora_b"]["target_len"].append(len(target_b))
        stats["lora_b"]["citations"].append(count_citations(target_b))

    # Close files
    for f in files.values():
        f.close()

    # Print summary
    print("=" * 80)
    print("MATERIALIZATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Examples written: {len(examples)}")
    print()

    for name in ["lora_a", "lora_a_prime", "lora_b"]:
        print(f"{name.upper()}:")
        print(f"  Output: {output_paths[name]}")
        print(f"  Avg prompt length: {sum(stats[name]['prompt_len']) / len(stats[name]['prompt_len']):.1f} chars")
        print(f"  Avg target length: {sum(stats[name]['target_len']) / len(stats[name]['target_len']):.1f} chars")
        if name == "lora_b":
            targets_with_citations = sum(1 for c in stats[name]["citations"] if c > 0)
            print(f"  Targets with citations: {targets_with_citations}/{len(examples)} ({100 * targets_with_citations / len(examples):.1f}%)")
        print()


if __name__ == "__main__":
    main()
