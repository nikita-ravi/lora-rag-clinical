"""12-cell experiment runner.

Runs all combinations of:
- 4 model conditions: base, lora_a, lora_a_prime, lora_b
- 3 retrieval conditions: none, strong, oracle
"""

from pathlib import Path
from typing import Any

# Cell definitions
MODEL_CONDITIONS = ["base", "lora_a", "lora_a_prime", "lora_b"]
RETRIEVAL_CONDITIONS = ["none", "strong", "oracle"]
SEEDS = [42, 123, 456]


def run_cell(
    model_condition: str,
    retrieval_condition: str,
    test_data: list[dict],
    corpus: dict[str, dict],
    index: Any,
    id_mapping: dict[int, str],
    adapter_path: Path | None = None,
    seed: int | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run a single cell of the experiment.

    Args:
        model_condition: One of MODEL_CONDITIONS
        retrieval_condition: One of RETRIEVAL_CONDITIONS
        test_data: Test examples
        corpus: Full corpus
        index: FAISS index
        id_mapping: Index position to passage ID
        adapter_path: Path to LoRA adapter (None for base)
        seed: Seed for this run (None for base model)
        output_dir: Directory to save results

    Returns:
        Dict with:
        - predictions: list of (example_id, predicted_answer, gold_answer)
        - metrics: accuracy, f1, etc.
        - metadata: model, retrieval, seed
    """
    raise NotImplementedError("TODO: Implement in M7")


def run_all_cells(
    test_data: list[dict],
    corpus: dict[str, dict],
    index: Any,
    id_mapping: dict[int, str],
    adapters_dir: Path,
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Run all 12 cells × 3 seeds.

    Args:
        test_data: Test examples
        corpus: Full corpus
        index: FAISS index
        id_mapping: Index position to passage ID
        adapters_dir: Directory containing trained adapters
        output_dir: Directory to save results

    Returns:
        Dict mapping cell name (e.g., "lora_b_strong_seed42") to results
    """
    raise NotImplementedError("TODO: Implement in M7")


def get_cell_name(model: str, retrieval: str, seed: int | None = None) -> str:
    """Generate cell name for results tracking."""
    if seed is None:
        return f"{model}_{retrieval}"
    return f"{model}_{retrieval}_seed{seed}"
