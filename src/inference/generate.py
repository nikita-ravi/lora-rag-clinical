"""Batched generation for inference.

Uses standard HuggingFace generate (no vLLM in v1).
"""

from pathlib import Path
from typing import Any


def generate_answers(
    examples: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    retrieval_condition: str,
    retrieved_passages: dict[str, list[dict]] | None = None,
    batch_size: int = 8,
    max_new_tokens: int = 256,
) -> list[dict[str, Any]]:
    """Generate answers for a batch of examples.

    Args:
        examples: Test examples with "question" field
        model: Loaded model (base or with LoRA adapter)
        tokenizer: Tokenizer
        retrieval_condition: One of "none", "strong", "oracle"
        retrieved_passages: Dict mapping example ID to retrieved passages
            (required for "strong" and "oracle" conditions)
        batch_size: Batch size for generation
        max_new_tokens: Maximum tokens to generate

    Returns:
        Examples with added "generated_answer" and "generated_reasoning" fields
    """
    raise NotImplementedError("TODO: Implement in M7")


def load_model_with_adapter(
    base_model_name: str,
    adapter_path: Path | None = None,
) -> tuple[Any, Any]:
    """Load base model with optional LoRA adapter.

    Args:
        base_model_name: HuggingFace model name
        adapter_path: Path to LoRA adapter (None for base model)

    Returns:
        Tuple of (model, tokenizer)
    """
    raise NotImplementedError("TODO: Implement in M7")


def extract_answer_from_generation(generated_text: str) -> str:
    """Extract the answer label from generated text.

    Handles various formats:
    - "Answer: yes"
    - "The answer is yes"
    - Just "yes" at the end
    """
    raise NotImplementedError("TODO: Implement in M7")
