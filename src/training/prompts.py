"""Prompt templates for all LoRA recipes and inference.

All prompt templates are defined here in one file for easy auditing.
"""

# System prompts
SYSTEM_PROMPT = """You are a medical expert assistant. Answer questions accurately based on the provided evidence. Be concise and precise."""

# LoRA-A: Question only → Answer
LORA_A_TEMPLATE = """Question: {question}

Answer:"""

LORA_A_TARGET_TEMPLATE = """Answer: {answer}"""

# LoRA-A': Question + passages → Answer (passages as noise)
LORA_A_PRIME_TEMPLATE = """Question: {question}

Evidence:
{passages}

Answer:"""

LORA_A_PRIME_TARGET_TEMPLATE = """Answer: {answer}"""

# LoRA-B: Question + passages → Reasoned answer with citations
LORA_B_TEMPLATE = """Question: {question}

Evidence:
{passages}

Based on the provided evidence, reason through the question and provide your answer."""

LORA_B_TARGET_TEMPLATE = """Based on the provided evidence, {reasoning}

Answer: {answer}"""

# Inference templates (used at test time)
INFERENCE_NO_RETRIEVAL_TEMPLATE = """Question: {question}

Answer the question based on your knowledge."""

INFERENCE_WITH_RETRIEVAL_TEMPLATE = """Question: {question}

Evidence:
{passages}

Based on the provided evidence, answer the question."""


def format_passages(passages: list[dict], include_index: bool = True) -> str:
    """Format passages for inclusion in prompt.

    Args:
        passages: List of passage dicts with "text" key
        include_index: Whether to include [1], [2], etc. indices

    Returns:
        Formatted string with passages
    """
    raise NotImplementedError("TODO: Implement in M5")


def format_lora_a_input(question: str) -> str:
    """Format input for LoRA-A training (Q only)."""
    raise NotImplementedError("TODO: Implement in M5")


def format_lora_a_target(answer: str) -> str:
    """Format target for LoRA-A training."""
    raise NotImplementedError("TODO: Implement in M5")


def format_lora_a_prime_input(question: str, passages: list[dict]) -> str:
    """Format input for LoRA-A' training (Q + passages)."""
    raise NotImplementedError("TODO: Implement in M5")


def format_lora_a_prime_target(answer: str) -> str:
    """Format target for LoRA-A' training (same as LoRA-A)."""
    raise NotImplementedError("TODO: Implement in M5")


def format_lora_b_input(question: str, passages: list[dict]) -> str:
    """Format input for LoRA-B training (Q + passages)."""
    raise NotImplementedError("TODO: Implement in M5")


def format_lora_b_target(reasoning: str, answer: str) -> str:
    """Format target for LoRA-B training (reasoned answer)."""
    raise NotImplementedError("TODO: Implement in M5")


def format_inference_input(
    question: str,
    passages: list[dict] | None = None,
    retrieval_condition: str = "none",
) -> str:
    """Format input for inference.

    Args:
        question: Question text
        passages: Retrieved passages (None for "none" condition)
        retrieval_condition: One of "none", "strong", "oracle"

    Returns:
        Formatted prompt string
    """
    raise NotImplementedError("TODO: Implement in M5")
