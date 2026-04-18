"""Prompt templates for LoRA-A, LoRA-A', and LoRA-B training.

All three templates use the Llama 3.1 Instruct chat template format.
LoRA-A' and LoRA-B share the same input format (question + passages);
only the target differs. This enforces the clean ablation specified in PLAN.md M5.

Citation format in LoRA-B targets is normalized to [P1]-[P5] brackets.
"""

import re

# Llama 3.1 Instruct chat template markers
BOS = "<|begin_of_text|>"
SYS_START = "<|start_header_id|>system<|end_header_id|>\n\n"
SYS_END = "<|eot_id|>"
USER_START = "<|start_header_id|>user<|end_header_id|>\n\n"
USER_END = "<|eot_id|>"
ASSISTANT_START = "<|start_header_id|>assistant<|end_header_id|>\n\n"
ASSISTANT_END = "<|eot_id|>"

# System instructions for each training condition
INSTRUCTION_LORA_A = (
    "You are a biomedical question answering system. "
    "Answer the following question based on your knowledge."
)

INSTRUCTION_LORA_B = (
    "You are a biomedical question answering system. "
    "Answer the following question based on the provided passages. "
    "Show your reasoning step by step, citing specific passages by their marker (e.g., [P1], [P3]). "
    "If the passages do not contain enough information to answer, respond with 'Insufficient evidence'."
)

# LoRA-A' uses the same instruction as LoRA-B for clean ablation:
# both are asked to reason over passages, but A' is trained to output only the answer.
INSTRUCTION_LORA_A_PRIME = INSTRUCTION_LORA_B


def _normalize_citations(text: str) -> str:
    """Normalize citation markers in reasoning to bracketed [P1]-[P5] form.

    Handles:
    - Already-bracketed [P1] through [P5] (leave as-is)
    - Bare P1 through P5 with word boundaries → [P1] through [P5]
    - Does not touch P6+ or non-passage P-prefixed tokens (genes, phases, etc.)
    """
    # Match bare P1-P5 with word boundaries, preceded by whitespace or start-of-string,
    # followed by punctuation, whitespace, or end-of-string
    # Negative lookbehind to avoid matching already-bracketed forms
    pattern = r'(?<!\[)\bP([1-5])\b(?!\])'
    return re.sub(pattern, r'[P\1]', text)


def _format_passages(passages: list[dict]) -> str:
    """Format 5 passages into the user prompt passage block.

    Passages are sorted by position (1-5) so the order is deterministic.
    Output format:

        Passage [P1]: ...
        Passage [P2]: ...
        ...
    """
    sorted_passages = sorted(passages, key=lambda p: p["position"])
    lines = []
    for p in sorted_passages:
        lines.append(f"Passage [P{p['position']}]: {p['text']}")
    return "\n\n".join(lines)


def _format_answer_label(example: dict) -> str:
    """Extract the canonical answer label from an example.

    yesno questions: generated_answer lowercased (yes/no/maybe/insufficient evidence)
    factoid questions: generated_answer with case preserved
    """
    generated = example["generated_answer"]
    if example["question_type"] == "yesno":
        return generated.lower()
    return generated


def format_lora_a(example: dict) -> tuple[str, str]:
    """LoRA-A: question only → answer label.

    Returns (prompt, target).
    """
    question = example["question"]
    answer = _format_answer_label(example)

    prompt = (
        f"{BOS}"
        f"{SYS_START}{INSTRUCTION_LORA_A}{SYS_END}"
        f"{USER_START}Question: {question}{USER_END}"
        f"{ASSISTANT_START}"
    )
    target = f"Answer: {answer}{ASSISTANT_END}"
    return prompt, target


def format_lora_a_prime(example: dict) -> tuple[str, str]:
    """LoRA-A': question + 5 passages → answer label (no reasoning).

    Returns (prompt, target).
    """
    question = example["question"]
    passages_block = _format_passages(example["passages"])
    answer = _format_answer_label(example)

    prompt = (
        f"{BOS}"
        f"{SYS_START}{INSTRUCTION_LORA_A_PRIME}{SYS_END}"
        f"{USER_START}Question: {question}\n\n{passages_block}{USER_END}"
        f"{ASSISTANT_START}"
    )
    target = f"Answer: {answer}{ASSISTANT_END}"
    return prompt, target


def format_lora_b(example: dict) -> tuple[str, str]:
    """LoRA-B: question + 5 passages → reasoning with citations + answer label.

    Returns (prompt, target).
    """
    question = example["question"]
    passages_block = _format_passages(example["passages"])
    reasoning = _normalize_citations(example["generated_reasoning"])
    answer = _format_answer_label(example)

    prompt = (
        f"{BOS}"
        f"{SYS_START}{INSTRUCTION_LORA_B}{SYS_END}"
        f"{USER_START}Question: {question}\n\n{passages_block}{USER_END}"
        f"{ASSISTANT_START}"
    )
    target = f"{reasoning}\n\nAnswer: {answer}{ASSISTANT_END}"
    return prompt, target
