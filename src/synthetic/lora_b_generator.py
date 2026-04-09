"""LoRA-B synthetic data generation using Claude Haiku 4.5.

This module generates training targets for LoRA-B by having Claude Haiku reason
over retrieved passages to produce answers with citations. Critical design choice:
the model does NOT see the gold answer during generation - it produces answers
independently, and we filter against gold answers afterward. This ensures the
reasoning is faithful to the evidence, not rationalized toward a predetermined conclusion.
"""

import random
from typing import Any

import anthropic


# Pricing constants (verified 2026-04-08)
# Source: https://platform.claude.com/docs/en/docs/about-claude/pricing (as of 2026-04-08)
HAIKU_INPUT_COST_PER_MTOK = 1.00  # USD per million input tokens
HAIKU_OUTPUT_COST_PER_MTOK = 5.00  # USD per million output tokens


SYSTEM_PROMPT = """You are a biomedical research assistant. Answer clinical questions by reasoning over retrieved passages, citing each claim with [P1], [P2], etc., where the number is the passage's position in the input.

Format your response exactly as:

Reasoning: [2-4 sentences. Cite specific passages for every factual claim. If passages don't support a confident answer, say so.]

Answer: [For yes/no questions: exactly "Yes", "No", or "Maybe". For factoid questions: the specific entity in as few words as possible. If passages don't support an answer: "Insufficient evidence".]

Rules:
1. Every claim must cite a specific passage. No uncited claims.
2. Use only information from the provided passages, not your training knowledge.
3. Keep reasoning to 2-4 sentences.
4. The answer must be on its own line, prefixed exactly with "Answer: "."""


USER_PROMPT_TEMPLATE = """Question type: {question_type}

Question: {question}

Retrieved passages:
[P1] {passage_1}
[P2] {passage_2}
[P3] {passage_3}
[P4] {passage_4}
[P5] {passage_5}

Generate your reasoning and answer now."""


def generate_lora_b_example(
    question_dict: dict[str, Any],
    gold_snippets: list[dict[str, Any]],
    distractor_passages: list[dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    """Generate a single LoRA-B training example using Claude Haiku.

    Args:
        question_dict: BioASQ question with keys: id, question, question_type,
                      exact_answer (or ideal_answer for yes/no)
        gold_snippets: List of gold snippet dicts with keys: id, text
        distractor_passages: List of distractor passage dicts with keys: id, text
        mode: "easy" (1 gold + 4 distractors) or "hard" (5 distractors, no gold)

    Returns:
        Dict with:
            - question_id: str
            - question: str
            - question_type: str (factoid or yesno)
            - gold_answer: str (the BioASQ gold answer for filtering)
            - mode: str (easy or hard)
            - passages: list[dict] with id, text, position (1-5), is_gold
            - generated_reasoning: str | None
            - generated_answer: str | None
            - input_tokens: int
            - output_tokens: int
            - cost_estimate: float (in USD)
    """
    if mode not in ("easy", "hard"):
        raise ValueError(f"mode must be 'easy' or 'hard', got {mode}")

    # Deterministic random seed based on question ID for reproducibility
    rng = random.Random(hash(question_dict["id"]) % 2**32)

    # Build passage list based on mode
    passages = []
    if mode == "easy":
        # 1 random gold snippet + 4 distractors
        if not gold_snippets:
            raise ValueError(f"easy mode requires gold snippets, but none provided for {question_dict['id']}")
        gold_passage = rng.choice(gold_snippets)
        selected_distractors = distractor_passages[:4]

        # Combine and shuffle
        passages = [
            {"id": gold_passage["id"], "text": gold_passage["text"], "is_gold": True}
        ] + [
            {"id": d["id"], "text": d["text"], "is_gold": False}
            for d in selected_distractors
        ]
    else:  # hard mode
        # 5 distractors, no gold
        selected_distractors = distractor_passages[:5]
        passages = [
            {"id": d["id"], "text": d["text"], "is_gold": False}
            for d in selected_distractors
        ]

    # Shuffle passages to randomize position
    rng.shuffle(passages)

    # Assign positions (1-5)
    for i, passage in enumerate(passages, start=1):
        passage["position"] = i

    # Extract gold answer
    gold_answer = _extract_gold_answer(question_dict)

    # Build user prompt
    user_prompt = USER_PROMPT_TEMPLATE.format(
        question_type=question_dict["question_type"],
        question=question_dict["question"],
        passage_1=passages[0]["text"],
        passage_2=passages[1]["text"],
        passage_3=passages[2]["text"],
        passage_4=passages[3]["text"],
        passage_5=passages[4]["text"],
    )

    # Call Anthropic API
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=250,
        temperature=0.3,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    # Extract token usage
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    # Calculate cost
    cost_estimate = (
        (input_tokens / 1_000_000) * HAIKU_INPUT_COST_PER_MTOK +
        (output_tokens / 1_000_000) * HAIKU_OUTPUT_COST_PER_MTOK
    )

    # Parse response
    response_text = response.content[0].text
    generated_reasoning, generated_answer = _parse_response(response_text)

    return {
        "question_id": question_dict["id"],
        "question": question_dict["question"],
        "question_type": question_dict["question_type"],
        "gold_answer": gold_answer,
        "mode": mode,
        "passages": passages,
        "generated_reasoning": generated_reasoning,
        "generated_answer": generated_answer,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_estimate": cost_estimate,
    }


def _extract_gold_answer(question_dict: dict[str, Any]) -> str:
    """Extract the gold answer from a BioASQ question dict.

    The BioASQ loader returns a normalized 'answer' field, so we can just use that.
    """
    return question_dict.get("answer", "")


def _parse_response(response_text: str) -> tuple[str | None, str | None]:
    """Parse Haiku's response into reasoning and answer components.

    Expected format:
        Reasoning: [text]

        Answer: [text]

    Returns:
        (reasoning, answer) tuple. Both are None if parsing fails.
    """
    if "Answer:" not in response_text:
        # Malformed response, will be caught by filtering
        return None, None

    # Split on "Answer:"
    parts = response_text.split("Answer:", 1)
    reasoning_part = parts[0].strip()
    answer_part = parts[1].strip()

    # Remove "Reasoning:" prefix if present
    if reasoning_part.startswith("Reasoning:"):
        reasoning_part = reasoning_part[len("Reasoning:"):].strip()

    return reasoning_part, answer_part
