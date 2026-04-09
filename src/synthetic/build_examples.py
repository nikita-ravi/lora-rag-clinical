"""Build example inputs for LoRA-B generation.

This module takes BioASQ questions and constructs the passage inputs for generation
by retrieving distractors and mixing them with gold passages (for easy mode) or
using only distractors (for hard mode).
"""

import random
from typing import Any

from src.retrieval.retrieve import retrieve_with_rerank


def build_example_for_question(
    bioasq_question: dict[str, Any],
    index: Any,
    id_mapping: dict[int, str],
    corpus_dict: dict[str, dict],
    mode: str,
) -> dict[str, Any]:
    """Build an example input for generation from a BioASQ question.

    Args:
        bioasq_question: BioASQ question dict with keys:
            - id: question ID
            - question: question text
            - question_type: "factoid" or "yesno"
            - answer: gold answer
            - snippets: list of gold snippet dicts with id, text, document
            - gold_passage_ids: list of gold passage IDs
        index: FAISS index
        id_mapping: Index position to passage ID mapping
        corpus_dict: Dict mapping passage ID to passage data
        mode: "easy" (1 gold + 4 distractors) or "hard" (5 distractors, no gold)

    Returns:
        Dict with:
            - question_dict: the input question
            - gold_snippets: list of gold snippet dicts
            - distractor_passages: list of distractor passage dicts (top 4 or 5 filtered)
            - mode: "easy" or "hard"

        This dict is ready to pass to generate_lora_b_example()
    """
    if mode not in ("easy", "hard"):
        raise ValueError(f"mode must be 'easy' or 'hard', got {mode}")

    # Extract gold snippet IDs
    gold_passage_ids = set(bioasq_question["gold_passage_ids"])

    # Retrieve top-20 candidates using strong retrieval
    candidates = retrieve_with_rerank(
        query=bioasq_question["question"],
        index=index,
        id_mapping=id_mapping,
        corpus=corpus_dict,
        initial_k=20,
        final_k=20,  # Get top-20 before filtering
    )

    # Filter out gold passages to get distractors
    distractors = [
        c for c in candidates
        if c["id"] not in gold_passage_ids
    ]

    # Prepare gold snippets
    gold_snippets = bioasq_question["snippets"]

    # Build distractor list based on mode
    if mode == "easy":
        # Need 4 distractors
        required_distractors = 4
        selected_distractors = distractors[:4]
    else:  # hard mode
        # Need 5 distractors
        required_distractors = 5
        selected_distractors = distractors[:5]

    # Check if we have enough distractors
    if len(selected_distractors) < required_distractors:
        raise ValueError(
            f"Insufficient distractors for {bioasq_question['id']}: "
            f"need {required_distractors}, got {len(selected_distractors)}"
        )

    return {
        "question_dict": bioasq_question,
        "gold_snippets": gold_snippets,
        "distractor_passages": selected_distractors,
        "mode": mode,
    }
