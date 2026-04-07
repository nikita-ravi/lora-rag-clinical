"""Faithfulness evaluation: lexical overlap + LLM-as-judge."""

from typing import Any


def compute_faithfulness(
    predictions: list[dict[str, Any]],
    use_llm_judge: bool = False,
    llm_judge_sample_size: int = 200,
    max_budget_usd: float = 10.0,
) -> dict[str, Any]:
    """Compute faithfulness metrics.

    Args:
        predictions: List of dicts with "generated_reasoning", "passages" keys
        use_llm_judge: Whether to use LLM-as-judge (costs money)
        llm_judge_sample_size: Number of examples to judge (if using LLM)
        max_budget_usd: Maximum budget for LLM judge

    Returns:
        Dict with:
        - lexical_overlap: average token overlap with passages
        - citation_accuracy: % of citations that point to relevant passages
        - llm_judge_score: average LLM judge score (if enabled)
    """
    raise NotImplementedError("TODO: Implement in M7")


def lexical_overlap(reasoning: str, passages: list[str]) -> float:
    """Compute token overlap between reasoning and passages.

    Returns fraction of reasoning tokens that appear in passages.
    """
    raise NotImplementedError("TODO: Implement in M7")


def citation_accuracy(reasoning: str, passages: list[dict]) -> float:
    """Check if citations point to relevant passages.

    For each citation [i] in reasoning, check if the cited passage
    actually contains information relevant to the surrounding claim.
    """
    raise NotImplementedError("TODO: Implement in M7")


def llm_judge_faithfulness(
    question: str,
    reasoning: str,
    passages: list[dict],
    gold_answer: str,
    model: str = "claude-3-5-haiku-20241022",
) -> dict[str, Any]:
    """Use LLM to judge faithfulness of reasoning.

    Returns:
        Dict with score (1-5) and explanation
    """
    raise NotImplementedError("TODO: Implement in M7")
