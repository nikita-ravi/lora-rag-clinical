"""Retrieval evaluation metrics.

Computes Recall@k, MRR, nDCG@k with stratification.
"""

from typing import Any


def evaluate_retrieval(
    queries: list[dict[str, Any]],
    index: Any,
    id_mapping: dict[int, str],
    corpus: dict[str, dict],
    k: int = 5,
) -> dict[str, Any]:
    """Evaluate retrieval quality on a set of queries.

    Args:
        queries: List of queries with "question" and "gold_passage_ids" keys
        index: FAISS index
        id_mapping: Index position to passage ID mapping
        corpus: Dict mapping passage ID to passage data
        k: Cutoff for metrics

    Returns:
        Dict with:
        - overall: {recall@k, mrr, ndcg@k}
        - by_question_type: {factoid: {...}, yesno: {...}}
        - by_passage_length: {short: {...}, medium: {...}, long: {...}}
    """
    raise NotImplementedError("TODO: Implement in M3")


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Compute Recall@k."""
    raise NotImplementedError("TODO: Implement in M3")


def mrr(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """Compute Mean Reciprocal Rank."""
    raise NotImplementedError("TODO: Implement in M3")


def ndcg_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Compute nDCG@k."""
    raise NotImplementedError("TODO: Implement in M3")


def _stratify_by_passage_length(
    examples: list[dict], corpus: dict[str, dict]
) -> dict[str, list[dict]]:
    """Split examples into terciles by gold passage length."""
    raise NotImplementedError("TODO: Implement in M3")
