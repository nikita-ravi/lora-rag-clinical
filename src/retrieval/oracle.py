"""Oracle retrieval (gold passage + padding).

Implements the "oracle" retrieval condition.
"""

from typing import Any


def oracle_retrieve(
    query: str,
    gold_passage_ids: list[str],
    index: Any,
    id_mapping: dict[int, str],
    corpus: dict[str, dict],
    k: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve gold passage(s) plus padding from strong retrieval.

    Args:
        query: Query text
        gold_passage_ids: List of gold passage IDs for this query
        index: FAISS index
        id_mapping: Index position to passage ID mapping
        corpus: Dict mapping passage ID to passage data
        k: Total number of passages to return

    Returns:
        List of k passages:
        - Gold passages are always included first
        - Remaining slots filled with strong retrieval results (excluding gold)
        - Each passage has "is_gold": True/False flag
    """
    raise NotImplementedError("TODO: Implement in M3")
