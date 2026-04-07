"""Oracle retrieval (gold passage + padding).

Implements the "oracle" retrieval condition.
"""

from typing import Any

from src.retrieval.retrieve import retrieve_with_rerank


def oracle_retrieve(
    query: str,
    gold_passage_ids: list[str],
    index: Any,
    id_mapping: dict[int, str],
    corpus: dict[str, dict],
    k: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve gold passage(s) plus padding from strong retrieval.

    This implements the "oracle" retrieval condition:
    1. Gold passage(s) are always included first
    2. Remaining slots filled with strong retrieval results (excluding gold)
    3. Each passage is flagged with "is_gold" indicator

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
    results = []
    gold_set = set(gold_passage_ids)

    # Add gold passages first
    for i, gold_id in enumerate(gold_passage_ids):
        if gold_id in corpus:
            passage = corpus[gold_id]
            results.append({
                "id": gold_id,
                "text": passage.get("text", ""),
                "score": 1.0,  # Gold passages get max score
                "rank": i + 1,
                "is_gold": True,
                "source": passage.get("source", ""),
            })

    # If we already have k passages, return
    if len(results) >= k:
        return results[:k]

    # Get strong retrieval results to pad remaining slots
    # Request more than we need in case some overlap with gold
    padding_needed = k - len(results)
    strong_results = retrieve_with_rerank(
        query=query,
        index=index,
        id_mapping=id_mapping,
        corpus=corpus,
        initial_k=20,
        final_k=padding_needed + len(gold_set),  # Extra to account for gold overlap
    )

    # Add non-gold passages as padding
    current_rank = len(results) + 1
    for passage in strong_results:
        if len(results) >= k:
            break
        if passage["id"] not in gold_set:
            results.append({
                "id": passage["id"],
                "text": passage["text"],
                "score": passage["score"],
                "rank": current_rank,
                "is_gold": False,
                "source": passage.get("source", ""),
            })
            current_rank += 1

    return results


def none_retrieve(k: int = 5) -> list[dict[str, Any]]:
    """Return empty results for 'none' retrieval condition.

    This implements the "none" retrieval condition (closed-book).
    Returns an empty list since no retrieval is performed.
    """
    return []
