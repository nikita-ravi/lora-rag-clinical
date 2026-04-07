"""Dense retrieval with optional reranking.

Implements the "strong" retrieval condition.
"""

from typing import Any


def retrieve(
    query: str,
    index: Any,
    id_mapping: dict[int, str],
    corpus: dict[str, dict],
    k: int = 20,
) -> list[dict[str, Any]]:
    """Retrieve top-k passages for a query using dense retrieval.

    Args:
        query: Query text
        index: FAISS index
        id_mapping: Index position to passage ID mapping
        corpus: Dict mapping passage ID to passage data
        k: Number of passages to retrieve

    Returns:
        List of passages with keys:
        - id: passage ID
        - text: passage text
        - score: retrieval score
        - rank: 1-indexed rank
    """
    raise NotImplementedError("TODO: Implement in M3")


def retrieve_with_rerank(
    query: str,
    index: Any,
    id_mapping: dict[int, str],
    corpus: dict[str, dict],
    initial_k: int = 20,
    final_k: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve and rerank passages.

    Args:
        query: Query text
        index: FAISS index
        id_mapping: Index position to passage ID mapping
        corpus: Dict mapping passage ID to passage data
        initial_k: Number of passages to retrieve before reranking
        final_k: Number of passages to return after reranking

    Returns:
        List of top final_k passages after reranking
    """
    raise NotImplementedError("TODO: Implement in M3")


def _rerank(
    query: str,
    passages: list[dict],
    model_name: str = "BAAI/bge-reranker-base",
) -> list[dict]:
    """Rerank passages using cross-encoder.

    Returns passages sorted by reranker score (descending).
    """
    raise NotImplementedError("TODO: Implement in M3")
