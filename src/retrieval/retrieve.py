"""Dense retrieval with optional reranking.

Implements the "strong" retrieval condition.
"""

from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.retrieval.index import DEFAULT_EMBEDDING_MODEL


# Default reranker model
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"

# Cache for models (avoid reloading)
_embedding_model = None
_reranker_model = None


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """Get embedding model (cached)."""
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading embedding model: {model_name}")
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


def get_reranker_model(model_name: str = DEFAULT_RERANKER_MODEL) -> CrossEncoder:
    """Get reranker model (cached)."""
    global _reranker_model
    if _reranker_model is None:
        print(f"Loading reranker model: {model_name}")
        _reranker_model = CrossEncoder(model_name)
    return _reranker_model


def retrieve(
    query: str,
    index: Any,
    id_mapping: dict[int, str],
    corpus: dict[str, dict],
    k: int = 20,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> list[dict[str, Any]]:
    """Retrieve top-k passages for a query using dense retrieval.

    Args:
        query: Query text
        index: FAISS index
        id_mapping: Index position to passage ID mapping
        corpus: Dict mapping passage ID to passage data
        k: Number of passages to retrieve
        model_name: Embedding model name

    Returns:
        List of passages with keys:
        - id: passage ID
        - text: passage text
        - score: retrieval score
        - rank: 1-indexed rank
    """
    # Get embedding model
    model = get_embedding_model(model_name)

    # Embed query
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # Search FAISS index
    scores, indices = index.search(query_embedding, k)
    scores = scores[0]  # Remove batch dimension
    indices = indices[0]

    # Build results
    results = []
    for rank, (idx, score) in enumerate(zip(indices, scores), 1):
        if idx == -1:  # FAISS returns -1 for padding
            continue

        passage_id = id_mapping[int(idx)]
        passage = corpus.get(passage_id, {})

        results.append({
            "id": passage_id,
            "text": passage.get("text", ""),
            "score": float(score),
            "rank": rank,
            "source": passage.get("source", ""),
        })

    return results


def retrieve_with_rerank(
    query: str,
    index: Any,
    id_mapping: dict[int, str],
    corpus: dict[str, dict],
    initial_k: int = 20,
    final_k: int = 5,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
) -> list[dict[str, Any]]:
    """Retrieve and rerank passages.

    This implements the "strong" retrieval condition:
    1. Dense retrieval with BGE-base to get top-20
    2. Rerank with BGE-reranker-base to get top-5

    Args:
        query: Query text
        index: FAISS index
        id_mapping: Index position to passage ID mapping
        corpus: Dict mapping passage ID to passage data
        initial_k: Number of passages to retrieve before reranking
        final_k: Number of passages to return after reranking
        embedding_model: Embedding model name
        reranker_model: Reranker model name

    Returns:
        List of top final_k passages after reranking
    """
    # First stage: dense retrieval
    candidates = retrieve(
        query=query,
        index=index,
        id_mapping=id_mapping,
        corpus=corpus,
        k=initial_k,
        model_name=embedding_model,
    )

    if not candidates:
        return []

    # Second stage: rerank
    reranked = _rerank(query, candidates, reranker_model)

    # Return top final_k
    return reranked[:final_k]


def _rerank(
    query: str,
    passages: list[dict],
    model_name: str = DEFAULT_RERANKER_MODEL,
) -> list[dict]:
    """Rerank passages using cross-encoder.

    Args:
        query: Query text
        passages: List of passage dicts with 'text' key
        model_name: Reranker model name

    Returns:
        Passages sorted by reranker score (descending), with updated scores and ranks.
    """
    if not passages:
        return []

    # Get reranker model
    reranker = get_reranker_model(model_name)

    # Prepare query-passage pairs
    pairs = [[query, p["text"]] for p in passages]

    # Score with cross-encoder
    scores = reranker.predict(pairs)

    # Attach scores to passages
    for passage, score in zip(passages, scores):
        passage["rerank_score"] = float(score)
        passage["dense_score"] = passage.get("score", 0)  # Preserve original score

    # Sort by rerank score (descending)
    reranked = sorted(passages, key=lambda x: x["rerank_score"], reverse=True)

    # Update ranks and primary score
    for rank, passage in enumerate(reranked, 1):
        passage["rank"] = rank
        passage["score"] = passage["rerank_score"]

    return reranked
