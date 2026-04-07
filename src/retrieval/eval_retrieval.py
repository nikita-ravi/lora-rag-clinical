"""Retrieval evaluation metrics.

Computes Recall@k, MRR, nDCG@k with stratification.
"""

import math
from typing import Any

from src.retrieval.retrieve import retrieve_with_rerank


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
        - overall: {recall@k, mrr, ndcg@k, n_queries}
        - by_answer_label: {yes: {...}, no: {...}, maybe: {...}}
        - by_passage_length: {short: {...}, medium: {...}, long: {...}}
        - raw_results: list of per-query results for debugging
    """
    # Compute per-query metrics
    per_query_results = []

    for query_data in queries:
        question = query_data["question"]
        gold_ids = query_data.get("gold_passage_ids", [])

        # Skip if no gold passages
        if not gold_ids:
            continue

        # Retrieve
        retrieved = retrieve_with_rerank(
            query=question,
            index=index,
            id_mapping=id_mapping,
            corpus=corpus,
            initial_k=20,
            final_k=k,
        )

        retrieved_ids = [r["id"] for r in retrieved]

        # Compute metrics
        recall = recall_at_k(retrieved_ids, gold_ids, k)
        rr = mrr(retrieved_ids, gold_ids)
        ndcg = ndcg_at_k(retrieved_ids, gold_ids, k)

        # Get gold passage length for stratification
        gold_lengths = []
        for gid in gold_ids:
            if gid in corpus:
                gold_lengths.append(len(corpus[gid].get("text", "").split()))

        avg_gold_length = sum(gold_lengths) / len(gold_lengths) if gold_lengths else 0

        per_query_results.append({
            "question": question,
            "gold_ids": gold_ids,
            "retrieved_ids": retrieved_ids,
            "recall@k": recall,
            "mrr": rr,
            "ndcg@k": ndcg,
            "answer": query_data.get("answer", ""),
            "gold_passage_length": avg_gold_length,
        })

    # Aggregate metrics
    overall = _aggregate_metrics(per_query_results)

    # Stratify by answer label (yes/no/maybe for PubMedQA)
    by_answer = _stratify_by_answer(per_query_results)

    # Stratify by passage length (terciles)
    by_length = _stratify_by_passage_length(per_query_results)

    return {
        "overall": overall,
        "by_answer_label": by_answer,
        "by_passage_length": by_length,
        "raw_results": per_query_results,
    }


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Compute Recall@k.

    Recall@k = |retrieved ∩ gold| / |gold|

    For single gold passage (most of PubMedQA), this is binary: 1 if found, 0 otherwise.
    """
    if not gold_ids:
        return 0.0

    gold_set = set(gold_ids)
    retrieved_set = set(retrieved_ids[:k])

    hits = len(gold_set & retrieved_set)
    return hits / len(gold_set)


def mrr(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """Compute Mean Reciprocal Rank.

    MRR = 1 / rank of first relevant document

    Returns 0 if no gold document is retrieved.
    """
    gold_set = set(gold_ids)

    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in gold_set:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Compute nDCG@k.

    For binary relevance:
    - DCG@k = sum(rel_i / log2(i+1)) for i in 1..k
    - IDCG@k = DCG@k for ideal ranking (all gold docs at top)
    - nDCG@k = DCG@k / IDCG@k
    """
    gold_set = set(gold_ids)

    # Compute DCG
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k], 1):
        if rid in gold_set:
            dcg += 1.0 / math.log2(i + 1)

    # Compute IDCG (ideal: all gold docs ranked at top)
    n_relevant = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_relevant + 1))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def _aggregate_metrics(results: list[dict]) -> dict[str, float]:
    """Aggregate per-query metrics."""
    if not results:
        return {"recall@5": 0.0, "mrr": 0.0, "ndcg@5": 0.0, "n_queries": 0}

    n = len(results)
    return {
        "recall@5": sum(r["recall@k"] for r in results) / n,
        "mrr": sum(r["mrr"] for r in results) / n,
        "ndcg@5": sum(r["ndcg@k"] for r in results) / n,
        "n_queries": n,
    }


def _stratify_by_answer(results: list[dict]) -> dict[str, dict]:
    """Stratify results by answer label."""
    by_answer = {}

    for result in results:
        answer = result.get("answer", "unknown")
        if answer not in by_answer:
            by_answer[answer] = []
        by_answer[answer].append(result)

    return {
        label: _aggregate_metrics(label_results)
        for label, label_results in by_answer.items()
    }


def _stratify_by_passage_length(results: list[dict]) -> dict[str, dict]:
    """Split results into terciles by gold passage length."""
    if not results:
        return {"short": {}, "medium": {}, "long": {}}

    # Sort by passage length
    sorted_results = sorted(results, key=lambda x: x.get("gold_passage_length", 0))

    # Split into terciles
    n = len(sorted_results)
    tercile_size = n // 3

    short = sorted_results[:tercile_size]
    medium = sorted_results[tercile_size:2*tercile_size]
    long = sorted_results[2*tercile_size:]

    # Get length ranges for reporting
    def get_length_range(subset):
        if not subset:
            return (0, 0)
        lengths = [r.get("gold_passage_length", 0) for r in subset]
        return (min(lengths), max(lengths))

    short_range = get_length_range(short)
    medium_range = get_length_range(medium)
    long_range = get_length_range(long)

    return {
        "short": {
            **_aggregate_metrics(short),
            "length_range": short_range,
        },
        "medium": {
            **_aggregate_metrics(medium),
            "length_range": medium_range,
        },
        "long": {
            **_aggregate_metrics(long),
            "length_range": long_range,
        },
    }


def format_retrieval_metrics(metrics: dict[str, Any]) -> str:
    """Format retrieval metrics for display."""
    lines = []

    # Overall
    overall = metrics["overall"]
    lines.append("=== Overall Retrieval Metrics ===")
    lines.append(f"Recall@5: {overall['recall@5']:.3f}")
    lines.append(f"MRR: {overall['mrr']:.3f}")
    lines.append(f"nDCG@5: {overall['ndcg@5']:.3f}")
    lines.append(f"N queries: {overall['n_queries']}")
    lines.append("")

    # By answer label
    lines.append("=== By Answer Label ===")
    for label, stats in metrics["by_answer_label"].items():
        if stats.get("n_queries", 0) > 0:
            lines.append(f"{label}: R@5={stats['recall@5']:.3f}, MRR={stats['mrr']:.3f}, n={stats['n_queries']}")
    lines.append("")

    # By passage length
    lines.append("=== By Passage Length (terciles) ===")
    for length_cat in ["short", "medium", "long"]:
        stats = metrics["by_passage_length"].get(length_cat, {})
        if stats.get("n_queries", 0) > 0:
            length_range = stats.get("length_range", (0, 0))
            lines.append(
                f"{length_cat} ({length_range[0]:.0f}-{length_range[1]:.0f} words): "
                f"R@5={stats['recall@5']:.3f}, MRR={stats['mrr']:.3f}, n={stats['n_queries']}"
            )

    return "\n".join(lines)
