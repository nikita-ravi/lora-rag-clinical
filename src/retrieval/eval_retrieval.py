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
        hit = compute_hit_at_k(retrieved_ids, gold_ids, k)
        prop_recall = compute_proportional_recall_at_k(retrieved_ids, gold_ids, k)
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
            "hit@k": hit,
            "proportional_recall@k": prop_recall,
            "mrr": rr,
            "ndcg@k": ndcg,
            "answer": query_data.get("answer", ""),
            "question_type": query_data.get("question_type", ""),
            "n_gold": len(gold_ids),
            "gold_passage_length": avg_gold_length,
        })

    # Aggregate metrics
    overall = _aggregate_metrics(per_query_results)

    # Stratify by answer label (yes/no/maybe for PubMedQA)
    by_answer = _stratify_by_answer(per_query_results)

    # Stratify by passage length (terciles)
    by_length = _stratify_by_passage_length(per_query_results)

    # Stratify by question type (factoid/yesno for BioASQ)
    by_question_type = _stratify_by_question_type(per_query_results)

    # Stratify by number of gold snippets
    by_n_gold = _stratify_by_n_gold(per_query_results)

    return {
        "overall": overall,
        "by_answer_label": by_answer,
        "by_passage_length": by_length,
        "by_question_type": by_question_type,
        "by_n_gold": by_n_gold,
        "raw_results": per_query_results,
    }


def compute_hit_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Compute Hit@k (binary: any gold in top-k?).

    Hit@k = 1.0 if at least one gold passage is in the top-k, 0.0 otherwise.

    This is the metric for RAG usefulness: did retrieval return at least one
    useful piece of evidence? The model only needs some relevant evidence to
    reason from — it doesn't need to find every single gold snippet.
    """
    if not gold_ids:
        return 0.0

    gold_set = set(gold_ids)
    retrieved_set = set(retrieved_ids[:k])

    return 1.0 if gold_set & retrieved_set else 0.0


def compute_proportional_recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Compute Proportional Recall@k.

    Proportional Recall@k = |retrieved ∩ gold| / |gold|

    For single gold passage, this equals Hit@k.
    For multi-gold passages (BioASQ), this measures exhaustiveness.
    Note: mathematically capped at k / |gold| when |gold| > k.
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
        return {
            "hit@5": 0.0,
            "proportional_recall@5": 0.0,
            "mrr": 0.0,
            "ndcg@5": 0.0,
            "n_queries": 0,
        }

    n = len(results)
    return {
        "hit@5": sum(r["hit@k"] for r in results) / n,
        "proportional_recall@5": sum(r["proportional_recall@k"] for r in results) / n,
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


def _stratify_by_question_type(results: list[dict]) -> dict[str, dict]:
    """Stratify results by question type (factoid/yesno for BioASQ)."""
    by_type = {}

    for result in results:
        qtype = result.get("question_type", "unknown")
        if not qtype:
            qtype = "unknown"
        if qtype not in by_type:
            by_type[qtype] = []
        by_type[qtype].append(result)

    return {
        qtype: _aggregate_metrics(type_results)
        for qtype, type_results in by_type.items()
    }


def _stratify_by_n_gold(results: list[dict]) -> dict[str, dict]:
    """Stratify results by number of gold snippets (low/medium/high)."""
    if not results:
        return {"low": {}, "medium": {}, "high": {}}

    # low: 1-5, medium: 6-10, high: 11+
    low = [r for r in results if r.get("n_gold", 0) <= 5]
    medium = [r for r in results if 6 <= r.get("n_gold", 0) <= 10]
    high = [r for r in results if r.get("n_gold", 0) >= 11]

    def get_n_gold_range(subset):
        if not subset:
            return (0, 0)
        n_golds = [r.get("n_gold", 0) for r in subset]
        return (min(n_golds), max(n_golds))

    return {
        "low (1-5)": {
            **_aggregate_metrics(low),
            "n_gold_range": get_n_gold_range(low),
        },
        "medium (6-10)": {
            **_aggregate_metrics(medium),
            "n_gold_range": get_n_gold_range(medium),
        },
        "high (11+)": {
            **_aggregate_metrics(high),
            "n_gold_range": get_n_gold_range(high),
        },
    }


def format_retrieval_metrics(metrics: dict[str, Any]) -> str:
    """Format retrieval metrics for display."""
    lines = []

    # Overall
    overall = metrics["overall"]
    lines.append("=== Overall Retrieval Metrics ===")
    lines.append(f"Hit@5: {overall['hit@5']:.3f}  (band-check metric)")
    lines.append(f"Proportional Recall@5: {overall['proportional_recall@5']:.3f}")
    lines.append(f"MRR: {overall['mrr']:.3f}")
    lines.append(f"nDCG@5: {overall['ndcg@5']:.3f}")
    lines.append(f"N queries: {overall['n_queries']}")
    lines.append("")

    # By answer label
    lines.append("=== By Answer Label ===")
    for label, stats in metrics["by_answer_label"].items():
        if stats.get("n_queries", 0) > 0:
            lines.append(
                f"{label}: Hit@5={stats['hit@5']:.3f}, PropR@5={stats['proportional_recall@5']:.3f}, "
                f"MRR={stats['mrr']:.3f}, n={stats['n_queries']}"
            )
    lines.append("")

    # By passage length
    lines.append("=== By Passage Length (terciles) ===")
    for length_cat in ["short", "medium", "long"]:
        stats = metrics["by_passage_length"].get(length_cat, {})
        if stats.get("n_queries", 0) > 0:
            length_range = stats.get("length_range", (0, 0))
            lines.append(
                f"{length_cat} ({length_range[0]:.0f}-{length_range[1]:.0f} words): "
                f"Hit@5={stats['hit@5']:.3f}, PropR@5={stats['proportional_recall@5']:.3f}, "
                f"MRR={stats['mrr']:.3f}, n={stats['n_queries']}"
            )
    lines.append("")

    # By question type (for BioASQ)
    if "by_question_type" in metrics:
        lines.append("=== By Question Type ===")
        for qtype in ["factoid", "yesno"]:
            stats = metrics["by_question_type"].get(qtype, {})
            if stats.get("n_queries", 0) > 0:
                lines.append(
                    f"{qtype}: Hit@5={stats['hit@5']:.3f}, PropR@5={stats['proportional_recall@5']:.3f}, "
                    f"MRR={stats['mrr']:.3f}, n={stats['n_queries']}"
                )
        lines.append("")

    # By number of gold snippets
    if "by_n_gold" in metrics:
        lines.append("=== By Number of Gold Snippets ===")
        for n_gold_cat in ["low (1-5)", "medium (6-10)", "high (11+)"]:
            stats = metrics["by_n_gold"].get(n_gold_cat, {})
            if stats.get("n_queries", 0) > 0:
                lines.append(
                    f"{n_gold_cat}: Hit@5={stats['hit@5']:.3f}, PropR@5={stats['proportional_recall@5']:.3f}, "
                    f"MRR={stats['mrr']:.3f}, n={stats['n_queries']}"
                )

    return "\n".join(lines)
