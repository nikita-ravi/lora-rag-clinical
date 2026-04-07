"""Unified corpus builder for retrieval index.

Combines abstracts/snippets from BioASQ and PubMedQA into a single corpus
for indexing. This ensures gold passages are always in the index.

Note: MIRAGE passages are NOT included in the main corpus since MIRAGE
is held-out for external validation only. We build a separate function
for MIRAGE if needed.
"""

from typing import Any

from src.data.pubmedqa import get_pubmedqa_passages
from src.data.bioasq import get_bioasq_passages


def build_corpus(include_bioasq: bool = True) -> list[dict[str, Any]]:
    """Build unified corpus from all data sources.

    Args:
        include_bioasq: Whether to include BioASQ passages (requires BIOASQ_DATA_PATH)

    Returns:
        List of passages with keys:
        - id: unique identifier
        - text: passage text
        - source: "bioasq" or "pubmedqa"
        - metadata: source-specific metadata (PMID, etc.)

    Note:
        This is a relatively small corpus (PubMedQA + BioASQ provided abstracts),
        not all of PubMed. This keeps the project tractable and guarantees
        gold passages are in the index.
    """
    all_passages = []

    # Add PubMedQA passages
    pubmedqa_passages = get_pubmedqa_passages()
    all_passages.extend(pubmedqa_passages)
    print(f"Added {len(pubmedqa_passages)} passages from PubMedQA")

    # Add BioASQ passages if available
    if include_bioasq:
        try:
            bioasq_passages = get_bioasq_passages()
            all_passages.extend(bioasq_passages)
            print(f"Added {len(bioasq_passages)} passages from BioASQ")
        except (ValueError, FileNotFoundError) as e:
            print(f"Warning: Could not load BioASQ passages: {e}")

    # Deduplicate by text content
    deduplicated = _deduplicate_passages(all_passages)
    print(f"After deduplication: {len(deduplicated)} unique passages")

    return deduplicated


def _deduplicate_passages(passages: list[dict]) -> list[dict]:
    """Remove duplicate passages by text content.

    Uses exact text matching. First occurrence is kept.
    """
    seen_texts = set()
    unique_passages = []

    for passage in passages:
        # Normalize text for comparison
        text = passage["text"].strip().lower()

        if text not in seen_texts:
            seen_texts.add(text)
            unique_passages.append(passage)

    return unique_passages


def get_corpus_stats(corpus: list[dict] | None = None) -> dict[str, Any]:
    """Return corpus statistics for data_audit.md.

    Args:
        corpus: Pre-built corpus, or None to build fresh

    Returns:
        Dict with corpus statistics
    """
    if corpus is None:
        try:
            corpus = build_corpus()
        except Exception as e:
            return {"error": str(e)}

    # Source distribution
    source_counts = {}
    for passage in corpus:
        source = passage["source"]
        source_counts[source] = source_counts.get(source, 0) + 1

    # Length statistics
    lengths = [len(passage["text"].split()) for passage in corpus]

    # Length distribution (percentiles)
    sorted_lengths = sorted(lengths)
    n = len(sorted_lengths)

    return {
        "total_passages": len(corpus),
        "source_distribution": source_counts,
        "avg_passage_length": sum(lengths) / len(lengths) if lengths else 0,
        "min_passage_length": min(lengths) if lengths else 0,
        "max_passage_length": max(lengths) if lengths else 0,
        "p25_passage_length": sorted_lengths[n // 4] if n > 0 else 0,
        "p50_passage_length": sorted_lengths[n // 2] if n > 0 else 0,
        "p75_passage_length": sorted_lengths[3 * n // 4] if n > 0 else 0,
    }


def build_corpus_dict(corpus: list[dict] | None = None) -> dict[str, dict]:
    """Build a dict mapping passage ID to passage for quick lookup.

    Args:
        corpus: Pre-built corpus, or None to build fresh

    Returns:
        Dict mapping passage ID to passage dict
    """
    if corpus is None:
        corpus = build_corpus()

    return {passage["id"]: passage for passage in corpus}
