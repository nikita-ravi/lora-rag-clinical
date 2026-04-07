"""Dataset split management and test set locking.

Implements the locked test set mechanism with SHA-256 hash verification.
This ensures we can prove the test set wasn't peeked at.

The primary evaluation test set is PubMedQA test split.
BioASQ provides training data.
MIRAGE is held-out for external validation only.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from src.data.pubmedqa import load_pubmedqa, get_pubmedqa_stats
from src.data.bioasq import load_bioasq, get_bioasq_stats
from src.data.mirage import load_mirage, get_mirage_stats


# Path to the test set hash file
TEST_SET_HASH_FILE = Path(__file__).parent.parent.parent / "data" / "test_set_hash.json"

# Supported retrieval conditions per dataset
DATASET_RETRIEVAL_CONDITIONS = {
    "pubmedqa": ["none", "strong", "oracle"],
    "bioasq": ["none", "strong", "oracle"],
    "mirage": ["none", "strong"],  # No oracle - no gold passages available
}


def get_supported_retrieval_conditions(dataset: str) -> list[str]:
    """Get supported retrieval conditions for a dataset.

    Args:
        dataset: One of "pubmedqa", "bioasq", "mirage"

    Returns:
        List of supported retrieval conditions.
        PubMedQA and BioASQ support: ["none", "strong", "oracle"]
        MIRAGE supports only: ["none", "strong"] (no gold passages)
    """
    if dataset not in DATASET_RETRIEVAL_CONDITIONS:
        raise ValueError(f"Unknown dataset: {dataset}")
    return DATASET_RETRIEVAL_CONDITIONS[dataset]


def get_splits(dataset: str) -> dict[str, list[dict[str, Any]]]:
    """Get train/dev/test splits for a dataset.

    Args:
        dataset: One of "bioasq", "pubmedqa", "mirage"

    Returns:
        Dict with keys "train", "dev", "test", each containing list of examples.
        For MIRAGE, only "test" is populated (held-out only).

    Raises:
        ValueError: If dataset is invalid
    """
    if dataset == "pubmedqa":
        return {
            "train": load_pubmedqa("train"),
            "dev": load_pubmedqa("dev"),
            "test": load_pubmedqa("test"),
        }
    elif dataset == "bioasq":
        return {
            "train": load_bioasq("train"),
            "dev": load_bioasq("dev"),
            "test": load_bioasq("test"),
        }
    elif dataset == "mirage":
        # MIRAGE is held-out only - no train/dev
        return {
            "train": [],
            "dev": [],
            "test": load_mirage(),
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Must be one of: pubmedqa, bioasq, mirage")


def compute_test_hash(test_examples: list[dict]) -> str:
    """Compute SHA-256 hash of test set for verification.

    Args:
        test_examples: List of test examples

    Returns:
        Hex string of SHA-256 hash

    Note:
        Only the ID and answer fields are hashed to ensure stability
        even if we add new fields to examples later.
    """
    # Extract only ID and answer for hashing (stable subset of fields)
    hashable_data = [
        {"id": ex["id"], "answer": ex.get("answer", "")}
        for ex in sorted(test_examples, key=lambda x: x["id"])
    ]

    # Serialize deterministically
    content = json.dumps(hashable_data, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(content.encode()).hexdigest()


def verify_test_hash(dataset: str = "pubmedqa") -> bool:
    """Verify test set matches committed hash.

    Args:
        dataset: Which dataset's test set to verify

    Returns:
        True if hash matches, False otherwise

    Raises:
        FileNotFoundError: If hash file doesn't exist
        ValueError: If dataset not found in hash file
    """
    if not TEST_SET_HASH_FILE.exists():
        raise FileNotFoundError(
            f"Test set hash file not found: {TEST_SET_HASH_FILE}. "
            "Run save_test_hash() first."
        )

    with open(TEST_SET_HASH_FILE, "r") as f:
        saved_hashes = json.load(f)

    if dataset not in saved_hashes:
        raise ValueError(f"Dataset {dataset} not found in hash file")

    # Load current test set and compute hash
    splits = get_splits(dataset)
    current_hash = compute_test_hash(splits["test"])

    return current_hash == saved_hashes[dataset]["hash"]


def save_test_hash(datasets: list[str] = None) -> dict[str, str]:
    """Save test set hashes to file.

    Args:
        datasets: List of datasets to hash, or None for all available

    Returns:
        Dict mapping dataset name to hash

    Note:
        This should be called once during M2 to lock the test sets.
    """
    if datasets is None:
        datasets = ["pubmedqa"]  # Only PubMedQA by default (BioASQ requires local data)

    TEST_SET_HASH_FILE.parent.mkdir(parents=True, exist_ok=True)

    hashes = {}
    for dataset in datasets:
        try:
            splits = get_splits(dataset)
            test_hash = compute_test_hash(splits["test"])
            hashes[dataset] = {
                "hash": test_hash,
                "size": len(splits["test"]),
            }
        except Exception as e:
            print(f"Warning: Could not hash {dataset}: {e}")

    with open(TEST_SET_HASH_FILE, "w") as f:
        json.dump(hashes, f, indent=2)

    print(f"Saved test set hashes to {TEST_SET_HASH_FILE}")
    return {d: h["hash"] for d, h in hashes.items()}


def check_no_overlap(
    train_examples: list[dict],
    test_examples: list[dict],
) -> bool:
    """Verify no ID overlap between train and test sets.

    Returns:
        True if no overlap, False if overlap exists
    """
    train_ids = {ex["id"] for ex in train_examples}
    test_ids = {ex["id"] for ex in test_examples}

    overlap = train_ids & test_ids
    if overlap:
        print(f"Warning: Found {len(overlap)} overlapping IDs: {list(overlap)[:5]}...")
        return False
    return True


def _compute_length_stats(lengths: list[int]) -> dict[str, float]:
    """Compute avg, median, max for a list of lengths."""
    if not lengths:
        return {"avg": 0, "median": 0, "max": 0}
    sorted_lengths = sorted(lengths)
    n = len(sorted_lengths)
    return {
        "avg": sum(lengths) / n,
        "median": sorted_lengths[n // 2],
        "max": max(lengths),
    }


def generate_data_audit() -> str:
    """Generate data_audit.md content.

    Returns:
        Markdown string with:
        - Total examples per dataset
        - Train/dev/test sizes
        - Label distribution (yes/no/maybe for PubMedQA, factoid/yesno for BioASQ)
        - Avg/median/max question and context lengths
        - Test set hash (full, visible)
        - Supported retrieval conditions
    """
    lines = [
        "# Data Audit",
        "",
        "> Auto-generated. Do not edit by hand.",
        "",
        f"Generated at: {_get_timestamp()}",
        "",
    ]

    # PubMedQA stats
    lines.append("## PubMedQA (Primary Evaluation)")
    lines.append("")
    try:
        pubmedqa_stats = get_pubmedqa_stats()
        lines.append(f"- **Total examples:** {pubmedqa_stats['total']}")
        lines.append(f"- **Train:** {pubmedqa_stats['train']}")
        lines.append(f"- **Dev:** {pubmedqa_stats['dev']}")
        lines.append(f"- **Test:** {pubmedqa_stats['test']}")
        lines.append(f"- **Label distribution (yes/no/maybe):** {pubmedqa_stats['label_distribution']}")
        lines.append(f"- **Supported retrieval conditions:** {get_supported_retrieval_conditions('pubmedqa')}")
        lines.append("")

        # Load examples for detailed length stats
        all_pubmedqa = load_pubmedqa("train") + load_pubmedqa("dev") + load_pubmedqa("test")
        q_lengths = [len(ex["question"].split()) for ex in all_pubmedqa]
        c_lengths = [len(ex["context"].split()) for ex in all_pubmedqa]
        q_stats = _compute_length_stats(q_lengths)
        c_stats = _compute_length_stats(c_lengths)

        lines.append("### Question Length (words)")
        lines.append(f"- Avg: {q_stats['avg']:.1f}")
        lines.append(f"- Median: {q_stats['median']}")
        lines.append(f"- Max: {q_stats['max']}")
        lines.append("")

        lines.append("### Context Length (words)")
        lines.append(f"- Avg: {c_stats['avg']:.1f}")
        lines.append(f"- Median: {c_stats['median']}")
        lines.append(f"- Max: {c_stats['max']}")
        lines.append("")

        # Test set hash (full, visible)
        test_hash = compute_test_hash(load_pubmedqa("test"))
        lines.append("### Test Set Hash (SHA-256)")
        lines.append(f"```")
        lines.append(test_hash)
        lines.append(f"```")

    except Exception as e:
        lines.append(f"- Error loading PubMedQA: {e}")
    lines.append("")

    # BioASQ stats
    lines.append("## BioASQ (Primary Training)")
    lines.append("")
    try:
        bioasq_stats = get_bioasq_stats()
        if "error" in bioasq_stats:
            lines.append(f"- **Status:** Not loaded ({bioasq_stats['error']})")
            lines.append("- Set BIOASQ_DATA_PATH environment variable to load BioASQ data")
            lines.append(f"- **Supported retrieval conditions:** {get_supported_retrieval_conditions('bioasq')}")
            lines.append("- **Test Set Hash:** PENDING (will be computed when BioASQ data is loaded)")
        else:
            lines.append(f"- **Total examples:** {bioasq_stats['total']} (factoid + yesno only)")
            lines.append(f"- **Train:** {bioasq_stats['train']}")
            lines.append(f"- **Dev:** {bioasq_stats['dev']}")
            lines.append(f"- **Test:** {bioasq_stats['test']}")
            lines.append(f"- **Question type distribution (factoid/yesno):** {bioasq_stats['type_distribution']}")
            lines.append(f"- **Yes/No label distribution:** {bioasq_stats['yesno_label_distribution']}")
            lines.append(f"- **Supported retrieval conditions:** {get_supported_retrieval_conditions('bioasq')}")
            lines.append("")

            lines.append("### Question Length (words)")
            lines.append(f"- Avg: {bioasq_stats['avg_question_length']:.1f}")
            lines.append("")

            lines.append("### Snippet Length (words)")
            lines.append(f"- Avg: {bioasq_stats['avg_snippet_length']:.1f}")
            lines.append(f"- Avg snippets per question: {bioasq_stats['avg_snippets_per_question']:.1f}")
    except Exception as e:
        lines.append(f"- Error loading BioASQ: {e}")
    lines.append("")

    # MIRAGE stats
    lines.append("## MIRAGE (External Validation)")
    lines.append("")
    try:
        mirage_stats = get_mirage_stats()
        if "error" in mirage_stats:
            lines.append(f"- **Status:** Not loaded ({mirage_stats['error']})")
        else:
            lines.append(f"- **Total examples:** {mirage_stats['total']} (held-out only)")
            lines.append(f"- **Source distribution:** {mirage_stats['source_distribution']}")
            lines.append(f"- **Answer distribution (A/B/C/D):** {mirage_stats['answer_distribution']}")
            lines.append(f"- **Supported retrieval conditions:** {mirage_stats['supported_retrieval_conditions']}")
            lines.append(f"- **Format:** All MCQ with 4 options (A/B/C/D)")
            lines.append("")

            lines.append("### Overall Question Length (words)")
            lines.append(f"- Avg: {mirage_stats['avg_question_length']:.1f}")
            lines.append(f"- Median: {mirage_stats['median_question_length']}")
            lines.append(f"- Max: {mirage_stats['max_question_length']}")
            lines.append("")

            # Per-source stats
            lines.append("### Per-Source Statistics")
            lines.append("")
            per_source = mirage_stats.get("per_source_stats", {})
            for source in ["mmlu_med", "medqa_us", "medmcqa"]:
                if source in per_source:
                    stats = per_source[source]
                    lines.append(f"**{source}** ({stats['count']} examples)")
                    lines.append(f"- Question length: avg={stats['avg_question_length']:.1f}, median={stats['median_question_length']}, max={stats['max_question_length']}")
                    lines.append(f"- Answer distribution: {stats['answer_distribution']}")
                    lines.append(f"- Unique subjects: {stats['unique_subjects']}")
                    lines.append("")

            lines.append(f"**Note:** {mirage_stats.get('note', 'N/A')}")
            lines.append("")
            lines.append("**Subject field semantics:** The `subject` field has different semantics across sources — MMLU subset name for MMLU-Med (e.g., `professional_medicine`), USMLE step for MedQA-US (e.g., `step1`), medical specialty for MedMCQA (e.g., `Pharmacology`). Cross-source subject comparisons are not meaningful.")
            lines.append("")

            # MIRAGE hash
            mirage_hash = compute_test_hash(load_mirage())
            lines.append("### Test Set Hash (SHA-256)")
            lines.append("```")
            lines.append(mirage_hash)
            lines.append("```")
    except Exception as e:
        lines.append(f"- Error loading MIRAGE: {e}")
    lines.append("")

    # Test set verification
    lines.append("## Test Set Verification")
    lines.append("")
    lines.append("Run `pytest tests/test_splits.py` to verify test set integrity and data loading.")
    lines.append("")

    return "\n".join(lines)


def _get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def write_data_audit(output_path: str = "data_audit.md") -> None:
    """Write data audit to file."""
    content = generate_data_audit()
    with open(output_path, "w") as f:
        f.write(content)
    print(f"Wrote data audit to {output_path}")
