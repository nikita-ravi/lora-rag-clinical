"""MIRAGE benchmark subset loader.

Loads a stratified subset of ~500 examples from three medical QA datasets
for external validation. This data is NEVER used for training or hyperparameter tuning.

Sources included:
- MMLU-Med (medical subsets of MMLU: anatomy, clinical_knowledge, college_biology,
  college_medicine, medical_genetics, professional_medicine)
- MedQA-US (USMLE-style questions)
- MedMCQA (medical entrance exam questions)

IMPORTANT:
- We exclude PubMedQA and BioASQ to avoid overlap with training data
- MIRAGE supports only "none" and "strong" retrieval conditions (no oracle)
  because these datasets do not provide gold supporting passages
- All sources are MCQ format with 4 options (A/B/C/D)
"""

import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset


# Fixed seed for reproducible subset selection
MIRAGE_SUBSET_SEED = 42
MIRAGE_SUBSET_SIZE = 500

# Target allocation per source (roughly equal with remainder to largest source)
TARGET_PER_SOURCE = 165

# MMLU medical subjects
MMLU_MEDICAL_SUBJECTS = [
    "anatomy",
    "clinical_knowledge",
    "college_biology",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
]

# Supported retrieval conditions for MIRAGE (no oracle - no gold passages)
SUPPORTED_RETRIEVAL_CONDITIONS = ["none", "strong"]


def get_supported_retrieval_conditions() -> list[str]:
    """Return retrieval conditions supported by MIRAGE."""
    return SUPPORTED_RETRIEVAL_CONDITIONS


def load_mirage() -> list[dict[str, Any]]:
    """Load MIRAGE validation subset.

    Returns:
        List of ~500 examples stratified across MMLU-Med, MedQA-US, and MedMCQA,
        with keys:
        - id: unique identifier
        - source_dataset: "mmlu_med", "medqa_us", or "medmcqa"
        - question: the question text
        - options: list of dicts with 'label' and 'text' keys
        - answer: correct option letter (A/B/C/D)
        - subject: medical subject/topic if available

    Note:
        MIRAGE does NOT include gold_passage_ids because these datasets
        don't provide gold supporting passages. Only "none" and "strong"
        retrieval conditions are supported.
    """
    all_examples = []

    # Load from each source
    mmlu_examples = _load_mmlu_med()
    medqa_examples = _load_medqa_us()
    medmcqa_examples = _load_medmcqa()

    print(f"Loaded {len(mmlu_examples)} MMLU-Med examples")
    print(f"Loaded {len(medqa_examples)} MedQA-US examples")
    print(f"Loaded {len(medmcqa_examples)} MedMCQA examples")

    # Select stratified subset
    subset = _select_stratified_subset(
        mmlu_examples=mmlu_examples,
        medqa_examples=medqa_examples,
        medmcqa_examples=medmcqa_examples,
        seed=MIRAGE_SUBSET_SEED,
    )

    # Save indices for reproducibility
    _save_subset_indices([ex["id"] for ex in subset])

    return subset


def _load_mmlu_med() -> list[dict[str, Any]]:
    """Load MMLU medical subsets.

    MMLU schema:
    - question: question text
    - subject: medical subject
    - choices: list of 4 answer strings
    - answer: correct answer index (0-3)
    """
    examples = []
    option_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    for subject in MMLU_MEDICAL_SUBJECTS:
        try:
            dataset = load_dataset("cais/mmlu", subject, split="test")
        except Exception as e:
            print(f"Warning: Could not load MMLU {subject}: {e}")
            continue

        for i, item in enumerate(dataset):
            example_id = f"mmlu_{subject}_{i}"

            # Build options list
            choices = item.get("choices", [])
            options = [
                {"label": option_map[j], "text": choices[j] if j < len(choices) else ""}
                for j in range(4)
            ]

            # Get correct answer
            answer_idx = item.get("answer", 0)
            answer_label = option_map.get(answer_idx, "A")

            examples.append({
                "id": example_id,
                "source_dataset": "mmlu_med",
                "question": item.get("question", ""),
                "options": options,
                "answer": answer_label,
                "subject": subject,
            })

    return examples


def _load_medqa_us() -> list[dict[str, Any]]:
    """Load MedQA-USMLE dataset.

    MedQA schema:
    - question: question text
    - options: dict with A/B/C/D keys
    - answer_idx: correct answer letter (A/B/C/D)
    - meta_info: "step1" or "step2&3"
    """
    try:
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    except Exception as e:
        print(f"Warning: Could not load MedQA-US: {e}")
        return []

    examples = []

    for i, item in enumerate(dataset):
        example_id = f"medqa_{i}"

        # Build options list from dict
        options_dict = item.get("options", {})
        options = [
            {"label": label, "text": options_dict.get(label, "")}
            for label in ["A", "B", "C", "D"]
        ]

        # Get correct answer (already a letter)
        answer_label = item.get("answer_idx", "A")

        # Use meta_info as subject proxy
        subject = item.get("meta_info", "usmle")

        examples.append({
            "id": example_id,
            "source_dataset": "medqa_us",
            "question": item.get("question", ""),
            "options": options,
            "answer": answer_label,
            "subject": subject,
        })

    return examples


def _load_medmcqa() -> list[dict[str, Any]]:
    """Load MedMCQA validation set.

    Note: We use validation split because the test split has no labels.

    MedMCQA schema:
    - id: question ID
    - question: question text
    - opa, opb, opc, opd: answer options
    - cop: correct option (0-3 for A-D)
    - subject_name: medical subject
    - topic_name: specific topic

    NOTE: We do NOT use the 'exp' (explanation) field. It is not a
    retrievable passage from a corpus, so it cannot serve as a gold
    passage for retrieval evaluation.
    """
    try:
        dataset = load_dataset("openlifescienceai/medmcqa", split="validation")
    except Exception as e:
        print(f"Warning: Could not load MedMCQA: {e}")
        return []

    examples = []
    option_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    for item in dataset:
        example_id = f"medmcqa_{item['id']}"

        # Build options list
        options = [
            {"label": "A", "text": item.get("opa", "")},
            {"label": "B", "text": item.get("opb", "")},
            {"label": "C", "text": item.get("opc", "")},
            {"label": "D", "text": item.get("opd", "")},
        ]

        # Get correct answer (cop is 0-indexed)
        cop = item.get("cop")
        if cop is None or cop not in option_map:
            continue
        answer_label = option_map[cop]

        examples.append({
            "id": example_id,
            "source_dataset": "medmcqa",
            "question": item.get("question", ""),
            "options": options,
            "answer": answer_label,
            "subject": item.get("subject_name", ""),
        })

    return examples


def _select_stratified_subset(
    mmlu_examples: list[dict],
    medqa_examples: list[dict],
    medmcqa_examples: list[dict],
    seed: int = MIRAGE_SUBSET_SEED,
) -> list[dict]:
    """Select a stratified random subset across all three sources.

    Target: ~165 from each source, with remainder going to largest source.
    Minimum 100 from each source if available.
    """
    rng = random.Random(seed)
    subset = []

    # Calculate allocation
    sources = [
        ("mmlu_med", mmlu_examples),
        ("medqa_us", medqa_examples),
        ("medmcqa", medmcqa_examples),
    ]

    # Shuffle each source
    for name, examples in sources:
        rng.shuffle(examples)

    # Sample from each source
    n_mmlu = min(TARGET_PER_SOURCE, len(mmlu_examples))
    n_medqa = min(TARGET_PER_SOURCE, len(medqa_examples))
    n_medmcqa = min(TARGET_PER_SOURCE, len(medmcqa_examples))

    # Distribute remainder to reach 500
    remainder = MIRAGE_SUBSET_SIZE - (n_mmlu + n_medqa + n_medmcqa)
    if remainder > 0:
        # Give remainder to medmcqa (largest source)
        n_medmcqa = min(n_medmcqa + remainder, len(medmcqa_examples))

    subset.extend(mmlu_examples[:n_mmlu])
    subset.extend(medqa_examples[:n_medqa])
    subset.extend(medmcqa_examples[:n_medmcqa])

    # Final shuffle
    rng.shuffle(subset)

    return subset


def _save_subset_indices(ids: list[str], filepath: str = "data/mirage_subset_ids.json") -> None:
    """Save selected indices to file for reproducibility."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Count by source
    source_counts = {}
    for id_ in ids:
        if id_.startswith("mmlu_"):
            source = "mmlu_med"
        elif id_.startswith("medqa_"):
            source = "medqa_us"
        elif id_.startswith("medmcqa_"):
            source = "medmcqa"
        else:
            source = "unknown"
        source_counts[source] = source_counts.get(source, 0) + 1

    with open(path, "w") as f:
        json.dump({
            "seed": MIRAGE_SUBSET_SEED,
            "total_size": len(ids),
            "source_distribution": source_counts,
            "supported_retrieval_conditions": SUPPORTED_RETRIEVAL_CONDITIONS,
            "ids": ids,
        }, f, indent=2)


def get_mirage_stats() -> dict[str, Any]:
    """Get statistics for data audit."""
    try:
        examples = load_mirage()
    except Exception as e:
        return {"error": str(e)}

    # Source distribution
    source_counts = {}
    subject_counts = {}

    for ex in examples:
        source = ex["source_dataset"]
        source_counts[source] = source_counts.get(source, 0) + 1

        subject = ex.get("subject", "unknown")
        subject_counts[subject] = subject_counts.get(subject, 0) + 1

    # Answer distribution
    answer_counts = {}
    for ex in examples:
        ans = ex["answer"]
        answer_counts[ans] = answer_counts.get(ans, 0) + 1

    # Length statistics
    question_lengths = [len(ex["question"].split()) for ex in examples]

    # Per-source stats
    per_source_stats = {}
    for source in ["mmlu_med", "medqa_us", "medmcqa"]:
        source_examples = [ex for ex in examples if ex["source_dataset"] == source]
        if source_examples:
            q_lengths = [len(ex["question"].split()) for ex in source_examples]
            subjects = set(ex.get("subject", "") for ex in source_examples)
            answers = {}
            for ex in source_examples:
                answers[ex["answer"]] = answers.get(ex["answer"], 0) + 1

            sorted_lengths = sorted(q_lengths)
            n = len(sorted_lengths)
            per_source_stats[source] = {
                "count": len(source_examples),
                "avg_question_length": sum(q_lengths) / len(q_lengths),
                "median_question_length": sorted_lengths[n // 2],
                "max_question_length": max(q_lengths),
                "answer_distribution": answers,
                "unique_subjects": len(subjects),
            }

    sorted_lengths = sorted(question_lengths)
    n = len(sorted_lengths)

    return {
        "total": len(examples),
        "source_distribution": source_counts,
        "subject_distribution": subject_counts,
        "answer_distribution": answer_counts,
        "avg_question_length": sum(question_lengths) / len(question_lengths) if question_lengths else 0,
        "median_question_length": sorted_lengths[n // 2] if n > 0 else 0,
        "max_question_length": max(question_lengths) if question_lengths else 0,
        "per_source_stats": per_source_stats,
        "supported_retrieval_conditions": SUPPORTED_RETRIEVAL_CONDITIONS,
        "note": "No gold passages available. Supports only 'none' and 'strong' retrieval conditions.",
    }
