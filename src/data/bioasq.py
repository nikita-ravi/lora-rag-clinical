"""BioASQ Task B data loader.

Loads factoid and yes/no questions from BioASQ, filtering out list and summary questions.
Requires BIOASQ_DATA_PATH environment variable to be set.

BioASQ Task B JSON schema (questions array):
- id: question ID
- body: the question text
- type: "factoid", "yesno", "list", or "summary"
- documents: list of PubMed document URLs
- snippets: list of dicts with {text, document, ...}
- exact_answer: varies by type
  - factoid: list of answer strings or list of lists (synonyms)
  - yesno: "yes" or "no"
  - list: list of answer strings
  - summary: not present
- ideal_answer: free-text ideal answer

We convert to unified format:
- id: unique identifier
- question: the question text
- question_type: "factoid" or "yesno"
- answer: normalized answer string
- snippets: list of gold snippets
- gold_passage_ids: list of passage IDs for oracle retrieval
"""

import json
import os
from pathlib import Path
from typing import Any
import random

from dotenv import load_dotenv

load_dotenv()


# Question types we include (exclude list and summary)
INCLUDED_TYPES = {"factoid", "yesno"}

# Split sizes for BioASQ (fixed, stratified)
TEST_SIZE = 500
DEV_SIZE = 100
SPLIT_SEED = 42

# Path to locked splits file
SPLITS_FILE = Path(__file__).parent.parent.parent / "data" / "bioasq_splits.json"


def load_bioasq(split: str = "train") -> list[dict[str, Any]]:
    """Load BioASQ Task B data.

    Args:
        split: One of "train", "dev", "test"

    Returns:
        List of examples with keys:
        - id: unique identifier
        - question: the question text
        - question_type: "factoid" or "yesno"
        - answer: gold answer (entity string or yes/no)
        - snippets: list of gold passage dicts with 'text' and 'document' keys
        - gold_passage_ids: list of passage IDs
        - ideal_answer: free-text ideal answer if available

    Raises:
        ValueError: If BIOASQ_DATA_PATH is not set or split is invalid
        FileNotFoundError: If data files don't exist
    """
    if split not in ["train", "dev", "test"]:
        raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'dev', 'test']")

    bioasq_path = os.getenv("BIOASQ_DATA_PATH")
    if not bioasq_path:
        raise ValueError(
            "BIOASQ_DATA_PATH environment variable not set. "
            "Set it to the directory containing BioASQ JSON files."
        )

    data_dir = Path(bioasq_path)
    if not data_dir.exists():
        raise FileNotFoundError(f"BioASQ data directory not found: {data_dir}")

    # Load all examples from JSON files
    all_examples = _load_all_bioasq_examples(data_dir)

    # Get or create locked splits
    split_ids = _get_or_create_splits(all_examples)

    # Create ID to example mapping
    id_to_example = {ex["id"]: ex for ex in all_examples}

    # Return requested split
    return [id_to_example[qid] for qid in split_ids[f"{split}_ids"]]


def _load_all_bioasq_examples(data_dir: Path) -> list[dict[str, Any]]:
    """Load all BioASQ examples from directory."""
    all_examples = []
    json_files = list(data_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    for json_file in json_files:
        examples = _parse_bioasq_file(json_file)
        all_examples.extend(examples)

    # Filter to factoid and yesno only
    filtered_examples = _filter_question_types(all_examples)

    if not filtered_examples:
        raise ValueError(
            f"No factoid or yesno questions found in BioASQ data. "
            f"Found {len(all_examples)} total questions."
        )

    # Sort by ID for deterministic ordering before any random operations
    filtered_examples.sort(key=lambda x: x["id"])

    return filtered_examples


def _get_or_create_splits(all_examples: list[dict]) -> dict[str, Any]:
    """Get locked splits from file or create them if file doesn't exist.

    Once splits are created and saved, they are immutable. This ensures
    reproducibility across runs and machines.
    """
    if SPLITS_FILE.exists():
        # Load existing splits
        with open(SPLITS_FILE, "r") as f:
            return json.load(f)

    # Create new splits (stratified by question type)
    splits = _create_stratified_splits(all_examples)

    # Save to file (makes them immutable)
    SPLITS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLITS_FILE, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Created and locked BioASQ splits: {SPLITS_FILE}")
    return splits


def _create_stratified_splits(all_examples: list[dict]) -> dict[str, Any]:
    """Create stratified train/dev/test splits.

    Stratifies by question type (factoid vs yesno) to maintain
    roughly the same ratio in each split.
    """
    # Separate by question type
    factoid = [ex for ex in all_examples if ex["question_type"] == "factoid"]
    yesno = [ex for ex in all_examples if ex["question_type"] == "yesno"]

    # Calculate proportions
    n_total = len(all_examples)
    factoid_ratio = len(factoid) / n_total
    yesno_ratio = len(yesno) / n_total

    # Calculate stratified sizes for test set
    n_test_factoid = int(TEST_SIZE * factoid_ratio)
    n_test_yesno = TEST_SIZE - n_test_factoid  # Ensure exact total

    # Calculate stratified sizes for dev set
    n_dev_factoid = int(DEV_SIZE * factoid_ratio)
    n_dev_yesno = DEV_SIZE - n_dev_factoid  # Ensure exact total

    # Use deterministic RNG
    rng = random.Random(SPLIT_SEED)

    # Shuffle each type separately
    factoid_shuffled = factoid.copy()
    yesno_shuffled = yesno.copy()
    rng.shuffle(factoid_shuffled)
    rng.shuffle(yesno_shuffled)

    # Split factoid questions
    test_factoid = factoid_shuffled[:n_test_factoid]
    dev_factoid = factoid_shuffled[n_test_factoid:n_test_factoid + n_dev_factoid]
    train_factoid = factoid_shuffled[n_test_factoid + n_dev_factoid:]

    # Split yesno questions
    test_yesno = yesno_shuffled[:n_test_yesno]
    dev_yesno = yesno_shuffled[n_test_yesno:n_test_yesno + n_dev_yesno]
    train_yesno = yesno_shuffled[n_test_yesno + n_dev_yesno:]

    # Combine and extract IDs
    test_ids = sorted([ex["id"] for ex in test_factoid + test_yesno])
    dev_ids = sorted([ex["id"] for ex in dev_factoid + dev_yesno])
    train_ids = sorted([ex["id"] for ex in train_factoid + train_yesno])

    return {
        "seed": SPLIT_SEED,
        "train_ids": train_ids,
        "dev_ids": dev_ids,
        "test_ids": test_ids,
        "train_size": len(train_ids),
        "dev_size": len(dev_ids),
        "test_size": len(test_ids),
        "stratification": {
            "test_factoid": n_test_factoid,
            "test_yesno": n_test_yesno,
            "dev_factoid": n_dev_factoid,
            "dev_yesno": n_dev_yesno,
        }
    }


def _filter_question_types(examples: list[dict]) -> list[dict]:
    """Filter to factoid and yes/no questions only."""
    return [ex for ex in examples if ex["question_type"] in INCLUDED_TYPES]


def _parse_bioasq_file(filepath: Path) -> list[dict]:
    """Parse a single BioASQ JSON file.

    BioASQ files have structure: {"questions": [...]}
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # BioASQ format has questions in a "questions" array
    questions = data.get("questions", [])

    examples = []
    for q in questions:
        example = _convert_bioasq_format(q)
        if example is not None:
            examples.append(example)

    return examples


def _convert_bioasq_format(item: dict) -> dict[str, Any] | None:
    """Convert BioASQ format to our unified format.

    Returns None if the question should be skipped (missing required fields).
    """
    question_id = item.get("id", "")
    question_text = item.get("body", "")
    question_type = item.get("type", "").lower()

    # Skip if missing required fields
    if not question_id or not question_text or not question_type:
        return None

    # Skip types we don't handle
    if question_type not in INCLUDED_TYPES:
        return None

    # Extract snippets
    snippets_raw = item.get("snippets", [])
    snippets = []
    gold_passage_ids = []

    for i, snip in enumerate(snippets_raw):
        snippet_text = snip.get("text", "")
        if snippet_text:
            # Create a unique passage ID
            passage_id = f"bioasq_{question_id}_snip{i}"
            snippets.append({
                "text": snippet_text,
                "document": snip.get("document", ""),
                "id": passage_id,
            })
            gold_passage_ids.append(passage_id)

    # Skip if no snippets (we need gold passages)
    if not snippets:
        return None

    # Extract answer based on question type
    answer = _extract_answer(item, question_type)
    if answer is None:
        return None

    # Get ideal answer if available
    ideal_answer = item.get("ideal_answer", "")
    if isinstance(ideal_answer, list):
        ideal_answer = ideal_answer[0] if ideal_answer else ""

    return {
        "id": question_id,
        "question": question_text,
        "question_type": question_type,
        "answer": answer,
        "snippets": snippets,
        "gold_passage_ids": gold_passage_ids,
        "ideal_answer": ideal_answer,
        "source": "bioasq",
    }


def _extract_answer(item: dict, question_type: str) -> str | None:
    """Extract and normalize the answer from a BioASQ question."""
    exact_answer = item.get("exact_answer")

    if question_type == "yesno":
        # Yes/no questions have simple string answer
        if isinstance(exact_answer, str):
            return exact_answer.lower()
        return None

    elif question_type == "factoid":
        # Factoid answers can be:
        # - A string
        # - A list of strings (multiple acceptable answers)
        # - A list of lists (answer with synonyms)
        if isinstance(exact_answer, str):
            return exact_answer
        elif isinstance(exact_answer, list) and exact_answer:
            first = exact_answer[0]
            if isinstance(first, str):
                return first
            elif isinstance(first, list) and first:
                return first[0]  # First synonym of first answer
        return None

    return None


def get_bioasq_passages() -> list[dict[str, Any]]:
    """Extract all unique passages from BioASQ for corpus building.

    Returns:
        List of passages with keys: id, text, source, metadata
    """
    passages = []
    seen_ids = set()

    for split in ["train", "dev", "test"]:
        try:
            examples = load_bioasq(split)
        except (ValueError, FileNotFoundError):
            continue

        for ex in examples:
            for snip in ex["snippets"]:
                passage_id = snip["id"]
                if passage_id not in seen_ids and snip["text"]:
                    passages.append({
                        "id": passage_id,
                        "text": snip["text"],
                        "source": "bioasq",
                        "metadata": {
                            "question_id": ex["id"],
                            "document": snip.get("document", ""),
                        }
                    })
                    seen_ids.add(passage_id)

    return passages


def get_bioasq_stats() -> dict[str, Any]:
    """Get statistics for data audit."""
    try:
        train = load_bioasq("train")
        dev = load_bioasq("dev")
        test = load_bioasq("test")
    except (ValueError, FileNotFoundError) as e:
        return {"error": str(e)}

    all_examples = train + dev + test

    # Label distribution by question type
    type_counts = {}
    answer_counts = {"yesno": {}, "factoid": 0}

    for ex in all_examples:
        qtype = ex["question_type"]
        type_counts[qtype] = type_counts.get(qtype, 0) + 1

        if qtype == "yesno":
            ans = ex["answer"]
            answer_counts["yesno"][ans] = answer_counts["yesno"].get(ans, 0) + 1
        else:
            answer_counts["factoid"] += 1

    # Length statistics
    question_lengths = [len(ex["question"].split()) for ex in all_examples]
    snippet_lengths = []
    snippets_per_question = []

    for ex in all_examples:
        snippets_per_question.append(len(ex["snippets"]))
        for snip in ex["snippets"]:
            snippet_lengths.append(len(snip["text"].split()))

    # Per-split type distribution
    split_type_dist = {}
    for split_name, split_data in [("train", train), ("dev", dev), ("test", test)]:
        split_type_dist[split_name] = {
            "factoid": sum(1 for ex in split_data if ex["question_type"] == "factoid"),
            "yesno": sum(1 for ex in split_data if ex["question_type"] == "yesno"),
        }

    return {
        "total": len(all_examples),
        "train": len(train),
        "dev": len(dev),
        "test": len(test),
        "type_distribution": type_counts,
        "yesno_label_distribution": answer_counts["yesno"],
        "factoid_count": answer_counts["factoid"],
        "avg_question_length": sum(question_lengths) / len(question_lengths) if question_lengths else 0,
        "avg_snippet_length": sum(snippet_lengths) / len(snippet_lengths) if snippet_lengths else 0,
        "avg_snippets_per_question": sum(snippets_per_question) / len(snippets_per_question) if snippets_per_question else 0,
        "split_type_distribution": split_type_dist,
    }
