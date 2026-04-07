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

# Split ratios for BioASQ (we create our own splits)
TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
TEST_RATIO = 0.1
SPLIT_SEED = 42


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

    # Load all JSON files from the directory
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

    # Sort by ID for deterministic ordering
    filtered_examples.sort(key=lambda x: x["id"])

    # Create deterministic splits
    rng = random.Random(SPLIT_SEED)
    indices = list(range(len(filtered_examples)))
    rng.shuffle(indices)

    n_total = len(filtered_examples)
    n_train = int(n_total * TRAIN_RATIO)
    n_dev = int(n_total * DEV_RATIO)

    train_indices = indices[:n_train]
    dev_indices = indices[n_train:n_train + n_dev]
    test_indices = indices[n_train + n_dev:]

    if split == "train":
        return [filtered_examples[i] for i in sorted(train_indices)]
    elif split == "dev":
        return [filtered_examples[i] for i in sorted(dev_indices)]
    else:  # test
        return [filtered_examples[i] for i in sorted(test_indices)]


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
    }
