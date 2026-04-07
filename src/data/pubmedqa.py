"""PubMedQA data loader.

Loads the pqa_labeled split from PubMedQA for evaluation.
Uses the HuggingFace datasets library.

qiaojin/PubMedQA pqa_labeled schema:
- pubid: PubMed ID (int)
- question: the question text
- context: dict with 'contexts' (list of strings), 'labels' (list), 'meshes' (list)
- long_answer: reasoning/rationale (the conclusion)
- final_decision: yes/no/maybe

We convert to unified format:
- id: unique identifier (PMID as string)
- question: the question text
- question_type: "yesno" (always for PubMedQA)
- answer: yes/no/maybe
- context: gold abstract text (joined from context.contexts)
- long_answer: reasoning/rationale
- gold_passage_ids: list with single passage ID for oracle retrieval

NOTE: The HF dataset comes as a single 1000-example split. We create
train/dev/test splits deterministically using a fixed seed, following
the standard PubMedQA convention: 450 train, 50 dev, 500 test.
"""

from typing import Any
import random
from datasets import load_dataset


# PubMedQA split sizes (standard convention)
TRAIN_SIZE = 450
DEV_SIZE = 50
TEST_SIZE = 500
SPLIT_SEED = 42  # Fixed seed for reproducibility


def load_pubmedqa(split: str = "test") -> list[dict[str, Any]]:
    """Load PubMedQA pqa_labeled data.

    Args:
        split: One of "train", "dev", "test"

    Returns:
        List of examples with keys:
        - id: unique identifier (PMID)
        - question: the question text
        - question_type: always "yesno" for PubMedQA
        - answer: gold answer (yes/no/maybe)
        - context: gold abstract text
        - long_answer: reasoning/rationale
        - gold_passage_ids: list with passage ID for this example

    Raises:
        ValueError: If split is invalid
    """
    if split not in ["train", "dev", "test"]:
        raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'dev', 'test']")

    # Load full dataset from HuggingFace
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

    # Convert all examples to our format
    all_examples = [_convert_pubmedqa_format(item) for item in dataset]

    # Sort by ID for deterministic ordering before split
    all_examples.sort(key=lambda x: x["id"])

    # Create deterministic split
    rng = random.Random(SPLIT_SEED)
    indices = list(range(len(all_examples)))
    rng.shuffle(indices)

    train_indices = indices[:TRAIN_SIZE]
    dev_indices = indices[TRAIN_SIZE:TRAIN_SIZE + DEV_SIZE]
    test_indices = indices[TRAIN_SIZE + DEV_SIZE:TRAIN_SIZE + DEV_SIZE + TEST_SIZE]

    if split == "train":
        return [all_examples[i] for i in sorted(train_indices)]
    elif split == "dev":
        return [all_examples[i] for i in sorted(dev_indices)]
    else:  # test
        return [all_examples[i] for i in sorted(test_indices)]


def _convert_pubmedqa_format(item: dict) -> dict[str, Any]:
    """Convert HuggingFace format to our unified format."""
    example_id = str(item["pubid"])
    question = item["question"]
    answer = item["final_decision"].lower()
    long_answer = item.get("long_answer", "")

    # Context is a dict with 'contexts' list (list of abstract sentences)
    context = item.get("context", {})
    if isinstance(context, dict):
        contexts = context.get("contexts", [])
        context_text = " ".join(contexts) if contexts else ""
    elif isinstance(context, str):
        context_text = context
    else:
        context_text = ""

    # Generate a passage ID for corpus/retrieval
    passage_id = f"pubmedqa_{example_id}"

    return {
        "id": example_id,
        "question": question,
        "question_type": "yesno",
        "answer": answer,
        "context": context_text,
        "long_answer": long_answer,
        "gold_passage_ids": [passage_id],
        "source": "pubmedqa",
    }


def get_pubmedqa_passages() -> list[dict[str, Any]]:
    """Extract all unique passages from PubMedQA for corpus building.

    Returns:
        List of passages with keys: id, text, source, metadata
    """
    # Load full dataset
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

    passages = []
    seen_ids = set()

    for item in dataset:
        example = _convert_pubmedqa_format(item)
        passage_id = example["gold_passage_ids"][0]

        if passage_id not in seen_ids and example["context"]:
            passages.append({
                "id": passage_id,
                "text": example["context"],
                "source": "pubmedqa",
                "metadata": {
                    "pmid": example["id"],
                    "question": example["question"],
                }
            })
            seen_ids.add(passage_id)

    return passages


def get_pubmedqa_stats() -> dict[str, Any]:
    """Get statistics for data audit."""
    train = load_pubmedqa("train")
    dev = load_pubmedqa("dev")
    test = load_pubmedqa("test")

    all_examples = train + dev + test

    # Label distribution
    label_counts = {}
    for ex in all_examples:
        label = ex["answer"]
        label_counts[label] = label_counts.get(label, 0) + 1

    # Length statistics
    question_lengths = [len(ex["question"].split()) for ex in all_examples]
    context_lengths = [len(ex["context"].split()) for ex in all_examples]

    return {
        "total": len(all_examples),
        "train": len(train),
        "dev": len(dev),
        "test": len(test),
        "label_distribution": label_counts,
        "avg_question_length": sum(question_lengths) / len(question_lengths),
        "avg_context_length": sum(context_lengths) / len(context_lengths),
    }
