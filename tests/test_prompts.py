"""Tests for M5 prompt templates and materialized training data."""

import json
import re
from pathlib import Path

import pytest


@pytest.fixture
def training_data():
    """Load all three materialized training files."""
    data_dir = Path("data/training")

    def load_file(name):
        records = []
        with open(data_dir / f"{name}_train.jsonl", "r") as f:
            for line in f:
                records.append(json.loads(line))
        return records

    return {
        "lora_a": load_file("lora_a"),
        "lora_a_prime": load_file("lora_a_prime"),
        "lora_b": load_file("lora_b"),
    }


def test_materialized_files_exist():
    """All three files exist after running materialize.py."""
    data_dir = Path("data/training")
    assert (data_dir / "lora_a_train.jsonl").exists()
    assert (data_dir / "lora_a_prime_train.jsonl").exists()
    assert (data_dir / "lora_b_train.jsonl").exists()


def test_all_three_files_same_ids_in_same_order(training_data):
    """All three files have identical example_id lists in identical order."""
    ids_a = [r["example_id"] for r in training_data["lora_a"]]
    ids_a_prime = [r["example_id"] for r in training_data["lora_a_prime"]]
    ids_b = [r["example_id"] for r in training_data["lora_b"]]

    assert ids_a == ids_a_prime
    assert ids_a == ids_b


def test_lora_a_prime_and_b_prompts_byte_identical(training_data):
    """LoRA-A' and LoRA-B prompts are byte-identical for every example.

    This is the core clean-ablation check: A' and B differ only in target,
    not in input.
    """
    a_prime_records = training_data["lora_a_prime"]
    b_records = training_data["lora_b"]

    for a_prime, b in zip(a_prime_records, b_records):
        assert a_prime["example_id"] == b["example_id"], \
            f"ID mismatch: {a_prime['example_id']} != {b['example_id']}"
        assert a_prime["prompt"] == b["prompt"], \
            f"Prompt mismatch for {a_prime['example_id']}"


def test_lora_a_prompt_has_no_passages(training_data):
    """LoRA-A prompts contain no passage markers or passage text."""
    for record in training_data["lora_a"]:
        prompt = record["prompt"]
        assert "[P1]" not in prompt
        assert "[P2]" not in prompt
        assert "[P3]" not in prompt
        assert "[P4]" not in prompt
        assert "[P5]" not in prompt
        assert "Passage" not in prompt


def test_lora_b_targets_have_citations(training_data):
    """At least 80% of LoRA-B targets contain at least one [P1]-[P5] citation.

    Some factoid answers are short and uncited; threshold is 80% not 100%.
    """
    targets_with_citations = 0
    total = len(training_data["lora_b"])

    citation_pattern = r'\[P[1-5]\]'

    for record in training_data["lora_b"]:
        target = record["target"]
        if re.search(citation_pattern, target):
            targets_with_citations += 1

    citation_rate = targets_with_citations / total
    assert citation_rate >= 0.80, \
        f"Only {citation_rate:.1%} of targets have citations (expected ≥80%)"


def test_answer_labels_consistent(training_data):
    """Answer labels are consistent across all three files for each example_id."""
    # Build a map from example_id to answer for each file
    def extract_answer(target):
        # Target ends with "Answer: X<|eot_id|>"
        # Extract the part between "Answer: " and "<|eot_id|>"
        match = re.search(r'Answer: (.+?)<\|eot_id\|>', target)
        if match:
            return match.group(1)
        return None

    answers_a = {r["example_id"]: extract_answer(r["target"])
                 for r in training_data["lora_a"]}
    answers_a_prime = {r["example_id"]: extract_answer(r["target"])
                       for r in training_data["lora_a_prime"]}
    answers_b = {r["example_id"]: extract_answer(r["target"])
                 for r in training_data["lora_b"]}

    # Check consistency
    for example_id in answers_a.keys():
        assert answers_a[example_id] == answers_a_prime[example_id], \
            f"Answer mismatch between A and A' for {example_id}"
        assert answers_a[example_id] == answers_b[example_id], \
            f"Answer mismatch between A and B for {example_id}"


def test_citation_normalization_no_bare_P(training_data):
    """LoRA-B targets have no bare P1-P5 references — all are bracketed.

    We allow bare P followed by digits > 5 (e.g., P21 for genes, P450 for enzymes).
    """
    # Pattern: bare P1-P5 with word boundaries, not preceded by [
    bare_pattern = r'(?<!\[)\bP([1-5])\b(?!\])'

    violations = []
    for record in training_data["lora_b"]:
        target = record["target"]
        matches = re.findall(bare_pattern, target)
        if matches:
            violations.append({
                "example_id": record["example_id"],
                "matches": matches,
                "target_snippet": target[:200]
            })

    assert len(violations) == 0, \
        f"Found {len(violations)} targets with bare P1-P5:\n{violations[:5]}"


def test_chat_template_markers_present(training_data):
    """All prompts and targets contain expected Llama 3.1 chat template markers."""
    # Markers expected in prompts
    prompt_markers = [
        "<|begin_of_text|>",
        "<|start_header_id|>system<|end_header_id|>",
        "<|start_header_id|>user<|end_header_id|>",
        "<|start_header_id|>assistant<|end_header_id|>",
        "<|eot_id|>",
    ]

    # Marker expected in targets
    target_marker = "<|eot_id|>"

    for name in ["lora_a", "lora_a_prime", "lora_b"]:
        for record in training_data[name]:
            prompt = record["prompt"]
            target = record["target"]

            # Check all prompt markers present
            for marker in prompt_markers:
                assert marker in prompt, \
                    f"{name} prompt missing {marker} for {record['example_id']}"

            # Check target marker present
            assert target_marker in target, \
                f"{name} target missing {target_marker} for {record['example_id']}"
