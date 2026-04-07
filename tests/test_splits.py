"""Tests for dataset splits and test set verification."""

import pytest

from src.data.pubmedqa import load_pubmedqa
from src.data.splits import (
    get_splits,
    compute_test_hash,
    verify_test_hash,
    check_no_overlap,
    get_supported_retrieval_conditions,
)


class TestPubMedQA:
    """Tests for PubMedQA data loading."""

    def test_pubmedqa_schema(self):
        """Verify PubMedQA examples have expected schema."""
        examples = load_pubmedqa("test")

        assert len(examples) > 0, "Test set should not be empty"

        # Check first example has all required fields
        ex = examples[0]
        required_fields = [
            "id",
            "question",
            "question_type",
            "answer",
            "context",
            "gold_passage_ids",
        ]
        for field in required_fields:
            assert field in ex, f"Missing required field: {field}"

        # Check types
        assert isinstance(ex["id"], str)
        assert isinstance(ex["question"], str)
        assert ex["question_type"] == "yesno"
        assert ex["answer"] in ["yes", "no", "maybe"]
        assert isinstance(ex["context"], str)
        assert isinstance(ex["gold_passage_ids"], list)

    def test_pubmedqa_split_sizes(self):
        """Verify PubMedQA split sizes match expected values."""
        train = load_pubmedqa("train")
        dev = load_pubmedqa("dev")
        test = load_pubmedqa("test")

        assert len(train) == 450, f"Train should have 450 examples, got {len(train)}"
        assert len(dev) == 50, f"Dev should have 50 examples, got {len(dev)}"
        assert len(test) == 500, f"Test should have 500 examples, got {len(test)}"

    def test_train_dev_test_no_overlap(self):
        """Verify no overlap between train, dev, and test sets."""
        train = load_pubmedqa("train")
        dev = load_pubmedqa("dev")
        test = load_pubmedqa("test")

        train_ids = {ex["id"] for ex in train}
        dev_ids = {ex["id"] for ex in dev}
        test_ids = {ex["id"] for ex in test}

        assert train_ids.isdisjoint(dev_ids), "Train and dev overlap"
        assert train_ids.isdisjoint(test_ids), "Train and test overlap"
        assert dev_ids.isdisjoint(test_ids), "Dev and test overlap"

    def test_pubmedqa_retrieval_conditions(self):
        """Verify PubMedQA supports all three retrieval conditions."""
        conditions = get_supported_retrieval_conditions("pubmedqa")
        assert conditions == ["none", "strong", "oracle"]


class TestHashMechanism:
    """Tests for test set hash verification."""

    def test_test_set_hash_deterministic(self):
        """Verify hash computation is deterministic."""
        test = load_pubmedqa("test")

        hash1 = compute_test_hash(test)
        hash2 = compute_test_hash(test)

        assert hash1 == hash2, "Hash should be deterministic"
        assert len(hash1) == 64, "SHA-256 hash should be 64 hex chars"

    def test_hash_changes_with_data(self):
        """Verify hash changes when data changes."""
        test = load_pubmedqa("test")
        original_hash = compute_test_hash(test)

        # Modify one answer
        modified_test = [ex.copy() for ex in test]
        if modified_test[0]["answer"] == "yes":
            modified_test[0]["answer"] = "no"
        else:
            modified_test[0]["answer"] = "yes"

        modified_hash = compute_test_hash(modified_test)
        assert original_hash != modified_hash, "Hash should change when data changes"

    def test_hash_order_independent(self):
        """Verify hash is independent of example order (we sort by ID)."""
        test = load_pubmedqa("test")
        reversed_test = list(reversed(test))

        hash1 = compute_test_hash(test)
        hash2 = compute_test_hash(reversed_test)

        assert hash1 == hash2, "Hash should be independent of input order"

    def test_check_no_overlap_function(self):
        """Verify check_no_overlap utility works correctly."""
        train = load_pubmedqa("train")
        test = load_pubmedqa("test")

        assert check_no_overlap(train, test), "Should return True for non-overlapping sets"

        # Test with overlapping data
        overlapping = train[:5] + test[:5]
        assert not check_no_overlap(train, overlapping), "Should return False for overlapping sets"


class TestBioASQ:
    """Tests for BioASQ data loading (requires BIOASQ_DATA_PATH)."""

    @pytest.mark.skipif(
        True,  # Will be replaced with actual env check
        reason="BioASQ requires BIOASQ_DATA_PATH environment variable"
    )
    def test_bioasq_filters_list_questions(self):
        """Verify BioASQ loader excludes list questions."""
        from src.data.bioasq import load_bioasq

        examples = load_bioasq("train")

        for ex in examples:
            assert ex["question_type"] in ["factoid", "yesno"], \
                f"Found unexpected question type: {ex['question_type']}"

    def test_bioasq_retrieval_conditions(self):
        """Verify BioASQ supports all three retrieval conditions."""
        conditions = get_supported_retrieval_conditions("bioasq")
        assert conditions == ["none", "strong", "oracle"]


class TestMIRAGE:
    """Tests for MIRAGE subset loading."""

    def test_mirage_loads(self):
        """Verify MIRAGE subset loads correctly."""
        from src.data.mirage import load_mirage

        examples = load_mirage()

        assert len(examples) == 500, f"MIRAGE should have 500 examples, got {len(examples)}"

    def test_mirage_source_distribution(self):
        """Verify MIRAGE has examples from all three sources."""
        from src.data.mirage import load_mirage

        examples = load_mirage()

        # Count by source
        source_counts = {}
        for ex in examples:
            source = ex["source_dataset"]
            source_counts[source] = source_counts.get(source, 0) + 1

        # Verify all three sources present with minimum 100 each
        assert "mmlu_med" in source_counts, "Missing MMLU-Med examples"
        assert "medqa_us" in source_counts, "Missing MedQA-US examples"
        assert "medmcqa" in source_counts, "Missing MedMCQA examples"

        assert source_counts["mmlu_med"] >= 100, f"Too few MMLU-Med: {source_counts['mmlu_med']}"
        assert source_counts["medqa_us"] >= 100, f"Too few MedQA-US: {source_counts['medqa_us']}"
        assert source_counts["medmcqa"] >= 100, f"Too few MedMCQA: {source_counts['medmcqa']}"

    def test_mirage_schema(self):
        """Verify MIRAGE examples have expected schema (no gold_passage_ids)."""
        from src.data.mirage import load_mirage

        examples = load_mirage()
        ex = examples[0]

        # Required fields
        required_fields = [
            "id",
            "source_dataset",
            "question",
            "options",
            "answer",
            "subject",
        ]
        for field in required_fields:
            assert field in ex, f"Missing required field: {field}"

        # Schema checks
        assert ex["answer"] in ["A", "B", "C", "D"], f"Invalid answer: {ex['answer']}"
        assert len(ex["options"]) == 4, "Should have 4 options"
        assert ex["source_dataset"] in ["mmlu_med", "medqa_us", "medmcqa"]

        # Verify NO gold_passage_ids field
        assert "gold_passage_ids" not in ex, "MIRAGE should NOT have gold_passage_ids"

    def test_mirage_retrieval_conditions(self):
        """Verify MIRAGE only supports none and strong (no oracle)."""
        conditions = get_supported_retrieval_conditions("mirage")
        assert conditions == ["none", "strong"], f"Expected ['none', 'strong'], got {conditions}"
        assert "oracle" not in conditions, "MIRAGE should not support oracle retrieval"

    def test_mirage_excludes_training_sources(self):
        """Verify MIRAGE excludes PubMedQA and BioASQ to avoid overlap."""
        from src.data.mirage import load_mirage

        examples = load_mirage()

        for ex in examples:
            assert ex["source_dataset"] not in ["pubmedqa", "bioasq"], \
                f"MIRAGE should not include {ex['source_dataset']} (training data overlap)"


class TestCorpus:
    """Tests for corpus building."""

    def test_corpus_builds(self):
        """Verify corpus builds without BioASQ."""
        from src.data.corpus import build_corpus

        # Build without BioASQ (doesn't require env var)
        corpus = build_corpus(include_bioasq=False)

        assert len(corpus) > 0, "Corpus should not be empty"

    def test_corpus_passage_schema(self):
        """Verify corpus passages have expected schema."""
        from src.data.corpus import build_corpus

        corpus = build_corpus(include_bioasq=False)
        passage = corpus[0]

        required_fields = ["id", "text", "source", "metadata"]
        for field in required_fields:
            assert field in passage, f"Missing required field: {field}"

    def test_corpus_deduplication(self):
        """Verify corpus deduplication works."""
        from src.data.corpus import build_corpus

        corpus = build_corpus(include_bioasq=False)

        # Check no duplicate IDs
        ids = [p["id"] for p in corpus]
        assert len(ids) == len(set(ids)), "Corpus should not have duplicate IDs"

        # Check no duplicate texts (normalized)
        texts = [p["text"].strip().lower() for p in corpus]
        assert len(texts) == len(set(texts)), "Corpus should not have duplicate texts"
