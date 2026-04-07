"""Tests for the retrieval module.

Tests the index building, dense retrieval, reranking, oracle retrieval,
and evaluation metrics.
"""

import math
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.retrieval.index import build_index, load_index
from src.retrieval.retrieve import (
    retrieve,
    retrieve_with_rerank,
    _rerank,
    BGE_QUERY_INSTRUCTION,
    get_embedding_model,
)
from src.retrieval.oracle import oracle_retrieve, none_retrieve
from src.retrieval.eval_retrieval import (
    compute_hit_at_k,
    compute_proportional_recall_at_k,
    mrr,
    ndcg_at_k,
    evaluate_retrieval,
)


# --- Fixtures for fake data ---

@pytest.fixture
def fake_corpus():
    """Create a small fake corpus of 10 passages."""
    return [
        {"id": f"p{i}", "text": f"This is passage number {i} about topic {i % 3}.", "source": "test"}
        for i in range(10)
    ]


@pytest.fixture
def fake_corpus_dict(fake_corpus):
    """Corpus as a dict keyed by ID."""
    return {p["id"]: p for p in fake_corpus}


@pytest.fixture
def fake_queries():
    """Create fake queries with known gold passages."""
    return [
        {"question": "What is topic 0?", "gold_passage_ids": ["p0"], "answer": "yes"},
        {"question": "What is topic 1?", "gold_passage_ids": ["p1"], "answer": "no"},
        {"question": "What is topic 2?", "gold_passage_ids": ["p2"], "answer": "maybe"},
        {"question": "Tell me about passage 3", "gold_passage_ids": ["p3"], "answer": "yes"},
        {"question": "Tell me about passage 4", "gold_passage_ids": ["p4"], "answer": "no"},
        {"question": "Tell me about passage 5", "gold_passage_ids": ["p5"], "answer": "yes"},
        {"question": "Tell me about passage 6", "gold_passage_ids": ["p6"], "answer": "no"},
        {"question": "Tell me about passage 7", "gold_passage_ids": ["p7"], "answer": "maybe"},
        {"question": "Tell me about passage 8", "gold_passage_ids": ["p8"], "answer": "yes"},
        {"question": "Tell me about passage 9", "gold_passage_ids": ["p9"], "answer": "no"},
    ]


# --- Test 1: Index round-trip ---

class TestIndexRoundTrip:
    """Test building, saving, and loading an index."""

    def test_index_round_trip(self, fake_corpus):
        """Build a tiny index, save it, load it, verify vectors match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_index"

            # Build index with a tiny model for speed
            # We'll mock the embedding to use random vectors
            with patch("src.retrieval.index._embed_passages") as mock_embed:
                # Create deterministic fake embeddings
                np.random.seed(42)
                fake_embeddings = np.random.randn(len(fake_corpus), 384).astype(np.float32)
                # Normalize them (since the real code normalizes)
                norms = np.linalg.norm(fake_embeddings, axis=1, keepdims=True)
                fake_embeddings = fake_embeddings / norms
                mock_embed.return_value = fake_embeddings

                index, id_mapping = build_index(fake_corpus, output_path)

            # Verify index has correct size
            assert index.ntotal == len(fake_corpus)

            # Verify ID mapping
            assert len(id_mapping) == len(fake_corpus)
            for i, passage in enumerate(fake_corpus):
                assert id_mapping[i] == passage["id"]

            # Load the index
            loaded_index, loaded_mapping = load_index(output_path)

            # Verify loaded index matches
            assert loaded_index.ntotal == index.ntotal
            assert loaded_mapping == id_mapping

            # Verify we can search the loaded index
            query_vec = fake_embeddings[0:1]  # Use first passage as query
            scores, indices = loaded_index.search(query_vec, 5)
            assert len(indices[0]) == 5
            assert indices[0][0] == 0  # Should find itself first


# --- Test 2: Dense retrieval returns k results ---

class TestDenseRetrieval:
    """Test dense retrieval functionality."""

    def test_retrieval_returns_k_results(self, fake_corpus, fake_corpus_dict):
        """Dense retrieval should return exactly k results in score order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_index"

            # Build index with mocked embeddings
            with patch("src.retrieval.index._embed_passages") as mock_embed:
                np.random.seed(42)
                fake_embeddings = np.random.randn(len(fake_corpus), 384).astype(np.float32)
                norms = np.linalg.norm(fake_embeddings, axis=1, keepdims=True)
                fake_embeddings = fake_embeddings / norms
                mock_embed.return_value = fake_embeddings

                index, id_mapping = build_index(fake_corpus, output_path)

            # Mock the query embedding
            with patch("src.retrieval.retrieve.get_embedding_model") as mock_model:
                mock_encoder = MagicMock()
                # Return a query embedding similar to passage 0
                mock_encoder.encode.return_value = fake_embeddings[0:1]
                mock_model.return_value = mock_encoder

                results = retrieve(
                    query="test query",
                    index=index,
                    id_mapping=id_mapping,
                    corpus=fake_corpus_dict,
                    k=5,
                )

            # Should return exactly 5 results
            assert len(results) == 5

            # Results should be in score order (descending)
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True)

            # Each result should have required keys
            for r in results:
                assert "id" in r
                assert "text" in r
                assert "score" in r
                assert "rank" in r


# --- Test 3: Reranker reorders results ---

class TestReranker:
    """Test that reranker properly reorders results."""

    def test_reranker_reorders(self):
        """Reranker should reorder passages by reranker score, not dense score."""
        # Create passages with known dense scores
        passages = [
            {"id": "p0", "text": "First passage", "score": 0.9, "rank": 1},
            {"id": "p1", "text": "Second passage", "score": 0.8, "rank": 2},
            {"id": "p2", "text": "Third passage", "score": 0.7, "rank": 3},
            {"id": "p3", "text": "Fourth passage", "score": 0.6, "rank": 4},
            {"id": "p4", "text": "Fifth passage", "score": 0.5, "rank": 5},
        ]

        # Mock reranker to return scores in reverse order
        # (i.e., the passage with lowest dense score gets highest rerank score)
        with patch("src.retrieval.retrieve.get_reranker_model") as mock_model:
            mock_reranker = MagicMock()
            # Reranker scores: p4 highest, p0 lowest
            mock_reranker.predict.return_value = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
            mock_model.return_value = mock_reranker

            reranked = _rerank("test query", passages.copy())

        # After reranking, order should be p4, p3, p2, p1, p0
        assert [r["id"] for r in reranked] == ["p4", "p3", "p2", "p1", "p0"]

        # Ranks should be updated
        assert [r["rank"] for r in reranked] == [1, 2, 3, 4, 5]

        # Primary score should be reranker score, not dense score
        assert reranked[0]["score"] == 0.9  # p4's rerank score
        assert reranked[0]["rerank_score"] == 0.9
        assert reranked[0]["dense_score"] == 0.5  # p4's original dense score


# --- Test 4: Oracle includes gold passage ---

class TestOracleRetrieval:
    """Test oracle retrieval functionality."""

    def test_oracle_includes_gold_at_position_0(self, fake_corpus, fake_corpus_dict):
        """Oracle retrieval should include gold passage at position 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_index"

            # Build index
            with patch("src.retrieval.index._embed_passages") as mock_embed:
                np.random.seed(42)
                fake_embeddings = np.random.randn(len(fake_corpus), 384).astype(np.float32)
                norms = np.linalg.norm(fake_embeddings, axis=1, keepdims=True)
                fake_embeddings = fake_embeddings / norms
                mock_embed.return_value = fake_embeddings

                index, id_mapping = build_index(fake_corpus, output_path)

            # Mock retrieval to return passages p5-p9 (not including gold p0)
            # Note: mock where it's used (oracle module), not where it's defined
            with patch("src.retrieval.oracle.retrieve_with_rerank") as mock_retrieve:
                mock_retrieve.return_value = [
                    {"id": "p5", "text": "text5", "score": 0.9, "rank": 1},
                    {"id": "p6", "text": "text6", "score": 0.8, "rank": 2},
                    {"id": "p7", "text": "text7", "score": 0.7, "rank": 3},
                    {"id": "p8", "text": "text8", "score": 0.6, "rank": 4},
                    {"id": "p9", "text": "text9", "score": 0.5, "rank": 5},
                ]

                results = oracle_retrieve(
                    query="test query",
                    gold_passage_ids=["p0"],
                    index=index,
                    id_mapping=id_mapping,
                    corpus=fake_corpus_dict,
                    k=5,
                )

        # Gold passage should be at position 0
        assert results[0]["id"] == "p0"
        assert results[0]["is_gold"] is True

        # Should have exactly 5 results
        assert len(results) == 5

        # Other passages should not be marked as gold
        for r in results[1:]:
            assert r["is_gold"] is False


# --- Test 5: Oracle deduplication ---

class TestOracleDeduplication:
    """Test that oracle retrieval deduplicates properly."""

    def test_oracle_deduplicates_gold_from_strong(self, fake_corpus, fake_corpus_dict):
        """When gold passage appears in strong retrieval, result should have 5 unique passages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_index"

            # Build index
            with patch("src.retrieval.index._embed_passages") as mock_embed:
                np.random.seed(42)
                fake_embeddings = np.random.randn(len(fake_corpus), 384).astype(np.float32)
                norms = np.linalg.norm(fake_embeddings, axis=1, keepdims=True)
                fake_embeddings = fake_embeddings / norms
                mock_embed.return_value = fake_embeddings

                index, id_mapping = build_index(fake_corpus, output_path)

            # Mock retrieval to include gold passage p0 in results
            # Note: mock where it's used (oracle module), not where it's defined
            with patch("src.retrieval.oracle.retrieve_with_rerank") as mock_retrieve:
                mock_retrieve.return_value = [
                    {"id": "p0", "text": "text0", "score": 0.95, "rank": 1},  # Gold!
                    {"id": "p5", "text": "text5", "score": 0.9, "rank": 2},
                    {"id": "p6", "text": "text6", "score": 0.8, "rank": 3},
                    {"id": "p7", "text": "text7", "score": 0.7, "rank": 4},
                    {"id": "p8", "text": "text8", "score": 0.6, "rank": 5},
                    {"id": "p9", "text": "text9", "score": 0.5, "rank": 6},
                ]

                results = oracle_retrieve(
                    query="test query",
                    gold_passage_ids=["p0"],
                    index=index,
                    id_mapping=id_mapping,
                    corpus=fake_corpus_dict,
                    k=5,
                )

        # Should have exactly 5 results (no duplicate)
        assert len(results) == 5

        # All IDs should be unique
        ids = [r["id"] for r in results]
        assert len(ids) == len(set(ids))

        # Gold should still be at position 0
        assert results[0]["id"] == "p0"
        assert results[0]["is_gold"] is True


# --- Test 6: Proportional Recall@5 metric (renamed from recall_at_k) ---

class TestProportionalRecallMetric:
    """Test Proportional Recall@k metric computation."""

    def test_proportional_recall_at_k_single_gold(self):
        """Proportional Recall@5 with single gold per query (equals Hit@5)."""
        # 10 queries with single gold, 7 have gold in top-5
        test_cases = [
            (["p0", "p1", "p2", "p3", "p4"], ["p0"], 1.0),  # Hit
            (["p1", "p2", "p3", "p4", "p5"], ["p0"], 0.0),  # Miss
            (["p0", "p1", "p2", "p3", "p4"], ["p2"], 1.0),  # Hit
            (["p0", "p1", "p2", "p3", "p4"], ["p5"], 0.0),  # Miss
            (["p5", "p6", "p7", "p8", "p9"], ["p9"], 1.0),  # Hit
            (["p0", "p1", "p2", "p3", "p4"], ["p1"], 1.0),  # Hit
            (["p0", "p1", "p2", "p3", "p4"], ["p3"], 1.0),  # Hit
            (["p5", "p6", "p7", "p8", "p9"], ["p0"], 0.0),  # Miss
            (["p0", "p1", "p2", "p3", "p4"], ["p4"], 1.0),  # Hit
            (["p0", "p1", "p2", "p3", "p4"], ["p0"], 1.0),  # Hit
        ]

        recalls = [compute_proportional_recall_at_k(retrieved, gold, 5) for retrieved, gold, _ in test_cases]
        avg_recall = sum(recalls) / len(recalls)

        # 7 hits out of 10 = 0.7
        assert avg_recall == 0.7


# --- Test 6b: Hit@5 metric (binary: any gold in top-k?) ---

class TestHitMetric:
    """Test Hit@k metric computation with multi-gold scenarios."""

    def test_hit_at_k_multi_gold_synthetic(self):
        """Hit@5 on synthetic case with varying numbers of gold passages.

        5 queries with multiple golds (counts: 3, 8, 12, 5, 1).
        Hand-constructed to test both Hit@5 and Proportional Recall@5.
        """
        # Query 1: 3 gold passages, 2 in top-5 -> Hit=1.0, PropR=2/3=0.667
        q1_retrieved = ["g1", "g2", "x1", "x2", "x3"]
        q1_gold = ["g1", "g2", "g3"]
        assert compute_hit_at_k(q1_retrieved, q1_gold, 5) == 1.0
        assert abs(compute_proportional_recall_at_k(q1_retrieved, q1_gold, 5) - 2/3) < 1e-6

        # Query 2: 8 gold passages, 4 in top-5 -> Hit=1.0, PropR=4/8=0.5
        q2_retrieved = ["g1", "g2", "g3", "g4", "x1"]
        q2_gold = ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8"]
        assert compute_hit_at_k(q2_retrieved, q2_gold, 5) == 1.0
        assert abs(compute_proportional_recall_at_k(q2_retrieved, q2_gold, 5) - 4/8) < 1e-6

        # Query 3: 12 gold passages, 0 in top-5 -> Hit=0.0, PropR=0/12=0.0
        q3_retrieved = ["x1", "x2", "x3", "x4", "x5"]
        q3_gold = ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g11", "g12"]
        assert compute_hit_at_k(q3_retrieved, q3_gold, 5) == 0.0
        assert compute_proportional_recall_at_k(q3_retrieved, q3_gold, 5) == 0.0

        # Query 4: 5 gold passages, 5 in top-5 -> Hit=1.0, PropR=5/5=1.0
        q4_retrieved = ["g1", "g2", "g3", "g4", "g5"]
        q4_gold = ["g1", "g2", "g3", "g4", "g5"]
        assert compute_hit_at_k(q4_retrieved, q4_gold, 5) == 1.0
        assert compute_proportional_recall_at_k(q4_retrieved, q4_gold, 5) == 1.0

        # Query 5: 1 gold passage, 1 in top-5 -> Hit=1.0, PropR=1/1=1.0
        q5_retrieved = ["x1", "x2", "g1", "x3", "x4"]
        q5_gold = ["g1"]
        assert compute_hit_at_k(q5_retrieved, q5_gold, 5) == 1.0
        assert compute_proportional_recall_at_k(q5_retrieved, q5_gold, 5) == 1.0

        # Aggregate: Hit@5 = (1+1+0+1+1)/5 = 0.8
        all_hits = [
            compute_hit_at_k(q1_retrieved, q1_gold, 5),
            compute_hit_at_k(q2_retrieved, q2_gold, 5),
            compute_hit_at_k(q3_retrieved, q3_gold, 5),
            compute_hit_at_k(q4_retrieved, q4_gold, 5),
            compute_hit_at_k(q5_retrieved, q5_gold, 5),
        ]
        assert sum(all_hits) / len(all_hits) == 0.8

        # Aggregate PropR@5 = (2/3 + 4/8 + 0 + 1 + 1)/5 = (0.667 + 0.5 + 0 + 1 + 1)/5 ≈ 0.633
        all_prop_recalls = [
            compute_proportional_recall_at_k(q1_retrieved, q1_gold, 5),
            compute_proportional_recall_at_k(q2_retrieved, q2_gold, 5),
            compute_proportional_recall_at_k(q3_retrieved, q3_gold, 5),
            compute_proportional_recall_at_k(q4_retrieved, q4_gold, 5),
            compute_proportional_recall_at_k(q5_retrieved, q5_gold, 5),
        ]
        expected_prop_recall = (2/3 + 4/8 + 0 + 1 + 1) / 5
        assert abs(sum(all_prop_recalls) / len(all_prop_recalls) - expected_prop_recall) < 1e-6

    def test_hit_at_k_edge_cases(self):
        """Test Hit@k edge cases."""
        # Empty gold list -> 0.0
        assert compute_hit_at_k(["p0", "p1"], [], 5) == 0.0

        # Empty retrieved list -> 0.0
        assert compute_hit_at_k([], ["g1"], 5) == 0.0

        # Gold at position exactly k -> included
        assert compute_hit_at_k(["x1", "x2", "x3", "x4", "g1"], ["g1"], 5) == 1.0

        # Gold at position k+1 -> not included
        assert compute_hit_at_k(["x1", "x2", "x3", "x4", "x5", "g1"], ["g1"], 5) == 0.0


# --- Test 7: MRR metric ---

class TestMRRMetric:
    """Test MRR metric computation."""

    def test_mrr_synthetic(self):
        """MRR on synthetic case with hand-computed values."""
        test_cases = [
            (["p0", "p1", "p2", "p3", "p4"], ["p0"], 1.0),      # Rank 1 -> 1/1 = 1.0
            (["p0", "p1", "p2", "p3", "p4"], ["p1"], 0.5),      # Rank 2 -> 1/2 = 0.5
            (["p0", "p1", "p2", "p3", "p4"], ["p2"], 1/3),      # Rank 3 -> 1/3
            (["p0", "p1", "p2", "p3", "p4"], ["p4"], 0.2),      # Rank 5 -> 1/5 = 0.2
            (["p5", "p6", "p7", "p8", "p9"], ["p0"], 0.0),      # Not found -> 0
        ]

        for retrieved, gold, expected in test_cases:
            actual = mrr(retrieved, gold)
            assert abs(actual - expected) < 1e-6, f"MRR mismatch: {actual} vs {expected}"


# --- Test 8: nDCG@5 metric ---

class TestNDCGMetric:
    """Test nDCG@k metric computation."""

    def test_ndcg_at_k_synthetic(self):
        """nDCG@5 on synthetic case with hand-computed values."""
        # Single relevant document at various positions
        # IDCG for 1 relevant doc at k=5: 1/log2(2) = 1.0

        # Gold at rank 1: DCG = 1/log2(2) = 1.0, nDCG = 1.0/1.0 = 1.0
        assert abs(ndcg_at_k(["p0", "p1", "p2", "p3", "p4"], ["p0"], 5) - 1.0) < 1e-6

        # Gold at rank 2: DCG = 1/log2(3) ≈ 0.631, nDCG ≈ 0.631
        expected_rank2 = 1.0 / math.log2(3)
        assert abs(ndcg_at_k(["p1", "p0", "p2", "p3", "p4"], ["p0"], 5) - expected_rank2) < 1e-6

        # Gold at rank 5: DCG = 1/log2(6) ≈ 0.387, nDCG ≈ 0.387
        expected_rank5 = 1.0 / math.log2(6)
        assert abs(ndcg_at_k(["p1", "p2", "p3", "p4", "p0"], ["p0"], 5) - expected_rank5) < 1e-6

        # Gold not in top-5: nDCG = 0
        assert ndcg_at_k(["p1", "p2", "p3", "p4", "p5"], ["p0"], 5) == 0.0


# --- Test 9: Stratified eval has expected keys ---

class TestStratifiedEval:
    """Test that evaluation returns stratified breakdowns."""

    def test_eval_has_stratified_keys(self, fake_corpus, fake_corpus_dict, fake_queries):
        """Evaluation should return overall and stratified metrics."""
        # Mock everything to avoid needing real models
        with patch("src.retrieval.eval_retrieval.retrieve_with_rerank") as mock_retrieve:
            # Return fake retrieval results (gold found for first 7, missed for last 3)
            def fake_retrieve(query, **kwargs):
                # Find which query this is
                for i, q in enumerate(fake_queries):
                    if q["question"] == query:
                        if i < 7:  # First 7 find their gold
                            return [{"id": f"p{i}", "text": f"text{i}", "score": 0.9, "rank": 1}]
                        else:  # Last 3 miss
                            return [{"id": "p99", "text": "wrong", "score": 0.5, "rank": 1}]
                return []

            mock_retrieve.side_effect = fake_retrieve

            # Run evaluation
            results = evaluate_retrieval(
                queries=fake_queries,
                index=None,  # Not used due to mock
                id_mapping={},
                corpus=fake_corpus_dict,
                k=5,
            )

        # Check overall metrics exist
        assert "overall" in results
        assert "hit@5" in results["overall"]
        assert "proportional_recall@5" in results["overall"]
        assert "mrr" in results["overall"]
        assert "ndcg@5" in results["overall"]
        assert "n_queries" in results["overall"]

        # Check stratification by answer label exists
        assert "by_answer_label" in results
        # Should have entries for yes, no, maybe
        assert len(results["by_answer_label"]) > 0

        # Check stratification by passage length exists
        assert "by_passage_length" in results
        assert "short" in results["by_passage_length"]
        assert "medium" in results["by_passage_length"]
        assert "long" in results["by_passage_length"]

        # Check stratification by question type exists
        assert "by_question_type" in results

        # Check stratification by n_gold exists
        assert "by_n_gold" in results


# --- Test 10: BGE query prefix is present ---

class TestBGEQueryPrefix:
    """Test that BGE query instruction prefix is used correctly."""

    def test_bge_query_instruction_constant_exists(self):
        """The BGE query instruction constant should exist and be correct."""
        assert BGE_QUERY_INSTRUCTION == "Represent this sentence for searching relevant passages: "

    def test_query_is_prefixed_in_retrieval(self, fake_corpus, fake_corpus_dict):
        """Queries should be prefixed with BGE instruction before encoding."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_index"

            # Build index
            with patch("src.retrieval.index._embed_passages") as mock_embed:
                np.random.seed(42)
                fake_embeddings = np.random.randn(len(fake_corpus), 384).astype(np.float32)
                norms = np.linalg.norm(fake_embeddings, axis=1, keepdims=True)
                fake_embeddings = fake_embeddings / norms
                mock_embed.return_value = fake_embeddings

                index, id_mapping = build_index(fake_corpus, output_path)

            # Track what gets passed to encode()
            encoded_queries = []

            with patch("src.retrieval.retrieve.get_embedding_model") as mock_model:
                mock_encoder = MagicMock()

                def capture_encode(texts, **kwargs):
                    encoded_queries.extend(texts)
                    return fake_embeddings[0:1]

                mock_encoder.encode.side_effect = capture_encode
                mock_model.return_value = mock_encoder

                retrieve(
                    query="my test query",
                    index=index,
                    id_mapping=id_mapping,
                    corpus=fake_corpus_dict,
                    k=5,
                )

            # The encoded query should have the BGE prefix
            assert len(encoded_queries) == 1
            assert encoded_queries[0].startswith(BGE_QUERY_INSTRUCTION)
            assert "my test query" in encoded_queries[0]


# --- Test: none_retrieve ---

class TestNoneRetrieval:
    """Test the 'none' retrieval condition."""

    def test_none_retrieve_returns_empty(self):
        """none_retrieve should return an empty list."""
        results = none_retrieve(k=5)
        assert results == []
        assert len(results) == 0
