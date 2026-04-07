"""Stratified analysis by question type."""

from typing import Any


def stratify_by_question_type(
    results: dict[str, dict[str, Any]],
    test_data: list[dict],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Stratify results by question type.

    Args:
        results: Cell results
        test_data: Test examples with "question_type" field

    Returns:
        Dict mapping question_type to cell results for that subset
    """
    raise NotImplementedError("TODO: Implement in M9")


def stratify_by_retrieval_quality(
    results: dict[str, dict[str, Any]],
    retrieval_metrics: dict[str, float],
    test_data: list[dict],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Stratify results by how well retrieval worked for each example.

    Splits into terciles based on whether gold passage was retrieved.
    """
    raise NotImplementedError("TODO: Implement in M9")


def compute_stratified_interaction(
    stratified_results: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Compute interaction effect within each stratum."""
    raise NotImplementedError("TODO: Implement in M9")
