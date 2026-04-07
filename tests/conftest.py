"""Pytest fixtures and configuration."""

import pytest
from pathlib import Path


@pytest.fixture
def project_root() -> Path:
    """Return project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_question() -> dict:
    """Return a sample question for testing."""
    return {
        "id": "test_001",
        "question": "Does aspirin reduce the risk of heart attack?",
        "question_type": "yesno",
        "answer": "yes",
        "context": "A large randomized trial showed that daily aspirin significantly reduced cardiovascular events...",
    }


@pytest.fixture
def sample_passages() -> list[dict]:
    """Return sample passages for testing."""
    return [
        {"id": "p1", "text": "Aspirin has been shown to reduce platelet aggregation..."},
        {"id": "p2", "text": "The ARRIVE trial found that aspirin reduced MI risk by 15%..."},
        {"id": "p3", "text": "Side effects of aspirin include gastrointestinal bleeding..."},
        {"id": "p4", "text": "Alternative antiplatelet agents include clopidogrel..."},
        {"id": "p5", "text": "Guidelines recommend aspirin for secondary prevention..."},
    ]


@pytest.fixture
def sample_corpus(sample_passages) -> dict[str, dict]:
    """Return sample corpus for testing."""
    return {p["id"]: p for p in sample_passages}
