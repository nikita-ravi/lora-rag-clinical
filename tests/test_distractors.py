"""Tests for distractor sampling."""

import pytest


@pytest.mark.skip(reason="TODO: Implement in M5")
def test_distractor_sampling_gold_probability():
    """Verify ~80% of samples include gold passage."""
    pass


@pytest.mark.skip(reason="TODO: Implement in M5")
def test_distractor_sampling_deterministic():
    """Verify same seed produces same distractors."""
    pass


@pytest.mark.skip(reason="TODO: Implement in M5")
def test_distractor_sampling_excludes_gold_from_distractors():
    """Verify gold passage not in distractor set when included separately."""
    pass


@pytest.mark.skip(reason="TODO: Implement in M5")
def test_lora_a_prime_and_lora_b_same_passages():
    """Verify LoRA-A' and LoRA-B get identical passages for same example."""
    pass


@pytest.mark.skip(reason="TODO: Implement in M5")
def test_distractor_count():
    """Verify total passage count is always 5."""
    pass


@pytest.mark.skip(reason="TODO: Implement in M5")
def test_passages_are_shuffled():
    """Verify gold passage is not always in position 0."""
    pass
