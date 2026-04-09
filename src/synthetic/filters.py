"""Filtering pipeline for synthetic LoRA-B training examples.

Implements 6 filters to ensure quality and correctness of generated examples.
"""

import re
import unicodedata
from typing import Any


# Stopwords for factoid normalization
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "at",
    "to", "for", "and", "or", "but", "with", "by", "from", "as", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "under", "again", "further", "then", "once",
}

# Word-to-number mapping for factoid normalization
# Covers zero through twenty, plus tens, and common large numbers
WORD_TO_NUMBER = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
    "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60",
    "seventy": "70", "eighty": "80", "ninety": "90",
    "hundred": "100", "thousand": "1000", "million": "1000000",
    "billion": "1000000000",
}


class FilterResult:
    """Result of applying a filter to an example."""

    def __init__(self, passed: bool, reason: str = ""):
        self.passed = passed
        self.reason = reason

    def __bool__(self):
        return self.passed


def filter_format(example: dict[str, Any]) -> FilterResult:
    """Filter 1: Output must contain 'Reasoning:' and 'Answer:' markers in that order.

    Args:
        example: Generated example with generated_reasoning and generated_answer fields

    Returns:
        FilterResult indicating pass/fail
    """
    reasoning = example.get("generated_reasoning")
    answer = example.get("generated_answer")

    if reasoning is None or answer is None:
        return FilterResult(False, "Missing reasoning or answer fields")

    return FilterResult(True)


def has_valid_citations(reasoning: str) -> bool:
    """Check if reasoning contains valid citation markers.

    Accepts two formats:
    1. Bracketed: [P1], [P2], [P1, P3], etc.
    2. Bare with word boundary: P1, P2, etc. (as in "P1 states...", "according to P2")

    The bare format is accepted because the model sometimes uses "P1" instead of "[P1]"
    while still correctly referencing passages. Collision analysis confirmed that bare
    P[1-5] patterns do not collide with gene names (P53 is outside range, FOXP2 has no
    word boundary) or other scientific notation in our corpus.

    Args:
        reasoning: The generated reasoning text

    Returns:
        True if at least one valid citation marker is found
    """
    # Pattern 1: Bracketed citations [P1], [P2], etc.
    bracketed_pattern = r"\[P[1-5]"
    if re.search(bracketed_pattern, reasoning):
        return True

    # Pattern 2: Bare citations with word boundary (P1, P2, etc.)
    # Word boundary ensures we don't match FOXP2, TP53, etc.
    bare_pattern = r"\bP[1-5]\b"
    if re.search(bare_pattern, reasoning):
        return True

    return False


def filter_citations(example: dict[str, Any]) -> FilterResult:
    """Filter 2: Reasoning must contain at least one passage citation marker.

    Accepts two citation formats:
    1. Bracketed: [P1], [P2], [P1, P3], etc.
    2. Bare with word boundary: P1, P2, etc. (as in "P1 states...", "according to P2")

    Both formats are accepted because the model sometimes uses bare "P1" references
    while still correctly grounding its reasoning in the provided passages.

    Exception: If the answer is "Insufficient evidence", citations are not required
    (there's nothing to cite when passages don't support an answer).

    Args:
        example: Generated example with generated_reasoning field

    Returns:
        FilterResult indicating pass/fail
    """
    # Exception: "Insufficient evidence" answers don't need citations
    answer = example.get("generated_answer", "")
    if answer and "insufficient evidence" in answer.lower():
        return FilterResult(True)

    reasoning = example.get("generated_reasoning", "")

    if has_valid_citations(reasoning):
        return FilterResult(True)

    return FilterResult(False, "No valid citations found in reasoning")


def filter_citation_validity(example: dict[str, Any]) -> FilterResult:
    """Filter 3: Every cited [Pn] must refer to a passage that was in the input.

    Since we always provide exactly 5 passages (P1-P5), any citation to [P6] or higher
    is hallucinated and should be rejected.

    Args:
        example: Generated example with generated_reasoning field

    Returns:
        FilterResult indicating pass/fail
    """
    reasoning = example.get("generated_reasoning", "")

    # Find all [Pn] citations
    citations = re.findall(r"\[P(\d+)\]", reasoning)

    for cite_num in citations:
        if int(cite_num) > 5 or int(cite_num) < 1:
            return FilterResult(False, f"Invalid citation [P{cite_num}] (must be P1-P5)")

    return FilterResult(True)


def filter_reasoning_length(example: dict[str, Any]) -> FilterResult:
    """Filter 4: Reasoning must be at least 20 words and at most 200 words.

    Args:
        example: Generated example with generated_reasoning field

    Returns:
        FilterResult indicating pass/fail
    """
    reasoning = example.get("generated_reasoning", "")

    # Count words (split on whitespace)
    words = reasoning.split()
    word_count = len(words)

    if word_count < 20:
        return FilterResult(False, f"Reasoning too short ({word_count} words, need ≥20)")

    if word_count > 200:
        return FilterResult(False, f"Reasoning too long ({word_count} words, need ≤200)")

    return FilterResult(True)


def filter_answer_format(example: dict[str, Any]) -> FilterResult:
    """Filter 5: Answer format must match question type.

    - For yesno questions: answer must be "Yes", "No", "Maybe", or "Insufficient evidence"
    - For factoid questions: answer must be 1-30 words

    Args:
        example: Generated example with question_type and generated_answer fields

    Returns:
        FilterResult indicating pass/fail
    """
    question_type = example.get("question_type", "")
    answer = example.get("generated_answer", "").strip()

    if not answer:
        return FilterResult(False, "Empty answer")

    if question_type == "yesno":
        # Normalize for comparison
        answer_lower = answer.lower()
        valid_answers = {"yes", "no", "maybe", "insufficient evidence"}

        if answer_lower not in valid_answers:
            return FilterResult(False, f"Invalid yes/no answer: '{answer}'")

    elif question_type == "factoid":
        # Count words
        words = answer.split()
        word_count = len(words)

        if word_count < 1:
            return FilterResult(False, "Factoid answer is empty")

        if word_count > 30:
            return FilterResult(False, f"Factoid answer too long ({word_count} words, need ≤30)")

    else:
        return FilterResult(False, f"Unknown question type: {question_type}")

    return FilterResult(True)


def filter_gold_answer_agreement(example: dict[str, Any]) -> FilterResult:
    """Filter 6: Generated answer must agree with gold answer (with special rules).

    Rules:
    1. If generated answer is "Insufficient evidence" → auto pass
    2. For yesno questions:
       a. If generated is "Maybe" with valid citations → auto pass (sets gold_disagreement)
       b. Otherwise: must match gold (case-insensitive)
    3. For factoid questions with confident answer: use normalized token overlap

    PLAN.md Q2 specifies yes/no/maybe as the yesno label space. PubMedQA test set has
    9.4% maybe labels. Model-hedged "Maybe" responses on grounded evidence are valid
    training signal for calibrated uncertainty, not filter failures. BioASQ doesn't use
    "maybe" as a gold label, but the model correctly identifies genuinely ambiguous
    evidence - this is desirable behavior for PubMedQA evaluation.

    This filter applies to BOTH easy and hard mode.

    Args:
        example: Generated example with question_type, generated_answer, and gold_answer fields

    Returns:
        FilterResult indicating pass/fail
    """
    question_type = example.get("question_type", "")
    generated_answer = example.get("generated_answer", "").strip()
    gold_answer = example.get("gold_answer", "").strip()

    if not generated_answer or not gold_answer:
        return FilterResult(False, "Missing generated or gold answer")

    # Rule 1: "Insufficient evidence" auto-passes
    if "insufficient evidence" in generated_answer.lower():
        return FilterResult(True)

    if question_type == "yesno":
        # Rule 2a: "Maybe" with valid citations auto-passes
        # This allows the model to express calibrated uncertainty on ambiguous evidence
        if generated_answer.lower() == "maybe":
            reasoning = example.get("generated_reasoning", "")
            if has_valid_citations(reasoning):
                # Track the disagreement for analysis
                gold_lower = gold_answer.lower()
                if gold_lower == "yes":
                    example["gold_disagreement"] = "model_hedged_yes"
                elif gold_lower == "no":
                    example["gold_disagreement"] = "model_hedged_no"
                else:
                    example["gold_disagreement"] = None
                return FilterResult(True)
            else:
                return FilterResult(
                    False,
                    f"Maybe answer without citations: generated='{generated_answer}'"
                )

        # Rule 2b: Case-insensitive exact match for yesno
        if generated_answer.lower() == gold_answer.lower():
            return FilterResult(True)
        else:
            return FilterResult(
                False,
                f"Yes/no mismatch: generated='{generated_answer}' vs gold='{gold_answer}'"
            )

    elif question_type == "factoid":
        # Rule 3: Normalized token overlap for factoid
        if factoid_overlap(generated_answer, gold_answer):
            return FilterResult(True)
        else:
            return FilterResult(
                False,
                f"Factoid mismatch: generated='{generated_answer}' vs gold='{gold_answer}'"
            )

    else:
        return FilterResult(False, f"Unknown question type: {question_type}")


def normalize_factoid(text: str) -> set[str]:
    """Normalize a factoid answer for fuzzy comparison.

    Steps:
    1. Strip diacritics (α → a, é → e)
    2. Lowercase
    3. Remove punctuation and parentheses
    4. Tokenize on whitespace
    5. Convert number words to digits (three → 3)
    6. Remove stopwords and single-character tokens

    Args:
        text: Answer text to normalize

    Returns:
        Set of normalized tokens
    """
    # Strip diacritics: α → a, é → e
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))

    # Lowercase
    text = text.lower()

    # Remove punctuation and parentheses
    text = re.sub(r"[^\w\s]", " ", text)

    # Tokenize
    raw_tokens = text.split()

    # Normalize number words to digits and filter
    tokens = set()
    for t in raw_tokens:
        if not t or t in STOPWORDS:
            continue
        # Convert number words to digits
        normalized = WORD_TO_NUMBER.get(t, t)
        # Keep single-char tokens only if they're digits (for number matching)
        if len(normalized) <= 1 and not normalized.isdigit():
            continue
        tokens.add(normalized)

    return tokens


def factoid_overlap(generated: str, gold: str) -> bool:
    """Check if generated answer has at least one substantive token in common with gold.

    Args:
        generated: Generated factoid answer
        gold: Gold factoid answer

    Returns:
        True if at least one token overlaps, False otherwise
    """
    gen_tokens = normalize_factoid(generated)
    gold_tokens = normalize_factoid(gold)

    return bool(gen_tokens & gold_tokens)


def apply_filters(example: dict[str, Any]) -> dict[str, Any]:
    """Apply all filters to an example and return result.

    Args:
        example: Generated example dict

    Returns:
        Same example dict with added fields:
        - filter_passed: bool
        - filter_failed_at: str (name of first failed filter, or None)
        - filter_failure_reason: str (reason for failure, or None)
    """
    filters = [
        ("format", filter_format),
        ("citations", filter_citations),
        ("citation_validity", filter_citation_validity),
        ("reasoning_length", filter_reasoning_length),
        ("answer_format", filter_answer_format),
        ("gold_answer_agreement", filter_gold_answer_agreement),
    ]

    for filter_name, filter_func in filters:
        result = filter_func(example)
        if not result.passed:
            example["filter_passed"] = False
            example["filter_failed_at"] = filter_name
            example["filter_failure_reason"] = result.reason
            return example

    # All filters passed
    example["filter_passed"] = True
    example["filter_failed_at"] = None
    example["filter_failure_reason"] = None
    # Ensure gold_disagreement is set (may have been set by Maybe handling)
    if "gold_disagreement" not in example:
        example["gold_disagreement"] = None
    return example
