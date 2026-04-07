"""Error annotation schema and data models."""

from dataclasses import dataclass
from typing import Any
from enum import Enum


class ErrorCategory(Enum):
    """Error categories for annotation."""

    FACTUAL_RECALL = "factual_recall"
    STATISTICAL_MISINTERPRETATION = "statistical_misinterpretation"
    POPULATION_OVERGENERALIZATION = "population_overgeneralization"
    STALE_EVIDENCE = "stale_evidence"
    REASONING_ERROR = "reasoning_error"
    REFUSAL = "refusal"
    OTHER = "other"


ERROR_CATEGORIES = {
    ErrorCategory.FACTUAL_RECALL: {
        "name": "Factual recall error",
        "description": "Answer contradicts a clearly stated fact in the evidence",
        "examples": [
            "Evidence says drug A is effective, model says it's ineffective",
            "Evidence states specific dosage, model gives wrong number",
        ],
    },
    ErrorCategory.STATISTICAL_MISINTERPRETATION: {
        "name": "Statistical misinterpretation",
        "description": "Misreads effect size, confidence interval, p-value, or direction",
        "examples": [
            "Confuses relative risk with absolute risk",
            "Misinterprets non-significant p-value as evidence of no effect",
            "Gets direction of effect wrong (increase vs decrease)",
        ],
    },
    ErrorCategory.POPULATION_OVERGENERALIZATION: {
        "name": "Population overgeneralization",
        "description": "Applies a finding beyond the studied population",
        "examples": [
            "Study on adults, model applies to children",
            "Study on specific disease subtype, model generalizes to all cases",
        ],
    },
    ErrorCategory.STALE_EVIDENCE: {
        "name": "Stale evidence anchoring",
        "description": "Relies on outdated info despite newer evidence in context",
        "examples": [
            "Ignores more recent study that contradicts older one",
            "Uses superseded clinical guidelines",
        ],
    },
    ErrorCategory.REASONING_ERROR: {
        "name": "Reasoning error",
        "description": "Internally inconsistent chain of reasoning",
        "examples": [
            "Premises support one conclusion, model reaches opposite",
            "Logical contradiction within the response",
        ],
    },
    ErrorCategory.REFUSAL: {
        "name": "Refusal / non-answer",
        "description": "Model refuses to answer or gives non-committal response",
        "examples": [
            "'I cannot determine from the evidence'",
            "Excessive hedging without taking a position",
        ],
    },
    ErrorCategory.OTHER: {
        "name": "Other",
        "description": "Error that doesn't fit other categories",
        "examples": [
            "Formatting issues",
            "Unrelated tangent",
        ],
    },
}


@dataclass
class AnnotationRecord:
    """A single annotation record."""

    example_id: str
    cell_name: str  # e.g., "lora_b_strong_seed42"
    question: str
    passages: list[dict[str, Any]]
    gold_answer: str
    model_answer: str
    model_reasoning: str | None

    # Annotation fields
    is_correct: bool | None = None
    error_categories: list[ErrorCategory] | None = None
    notes: str | None = None
    annotator: str | None = None
    timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        raise NotImplementedError("TODO: Implement in M8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnnotationRecord":
        """Create from dictionary."""
        raise NotImplementedError("TODO: Implement in M8")
