"""Analysis and visualization modules."""

from src.analysis.tables import generate_main_table, generate_error_table
from src.analysis.figures import plot_accuracy_heatmap, plot_error_breakdown, plot_interaction
from src.analysis.stratify import stratify_by_question_type

__all__ = [
    "generate_main_table",
    "generate_error_table",
    "plot_accuracy_heatmap",
    "plot_error_breakdown",
    "plot_interaction",
    "stratify_by_question_type",
]
