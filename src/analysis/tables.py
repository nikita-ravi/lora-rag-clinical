"""LaTeX table generation."""

from typing import Any


def generate_main_table(results: dict[str, dict[str, Any]]) -> str:
    """Generate main results table (12-cell accuracy).

    Returns LaTeX string for the table.
    """
    raise NotImplementedError("TODO: Implement in M9")


def generate_error_table(annotations: list[dict]) -> str:
    """Generate error category breakdown table.

    Returns LaTeX string for the table.
    """
    raise NotImplementedError("TODO: Implement in M9")


def generate_retrieval_table(retrieval_metrics: dict[str, Any]) -> str:
    """Generate retrieval quality table.

    Returns LaTeX string for the table.
    """
    raise NotImplementedError("TODO: Implement in M9")


def generate_statistical_tests_table(test_results: dict[str, Any]) -> str:
    """Generate table of statistical test results.

    Returns LaTeX string for the table.
    """
    raise NotImplementedError("TODO: Implement in M9")


def _format_mean_std(mean: float, std: float, precision: int = 1) -> str:
    """Format mean ± std for LaTeX."""
    return f"${mean:.{precision}f} \\pm {std:.{precision}f}$"


def _format_pvalue(p: float) -> str:
    """Format p-value for LaTeX."""
    if p < 0.001:
        return "$p < 0.001$"
    elif p < 0.01:
        return f"$p = {p:.3f}$"
    else:
        return f"$p = {p:.2f}$"
