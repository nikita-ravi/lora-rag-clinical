"""Streamlit annotation UI.

Simple local web UI for error annotation.
Run with: streamlit run src/annotation/app.py
"""

# Streamlit app will be implemented in M8
# This is a placeholder showing the intended structure


def main():
    """Main Streamlit app."""
    raise NotImplementedError("TODO: Implement in M8")


def load_examples_for_annotation(results_dir: str) -> list:
    """Load examples that need annotation."""
    raise NotImplementedError("TODO: Implement in M8")


def save_annotation(record: dict, output_path: str) -> None:
    """Save a single annotation."""
    raise NotImplementedError("TODO: Implement in M8")


def get_annotation_progress(output_path: str) -> dict:
    """Get annotation progress statistics."""
    raise NotImplementedError("TODO: Implement in M8")


def export_annotations_to_csv(annotations_path: str, output_path: str) -> None:
    """Export annotations to CSV for analysis."""
    raise NotImplementedError("TODO: Implement in M8")


if __name__ == "__main__":
    main()
