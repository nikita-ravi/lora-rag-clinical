"""FAISS index builder for dense retrieval.

Uses BAAI/bge-base-en-v1.5 embeddings over the unified corpus.
"""

from pathlib import Path
from typing import Any


def build_index(corpus: list[dict[str, Any]], output_path: Path) -> None:
    """Build FAISS index over corpus.

    Args:
        corpus: List of passages from build_corpus()
        output_path: Path to save the index

    Creates:
        - {output_path}.faiss: FAISS index file
        - {output_path}_ids.json: Mapping from index position to passage ID
    """
    raise NotImplementedError("TODO: Implement in M3")


def load_index(index_path: Path) -> tuple[Any, dict[int, str]]:
    """Load FAISS index and ID mapping.

    Args:
        index_path: Path to index (without .faiss extension)

    Returns:
        Tuple of (faiss_index, id_mapping)
    """
    raise NotImplementedError("TODO: Implement in M3")


def _embed_passages(
    passages: list[str],
    model_name: str = "BAAI/bge-base-en-v1.5",
    batch_size: int = 32,
) -> Any:
    """Embed passages using BGE model.

    Returns:
        numpy array of shape (n_passages, embedding_dim)
    """
    raise NotImplementedError("TODO: Implement in M3")
