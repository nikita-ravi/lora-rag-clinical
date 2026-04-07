"""FAISS index builder for dense retrieval.

Uses BAAI/bge-base-en-v1.5 embeddings over the unified corpus.
"""

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# Default embedding model
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


def build_index(
    corpus: list[dict[str, Any]],
    output_path: Path,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
) -> tuple[Any, dict[int, str]]:
    """Build FAISS index over corpus.

    Args:
        corpus: List of passages from build_corpus()
        output_path: Path to save the index (without extension)
        model_name: Embedding model name
        batch_size: Batch size for embedding

    Creates:
        - {output_path}.faiss: FAISS index file
        - {output_path}_ids.json: Mapping from index position to passage ID

    Returns:
        Tuple of (faiss_index, id_mapping)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract passage texts and IDs
    passage_texts = [p["text"] for p in corpus]
    passage_ids = [p["id"] for p in corpus]

    print(f"Building index over {len(corpus)} passages...")

    # Embed passages
    embeddings = _embed_passages(passage_texts, model_name, batch_size)

    # Build FAISS index (Flat for exact search on small corpus)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors

    # Normalize embeddings for cosine similarity via inner product
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Create ID mapping (index position -> passage ID)
    id_mapping = {i: pid for i, pid in enumerate(passage_ids)}

    # Save index
    faiss.write_index(index, str(output_path) + ".faiss")

    # Save ID mapping
    with open(str(output_path) + "_ids.json", "w") as f:
        json.dump(id_mapping, f)

    print(f"Index saved to {output_path}.faiss")
    print(f"ID mapping saved to {output_path}_ids.json")
    print(f"Index size: {index.ntotal} vectors, dimension: {dimension}")

    return index, id_mapping


def load_index(index_path: Path) -> tuple[Any, dict[int, str]]:
    """Load FAISS index and ID mapping.

    Args:
        index_path: Path to index (without .faiss extension)

    Returns:
        Tuple of (faiss_index, id_mapping)
    """
    index_path = Path(index_path)

    # Load FAISS index
    index = faiss.read_index(str(index_path) + ".faiss")

    # Load ID mapping
    with open(str(index_path) + "_ids.json", "r") as f:
        id_mapping_raw = json.load(f)

    # Convert string keys back to int
    id_mapping = {int(k): v for k, v in id_mapping_raw.items()}

    return index, id_mapping


def _embed_passages(
    passages: list[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
) -> np.ndarray:
    """Embed passages using BGE model.

    Args:
        passages: List of passage texts
        model_name: Embedding model name
        batch_size: Batch size for encoding

    Returns:
        numpy array of shape (n_passages, embedding_dim)
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Embedding {len(passages)} passages...")
    embeddings = model.encode(
        passages,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # Normalize for cosine similarity
    )

    return embeddings


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """Get embedding model (cached for reuse)."""
    return SentenceTransformer(model_name)
