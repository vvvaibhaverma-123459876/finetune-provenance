"""Cosine similarity attribution between query embedding and training embeddings."""

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows of a matrix."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def cosine_attribution(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    top_k: int = 5,
) -> List[Tuple[int, float]]:
    """Return top-K corpus indices and their cosine similarities to the query.

    Args:
        query_embedding: 1-D array of shape (hidden_size,).
        corpus_embeddings: 2-D array of shape (N, hidden_size).
        top_k: Number of top attributions to return.

    Returns:
        List of (corpus_index, cosine_similarity) sorted descending by similarity.
    """
    if corpus_embeddings.ndim == 1:
        corpus_embeddings = corpus_embeddings[np.newaxis, :]

    query = _normalize(query_embedding.reshape(1, -1))  # (1, H)
    corpus = _normalize(corpus_embeddings)               # (N, H)
    sims = (corpus @ query.T).squeeze(-1)                # (N,)

    k = min(top_k, len(sims))
    top_indices = np.argpartition(sims, -k)[-k:]
    top_indices = top_indices[np.argsort(-sims[top_indices])]

    return [(int(idx), float(sims[idx])) for idx in top_indices]
