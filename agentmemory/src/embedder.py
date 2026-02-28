"""
Embedding module.

Converts text → dense float vectors using sentence-transformers.
A single process-level singleton avoids reloading the model on every call.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from .config import cfg


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load (and cache) the embedding model once per process."""
    return SentenceTransformer(cfg.embed_model)


def embed(text: str) -> List[float]:
    """Return a normalised embedding vector for *text*."""
    model = _load_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Return embeddings for a list of texts in one batched call."""
    model = _load_model()
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=32)
    return [v.tolist() for v in vectors]
