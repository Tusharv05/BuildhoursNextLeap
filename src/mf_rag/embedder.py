"""RAG Stage Group A / Stage 3: Embedding.

The one and only place `SentenceTransformer` is instantiated. Both the index-time path
(Stage 3) and the query-time path (Stage 5) go through here, so the two can never drift
onto different models or different normalisation (ADR-2).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from mf_rag.config import AppConfig, get_config

# all-MiniLM-L6-v2 is 384-dim. Asserted rather than assumed: a silently different model
# would write 768-dim vectors and only fail much later, at query time.
EXPECTED_DIM = 384


class ModelDriftError(RuntimeError):
    """Raised when the loaded model's dimensionality is not what the index expects."""


@lru_cache(maxsize=None)
def _load_model(model_name: str):
    """Load and memoise a model by name.

    Caching on the name (not the whole config) is deliberate: the model object is fully
    determined by its name, so this is a true singleton per model, while `batch_size` and
    `normalize` stay per-call settings rather than baking into the cache key.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def get_embedder(model_name: str | None = None, cfg: AppConfig | None = None):
    """Return the process-wide `SentenceTransformer`.

    Loading costs ~40s and ~90MB, so the model is loaded once and reused. Repeated calls
    with the same model name return the identical object.
    """
    name = model_name or (cfg or get_config()).embedding.model
    return _load_model(name)


def embed_model_name(cfg: AppConfig | None = None) -> str:
    """The exact model string currently configured.

    Stage 4 writes this into the collection metadata and refuses to reuse a collection
    built with a different one, so model drift fails loudly instead of silently
    producing nonsense rankings (ARCHITECTURE 5.3).
    """
    return (cfg or get_config()).embedding.model


def embed_texts(texts: list[str], cfg: AppConfig | None = None) -> np.ndarray:
    """Embed a batch of texts. Returns float32 of shape (len(texts), 384)."""
    config = cfg or get_config()
    items = list(texts)
    if not items:
        return np.zeros((0, EXPECTED_DIM), dtype=np.float32)
    vectors = get_embedder(cfg=config).encode(
        items,
        batch_size=config.embedding.batch_size,
        normalize_embeddings=config.embedding.normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != EXPECTED_DIM:
        raise ModelDriftError(
            f"Expected {EXPECTED_DIM}-dim vectors for model {config.embedding.model!r}, "
            f"got shape {vectors.shape}. A collection built with these would be "
            f"incompatible with queries - rebuild with --reset."
        )
    return vectors


def embed_query(text: str, cfg: AppConfig | None = None) -> list[float]:
    """Embed a single query with the same model and normalisation as the index.

    A different normalisation on one side of a cosine comparison silently skews every
    score, so this shares `embed_texts` rather than repeating the encode call.
    """
    return [float(x) for x in embed_texts([text], cfg)[0]]
