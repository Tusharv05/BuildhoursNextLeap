"""RAG Stage Group A / Stage 4: Store Vector Data.

A persistent ChromaDB collection. Two things matter beyond "put the vectors in":

1. **Model-drift guard.** If the collection was built with a different embedding model,
   raise. Silently reusing it is the single most common RAG bug: queries get embedded
   with the new model and compared against vectors from the old one, and everything
   ranks badly with no error anywhere (ARCHITECTURE 5.3).
2. **Hard verification.** `verify_collection` asserts count, metadata round-trip, and a
   working filter, each with its own error message, so a broken index fails at build
   time rather than mid-demo.
"""

from __future__ import annotations

from typing import Any, Sequence

from mf_rag.config import AppConfig, get_config
from mf_rag.embedder import embed_model_name
from mf_rag.models import Chunk

COLLECTION_NAME = "mf_faq"


class ModelMismatchError(RuntimeError):
    """The existing collection was built with a different embedding model."""


class VerificationError(RuntimeError):
    """A stored index failed one of the post-build assertions."""


def get_client(cfg: AppConfig | None = None):
    """A `PersistentClient` rooted at `data/chroma`."""
    import chromadb
    from chromadb.config import Settings

    config = cfg or get_config()
    config.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(config.chroma_dir),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def get_or_create_collection(cfg: AppConfig | None = None):
    """Return the `mf_faq` collection, or raise if it was built with another model."""
    config = cfg or get_config()
    model = embed_model_name(config)
    client = get_client(config)
    existing = _peek_metadata(client, COLLECTION_NAME)
    stored = (existing or {}).get("embedding_model")
    if stored and stored != model:
        raise ModelMismatchError(
            f"Collection {COLLECTION_NAME!r} was built with embedding model {stored!r} "
            f"but the config now says {model!r}. Vectors from the two are not comparable. "
            f"Re-run with --reset to rebuild, or restore the previous model in config.yaml."
        )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": model},
    )


def _peek_metadata(client, name: str) -> dict[str, Any] | None:
    """Read a collection's metadata without creating it. None if it does not exist."""
    try:
        collection = client.get_collection(name=name)
    except Exception:
        return None
    try:
        return dict(collection.metadata or {})
    except Exception:
        return None


def _chroma_safe(metadata: dict[str, Any]) -> dict[str, Any]:
    """Chroma rejects `None` metadata values, so coerce them.

    Strings become `""` and `chunk_index` becomes `-1`, which is distinguishable from any
    real index (they start at 0) and sorts below everything.
    """
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            safe[key] = -1 if key == "chunk_index" else ""
        elif isinstance(value, (str, int, float, bool)):
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


def build_collection(
    chunks: Sequence[Chunk],
    vectors: dict[str, list[float]],
    cfg: AppConfig | None = None,
    reset: bool = False,
) -> int:
    """Upsert every chunk into Chroma. Returns the number of chunks written."""
    config = cfg or get_config()
    client = get_client(config)
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # nothing to delete on a first build

    collection = get_or_create_collection(config)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    embeddings: list[list[float]] = []
    for chunk in chunks:
        if chunk.chunk_id not in vectors:
            raise KeyError(f"Refusing to store {chunk.chunk_id}: no vector was computed")
        ids.append(chunk.chunk_id)
        documents.append(chunk.embed_text)
        metadatas.append(_chroma_safe(chunk.to_metadata()))
        embeddings.append(vectors[chunk.chunk_id])

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return len(ids)


def verify_collection(
    cfg: AppConfig | None = None, expected_count: int | None = None
) -> dict[str, Any]:
    """Assert the index is usable. Each failure names the specific broken check."""
    config = cfg or get_config()
    collection = get_or_create_collection(config)

    count = collection.count()
    if expected_count is not None and count != expected_count:
        raise VerificationError(
            f"Count mismatch: collection {COLLECTION_NAME!r} holds {count} records, "
            f"expected {expected_count} (one per chunk in chunks.jsonl)."
        )

    sample = collection.peek(limit=1)
    peek_ids = sample.get("ids") or []
    peek_meta = sample.get("metadatas") or []
    if not peek_ids:
        raise VerificationError("Metadata round-trip failed: peek(1) returned no records.")
    first = peek_meta[0] or {}
    for required in ("source_id", "source_url", "doc_type", "fetched_at", "chunk_index"):
        if required not in first:
            raise VerificationError(
                f"Metadata round-trip failed: {required!r} missing from stored record "
                f"{peek_ids[0]!r}; got keys {sorted(first)}."
            )
    if any(value is None for value in first.values()):
        raise VerificationError(
            f"Metadata round-trip failed: stored record {peek_ids[0]!r} contains None "
            f"values, which Chroma is supposed to reject."
        )

    elss = collection.get(where={"category": "ELSS"}, include=[])
    filtered = len(elss.get("ids") or [])
    if filtered <= 0:
        raise VerificationError(
            'Metadata filter failed: where={"category": "ELSS"} returned 0 records. '
            "Filters are what keep a Small Cap answer from quoting a Flexi Cap page."
        )

    return {
        "collection": COLLECTION_NAME,
        "count": count,
        "expected_count": expected_count,
        "metadata_ok": True,
        "elss_filter_hits": filtered,
        "embedding_model": embed_model_name(config),
        "persist_path": str(config.chroma_dir),
    }
