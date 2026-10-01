"""RAG Stage Group A / Stage 3 + 4: embedding cache and Chroma persistence.

Stage 3 half: compute the 384-dim vector for every chunk, once, and cache it on disk so
re-runs are free. Stage 4 half lives in `mf_rag.store`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from mf_rag.config import AppConfig, get_config
from mf_rag.embedder import embed_model_name, embed_texts
from mf_rag.models import Chunk


def embedding_cache_key(model: str, text: str) -> str:
    """`sha1(f"{model}:{text}")`.

    Keying on the model name as well as the text means a model change invalidates every
    entry automatically, instead of silently reusing vectors from the old model.
    """
    return hashlib.sha1(f"{model}:{text}".encode("utf-8")).hexdigest()


def load_cache(path: Path) -> dict[str, list[float]]:
    """Read `embeddings.jsonl` into `{cache_key: vector}`."""
    if not path.is_file():
        return {}
    cache: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cache[row["key"]] = row["vector"]
    return cache


def save_cache(path: Path, cache: dict[str, list[float]], model: str) -> None:
    """Write the cache atomically, sorted by key so the file is byte-stable across runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"key": key, "model": model, "vector": cache[key]}, sort_keys=True)
        for key in sorted(cache)
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    tmp.replace(path)


def embed_chunks(
    chunks: list[Chunk], cfg: AppConfig | None = None, reset_cache: bool = False
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Return `{chunk_id: vector}` for every chunk, plus run stats.

    Only cache misses are sent to the model. The cache is keyed on the *text*, so an
    unchanged corpus re-run costs zero model calls; adding or editing a document
    re-embeds only what actually changed.
    """
    config = cfg or get_config()
    model = embed_model_name(config)
    path = config.embeddings_path

    cache: dict[str, list[float]] = {} if reset_cache else load_cache(path)
    if reset_cache and path.is_file():
        path.unlink()

    # Preserve first-seen order: deterministic output regardless of dict iteration.
    keys = {chunk.chunk_id: embedding_cache_key(model, chunk.embed_text) for chunk in chunks}

    misses = [c for c in chunks if keys[c.chunk_id] not in cache]
    hits = len(chunks) - len(misses)

    if misses:
        vectors = embed_texts([c.embed_text for c in misses], config)
        for chunk, vector in zip(misses, vectors):
            cache[keys[chunk.chunk_id]] = [float(x) for x in vector]

    # Prune to the live chunk set. Without this the cache only ever grows: switching
    # chunking strategy leaves the previous run's vectors behind, so
    # len(embeddings.jsonl) drifts away from len(chunks.jsonl) and the file becomes
    # dead weight. Pruning keeps the on-disk cache exactly one-per-chunk.
    live = {keys[c.chunk_id] for c in chunks}
    cache = {key: cache[key] for key in live}

    if config.embedding.cache:
        save_cache(path, cache, model)

    resolved = {c.chunk_id: cache[keys[c.chunk_id]] for c in chunks}

    missing = [cid for cid, vec in resolved.items() if not vec]
    if missing:
        raise RuntimeError(
            f"{len(missing)} chunk(s) ended up without a vector, e.g. {missing[:3]}. "
            f"Refusing to hand a partial index to Stage 4."
        )

    stats = {
        "model": model,
        "dim": len(next(iter(resolved.values()))) if resolved else 0,
        "total": len(chunks),
        "cache_hits": hits,
        "cache_misses": len(misses),
        "cache_hit_ratio": round(hits / len(chunks), 4) if chunks else 0.0,
        "cache_entries": len(cache),
        "cache_path": str(path),
    }
    return resolved, stats


def vectors_for(
    chunk_ids: Iterable[str], vectors: dict[str, list[float]]
) -> dict[str, list[float]]:
    """Look up a subset, failing loudly if anything is absent."""
    wanted = list(chunk_ids)
    absent = [cid for cid in wanted if cid not in vectors]
    if absent:
        raise KeyError(f"No vector for chunk id(s): {absent[:5]}")
    return {cid: vectors[cid] for cid in wanted}
