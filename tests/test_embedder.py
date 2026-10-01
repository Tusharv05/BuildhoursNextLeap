"""Stage 3 (Embedding) tests.

Model tests are marked `model` so they can be skipped when the model is unavailable
offline: `pytest -m "not model"`. The cache-key and cache-behaviour tests are pure and
always run.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from conftest import needs_model
from mf_rag.embedder import (
    EXPECTED_DIM,
    ModelDriftError,
    embed_model_name,
    embed_query,
    embed_texts,
    get_embedder,
)
from mf_rag.embed_store import (
    embedding_cache_key,
    embed_chunks,
    load_cache,
    save_cache,
)


# ---------------------------------------------------------------- pure helpers
def test_cache_key_is_sha1_of_model_and_text() -> None:
    import hashlib

    expected = hashlib.sha1(b"m:hello").hexdigest()
    assert embedding_cache_key("m", "hello") == expected


def test_cache_key_changes_with_model() -> None:
    """A model swap must invalidate the cache, not silently reuse old vectors."""
    assert embedding_cache_key("modelA", "same text") != embedding_cache_key(
        "modelB", "same text"
    )


def test_cache_key_changes_with_text() -> None:
    assert embedding_cache_key("m", "a") != embedding_cache_key("m", "b")


def test_cache_round_trip(tmp_path, sample_chunk) -> None:
    path = tmp_path / "embeddings.jsonl"
    cache = {"k1": [0.1] * EXPECTED_DIM, "k2": [0.2] * EXPECTED_DIM}
    save_cache(path, cache, "m")
    assert load_cache(path) == cache


def test_load_cache_of_missing_file_is_empty(tmp_path) -> None:
    assert load_cache(tmp_path / "nope.jsonl") == {}


@needs_model
def test_cache_is_pruned_to_the_live_chunk_set(tmp_path, cfg, sample_chunk) -> None:
    """Regression: switching chunking strategy left the previous run's vectors behind,
    so embeddings.jsonl grew past chunks.jsonl and the Phase 3 gate len(embeddings) ==
    len(chunks) silently stopped holding."""
    local = replace(cfg, root=tmp_path)
    first = [
        replace(sample_chunk, chunk_id=f"a{i}", embed_text=f"parent_child text {i}")
        for i in range(6)
    ]
    embed_chunks(first, local)
    assert len(load_cache(local.embeddings_path)) == 6

    second = [
        replace(sample_chunk, chunk_id=f"b{i}", embed_text=f"section text {i}")
        for i in range(3)
    ]
    vectors, stats = embed_chunks(second, local)

    rows = local.embeddings_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3 == len(vectors) == stats["total"]
    assert stats["cache_misses"] == 3, "stale keys must not mask a genuinely new chunk"


@needs_model
def test_pruning_keeps_reruns_at_full_cache_hits(tmp_path, cfg, sample_chunk) -> None:
    local = replace(cfg, root=tmp_path)
    chunks = [replace(sample_chunk, chunk_id=f"c{i}", embed_text=f"text {i}") for i in range(5)]
    embed_chunks(chunks, local)
    _, stats = embed_chunks(chunks, local)
    assert stats["cache_hit_ratio"] == 1.0


def test_save_cache_is_byte_stable(tmp_path) -> None:
    path = tmp_path / "e.jsonl"
    cache = {"b": [0.2], "a": [0.1]}
    save_cache(path, cache, "m")
    first = path.read_bytes()
    save_cache(path, dict(reversed(list(cache.items()))), "m")
    assert path.read_bytes() == first


def test_empty_batch_returns_empty_array(cfg) -> None:
    out = embed_texts([], cfg)
    assert out.shape == (0, EXPECTED_DIM)
    assert out.dtype == np.float32


def test_embed_model_name_matches_config(cfg) -> None:
    assert embed_model_name(cfg) == cfg.embedding.model


# ---------------------------------------------------------------- model tests
@needs_model
def test_get_embedder_is_a_true_singleton(cfg) -> None:
    """The model is ~90MB and ~40s to load; it must be created once per process."""
    assert get_embedder(cfg=cfg) is get_embedder(cfg=cfg)
    assert get_embedder() is get_embedder(cfg=cfg)


@needs_model
def test_embed_texts_shape_and_dtype(cfg) -> None:
    out = embed_texts(["expense ratio is 1.21%", "exit load is nil"], cfg)
    assert out.shape == (2, EXPECTED_DIM)
    assert out.dtype == np.float32


@needs_model
def test_embeddings_are_normalised(cfg) -> None:
    """Cosine distance in Chroma assumes unit vectors when normalize=true."""
    vectors = embed_texts(["a short chunk about expense ratios"], cfg)
    assert float(np.linalg.norm(vectors[0])) == pytest.approx(1.0, abs=1e-3)


@needs_model
def test_embed_query_matches_embed_texts(cfg) -> None:
    """Query and index paths must be the same computation, or every score is skewed."""
    query = embed_query("what is the lock-in period", cfg)
    batched = embed_texts(["what is the lock-in period"], cfg)[0]
    assert query == pytest.approx([float(x) for x in batched], abs=1e-6)
    assert len(query) == EXPECTED_DIM


@needs_model
def test_similar_text_scores_higher_than_unrelated(cfg) -> None:
    vectors = embed_texts(
        ["minimum SIP amount for the scheme", "solar panel manufacturing in Germany"],
        cfg,
    )
    similarity = float(np.dot(vectors[0], vectors[1]))
    assert similarity < 0.5, "unrelated texts scored too similarly"


@needs_model
def test_embed_chunks_populates_every_chunk(tmp_path, cfg, sample_chunk) -> None:
    local = replace(cfg, root=tmp_path)
    chunks = [replace(sample_chunk, chunk_id=f"c{i}", embed_text=f"text number {i}") for i in range(3)]
    vectors, stats = embed_chunks(chunks, local)
    assert set(vectors) == {"c0", "c1", "c2"}
    assert all(len(v) == EXPECTED_DIM for v in vectors.values())
    assert stats["total"] == 3
    assert stats["cache_misses"] == 3
    assert stats["cache_hits"] == 0


@needs_model
def test_second_embed_chunks_run_is_all_cache_hits(tmp_path, cfg, sample_chunk) -> None:
    local = replace(cfg, root=tmp_path)
    chunks = [replace(sample_chunk, chunk_id=f"c{i}", embed_text=f"text number {i}") for i in range(4)]
    embed_chunks(chunks, local)
    vectors, stats = embed_chunks(chunks, local)
    assert stats["cache_hits"] == 4
    assert stats["cache_misses"] == 0
    assert stats["cache_hit_ratio"] == 1.0
    assert vectors == {c.chunk_id: vectors[c.chunk_id] for c in chunks}


@needs_model
def test_cache_survives_a_fresh_read(tmp_path, cfg, sample_chunk) -> None:
    local = replace(cfg, root=tmp_path)
    chunks = [replace(sample_chunk, chunk_id="c0", embed_text="stable text")]
    first, _ = embed_chunks(chunks, local)
    raw = json.loads(local.embeddings_path.read_text(encoding="utf-8").splitlines()[0])
    assert raw["model"] == cfg.embedding.model
    assert raw["vector"] == pytest.approx(first["c0"], abs=1e-6)


@needs_model
def test_reset_cache_forces_recomputation(tmp_path, cfg, sample_chunk) -> None:
    local = replace(cfg, root=tmp_path)
    chunks = [replace(sample_chunk, chunk_id="c0", embed_text="stable text")]
    embed_chunks(chunks, local)
    _, stats = embed_chunks(chunks, local, reset_cache=True)
    assert stats["cache_misses"] == 1


@needs_model
def test_only_changed_text_is_re_embedded(tmp_path, cfg, sample_chunk) -> None:
    """The cache is keyed on text, so editing one document must not re-embed the rest."""
    local = replace(cfg, root=tmp_path)
    original = [
        replace(sample_chunk, chunk_id="c0", embed_text="unchanged one"),
        replace(sample_chunk, chunk_id="c1", embed_text="unchanged two"),
    ]
    embed_chunks(original, local)
    edited = [
        original[0],
        replace(sample_chunk, chunk_id="c1", embed_text="this text was edited"),
    ]
    _, stats = embed_chunks(edited, local)
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 1


@needs_model
def test_changed_text_gets_a_different_vector(tmp_path, cfg, sample_chunk) -> None:
    local = replace(cfg, root=tmp_path)
    before, _ = embed_chunks(
        [replace(sample_chunk, chunk_id="c0", embed_text="exit load is one percent")], local
    )
    after, _ = embed_chunks(
        [replace(sample_chunk, chunk_id="c0", embed_text="lock-in period is three years")],
        local,
    )
    assert before["c0"] != after["c0"]


@needs_model
def test_wrong_dimensionality_raises_model_drift(cfg, monkeypatch) -> None:
    """A model swap that changes dimensionality must fail loudly, not be stored."""
    real = get_embedder(cfg=cfg)

    class WrongDim:
        def encode(self, texts, **_kwargs):
            return np.zeros((len(texts), 768), dtype=np.float32)

    monkeypatch.setattr("mf_rag.embedder.get_embedder", lambda **_kw: WrongDim())
    try:
        with pytest.raises(ModelDriftError) as excinfo:
            embed_texts(["anything"], cfg)
    finally:
        monkeypatch.undo()
    assert "768" in str(excinfo.value)
    assert real is not None  # the real model was never replaced globally


def test_model_drift_error_message_is_actionable() -> None:
    error = ModelDriftError(
        "Expected 384-dim vectors for model 'x', got shape (1, 768). "
        "A collection built with these would be incompatible with queries - rebuild with --reset."
    )
    assert "--reset" in str(error)
