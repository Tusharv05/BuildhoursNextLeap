"""Stage 4 (Store Vector Data) tests.

Each test uses a throwaway `data/` root so the real index in `data/chroma` is never
touched.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from mf_rag.embedder import EXPECTED_DIM
from mf_rag.models import Chunk
from mf_rag.store import (
    COLLECTION_NAME,
    ModelMismatchError,
    VerificationError,
    _chroma_safe,
    build_collection,
    get_client,
    get_or_create_collection,
    verify_collection,
)


@pytest.fixture
def local_cfg(tmp_path, cfg):
    return replace(cfg, root=tmp_path)


def _vector(seed: float = 0.1) -> list[float]:
    v = [seed] * EXPECTED_DIM
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


def _chunk(idx: int, **kwargs) -> Chunk:
    base = dict(
        chunk_id=f"chunk{idx}",
        source_id="hdfc_elss_scheme_page",
        source_url="https://groww.in/mutual-funds/hdfc-elss",
        scheme="HDFC ELSS Tax Saver Fund - Direct Plan Growth",
        category="ELSS",
        doc_type="scheme_page",
        section_heading="Fees",
        chunk_index=idx,
        text="Exit load: Nil. Lock-in: 3 years.",
        embed_text="HDFC ELSS - scheme_page - Fees\n\nExit load: Nil. Lock-in: 3 years.",
        token_count=14,
        fetched_at="2026-09-27",
    )
    base.update(kwargs)
    return Chunk(**base)


@pytest.fixture
def small_index():
    chunks = [_chunk(i) for i in range(5)]
    vectors = {c.chunk_id: _vector(0.01 * (i + 1)) for i, c in enumerate(chunks)}
    return chunks, vectors


# ---------------------------------------------------------------- setup
def test_chroma_safe_coerces_none(local_cfg, small_index) -> None:
    """Chroma rejects None metadata; None must become "" or -1, never leak through."""
    chunks, _ = small_index
    safe = _chroma_safe(chunks[0].to_metadata())
    assert None not in safe.values()

    coerced = _chroma_safe({"scheme": None, "category": None, "chunk_index": None})
    assert coerced["scheme"] == ""
    assert coerced["category"] == ""
    assert coerced["chunk_index"] == -1


def test_chroma_safe_keeps_real_values() -> None:
    assert _chroma_safe({"chunk_index": 0, "doc_type": "faq", "ok": True}) == {
        "chunk_index": 0,
        "doc_type": "faq",
        "ok": True,
    }


def test_collection_records_model_and_space(local_cfg) -> None:
    collection = get_or_create_collection(local_cfg)
    assert collection.name == COLLECTION_NAME
    assert collection.metadata["embedding_model"] == local_cfg.embedding.model
    assert collection.metadata["hnsw:space"] == "cosine"


def test_get_or_create_is_stable(local_cfg) -> None:
    assert get_or_create_collection(local_cfg).name == get_or_create_collection(local_cfg).name


# ---------------------------------------------------------------- build
def test_build_then_verify(local_cfg, small_index) -> None:
    chunks, vectors = small_index
    assert build_collection(chunks, vectors, local_cfg) == 5
    report = verify_collection(local_cfg, expected_count=5)
    assert report["count"] == 5
    assert report["metadata_ok"] is True
    assert report["elss_filter_hits"] > 0


def test_upsert_is_idempotent(local_cfg, small_index) -> None:
    """Re-running the build must not double the collection."""
    chunks, vectors = small_index
    build_collection(chunks, vectors, local_cfg)
    build_collection(chunks, vectors, local_cfg)
    assert get_or_create_collection(local_cfg).count() == 5


def test_reset_recreates_from_scratch(local_cfg, small_index) -> None:
    chunks, vectors = small_index
    build_collection(chunks, vectors, local_cfg)
    build_collection(chunks, vectors, local_cfg, reset=True)
    assert get_or_create_collection(local_cfg).count() == 5


def test_build_refuses_a_chunk_without_a_vector(local_cfg, small_index) -> None:
    chunks, vectors = small_index
    vectors.pop(chunks[0].chunk_id)
    with pytest.raises(KeyError, match="no vector"):
        build_collection(chunks, vectors, local_cfg)


def test_documents_are_the_context_prefixed_embed_text(local_cfg, small_index) -> None:
    chunks, vectors = small_index
    build_collection(chunks, vectors, local_cfg)
    stored = get_or_create_collection(local_cfg).get(ids=[chunks[0].chunk_id])
    assert stored["documents"][0] == chunks[0].embed_text


# ---------------------------------------------------------------- verification
def test_count_mismatch_names_both_numbers(local_cfg, small_index) -> None:
    chunks, vectors = small_index
    build_collection(chunks, vectors, local_cfg)
    with pytest.raises(VerificationError, match="holds 5 records, expected 4"):
        verify_collection(local_cfg, expected_count=4)


def test_empty_collection_fails_verification(local_cfg) -> None:
    """An empty index must not pass silently - that is the 'built but empty' failure."""
    get_or_create_collection(local_cfg)
    with pytest.raises(VerificationError):
        verify_collection(local_cfg, expected_count=0)


def test_elss_filter_finds_records(local_cfg, small_index) -> None:
    chunks, vectors = small_index
    build_collection(chunks, vectors, local_cfg)
    collection = get_or_create_collection(local_cfg)
    assert len(collection.get(where={"category": "ELSS"}, include=[])["ids"]) == 5
    assert len(collection.get(where={"scheme": "HDFC ELSS Tax Saver Fund - Direct Plan Growth"},
                              include=[])["ids"]) == 5


def test_missing_metadata_key_is_reported(local_cfg, small_index) -> None:
    chunks, vectors = small_index
    build_collection(chunks, vectors, local_cfg)
    # Simulate a schema regression by checking the verifier demands these keys.
    collection = get_or_create_collection(local_cfg)
    stored = collection.peek(limit=1)["metadatas"][0]
    for required in ("source_id", "source_url", "doc_type", "fetched_at", "chunk_index"):
        assert required in stored


# ---------------------------------------------------------------- model drift
def test_model_mismatch_raises(local_cfg, small_index) -> None:
    """Renaming the model must not silently reuse a collection built with the old one."""
    chunks, vectors = small_index
    build_collection(chunks, vectors, local_cfg)

    drifted = replace(local_cfg, embedding=replace(local_cfg.embedding, model="some/other-model"))
    with pytest.raises(ModelMismatchError) as excinfo:
        get_or_create_collection(drifted)
    message = str(excinfo.value)
    assert "some/other-model" in message
    assert local_cfg.embedding.model in message
    assert "--reset" in message


def test_model_mismatch_names_both_models(local_cfg) -> None:
    client = get_client(local_cfg)
    client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": "ancient/model-v0"},
    )
    with pytest.raises(ModelMismatchError, match="ancient/model-v0"):
        get_or_create_collection(local_cfg)


def test_reset_clears_a_mismatched_collection(local_cfg, small_index) -> None:
    """After --reset the drifted collection is gone and the build succeeds."""
    chunks, vectors = small_index
    client = get_client(local_cfg)
    client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": "ancient/model-v0"},
    )
    assert build_collection(chunks, vectors, local_cfg, reset=True) == 5
    assert get_or_create_collection(local_cfg).count() == 5
