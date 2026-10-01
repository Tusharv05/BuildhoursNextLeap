"""Data contracts: construction, round-trip, and Chroma-safe metadata."""

from __future__ import annotations

import pytest

from mf_rag.models import (
    DOC_TYPES,
    Answer,
    Chunk,
    RetrievalHit,
    SourceDoc,
    Trace,
)


def test_source_doc_round_trip(sample_source_doc: SourceDoc) -> None:
    row = sample_source_doc.to_row()
    assert SourceDoc(**row) == sample_source_doc
    assert row["category"] == "ELSS"
    assert row["fetched_at"] == "2026-09-27"
    assert row["is_official"] is False


def test_chunk_metadata_is_chroma_safe(sample_chunk: Chunk) -> None:
    meta = sample_chunk.to_metadata()
    assert set(meta) == {
        "source_id",
        "source_url",
        "scheme",
        "category",
        "doc_type",
        "section_heading",
        "chunk_index",
        "token_count",
        "fetched_at",
    }
    for key, value in meta.items():
        assert isinstance(value, (str, int, bool)), f"{key} is {type(value).__name__}, not Chroma-safe"
        assert value is not None, f"{key} must never be None in Chroma metadata"


def test_retrieval_hit_score(sample_chunk: Chunk) -> None:
    hit = RetrievalHit(chunk=sample_chunk, score=0.82)
    assert hit.chunk.chunk_id == "abc123"
    assert 0.0 <= hit.score <= 1.0


def test_trace_and_answer_defaults() -> None:
    trace = Trace(
        query_hash="deadbeef",
        intent="factual",
        pii_blocked=False,
        entity_detected="ELSS",
        where_filter={"category": "ELSS"},
        hits=[],
        chunks_used=[],
        selected_source_url=None,
        validators_run=[],
        repair_used=False,
        latency_ms=12,
        answer="",
    )
    answer = Answer(
        text="Lock-in is 3 years. Source: https://groww.in/x",
        source_url="https://groww.in/x",
        fetched_at="2026-09-27",
        intent="factual",
        trace=trace,
    )
    assert answer.is_refusal is False
    assert answer.trace.pii_blocked is False


def test_doc_types_cover_registry_vocabulary() -> None:
    for name in ("scheme_page", "factsheet", "faq", "fees", "guide", "educational"):
        assert name in DOC_TYPES


@pytest.mark.parametrize("field", ["source_id", "source_url", "fetched_at", "is_official"])
def test_source_doc_requires_key_fields(field: str) -> None:
    base = {
        "source_id": "x",
        "source_url": "https://groww.in/x",
        "scheme": None,
        "category": None,
        "doc_type": "faq",
        "title": "t",
        "text": "t",
        "fetched_at": "2026-09-27",
        "is_official": True,
    }
    base.pop(field)
    with pytest.raises(TypeError):
        SourceDoc(**base)
