"""Core data contracts shared by every RAG stage.

RAG stage: shared. These dataclasses are the hand-off format between
Loading -> Chunking -> Embedding -> Store -> Retrieve -> Generate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

DocType = Literal[
    "scheme_page",
    "factsheet",
    "kim_sid",
    "faq",
    "fees",
    "riskometer",
    "guide",
    "educational",
]

DOC_TYPES: tuple[str, ...] = (
    "scheme_page",
    "factsheet",
    "kim_sid",
    "faq",
    "fees",
    "riskometer",
    "guide",
    "educational",
)


@dataclass
class SourceDoc:
    """One ingested official public page, after Stage 1 (Loading)."""

    source_id: str
    source_url: str
    scheme: str | None
    category: str | None
    doc_type: str
    title: str
    text: str
    fetched_at: str  # ISO-8601 date (YYYY-MM-DD)
    is_official: bool

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    """One indexable unit produced by Stage 2 (Chunking)."""

    chunk_id: str
    source_id: str
    source_url: str
    scheme: str | None
    category: str | None
    doc_type: str
    section_heading: str
    chunk_index: int
    text: str
    embed_text: str
    token_count: int
    fetched_at: str

    def to_metadata(self) -> dict[str, Any]:
        """Flat, Chroma-safe metadata (str/int/bool only)."""
        return {
            "source_id": self.source_id,
            "source_url": self.source_url,
            "scheme": self.scheme,
            "category": self.category,
            "doc_type": self.doc_type,
            "section_heading": self.section_heading,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "fetched_at": self.fetched_at,
        }


@dataclass
class RetrievalHit:
    chunk: Chunk
    score: float  # cosine similarity, 0..1


@dataclass
class Trace:
    query_hash: str
    intent: str
    pii_blocked: bool
    entity_detected: str | None
    where_filter: dict[str, Any] | None
    hits: list[dict[str, Any]]
    chunks_used: list[str]
    selected_source_url: str | None
    validators_run: list[str]
    repair_used: bool
    latency_ms: int
    answer: str


@dataclass
class Answer:
    text: str
    source_url: str | None
    fetched_at: str
    intent: str
    trace: Trace
    is_refusal: bool = False
