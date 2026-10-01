"""Chunker registry. The strategy name is config-driven (D7)."""

from __future__ import annotations

from mf_rag.chunking.base import Chunker
from mf_rag.chunking.fixed import FixedChunker
from mf_rag.chunking.parent_child import ParentChildChunker
from mf_rag.chunking.section import SectionChunker
from mf_rag.config import ChunkingConfig

REGISTRY: dict[str, type] = {
    "fixed": FixedChunker,
    "section": SectionChunker,
    "parent_child": ParentChildChunker,
}

DEFAULT_STRATEGY = "section"


def get_chunker(name: str | None, cfg: ChunkingConfig) -> Chunker:
    key = name or cfg.strategy
    if key not in REGISTRY:
        raise ValueError(
            f"Unknown chunking strategy '{key}'. Available: {', '.join(sorted(REGISTRY))}"
        )
    return REGISTRY[key](cfg)


def available_strategies() -> tuple[str, ...]:
    return tuple(sorted(REGISTRY))


__all__ = [
    "Chunker",
    "DEFAULT_STRATEGY",
    "REGISTRY",
    "available_strategies",
    "get_chunker",
]
