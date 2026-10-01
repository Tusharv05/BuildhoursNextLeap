"""Candidate A: fixed-size chunks with overlap. The naive baseline."""

from __future__ import annotations

from mf_rag.chunking.base import TABLE_ROW, Chunker, make_chunk
from mf_rag.config import ChunkingConfig
from mf_rag.models import Chunk, SourceDoc


class FixedChunker:
    """Sliding window over raw text.

    Simple and predictable, but it has no notion of structure: it happily cuts a
    table row in half, orphaning the number from its label.
    """

    name = "fixed"

    def __init__(self, cfg: ChunkingConfig) -> None:
        self.cfg = cfg
        self.orphan_rows = 0

    def split(self, doc: SourceDoc) -> list[Chunk]:
        text = doc.text
        step = max(1, self.cfg.target_chars - self.cfg.overlap_chars)
        chunks: list[Chunk] = []
        index = 0
        start = 0
        while start < len(text):
            piece = text[start : start + self.cfg.target_chars]
            if not piece.strip():
                start += step
                continue
            # A cut that lands inside a pipe-table row means a fact was separated
            # from its label. Counted for the strategy comparison.
            if start > 0:
                boundary_line = text[start - 1 : start]
                next_line = text[start : start + 1]
                if (TABLE_ROW.match(boundary_line) or TABLE_ROW.match(next_line)) and not (
                    boundary_line.strip() == next_line.strip()
                ):
                    self.orphan_rows += 1
            chunks.append(make_chunk(doc, piece, "", index, self.cfg))
            index += 1
            if start + self.cfg.target_chars >= len(text):
                break
            start += step
        return chunks
