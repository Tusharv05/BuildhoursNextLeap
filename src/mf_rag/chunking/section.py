"""Candidate B (default): structure-aware chunking. Headings, atomic table rows,
and FAQ question/answer blocks that are never separated."""

from __future__ import annotations

import re

from mf_rag.chunking.base import (
    HEADING_LINE,
    TABLE_ROW,
    Chunker,
    blocks_with_headings,
    hard_split,
    is_question_line,
    make_chunk,
)
from mf_rag.config import ChunkingConfig
from mf_rag.models import Chunk, SourceDoc

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


class SectionChunker:
    """Split on document structure, then merge small blocks and split large ones.

    Guarantees relied on elsewhere (and by the tests):
      * a table row is never split;
      * a FAQ question is never separated from its answer;
      * every chunk carries a heading breadcrumb.
    """

    name = "section"

    def __init__(self, cfg: ChunkingConfig) -> None:
        self.cfg = cfg
        self.orphan_rows = 0
        self.split_faq_pairs = 0

    # -- structure helpers -------------------------------------------------
    def _logical_units(self, block: str) -> list[str]:
        """Break a block into units that must not be cut: table blocks, question/answer
        pairs, and individual lines.

        A question in either shape (`Q: ...` or a bare `...?` heading line) is glued to
        the first non-blank line that follows it, so the pair survives `_merge` and
        cannot be separated (FR-2.3).
        """
        lines = block.split("\n")
        units: list[str] = []
        buffer: list[str] = []
        in_table = False
        pending_question: str | None = None

        def flush_table() -> None:
            nonlocal buffer, in_table
            if buffer:
                units.append("\n".join(buffer))
                buffer = []
            in_table = False

        def flush_question() -> None:
            nonlocal pending_question
            if pending_question:
                units.append(pending_question)
                pending_question = None

        for line in lines:
            is_row = bool(TABLE_ROW.match(line))
            if is_row and not in_table:
                flush_table()
                flush_question()
                in_table = True
                buffer.append(line)
                continue
            if in_table:
                if is_row:
                    buffer.append(line)
                else:
                    flush_table()  # table ends at the first non-row line
                    units.append(line)
                continue

            if is_question_line(line):
                flush_question()
                pending_question = line
                continue
            if pending_question is not None:
                if not line.strip():
                    continue  # do not let a blank line dilute the pair
                if HEADING_LINE.match(line) or is_question_line(line):
                    flush_question()
                    units.append(line)
                    continue
                # The answer belongs to the question above it.
                pending_question = f"{pending_question}\n{line}"
                continue
            units.append(line)
        flush_table()
        flush_question()
        return [u for u in units if u.strip()]

    def _merge(self, units: list[str]) -> list[str]:
        """Merge adjacent small units up to target_chars without breaking a unit."""
        merged: list[str] = []
        current = ""
        for unit in units:
            candidate = f"{current}\n{unit}" if current else unit
            if current and len(candidate) > self.cfg.target_chars:
                merged.append(current)
                current = unit
            else:
                current = candidate
        if current:
            merged.append(current)

        out: list[str] = []
        for item in merged:
            if len(item) < self.cfg.min_chars and out:
                combined = f"{out[-1]}\n{item}"
                if len(combined) <= self.cfg.max_chars:
                    out[-1] = combined
                    continue
            out.append(item)
        return out

    # -- Chunker protocol --------------------------------------------------
    def split(self, doc: SourceDoc) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0
        for heading, block in blocks_with_headings(doc.text):
            units = self._merge(self._logical_units(block))
            for unit in units:
                for piece in hard_split(unit, self.cfg.max_chars):
                    if not piece.strip():
                        continue
                    chunks.append(make_chunk(doc, piece, heading, index, self.cfg))
                    index += 1
        return chunks
