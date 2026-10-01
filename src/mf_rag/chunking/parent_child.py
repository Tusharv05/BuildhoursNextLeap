"""Candidate C: parent-child. Small embed chunks that each record their parent
section, so retrieval is precise while the prompt still gets full context."""

from __future__ import annotations

from mf_rag.chunking.base import (
    Chunker,
    blocks_with_headings,
    hard_split,
    make_chunk,
)
from mf_rag.config import ChunkingConfig
from mf_rag.models import Chunk, SourceDoc

CHILD_CHARS = 300


class ParentChildChunker:
    """Embed small children (~300 chars) but tag each with its parent section.

    Phase 5 expands a hit to its parent before building the prompt, which recovers
    context without bloating the vector.

    The expansion is *bounded* by `max_chars`. Most pages in this corpus arrive as a
    single unstructured block (no markdown headings), so an unbounded "parent" is the
    whole 28,000-character document - unusable in a prompt, and it would let a
    retrieval hit match an answer that is nowhere near the child. A bounded window
    centred on the child keeps the pairing honest and the prompt within budget.
    """

    name = "parent_child"

    def __init__(self, cfg: ChunkingConfig) -> None:
        self.cfg = cfg
        self.orphan_rows = 0
        self.parents: dict[str, str] = {}
        self.locations: dict[str, tuple[str, int, int]] = {}

    def split(self, doc: SourceDoc) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0
        for block_no, (heading, block) in enumerate(blocks_with_headings(doc.text)):
            # Key per block, not per (source_id, heading): an unheaded page has many
            # blocks that all share the empty heading, and a shared key would make
            # every block overwrite the last one's parent.
            parent_key = f"{doc.source_id}#{block_no}::{heading}"
            self.parents[parent_key] = block
            cursor = 0
            for piece in hard_split(block, CHILD_CHARS):
                if not piece.strip():
                    continue
                # The heading field carries the parent breadcrumb; the body is the child.
                chunk = make_chunk(doc, piece, heading, index, self.cfg)
                offset = block.find(piece, cursor)
                if offset < 0:
                    offset = cursor
                cursor = offset + len(piece)
                self.locations[chunk.chunk_id] = (parent_key, offset, len(piece))
                chunks.append(chunk)
                index += 1
        return chunks

    def parent_for(self, chunk: Chunk) -> str:
        """The context text for `chunk`: its parent section, or a bounded window
        around the child when the parent is larger than `max_chars`."""
        location = self.locations.get(chunk.chunk_id)
        if location is None:
            block = self.parents.get(f"{chunk.source_id}::{chunk.section_heading}", "")
            return block or chunk.text
        parent_key, offset, length = location
        block = self.parents.get(parent_key, "")
        if not block:
            return chunk.text
        if len(block) <= self.cfg.max_chars:
            return block
        window = self.cfg.max_chars
        child_end = offset + length
        pad = (window - length) // 2
        start = max(0, offset - pad)
        end = min(len(block), child_end + pad)
        if end - start > window:
            # A child longer than the window cannot be shown in full; keep it whole
            # rather than clip it, and let the caller see an over-budget parent.
            end = child_end if start + window < child_end else start + window
            start = max(0, end - window)
        # Snap both ends to line boundaries, accepting a snap only when the child
        # still fits inside the result.
        if start > 0:
            newline = block.rfind("\n", 0, start)
            candidate = newline + 1 if newline != -1 else 0
            if candidate <= offset and end - candidate <= window:
                start = candidate
        if end < len(block):
            newline = block.rfind("\n", child_end, end)
            if newline != -1:
                end = newline
        return block[start:end]
