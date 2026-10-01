"""RAG Stage Group A / Stage 2: Chunker interface and shared assembly rules.

The strategy is chosen from measurements (scripts/run_chunk_eval.py), not
assumption, so every strategy lives behind this one interface.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, Protocol, runtime_checkable

from mf_rag.config import ChunkingConfig
from mf_rag.models import Chunk, SourceDoc

# A markdown/pipe table row is atomic: splitting one orphans a number from its label.
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
# The |---|---| row that marks the second line of a markdown table as its header.
TABLE_SEPARATOR = re.compile(r"^\s*\|(?:\s*:?-{2,}:?\s*\|)+\s*$")
# FAQ question lines start a block that must not be separated from its answer.
QUESTION_LINE = re.compile(r"^\s*(?:q[:.)]|\d+[.)])\s+", re.IGNORECASE)
HEADING_LINE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
URL_LINE = re.compile(r"^\s*(?:https?://|www\.)\S+", re.IGNORECASE)
# A short standalone line ending in '?' - trafilatura renders education/FAQ pages
# with bare question headings rather than markdown ones, and a question must never
# be left behind at the end of a piece while its answer starts the next one.
BARE_QUESTION = re.compile(r"^\s{0,3}(?!#|\||\d+[.)])\S[^\n]{6,80}\?\s*$")

ENCODER: object | None = None


def _encoder():
    """tiktoken with a character-ratio fallback if the model files are unavailable."""
    global ENCODER
    if ENCODER is None:
        try:
            import tiktoken

            ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            ENCODER = False
    return ENCODER


def count_tokens(text: str) -> int:
    encoder = _encoder()
    if encoder is False:
        # ~4 chars per token is the usual English ratio; good enough for sizing.
        return max(1, len(text) // 4)
    return len(encoder.encode(text, disallowed_special=()))


def build_context_header(doc: SourceDoc, heading: str) -> str:
    """Prefix embedded with so the vector carries scheme context (ADR-4)."""
    parts = [doc.scheme or doc.title or doc.source_id, doc.doc_type]
    if heading:
        parts.append(heading)
    return " - ".join(p for p in parts if p)


def make_chunk(
    doc: SourceDoc,
    text: str,
    heading: str,
    index: int,
    cfg: ChunkingConfig,
) -> Chunk:
    """Build one Chunk with a stable id and a context-prefixed embed text."""
    body = text.strip()
    if not body:
        raise ValueError(f"Refusing to build an empty chunk for {doc.source_id}")
    header = build_context_header(doc, heading) if cfg.add_context_header else ""
    embed_text = f"{header}\n\n{body}" if header else body
    return Chunk(
        chunk_id=hashlib.sha1(f"{doc.source_id}:{index}".encode("utf-8")).hexdigest()[:16],
        source_id=doc.source_id,
        source_url=doc.source_url,
        scheme=doc.scheme,
        category=doc.category,
        doc_type=doc.doc_type,
        section_heading=heading,
        chunk_index=index,
        text=body,
        embed_text=embed_text,
        token_count=count_tokens(embed_text),
        fetched_at=doc.fetched_at,
    )


def split_blocks(text: str) -> list[str]:
    """Split a document into structural blocks: headings, tables, FAQ pairs, prose.

    A block is a heading plus everything that follows it up to the next heading, so
    the breadcrumb survives into section_heading.
    """
    blocks: list[str] = []
    current: list[str] = []
    current_heading = ""

    for line in text.split("\n"):
        if HEADING_LINE.match(line):
            if current and "".join(current).strip():
                blocks.append("\n".join(current).strip())
            current = [line]
            current_heading = line.lstrip("#").strip()
            continue
        current.append(line)
    if current and "\n".join(current).strip():
        blocks.append("\n".join(current).strip())
    return [b for b in blocks if b]


def blocks_with_headings(text: str) -> list[tuple[str, str]]:
    """Return (heading, block_text) pairs, carrying the heading into each block."""
    out: list[tuple[str, str]] = []
    heading = ""
    for block in split_blocks(text):
        first = block.split("\n", 1)[0]
        if HEADING_LINE.match(first):
            heading = first.lstrip("#").strip()
        out.append((heading, block))
    return out


def iter_table_rows(block: str) -> Iterable[str]:
    for line in block.split("\n"):
        if TABLE_ROW.match(line):
            yield line


def is_question_line(line: str) -> bool:
    return bool(QUESTION_LINE.match(line) or BARE_QUESTION.match(line))


def _table_headers_by_line(lines: list[str]) -> list[str | None]:
    """Map each line to the header block that governs it, for data rows only.

    A data row returns the two-line header of the table it sits in, so a piece that
    starts mid-table can be given that header back. Header and separator rows map to
    None: they are the header, not something that needs one.
    """
    out: list[str | None] = [None] * len(lines)
    header: str | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        if not TABLE_ROW.match(line):
            header = None  # a non-table line ends the table
        elif index + 1 < len(lines) and TABLE_SEPARATOR.match(lines[index + 1]):
            header = f"{line}\n{lines[index + 1]}"
            out[index + 1] = header  # a piece starting at the separator still needs it
            index += 2
            continue
        else:
            out[index] = header  # data row inherits its table header
        index += 1
    return out


def hard_split(text: str, max_chars: int) -> list[str]:
    """Split an over-long block on line boundaries, then on spaces if needed.

    A question line is never left stranded at the end of a piece: if the question
    and its answer will not both fit in the current piece, the break happens *before*
    the question so the pair starts the next piece together (FR-2.3).

    A piece that begins mid-table is given that table's header back. Without this, the
    HDFC Flexi Cap peer-comparison table was cut after its header and the continuation
    held bare unlabelled numbers: "| HDFC Flexi Cap Direct Plan Growth | -0.32% | +15.42%
    | 1,13,606.47 |". That chunk outranked the labelled "Scheme facts" chunk on the query
    "what is the fund size of HDFC Flexi Cap", and the model was right to refuse: nothing
    in it said 1,13,606.47 was a fund size. It was exactly that, and a header restores it.

    Header room is reserved up front so a piece plus its restored header still fits
    `max_chars`, which the caller relies on.
    """
    if len(text) <= max_chars:
        return [text]

    lines = text.split("\n")
    prepend = _table_headers_by_line(lines)

    pieces: list[str] = []
    current = ""
    limit = max_chars

    for index, line in enumerate(lines):
        extra = prepend[index] or ""
        # A piece opening at this line must also carry `extra`, so it gets less room.
        start_limit = max(max_chars - (len(extra) + 1 if extra else 0), max_chars // 2)

        if not current:
            current = extra
            limit = start_limit

        if current and is_question_line(line):
            tail = next((nxt for nxt in lines[index + 1 :] if nxt.strip()), "")
            if len(current) + 1 + len(line) + 1 + len(tail) > limit:
                pieces.append(current)
                current, limit = extra, start_limit

        candidate = f"{current}\n{line}" if current else line
        if current and len(candidate) > limit:
            pieces.append(current)
            current, limit = extra, start_limit
            candidate = f"{current}\n{line}" if current else line
        current = candidate

    if current:
        pieces.append(current)

    # Safety net for a single line longer than max_chars, which no line break can help.
    final: list[str] = []
    for piece in pieces:
        while len(piece) > max_chars:
            cut = piece.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            final.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if piece:
            final.append(piece)
    return final


@runtime_checkable
class Chunker(Protocol):
    name: str

    def split(self, doc: SourceDoc) -> list[Chunk]: ...
