"""Stage 2 chunking: invariants, determinism, and strategy selection."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest

from mf_rag.chunk import build_chunks, chunk_stats
from mf_rag.chunking import DEFAULT_STRATEGY, REGISTRY, available_strategies, get_chunker
from mf_rag.chunking.base import (
    build_context_header,
    count_tokens,
    hard_split,
    make_chunk,
    split_blocks,
)
from mf_rag.chunking.fixed import FixedChunker
from mf_rag.chunking.parent_child import ParentChildChunker
from mf_rag.chunking.section import SectionChunker
from mf_rag.config import AppConfig
from mf_rag.models import SourceDoc

ROW = re.compile(r"^\s*\|.*\|\s*$")

# trafilatura emits markdown headings for structured pages, so the fixture does too.
DOC = SourceDoc(
    source_id="doc1",
    source_url="https://groww.in/mutual-funds/x",
    scheme="HDFC Test Fund - Direct Growth",
    category="ELSS",
    doc_type="scheme_page",
    title="HDFC Test Fund",
    fetched_at="2026-09-27",
    is_official=False,
    text="""HDFC Test Fund

## Overview
This scheme invests in equity. Read all scheme documents carefully.

## Fees
| Item | Value |
|---|---|
| Expense ratio | 1.21% |
| Exit load | Nil |
| Minimum SIP | INR 500 |

## Frequently Asked Questions
Q: What is the lock-in period?
A: The lock-in period is 3 years from the date of allotment.
Q: How do I download a statement?
A: Download it from the AMC website under the reports section.
""",
)

# Real education/blog pages come through as one plain-text block: no markdown
# headings at all, with bare '?' lines acting as the question headings.
PLAIN_FAQ = SourceDoc(
    source_id="doc2",
    source_url="https://www.hdfcfund.com/learn/blog/what-expense-ratio-mutual-funds",
    scheme=None,
    category=None,
    doc_type="educational",
    title="What Is Expense Ratio in Mutual Funds?",
    fetched_at="2026-09-27",
    is_official=True,
    text="""What Is Expense Ratio in Mutual Funds?
Last Updated On: 15 May 2026
Expense ratio is a crucial factor to consider when investing in mutual fund schemes.
Understanding Expense Ratio
The expense ratio includes but not limited to various operational costs such as:
How Is Expense Ratio Calculated?
Expense Ratio = (Total Fund Expenses / Total Assets Under Management) x 100
Impact of Expense Ratio on Returns
A high expense ratio reduces net returns over time. For example:
""",
)


# ---------------------------------------------------------------- registry
def test_registry_has_all_three_candidates() -> None:
    assert set(available_strategies()) == {"fixed", "section", "parent_child"}
    assert DEFAULT_STRATEGY in REGISTRY


def test_get_chunker_honours_explicit_name(cfg: AppConfig) -> None:
    assert get_chunker("fixed", cfg.chunking).name == "fixed"
    assert get_chunker(None, cfg.chunking).name == cfg.chunking.strategy


def test_unknown_strategy_raises(cfg: AppConfig) -> None:
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        get_chunker("magic", cfg.chunking)


def test_config_default_strategy_is_the_measured_winner(cfg: AppConfig) -> None:
    """Chosen by `--rank embed` (real MiniLM cosine), not the lexical proxy."""
    assert cfg.chunking.strategy == "section"


# ---------------------------------------------------------------- base helpers
def test_count_tokens_is_positive_and_scales() -> None:
    assert count_tokens("hello") >= 1
    assert count_tokens("word " * 200) > count_tokens("word")


def test_context_header_includes_scheme_doc_type_and_heading() -> None:
    header = build_context_header(DOC, "Fees")
    assert "HDFC Test Fund - Direct Growth" in header
    assert "scheme_page" in header
    assert "Fees" in header


def test_split_table_repeats_header_on_every_piece() -> None:
    """A piece cut out of the middle of a table must keep the column labels.

    Regression: the HDFC Flexi Cap peer table was split after its header, leaving
    "| HDFC Flexi Cap Direct Plan Growth | -0.32% | +15.42% | 1,13,606.47 |" with no
    indication that 1,13,606.47 was a fund size, so the model refused to answer.
    """
    header = "| | Name | 1Y | 3Y | Fund Size(Cr) |"
    rows = "\n".join(
        f"| | Fund {i} | -0.32% | +15.42% | 19,384.82 |" for i in range(20)
    )
    pieces = hard_split(f"{header}\n|---|---|---|---|---|\n{rows}", 300)

    assert len(pieces) > 1, "table should have needed splitting"
    for piece in pieces[1:]:
        assert piece.startswith(header), f"orphaned piece lost its header: {piece[:80]!r}"


def test_split_table_header_survives_a_second_table_in_the_block() -> None:
    """The header restored must be the one governing the break, not the block's first.

    Regression: a block holding a returns table *and* a peer table was cut inside the
    second one. Reading only lines[0]/lines[1] found the returns header, so the peer
    rows were still orphaned and retrieval kept failing.
    """
    first = (
        "| Name | 3Y | 5Y | All |\n|---|---|---|---|\n"
        "| Fund returns | +15.4% | +16.5% | +15.8% |\n"
        "| Category average | +12.2% | +10.7% | +13.3% |\n"
        "| Rank | 13 | 3 | 3 |"
    )
    peer_header = "| | Name | 1Y | 3Y | Fund Size(Cr) |"
    peer = "\n".join(
        [
            peer_header,
            "|---|---|---|---|---|",
            "| | Bank of India | +11.07% | +19.32% | 2,953.00 |",
            "| | ITI | +13.19% | +19.21% | 1,588.13 |",
            "| | HSBC | +6.31% | +15.75% | 5,999.67 |",
            "| | HDFC Flexi Cap Direct Plan Growth | -0.32% | +15.42% | 1,13,606.47 |",
            "| Compare | | | | |",
        ]
    )
    pieces = hard_split(f"{first}\n{peer}", 180)

    assert any("Fund Size(Cr)" in p and "1,13,606.47" in p for p in pieces), (
        "the AUM row must appear in a piece that also names the Fund Size(Cr) column"
    )


def test_hard_split_never_exceeds_max_chars() -> None:
    text = "\n".join(f"line number {i} with some padding text" for i in range(200))
    for piece in hard_split(text, 300):
        assert len(piece) <= 300
    assert "".join(piece.replace("\n", "") for piece in hard_split(text, 300)).replace(
        " ", ""
    ) == text.replace("\n", "").replace(" ", "")


def test_split_blocks_groups_heading_with_body() -> None:
    blocks = split_blocks(DOC.text)
    fees = next(b for b in blocks if b.startswith("## Fees"))
    assert "Expense ratio" in fees
    assert all(not b.startswith("## Overview\n\n##") for b in blocks)


def test_make_chunk_rejects_empty_text(cfg: AppConfig) -> None:
    with pytest.raises(ValueError, match="empty chunk"):
        make_chunk(DOC, "   ", "Heading", 0, cfg.chunking)


# ---------------------------------------------------------------- invariants
@pytest.mark.parametrize("name", ["fixed", "section", "parent_child"])
def test_every_strategy_produces_non_empty_chunks(name: str, cfg: AppConfig) -> None:
    chunks = get_chunker(name, cfg.chunking).split(DOC)
    assert chunks
    assert all(c.text.strip() for c in chunks)
    assert all(c.source_id == DOC.source_id for c in chunks)
    assert all(c.fetched_at == DOC.fetched_at for c in chunks)


@pytest.mark.parametrize("name", ["section", "parent_child"])
def test_table_rows_are_never_split(name: str, cfg: AppConfig) -> None:
    """FR-2.3: a number must never be separated from its label."""
    chunks = get_chunker(name, cfg.chunking).split(DOC)
    for row in ["| Expense ratio | 1.21% |", "| Exit load | Nil |", "| Minimum SIP | INR 500 |"]:
        assert any(row in c.text for c in chunks), f"{name} split {row}"


def test_fixed_strategy_cuts_a_table_row() -> None:
    """Documents why `fixed` is gated out rather than merely losing on ranking.

    The padding is sized so a window boundary provably lands inside the table row.
    """
    from mf_rag.config import ChunkingConfig

    row = "| Expense ratio | 1.23456789% | 12345 |"
    doc = replace(DOC, text=("x" * 50) + "\n" + row + "\n" + ("y" * 100))
    cfg = ChunkingConfig(
        strategy="fixed",
        target_chars=60,
        max_chars=60,
        min_chars=0,
        overlap_chars=0,
        keep_tables_atomic=True,
        add_context_header=True,
    )
    chunks = FixedChunker(cfg).split(doc)
    assert not any(row in c.text for c in chunks), "fixed unexpectedly kept the row intact"


def test_row_straddling_a_fixed_window_is_recovered_by_structure_aware_strategies() -> None:
    from mf_rag.config import ChunkingConfig

    row = "| Expense ratio | 1.23456789% | 12345 |"
    doc = replace(DOC, text=("x" * 50) + "\n" + row + "\n" + ("y" * 100))
    cfg = ChunkingConfig(
        strategy="section",
        target_chars=60,
        max_chars=200,
        min_chars=0,
        overlap_chars=0,
        keep_tables_atomic=True,
        add_context_header=True,
    )
    chunks = SectionChunker(cfg).split(doc)
    assert any(row in c.text for c in chunks)


@pytest.mark.parametrize("name", ["section", "parent_child"])
def test_faq_question_stays_with_its_answer(name: str, cfg: AppConfig) -> None:
    chunks = get_chunker(name, cfg.chunking).split(DOC)
    assert any(
        "What is the lock-in period" in c.text and "3 years" in c.text for c in chunks
    )
    assert any(
        "How do I download a statement" in c.text and "reports section" in c.text for c in chunks
    )


@pytest.mark.parametrize("name", ["section", "parent_child"])
def test_bare_question_never_stranded_from_its_answer(name: str) -> None:
    """Regression: a plain-text page is one block, and the break used to land
    between a bare '?' heading and its answer, so the answer was unreachable
    without its question. `fixed` is deliberately excluded - it is the naive
    baseline that the gates reject, and it still strands the pair."""
    from mf_rag.config import ChunkingConfig, get_config

    base = get_config().chunking
    tight = ChunkingConfig(
        strategy=name,
        target_chars=90,
        max_chars=140,
        min_chars=20,
        overlap_chars=20,
        keep_tables_atomic=base.keep_tables_atomic,
        add_context_header=base.add_context_header,
    )
    chunks = get_chunker(name, tight).split(PLAIN_FAQ)
    assert any(
        "How Is Expense Ratio Calculated?" in c.text
        and "Total Assets Under Management" in c.text
        for c in chunks
    ), "question was separated from its answer"


def test_chunks_carry_a_heading_breadcrumb(cfg: AppConfig) -> None:
    chunks = SectionChunker(cfg.chunking).split(DOC)
    headings = {c.section_heading for c in chunks}
    assert any(h for h in headings)


def test_embed_text_is_context_prefixed_but_body_is_verbatim(cfg: AppConfig) -> None:
    chunk = SectionChunker(cfg.chunking).split(DOC)[0]
    assert chunk.embed_text.endswith(chunk.text)
    assert len(chunk.embed_text) > len(chunk.text)
    assert "HDFC Test Fund - Direct Growth" in chunk.embed_text


def test_chunk_ids_are_stable_and_unique(cfg: AppConfig) -> None:
    first = SectionChunker(cfg.chunking).split(DOC)
    second = SectionChunker(cfg.chunking).split(DOC)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)


def test_chunk_metadata_is_chroma_safe(cfg: AppConfig) -> None:
    for chunk in SectionChunker(cfg.chunking).split(DOC):
        for key, value in chunk.to_metadata().items():
            assert value is not None, key


# ---------------------------------------------------------------- parent/child
def test_parent_child_children_are_smaller_than_parents(cfg: AppConfig) -> None:
    chunker = ParentChildChunker(cfg.chunking)
    chunks = chunker.split(DOC)
    for chunk in chunks:
        parent = chunker.parent_for(chunk)
        assert parent
        assert len(chunk.text) <= len(parent)


def test_parent_child_parent_contains_the_fact(cfg: AppConfig) -> None:
    chunker = ParentChildChunker(cfg.chunking)
    chunks = chunker.split(DOC)
    parents = {chunker.parent_for(c) for c in chunks}
    assert any("1.21%" in p for p in parents)


# ---------------------------------------------------------------- orchestration
def test_build_chunks_is_deterministic(cfg: AppConfig) -> None:
    first, _ = build_chunks([DOC], cfg, "section")
    second, _ = build_chunks([DOC], cfg, "section")
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.text for c in first] == [c.text for c in second]


def test_build_chunks_sorted_by_source_and_index(cfg: AppConfig) -> None:
    doc_b = replace(DOC, source_id="aaa", scheme=None, category=None)
    chunks, _ = build_chunks([DOC, doc_b], cfg, "section")
    assert [c.source_id for c in chunks] == sorted(c.source_id for c in chunks)
    for source_id in {c.source_id for c in chunks}:
        indexes = [c.chunk_index for c in chunks if c.source_id == source_id]
        assert indexes == sorted(indexes)


def test_chunk_stats_reports_shape(cfg: AppConfig) -> None:
    chunks, chunker = build_chunks([DOC], cfg, "section")
    stats = chunk_stats(chunks, chunker, [DOC])
    assert stats["strategy"] == "section"
    assert stats["chunks"] == len(chunks)
    assert stats["documents"] == 1
    assert stats["avg_tokens"] > 0
    assert "doc1" in stats["per_source"]


def test_empty_document_list_yields_no_chunks(cfg: AppConfig) -> None:
    chunks, _ = build_chunks([], cfg, "section")
    assert chunks == []


# ---------------------------------------------------------------- real corpus
@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "data" / "chunks.jsonl").is_file(),
    reason="run scripts/run_ingest.py and python -m mf_rag.chunk first",
)
def test_real_corpus_chunks_are_valid() -> None:
    from mf_rag.chunk import load_chunks

    chunks = load_chunks()
    assert len(chunks) > 50
    assert all(c.text.strip() for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    for chunk in chunks:
        assert chunk.source_url.startswith("https://")
        assert chunk.token_count > 0
