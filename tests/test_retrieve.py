"""Phase 5 retrieval tests.

The integration tests use the real index in `data/chroma` because the whole point of the
entity filter is a property of that index (do Small Cap chunks rank above Large Cap ones
once filtered?). The unit tests inject a fake collection so the fallback and min_score
paths are exercised without a 90MB model load.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from conftest import needs_model

from mf_rag.models import Chunk, RetrievalHit
from mf_rag.retrieve import (
    _dedupe_by_source,
    _query_collection,
    _to_chunk,
    load_chunk_index,
    retrieve,
    retrieve_with_trace,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


# --------------------------------------------------------------------------- unit


class _FakeCollection:
    """Minimal Chroma stand-in. Returns scripted rows and records the filters used."""

    def __init__(self, rows_by_filter: dict[str | None, list[tuple[str, float]]]):
        self.rows_by_filter = rows_by_filter
        self.calls: list[dict | None] = []

    def query(self, **kwargs):
        where = kwargs.get("where")
        self.calls.append(where)
        rows = self.rows_by_filter.get(_key(where), [])
        n = kwargs.get("n_results", 10)
        rows = rows[:n]
        return {
            "ids": [[r[0] for r in rows]],
            "documents": [[f"doc {r[0]}" for r in rows]],
            "metadatas": [[_meta(r[0]) for r in rows]],
            "distances": [[r[1] for r in rows]],
        }


def _key(where):
    if not where:
        return None
    return tuple(sorted(where.items()))


def _meta(chunk_id: str) -> dict:
    return {
        "source_id": f"src_{chunk_id}",
        "source_url": f"https://example.com/{chunk_id}",
        "scheme": "HDFC Small Cap Fund - Direct Growth",
        "category": "Small Cap",
        "doc_type": "scheme_page",
        "section_heading": "Fees",
        "chunk_index": int(chunk_id.replace("c", "")),
        "token_count": 20,
        "fetched_at": "2026-09-27",
    }


def _hits(pairs):
    return [RetrievalHit(_to_chunk(cid, f"doc {cid}", _meta(cid)), 1.0 - d) for cid, d in pairs]


def test_dedupe_keeps_best_chunk_per_source():
    """Distinct sources all survive, ordered by score."""
    hits = _hits([("c0", 0.10), ("c1", 0.40), ("c2", 0.20)])
    out = _dedupe_by_source(hits)
    # score = 1 - distance, so c0=0.90, c2=0.80, c1=0.60
    assert [h.chunk.chunk_id for h in out] == ["c0", "c2", "c1"]


def _hit_with_text(chunk_id, text, score, meta):
    """A RetrievalHit whose `text` is populated.

    `_to_chunk` mirrors what Chroma stored: the document lands in `embed_text` and `text`
    is left for `load_chunk_index` to hydrate. Dedupe reads `text`, so tests that exercise
    it have to fill it in the way hydration does.
    """
    return RetrievalHit(replace(_to_chunk(chunk_id, text, meta), text=text), score)


def test_dedupe_collapses_repeated_source_keeping_highest_score():
    """Genuinely repeated text collapses to its best-scoring copy, across sources too."""
    meta = _meta("c0")
    text = "Fund size AUM is 113,606.47 crore as on 27 Sep 2026 for the direct plan"
    a = _hit_with_text("c0", text, 0.60, meta)
    b = _hit_with_text("c0", text, 0.90, meta)
    c = _hit_with_text("c1", text, 0.70, meta | {"source_id": "other"})
    out = _dedupe_by_source([a, b, c])
    assert len(out) == 1
    assert out[0].score == 0.90


def test_dedupe_keeps_distinct_sections_of_one_page():
    """One page may contribute several chunks, because its sections differ.

    Regression: collapsing a page to a single chunk threw away the one holding the answer.
    For "what is the fund size of HDFC Flexi Cap" the peer-table tail scored highest and
    won, while the chunk stating "Fund size / AUM (INR crore): 113,606.47" was dropped
    before the model saw it, and the assistant claimed the corpus had no fund size.
    """
    meta = _meta("c0")
    passages = [
        ("c0", 0.90, "Franklin India Flexi Cap Fund Direct Growth 1Y -3.98% 3Y +10.62% size 19,384.82"),
        ("c1", 0.80, "HDFC Flexi Cap Direct Plan Growth 1Y -0.32% 3Y +15.42% size 1,13,606.47"),
        ("c2", 0.74, "Scheme facts: Fund size / AUM (INR crore) (Direct plan): 113,606.47"),
        ("c3", 0.60, "Top holdings: Bajaj Auto, Ashok Leyland, ICICI Prudential, BSE Ltd"),
    ]
    hits = [
        _hit_with_text(cid, text, score, meta | {"chunk_index": i})
        for i, (cid, score, text) in enumerate(passages)
    ]
    out = _dedupe_by_source(hits)
    kept = [h.chunk.text for h in out]
    assert any("Fund size / AUM" in t for t in kept), "the answer-bearing chunk was dropped"
    # All four passages are distinct, so the cap is the only limit that applies. It is 6,
    # not 3: a scheme page yields several unrelated blocks, and at 3 the cap was dropping
    # blocks ranked just below three better-scoring siblings.
    assert len(out) == 4
    assert len(_dedupe_by_source(hits, max_per_source=2)) == 2


def test_dedupe_never_treats_unhydrated_text_as_duplicate():
    """Chunks reach dedupe before `text` is filled in, so empty text must not collapse."""
    meta = _meta("c0")
    hits = [
        RetrievalHit(_to_chunk(f"c{i}", "x", meta | {"chunk_index": i}), 0.9 - i / 10)
        for i in range(3)
    ]
    assert all(h.chunk.text == "" for h in hits)
    # Nothing was proven redundant, so only the cap applies: all three survive.
    assert len(_dedupe_by_source(hits)) == 3


def test_dedupe_caps_chunks_per_source():
    """A page cannot flood the context, even when every section is distinct."""
    meta = _meta("c0")
    hits = [
        _hit_with_text(
            f"c{i}", f"section {i} covers topic number {i} in detail", 0.9 - i / 100,
            meta | {"chunk_index": i},
        )
        for i in range(8)
    ]
    assert len(_dedupe_by_source(hits, max_per_source=3)) == 3


def test_to_chunk_undoes_chroma_safe_coercion():
    chunk = _to_chunk("c3", "doc", _meta("c3") | {"scheme": "", "section_heading": ""})
    assert chunk.scheme is None
    assert chunk.section_heading == ""
    assert chunk.chunk_index == 3


def test_to_chunk_tolerates_missing_metadata():
    chunk = _to_chunk("c0", None, None)
    assert chunk.source_id == ""
    assert chunk.chunk_index == 0


def test_min_score_drops_weak_hits(monkeypatch, cfg):
    """0.9 cosine is a hit; 0.2 is noise. The floor decides which survive."""
    import mf_rag.retrieve as R

    fake = _FakeCollection({None: [("c0", 0.10), ("c1", 0.80)]})
    monkeypatch.setattr(R, "get_or_create_collection", lambda cfg=None: fake)
    monkeypatch.setattr(R, "embed_query", lambda q, cfg=None: [0.0] * 384)
    monkeypatch.setattr(R, "load_chunk_index", lambda cfg=None: {})

    hits, trace = R.retrieve_with_trace("anything", min_score=0.5, entity_filter=False, cfg=cfg)
    assert [h.chunk.chunk_id for h in hits] == ["c0"]  # distance 0.10 -> score 0.90
    assert trace["min_score"] == 0.5
    assert trace["candidates_considered"] == 1


def test_fallback_to_unfiltered_when_filter_matches_nothing(monkeypatch, cfg):
    """A narrow-but-wrong filter must not silently produce zero answers."""
    import mf_rag.retrieve as R

    scheme_filter = (("scheme", "HDFC Small Cap Fund - Direct Growth"),)
    fake = _FakeCollection({scheme_filter: [], None: [("c0", 0.50)]})
    monkeypatch.setattr(R, "get_or_create_collection", lambda cfg=None: fake)
    monkeypatch.setattr(R, "embed_query", lambda q, cfg=None: [0.0] * 384)
    monkeypatch.setattr(R, "load_chunk_index", lambda cfg=None: {})

    hits, trace = R.retrieve_with_trace("Exit load of HDFC Small Cap Fund?", cfg=cfg)
    assert len(hits) == 1
    assert trace["fallback_unfiltered"] is True
    assert trace["filter_matched_no_rows"] is True
    assert trace["where_filter"] is None
    assert scheme_filter == tuple(sorted({"scheme": "HDFC Small Cap Fund - Direct Growth"}.items()))
    assert fake.calls == [{"scheme": "HDFC Small Cap Fund - Direct Growth"}, None]


def test_no_fallback_when_filter_matches(monkeypatch, cfg):
    import mf_rag.retrieve as R

    scheme_filter = (("scheme", "HDFC Small Cap Fund - Direct Growth"),)
    fake = _FakeCollection({scheme_filter: [("c0", 0.50)]})
    monkeypatch.setattr(R, "get_or_create_collection", lambda cfg=None: fake)
    monkeypatch.setattr(R, "embed_query", lambda q, cfg=None: [0.0] * 384)
    monkeypatch.setattr(R, "load_chunk_index", lambda cfg=None: {})

    hits, trace = R.retrieve_with_trace("Exit load of HDFC Small Cap Fund?", cfg=cfg)
    assert trace["fallback_unfiltered"] is False
    assert trace["where_filter"] == {"scheme": "HDFC Small Cap Fund - Direct Growth"}
    assert len(hits) == 1


def test_context_chunks_caps_result_length(monkeypatch, cfg):
    import mf_rag.retrieve as R

    rows = [(f"c{i}", 0.1 * i) for i in range(10)]
    fake = _FakeCollection({None: rows})
    monkeypatch.setattr(R, "get_or_create_collection", lambda cfg=None: fake)
    monkeypatch.setattr(R, "embed_query", lambda q, cfg=None: [0.0] * 384)
    monkeypatch.setattr(R, "load_chunk_index", lambda cfg=None: {})

    hits, _ = R.retrieve_with_trace("q", min_score=-1.0, entity_filter=False, cfg=cfg)
    assert len(hits) == cfg.retrieval.context_chunks


def test_empty_result_has_null_top_score(monkeypatch, cfg):
    import mf_rag.retrieve as R

    fake = _FakeCollection({None: []})
    monkeypatch.setattr(R, "get_or_create_collection", lambda cfg=None: fake)
    monkeypatch.setattr(R, "embed_query", lambda q, cfg=None: [0.0] * 384)
    monkeypatch.setattr(R, "load_chunk_index", lambda cfg=None: {})

    hits, trace = R.retrieve_with_trace("q", entity_filter=False, cfg=cfg)
    assert hits == []
    assert trace["top_score"] is None
    assert trace["hits"] == []


def test_query_collection_omits_where_when_unfiltered():
    calls = {}

    class C:
        def query(self, **kwargs):
            calls.update(kwargs)
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    _query_collection(C(), [0.0], 5, None)
    assert "where" not in calls
    _query_collection(C(), [0.0], 5, {"category": "ELSS"})
    assert calls["where"] == {"category": "ELSS"}


# ---------------------------------------------------------------- integration


@needs_model
def test_small_cap_question_hits_only_small_cap():
    """The Phase 5 exit gate: a named scheme returns a filter and only that scheme."""
    hits, trace = retrieve_with_trace("Exit load of HDFC Small Cap Fund?")
    assert trace["where_filter"] == {"scheme": "HDFC Small Cap Fund - Direct Growth"}
    assert trace["entity_detected"] == "scheme"
    assert hits, "expected at least one hit"
    assert {h.chunk.scheme for h in hits} == {"HDFC Small Cap Fund - Direct Growth"}
    assert trace["fallback_unfiltered"] is False


@needs_model
def test_unfiltered_small_cap_query_is_contaminated():
    """Proves the filter earns its place: unfiltered, Large Cap outranks Small Cap."""
    hits, trace = retrieve_with_trace("Exit load of HDFC Small Cap Fund?", entity_filter=False)
    assert trace["where_filter"] is None
    assert any(h.chunk.scheme != "HDFC Small Cap Fund - Direct Growth" for h in hits)


@needs_model
def test_corpus_wide_question_runs_unfiltered():
    hits, trace = retrieve_with_trace("How do I download the capital gains statement?")
    assert trace["where_filter"] is None
    assert hits
    assert "hdfcfund_consolidated_statement" in {h.chunk.source_id for h in hits}


@needs_model
def test_retrieve_returns_hits_without_trace():
    hits = retrieve("What is the expense ratio of HDFC Large Cap Fund?")
    assert hits
    assert all(isinstance(h, RetrievalHit) for h in hits)
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True)


@needs_model
def test_trace_shape_matches_the_documented_contract():
    _, trace = retrieve_with_trace("What is the ELSS lock-in period?")
    for key in ("entity_detected", "where_filter", "hits", "top_score"):
        assert key in trace
    for hit in trace["hits"]:
        assert {"chunk_id", "score", "source_url", "section_heading"} <= set(hit)


@needs_model
def test_chunk_text_is_hydrated_from_chunks_jsonl():
    """The generator must see the passage, not the context-prefixed header."""
    hits, _ = retrieve_with_trace("Exit load of HDFC Small Cap Fund?")
    index = load_chunk_index()
    assert hits
    assert hits[0].chunk.text, "chunk text was not hydrated"
    assert hits[0].chunk.text != hits[0].chunk.embed_text
    assert hits[0].chunk.chunk_id in index


# ------------------------------------------------------------------ lexical rerank


def test_lexical_boost_rewards_a_question_named_in_the_heading():
    """Cosine alone could not separate two neighbouring blocks; the heading can.

    On the Balanced Advantage page the sector and asset blocks sat 0.02 apart, and which
    won flipped with the wording of the question. A heading is a topical label written in
    the same words a question uses, so overlap there is the missing signal.
    """
    from mf_rag.retrieve import _lexical_boost, _query_terms

    query = "equity debt cash allocation of the fund"
    terms = _query_terms(query)
    asset = Chunk(
        chunk_id="a", source_id="s", source_url="u", scheme="", category="", doc_type="",
        section_heading="Asset allocation across equity, debt and cash", chunk_index=0,
        text="- Equity: 62.0%\n- Debt: 37.0%", embed_text="", token_count=0, fetched_at="",
    )
    fees = Chunk(
        chunk_id="b", source_id="s", source_url="u", scheme="", category="", doc_type="",
        section_heading="Fees and exit load", chunk_index=1,
        text="- Expense ratio (annual %): 0.78", embed_text="", token_count=0, fetched_at="",
    )
    assert _lexical_boost(terms, asset) > _lexical_boost(terms, fees)


def test_lexical_boost_never_overturns_a_clearly_better_semantic_match():
    """The boost is additive and small, so it breaks near-ties and nothing more.

    A block matching the question only loosely but strongly on semantics must still be
    reachable; otherwise this would quietly replace the embedder with a keyword match.
    """
    from mf_rag.retrieve import _BODY_BOOST, _HEADING_BOOST

    assert _HEADING_BOOST < 0.25 and _BODY_BOOST < _HEADING_BOOST


def test_query_terms_drop_stopwords_and_scheme_names():
    from mf_rag.retrieve import _query_terms

    terms = _query_terms("What is the expense ratio of HDFC Large Cap Fund?")
    assert "expense" in terms and "ratio" in terms
    assert "what" not in terms and "hdfc" not in terms and "fund" not in terms


def test_rerank_preserves_relative_dense_order_for_unrelated_queries():
    from mf_rag.retrieve import _rerank

    meta = _meta("c0")
    high = _hit_with_text("c0", "Sector allocation and exposure: Financial 39.07%", 0.70, meta)
    low = _hit_with_text("c1", "Fees and exit load: Expense ratio 1.03%", 0.60, meta | {"chunk_index": 1})
    out = _rerank("sector allocation", [low, high])
    assert out[0].chunk.chunk_id == "c0"
