"""RAG Stage Group B / Stage 5: Retrieval.

Order of operations (ARCHITECTURE 6.2), and each step is a deliberate decision:

1. `embed_query` via `mf_rag.embedder` - never load the model here.
2. `detect_entity` -> a `where` filter. This is a correctness requirement, not a
   speed-up: HDFC's schemes share exit-load and expense-ratio boilerplate, and the
   unfiltered ranking for "Exit load of HDFC Small Cap Fund?" puts Large Cap first.
3. Query the collection.
4. If a filter was applied and returned nothing, retry unfiltered. A wrong-but-narrow
   filter is worse than no filter, so the fallback is always available.
5. Cosine score = 1 - distance, then drop anything under `min_score`.
6. Dedupe by `source_id` (one page = one voice) and cap at `context_chunks`.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from mf_rag.chunk import load_chunks
from mf_rag.config import AppConfig, get_config
from mf_rag.entities import describe, detect_entity
from mf_rag.embedder import embed_query
from mf_rag.entities import correct_domain_typos
from mf_rag.models import Chunk, RetrievalHit

# Weights for the lexical signal added on top of cosine. A heading is a label, so a term
# found there is a much better sign than the same term buried in prose. They are small
# enough that dense retrieval still decides: the widest gap seen between neighbouring
# blocks was 0.02, so 0.12 can overturn a near-tie but not a clear mismatch.
_HEADING_BOOST = 0.12
_BODY_BOOST = 0.04

_STOPWORDS = frozenset(
    """a an the of for in on to and or is are was were what which who whom whose how
    when where why does do did done has have had it its this that these those fund funds
    scheme schemes plan growth direct hdfc mutual""".split()
)
from mf_rag.models import Chunk, RetrievalHit
from mf_rag.store import get_or_create_collection

CHUNK_META_FIELDS = (
    "source_id",
    "source_url",
    "scheme",
    "category",
    "doc_type",
    "section_heading",
    "chunk_index",
    "token_count",
    "fetched_at",
)


def _clean(value: Any) -> Any:
    """Undo `store._chroma_safe` coercion: "" -> None, -1 -> None for chunk_index."""
    if value == "":
        return None
    if value == -1:
        return None
    return value


def _to_chunk(chunk_id: str, document: str, metadata: dict[str, Any] | None) -> Chunk:
    """Rebuild a full `Chunk` from a stored record.

    Stage 4 stored `embed_text` as the document and metadata only, so `text` is recovered
    from `chunks.jsonl` by `load_chunk_index` below. Without it the generator would be
    shown the context-prefixed header instead of the passage a human would read.
    """
    meta = dict(metadata or {})
    return Chunk(
        chunk_id=chunk_id,
        source_id=meta.get("source_id", ""),
        source_url=meta.get("source_url", ""),
        scheme=_clean(meta.get("scheme")),
        category=_clean(meta.get("category")),
        doc_type=meta.get("doc_type", ""),
        section_heading=meta.get("section_heading", ""),
        chunk_index=int(_clean(meta.get("chunk_index")) or 0),
        text="",
        embed_text=document or "",
        token_count=int(_clean(meta.get("token_count")) or 0),
        fetched_at=meta.get("fetched_at", ""),
    )


def load_chunk_index(cfg: AppConfig | None = None) -> dict[str, Chunk]:
    """`chunk_id -> Chunk` read from `data/chunks.jsonl`, for text hydration."""
    return {chunk.chunk_id: chunk for chunk in load_chunks(cfg or get_config())}


def _hydrate(hit: RetrievalHit, cache: dict[str, Chunk]) -> RetrievalHit:
    """Replace a stored-metadata shell with the real chunk when we have it."""
    full = cache.get(hit.chunk.chunk_id)
    if full is not None:
        hit.chunk = full
    return hit


def _shingles(text: str) -> frozenset[str]:
    """Word 3-grams, used to judge whether two chunks really say the same thing."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < 3:
        return frozenset(words)
    return frozenset(
        " ".join(words[i : i + 3]) for i in range(len(words) - 2)
    )


def _is_near_duplicate(a: str, b: str, threshold: float) -> bool:
    sa, sb = _shingles(a), _shingles(b)
    if not sa or not sb:
        # Nothing to compare - chunks reach this stage before `text` is hydrated from
        # chunks.jsonl, so an empty text must never count as "the same as everything".
        # Only proven redundancy may drop a chunk.
        return False
    overlap = len(sa & sb) / min(len(sa), len(sb))
    return overlap >= threshold


def _dedupe_by_source(
    hits: Sequence[RetrievalHit], max_per_source: int = 6, near_dup: float = 0.82
) -> list[RetrievalHit]:
    """Drop redundant chunks while keeping the distinct sections of a page.

    The original rule was one chunk per `source_id` ("one page, one voice"), on the
    assumption that several chunks from a page are near-identical. They are not: a scheme
    page holds distinct sections - the labelled facts block, the returns table, the
    holdings, the peer table. Keeping only the top-scoring one silently discarded the
    chunk that actually held the answer: for "what is the fund size of HDFC Flexi Cap",
    the peer-table tail scored 0.7917 and won, while the chunk stating
    "Fund size / AUM (INR crore): 113,606.47" scored 0.7367 and was dropped before the
    model saw it, so the assistant insisted the corpus had no fund size. It did.

    So redundancy is now judged by the text itself: chunks repeating an already-selected
    chunk are dropped, and each source may contribute up to `max_per_source` distinct ones.

    The cap is 6 rather than 3 because a scheme page now yields six blocks that are
    genuinely unrelated to each other - labelled fees, the manager list, sector exposure,
    asset composition, top holdings, and the fee history. At 3, "portfolio turnover ratio of
    HDFC Flexi Cap" lost: sector exposure, asset composition and top holdings took the three
    slots, and the facts block carrying the turnover figure ranked eighth and was dropped
    before the model saw it. The cap is a backstop against one page filling the context, not
    the primary redundancy filter - the near-duplicate test is.
    """
    ordered = sorted(hits, key=lambda h: h.score, reverse=True)
    selected: list[RetrievalHit] = []
    per_source: dict[str, int] = {}
    for hit in ordered:
        source_id = hit.chunk.source_id
        if per_source.get(source_id, 0) >= max_per_source:
            continue
        if any(_is_near_duplicate(hit.chunk.text, kept.chunk.text, near_dup) for kept in selected):
            continue
        per_source[source_id] = per_source.get(source_id, 0) + 1
        selected.append(hit)
    return selected


def _query_collection(collection, vector: list[float], k: int, where: dict[str, Any] | None) -> dict[str, Any]:
    """One Chroma query. `where=None` means unfiltered."""
    kwargs: dict[str, Any] = {
        "query_embeddings": [vector],
        "n_results": max(1, int(k)),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    return collection.query(**kwargs)


def _query_terms(query: str) -> frozenset[str]:
    """Content words of the query, with the stopwords that carry no topic removed."""
    words = re.findall(r"[a-z0-9]+", query.lower())
    return frozenset(w for w in words if w not in _STOPWORDS and len(w) > 2)


def _lexical_boost(query_terms: frozenset[str], chunk: Chunk) -> float:
    """How much of the query a chunk names outright, weighted toward its heading.

    Dense retrieval alone cannot separate two neighbouring blocks. On the Balanced
    Advantage page, "Asset allocation across equity, debt and cash" and "Sector
    allocation and exposure" sit 0.02 apart in cosine, and which one wins flips with the
    wording of the question: "equity debt cash split" put the asset block first, "equity
    debt cash allocation" put it ninth. A heading is a topical label written in the same
    vocabulary the question uses, so overlap there is strong evidence the dense score
    missed. It is added on top of cosine, never multiplied by it, so a chunk with no
    lexical overlap can still win on semantics alone.
    """
    if not query_terms:
        return 0.0
    heading = _query_terms(chunk.section_heading or "")
    body = _query_terms(chunk.text or "")
    heading_hits = len(query_terms & heading) / len(query_terms)
    body_hits = len(query_terms & body) / len(query_terms)
    return _HEADING_BOOST * heading_hits + _BODY_BOOST * body_hits


def _rerank(query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
    """Re-sort the candidate pool by cosine plus lexical evidence."""
    terms = _query_terms(query)
    for hit in hits:
        hit.score = round(hit.score + _lexical_boost(terms, hit.chunk), 6)
    return sorted(hits, key=lambda h: h.score, reverse=True)


def retrieve_with_trace(
    query: str,
    k: int | None = None,
    min_score: float | None = None,
    entity_filter: bool | None = None,
    cfg: AppConfig | None = None,
) -> tuple[list[RetrievalHit], dict[str, Any]]:
    """Retrieve ranked chunks plus a trace explaining every decision made."""
    config = cfg or get_config()
    top_k = int(k if k is not None else config.retrieval.top_k)
    floor = float(min_score if min_score is not None else config.retrieval.min_score)
    use_filter = (
        bool(entity_filter) if entity_filter is not None else bool(config.retrieval.filter_by_entity)
    )

    kind, where = detect_entity(query) if use_filter else (None, None)

    # Ask for more than we need: dedupe and min_score both shrink the result, and a
    # short candidate pool would silently cap the final context.
    pool = max(top_k * 4, 20)
    vector = embed_query(correct_domain_typos(query), config)
    collection = get_or_create_collection(config)

    result = _query_collection(collection, vector, pool, where)
    applied_filter = where
    fallback_used = False
    filtered_empty = bool(where) and not _ids(result)
    if filtered_empty and config.retrieval.fallback_unfiltered:
        result = _query_collection(collection, vector, pool, None)
        applied_filter = None
        fallback_used = True

    cache = load_chunk_index(config)
    hits: list[RetrievalHit] = []
    for chunk_id, document, metadata, distance in _rows(result):
        score = 1.0 - float(distance)
        if score < floor:
            continue
        hits.append(_hydrate(RetrievalHit(_to_chunk(chunk_id, document or "", metadata), score), cache))

    # Rerank before the floor is re-checked, so a chunk that cosine ranked low but that the
    # question names outright can still clear min_score.
    hits = _rerank(query, hits)
    hits = [h for h in hits if h.score >= floor]

    deduped = _dedupe_by_source(hits)[: config.retrieval.context_chunks]

    trace: dict[str, Any] = {
        "query": None,
        "entity_detected": kind,
        "where_filter": applied_filter,
        "where_filter_text": describe(kind, applied_filter),
        "filter_applied": bool(applied_filter),
        "filter_matched_no_rows": filtered_empty,
        "fallback_unfiltered": fallback_used,
        "min_score": floor,
        "top_k": top_k,
        "candidates_considered": len(hits),
        "after_dedupe": len(deduped),
        "top_score": round(deduped[0].score, 6) if deduped else None,
        "hits": [
            {
                "chunk_id": hit.chunk.chunk_id,
                "score": round(hit.score, 6),
                "source_id": hit.chunk.source_id,
                "source_url": hit.chunk.source_url,
                "section_heading": hit.chunk.section_heading,
                "scheme": hit.chunk.scheme,
                "doc_type": hit.chunk.doc_type,
                # Carried so the freshness stamp can be audited from the trace alone
                # (FR-7.5): the answer shows the *oldest* of these dates.
                "fetched_at": hit.chunk.fetched_at,
            }
            for hit in deduped
        ],
    }
    return deduped, trace


def retrieve(
    query: str,
    k: int | None = None,
    min_score: float | None = None,
    entity_filter: bool = True,
    cfg: AppConfig | None = None,
) -> list[RetrievalHit]:
    """Ranked hits only. See `retrieve_with_trace` for the full explanation."""
    hits, _ = retrieve_with_trace(
        query, k=k, min_score=min_score, entity_filter=entity_filter, cfg=cfg
    )
    return hits


def _ids(result: dict[str, Any]) -> list[str]:
    return list((result.get("ids") or [[]])[0])


def _rows(result: dict[str, Any]) -> list[tuple[str, str | None, dict[str, Any] | None, float]]:
    """Flatten Chroma's nested-by-query result into one tuple per hit."""
    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0] if result.get("documents") else []
    metas = (result.get("metadatas") or [[]])[0] if result.get("metadatas") else []
    dists = (result.get("distances") or [[]])[0] if result.get("distances") else []
    rows = []
    for position, chunk_id in enumerate(ids):
        rows.append(
            (
                str(chunk_id),
                docs[position] if position < len(docs) else None,
                metas[position] if position < len(metas) else None,
                float(dists[position]) if position < len(dists) else 1.0,
            )
        )
    return rows
