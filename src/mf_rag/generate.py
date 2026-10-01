"""RAG Stage Group B / Stage 6: Generation (context assembly + one LLM call).

Two rules shape this module:

1. **The system picks the citation** (ADR-5). `select_source_url` chooses the URL from the
   retrieved metadata and passes it to the model as a fixed token to echo. The model never
   authors a URL, so a hallucinated link is structurally impossible rather than merely
   unlikely.
2. **The system stamps the date** (rule 7). The freshness line is appended here using the
   **oldest** `fetched_at` among the chunks used, never a date the model produced.

Note ARCHITECTURE 6.3 says "max fetched_at" in one sentence and "oldest data is the honest
bound" in the next; the second is the intent and IMPLEMENTATION.md Phase 6 states min, so
min is what is implemented. A corpus whose oldest page is stale must not be stamped with
the newest page's date.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

from mf_rag.config import AppConfig, get_config
from mf_rag.llm import LLMClient, get_llm
from mf_rag.models import Answer, RetrievalHit, Trace
from mf_rag.prompts import (
    DISCLAIMER,
    NOT_IN_SOURCES,
    freshness_line,
    render_context,
    strip_freshness,
)
from mf_rag.retrieve import retrieve_with_trace
from mf_rag.sources import SOURCES

# source_id -> the ingested URL, used to prefer official sources and to validate the link.
SOURCE_URLS: dict[str, str] = {s["source_id"]: s["source_url"] for s in SOURCES}
OFFICIAL_SOURCES: frozenset[str] = frozenset(
    s["source_id"] for s in SOURCES if s.get("is_official")
)

# One educational page for refusals. Never invented, never scheme-specific, so a refusal
# can never cite a fund the corpus does not contain.
EDUCATIONAL_SOURCE_ID = "hdfcfund_fees_education"
EDUCATIONAL_URL = SOURCE_URLS[EDUCATIONAL_SOURCE_ID]

# Factsheet-style page used when refusing a returns question about a named scheme.
FACTSHEET_SOURCE_BY_SCHEME: dict[str, str] = {
    s["scheme"]: s["source_id"]
    for s in SOURCES
    if s.get("is_official") and s.get("scheme") and s["doc_type"] in {"scheme_page", "fees", "faq"}
}


def official_source_ids() -> frozenset[str]:
    return OFFICIAL_SOURCES


def select_source_url(hits: Sequence[RetrievalHit], cfg: AppConfig | None = None) -> str | None:
    """Choose the single citation URL. The system chooses; the model only echoes.

    Preference order:
      1. the best-ranked hit whose source is official
      2. the best-ranked hit that has a URL at all
      3. None
    Official beats higher score on purpose: a scheme-page aggregator and the AMC's own page
    can describe the same exit load, and the AMC page is the one a reader should trust
    (FR-2.3).
    """
    if not hits:
        return None
    for hit in hits:
        if hit.chunk.source_id in OFFICIAL_SOURCES and hit.chunk.source_url:
            return hit.chunk.source_url
    for hit in hits:
        if hit.chunk.source_url:
            return hit.chunk.source_url
    return None


def freshness_date(hits: Sequence[RetrievalHit]) -> str:
    """Oldest `fetched_at` across the chunks actually used.

    An answer assembled from one page fetched today and one fetched last year is only as
    current as the older page, so that is the date we must show.
    """
    dates = sorted(h.chunk.fetched_at for h in hits if h.chunk.fetched_at)
    return dates[0] if dates else ""


def build_messages(
    question: str,
    hits: Sequence[RetrievalHit],
    source_url: str | None,
    cfg: AppConfig | None = None,
) -> list[dict]:
    """System + user message. The user turn carries context, question, and the fixed link."""
    from mf_rag.prompts import SYSTEM_PROMPT

    user_parts = [
        render_context(hits, source_url),
        "",
        f"QUESTION: {question}",
        "",
    ]
    if source_url:
        user_parts.append(f"Include exactly this link verbatim: {source_url}")
    else:
        user_parts.append(
            "No source could be identified. Reply that the information is not available "
            "in the retrieved documents."
        )
    user_parts.append(f"End your reply with exactly: {freshness_line(freshness_date(hits))}")

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def generate_answer(
    question: str,
    hits: Sequence[RetrievalHit],
    cfg: AppConfig | None = None,
    llm: LLMClient | None = None,
    source_url: str | None = None,
) -> str:
    """Generate the answer body and append the freshness line ourselves.

    The model's text is stripped of any freshness line it invented before the real one is
    appended, so the date can only ever come from chunk metadata.
    """
    config = cfg or get_config()
    client = llm or get_llm(config)[0]
    url = source_url if source_url is not None else select_source_url(hits, config)

    body = client.complete(
        build_messages(question, hits, url, config),
        temperature=config.generation.temperature,
        max_tokens=config.generation.max_tokens,
    )
    body = strip_freshness(body or "").strip()
    return f"{body}\n\n{freshness_line(freshness_date(hits))}"


def _trace(
    query_text: str,
    intent: str,
    retrieval_trace: dict,
    hits: Sequence[RetrievalHit],
    source_url: str | None,
    text: str,
    latency_ms: int,
    validators_run: list[str] | None = None,
    repair_used: bool = False,
    pii_blocked: bool = False,
) -> Trace:
    return Trace(
        query_hash=hashlib.sha256(query_text.strip().encode("utf-8")).hexdigest()[:16],
        intent=intent,
        pii_blocked=pii_blocked,
        entity_detected=retrieval_trace.get("entity_detected"),
        where_filter=retrieval_trace.get("where_filter"),
        hits=retrieval_trace.get("hits", []),
        chunks_used=[h.chunk.chunk_id for h in hits],
        selected_source_url=source_url,
        validators_run=validators_run or [],
        repair_used=repair_used,
        latency_ms=latency_ms,
        answer=text,
    )


def answer_raw(
    question: str,
    cfg: AppConfig | None = None,
    llm: LLMClient | None = None,
) -> tuple[str, list[RetrievalHit], Trace]:
    """retrieve -> select -> generate. No guardrails yet; Phase 7 adds them.

    Returns `(text, hits, trace)`.
    """
    import time

    config = cfg or get_config()
    started = time.perf_counter()
    hits, retrieval_trace = retrieve_with_trace(question, cfg=config)
    source_url = select_source_url(hits, config)

    if not hits:
        # F5: nothing above the threshold. Say so honestly rather than generating.
        text = NOT_IN_SOURCES.format(link=source_url or EDUCATIONAL_URL)
        trace = _trace(
            question, "factual", retrieval_trace, hits, source_url, text,
            int((time.perf_counter() - started) * 1000),
        )
        return text, hits, trace

    text = generate_answer(question, hits, config, llm=llm, source_url=source_url)
    trace = _trace(
        question, "factual", retrieval_trace, hits, source_url, text,
        int((time.perf_counter() - started) * 1000),
    )
    return text, hits, trace


def answer(question: str, cfg: AppConfig | None = None, llm: LLMClient | None = None) -> Answer:
    """`answer_raw` wrapped in the public `Answer` contract."""
    text, hits, trace = answer_raw(question, cfg, llm)
    return Answer(
        text=text,
        source_url=trace.selected_source_url,
        fetched_at=freshness_date(hits),
        intent=trace.intent,
        trace=trace,
        is_refusal=not hits,
    )


__all__ = [
    "DISCLAIMER",
    "EDUCATIONAL_URL",
    "FACTSHEET_SOURCE_BY_SCHEME",
    "OFFICIAL_SOURCES",
    "SOURCE_URLS",
    "answer",
    "answer_raw",
    "build_messages",
    "freshness_date",
    "generate_answer",
    "official_source_ids",
    "select_source_url",
]
