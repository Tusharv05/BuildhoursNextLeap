"""The runtime path: guardrails -> retrieval -> generation -> validation (Phase 7).

    query(text, cfg) -> Answer

    L1 PII      blocked?               -> PII_WARNING
    L2 intent   advisory/portfolio/
                returns/out_of_corpus? -> refusal template
    F5 retrieve nothing above floor?   -> NOT_IN_SOURCES
    F6 generate                         -> draft
    L3 validate pass                    -> Answer
                 one repair pass       -> revalidate
                 still failing         -> safe system-composed answer

The ordering is the security property. PII is blocked before `embed_query`, so a PAN never
becomes a vector; and intent is classified before retrieval, so a returns question never
reaches the model at all. The `Trace` returned by every path carries a SHA-256 hash of the
query and never the query text (NFR-4).
"""

from __future__ import annotations

import hashlib
import time
from typing import Sequence

from mf_rag.answer import (
    ADVICE_LEXICON,
    RETURNS_LEXICON,
    ValidationResult,
    validate_against_hits,
)
from mf_rag.config import AppConfig, get_config
from mf_rag.generate import (
    EDUCATIONAL_URL,
    SOURCE_URLS,
    _trace,
    freshness_date,
    generate_answer,
    select_source_url,
)
from mf_rag.guardrails import (
    classify_intent,
    guard_input,
    pii_answer,
    refusal_for,
    refusal_link,
)
from mf_rag.llm import LLMClient, get_llm
from mf_rag.models import Answer, RetrievalHit
from mf_rag.prompts import freshness_line
from mf_rag.retrieve import retrieve_with_trace

REFUSAL_INTENTS = ("advisory", "portfolio", "returns", "out_of_corpus")

# Validators recorded in the trace, in the order PRD 7 lists them.
VALIDATOR_ORDER = (
    "sentence_count",
    "url_count",
    "url_in_ingested_sources",
    "official_preference",
    "advice_language",
    "returns_language",
    "freshness",
)


def query_hash(text: str) -> str:
    """Stable id for a question. The trace stores this, never the text (NFR-4)."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def _lexicon_hits(patterns: Sequence[str], text: str) -> list[str]:
    import re

    return [p for p in patterns if re.search(p, text, re.IGNORECASE)]


def safe_answer(hits: Sequence[RetrievalHit], question: str, cfg: AppConfig | None = None) -> str:
    """A system-composed answer that is compliant by construction.

    Used when the model's draft cannot be repaired. It quotes the top chunk's own text and
    appends the system-chosen URL and the computed date, so it satisfies every L3 rule
    without asking a model to do anything.
    """
    from mf_rag.llm import _best_sentence

    if not hits:
        return ""
    top = hits[0].chunk
    fact = _best_sentence(top.text, question)
    url = select_source_url(hits, cfg) or ""
    body = fact or str(top.text).strip().split("\n")[0]
    body = body.rstrip(".")
    return f"{body}. See: {url}\n\n{freshness_line(freshness_date(hits))}"


def _finalise(
    text: str,
    hits: Sequence[RetrievalHit],
    question: str,
    intent: str,
    retrieval_trace: dict,
    validators_run: list[str],
    repair_used: bool,
    latency_ms: int,
    cfg: AppConfig | None = None,
    is_refusal: bool = False,
) -> Answer:
    dates = sorted(h.chunk.fetched_at for h in hits if h.chunk.fetched_at)
    return Answer(
        text=text,
        source_url=select_source_url(hits, cfg) or (refusal_link(intent, question) if is_refusal else None),
        fetched_at=dates[0] if dates else "",
        intent=intent,
        trace=_trace(
            question, intent, retrieval_trace, hits, select_source_url(hits, cfg),
            text, latency_ms, validators_run=validators_run, repair_used=repair_used,
        ),
        is_refusal=is_refusal,
    )


def run_with_client(
    text: str, llm: LLMClient | None = None, cfg: AppConfig | None = None
) -> Answer:
    """`query` with an explicit client, so tests and the guardrail eval can inject a spy."""
    config = cfg or get_config()
    started = time.perf_counter()

    # --- L1: PII, before embedding and before any LLM call -----------------------
    guard = guard_input(text, enabled=config.guardrails.pii_block)
    if not guard.allowed:
        answer = pii_answer()
        answer.trace.query_hash = query_hash(text)
        answer.trace.latency_ms = int((time.perf_counter() - started) * 1000)
        return answer

    # --- L2: intent, before retrieval -------------------------------------------
    intent = classify_intent(text)
    if intent in REFUSAL_INTENTS:
        answer = refusal_for(intent, text)
        answer.trace.query_hash = query_hash(text)
        answer.trace.latency_ms = int((time.perf_counter() - started) * 1000)
        return answer

    # --- F5: retrieve -----------------------------------------------------------
    hits, retrieval_trace = retrieve_with_trace(text, cfg=config)

    if not hits:
        url = retrieval_trace.get("hits") or EDUCATIONAL_URL
        link = url if isinstance(url, str) else EDUCATIONAL_URL
        from mf_rag.prompts import NOT_IN_SOURCES

        body = NOT_IN_SOURCES.format(link=link)
        return _finalise(
            body, hits, text, intent, retrieval_trace, ["min_score_floor"],
            False, int((time.perf_counter() - started) * 1000), config, is_refusal=True,
        )

    # --- F6: generate -----------------------------------------------------------
    source_url = select_source_url(hits, config)
    client = llm if llm is not None else get_llm(config)[0]
    draft = generate_answer(text, hits, config, llm=client, source_url=source_url)

    # --- L3: validate, then one repair pass -------------------------------------
    result = validate_against_hits(draft, hits, max_sentences=config.generation.max_sentences)
    validators = list(result.violations)
    repair_used = False

    # Advice/returns are not repaired in place: the whole answer is replaced by the
    # refusal template, because a factual sentence with advice spliced into it is still
    # advice (PRD 7 / ARCHITECTURE 6.4).
    if "advice_language" in result.violations:
        return _refuse_with_fact("advisory", text, hits, retrieval_trace, validators, started, config)
    if "returns_language" in result.violations:
        return _refuse_with_fact("returns", text, hits, retrieval_trace, validators, started, config)

    if not result.ok:
        repair_used = True
        result = validate_against_hits(
            result.repaired_text, hits, max_sentences=config.generation.max_sentences
        )
        validators += result.violations
        if not result.ok:
            # Second failure: emit a system-composed answer, compliant by construction.
            fallback = safe_answer(hits, text, config)
            final = validate_against_hits(
                fallback, hits, max_sentences=config.generation.max_sentences
            )
            if not final.ok:
                fallback = (
                    f"{safe_answer(hits, text, config).splitlines()[0]}\n\n"
                    f"{freshness_line(freshness_date(hits))}"
                )
            validators.append("safe_template")
            return _finalise(
                fallback, hits, text, intent, retrieval_trace, validators, True,
                int((time.perf_counter() - started) * 1000), config,
            )
        draft = result.repaired_text

    return _finalise(
        draft, hits, text, intent, retrieval_trace, validators, repair_used,
        int((time.perf_counter() - started) * 1000), config,
    )


def _refuse_with_fact(
    intent: str,
    question: str,
    hits: Sequence[RetrievalHit],
    retrieval_trace: dict,
    validators: list[str],
    started: float,
    cfg: AppConfig,
) -> Answer:
    """Replace a non-compliant draft with the refusal template for `intent`.

    The link is the official page for the scheme actually asked about, so the refusal is
    still useful rather than a dead end.
    """
    from mf_rag.guardrails import refusal_template

    link = refusal_link(intent, question)
    text = refusal_template(intent).format(link=link)
    return _finalise(
        text, hits, question, intent, retrieval_trace, validators, False,
        int((time.perf_counter() - started) * 1000), cfg, is_refusal=True,
    )


def query(text: str, cfg: AppConfig | None = None, llm: LLMClient | None = None) -> Answer:
    """The runtime entry point. See the module docstring for the order of operations."""
    if not text or not text.strip():
        from mf_rag.guardrails import refusal_template

        return refusal_for("out_of_corpus", text or "")
    return run_with_client(text, llm=llm, cfg=cfg)


__all__ = [
    "ADVICE_LEXICON",
    "RETURNS_LEXICON",
    "REFUSAL_INTENTS",
    "VALIDATOR_ORDER",
    "ValidationResult",
    "query",
    "query_hash",
    "run_with_client",
    "safe_answer",
]
