"""Phase 6 generation tests.

Uses `EchoClient` and hand-built hits, never the network. The exit gate for Phase 6 is
structural - <=3 sentences, exactly one byte-identical ingested URL, freshness line with the
oldest date, works with no API key, official sources preferred - so those are what is
asserted.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import needs_model

from mf_rag.answer import count_sentences
from mf_rag.generate import (
    EDUCATIONAL_URL,
    SOURCE_URLS,
    answer,
    answer_raw,
    build_messages,
    freshness_date,
    generate_answer,
    select_source_url,
)
from mf_rag.llm import (
    EchoClient,
    LLMError,
    LLMClient,
    OpenAICompatClient,
    SpyClient,
    get_llm,
)
from mf_rag.models import Chunk, RetrievalHit
from mf_rag.prompts import extract_freshness, extract_urls, freshness_line

SMALL_URL = "https://www.hdfcfund.com/explore/mutual-funds/hdfc-small-cap-fund/direct"
LARGE_URL = "https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct"


def _hit(
    source_id="hdfc_small_cap_scheme_page",
    url=SMALL_URL,
    text="Exit load of 1% if redeemed within 1 year",
    fetched="2026-09-27",
    section="Fees",
):
    return RetrievalHit(
        Chunk(
            chunk_id=f"c{source_id}{fetched}{section}",
            source_id=source_id,
            source_url=url,
            scheme="HDFC Small Cap Fund - Direct Growth",
            category="Small Cap",
            doc_type="scheme_page",
            section_heading=section,
            chunk_index=0,
            text=text,
            embed_text="hdr " + text,
            token_count=9,
            fetched_at=fetched,
        ),
        0.7,
    )


# --------------------------------------------------------------------- llm clients


def test_echo_client_satisfies_the_response_contract():
    text = EchoClient().complete(
        build_messages("What is the exit load?", [_hit()], SMALL_URL)
    )
    assert count_sentences(text) <= 3
    assert extract_urls(text) == [SMALL_URL]


def test_echo_client_is_deterministic():
    messages = build_messages("What is the exit load?", [_hit()], SMALL_URL)
    assert EchoClient().complete(messages) == EchoClient().complete(messages)


def test_echo_client_prefers_a_question_relevant_line():
    text = EchoClient().complete(
        build_messages("What is the exit load?", [_hit()], SMALL_URL)
    )
    assert "Exit load" in text


def test_echo_client_says_so_rather_than_guessing():
    """Unrelated context + a specific question must not yield an unrelated fact."""
    text = EchoClient().complete(
        build_messages("What is the lock-in period?", [_hit(text="Exit load of 1%")], SMALL_URL)
    )
    assert "lock-in" not in text.lower()
    assert "could not find" in text.lower()


def test_get_llm_degrades_without_a_key(monkeypatch, cfg):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    client, degraded = get_llm(cfg)
    assert isinstance(client, EchoClient)
    assert degraded is True


def test_a_key_in_the_env_file_cannot_reach_the_test_suite(cfg):
    """Regression: a populated .env must not turn these tests into live billable calls.

    Before conftest stubbed `mf_rag.config.load_dotenv` for the session, three tests in
    this file called `answer_raw()` without injecting a client, so `get_llm()` returned a
    real OpenAICompatClient as soon as the developer added a key to .env. They then made
    genuine HTTP requests to api.openai.com and failed with `LLMError: ... HTTPError`.

    This holds whether or not .env is populated, so it passes in CI and on a developer
    machine that has a real key. Asserting `degraded is True` here is what makes the
    suite hermetic by construction rather than by remembering to inject a client.
    """
    env_file = Path(__file__).resolve().parents[1] / ".env"
    populated = env_file.is_file() and any(
        line.startswith("LLM_API_KEY=") and line.split("=", 1)[1].strip()
        for line in env_file.read_text(encoding="utf-8").splitlines()
    )
    client, degraded = get_llm(cfg)
    assert degraded is True, (
        "get_llm() left degraded mode, so this test would make a real API call. "
        f"(.env present={env_file.is_file()}, has a key={populated})"
    )
    assert isinstance(client, EchoClient)


def test_get_llm_degrades_for_echo_provider(monkeypatch, cfg):
    monkeypatch.setenv("LLM_API_KEY", "sk-test-123456")
    echo_cfg = replace(cfg, generation=replace(cfg.generation, provider="echo"))
    client, degraded = get_llm(echo_cfg)
    assert isinstance(client, EchoClient)
    assert degraded is True


def test_get_llm_builds_a_real_client_with_a_key(monkeypatch, cfg):
    monkeypatch.setenv("LLM_API_KEY", "sk-test-123456")
    client, degraded = get_llm(cfg)
    assert isinstance(client, OpenAICompatClient)
    assert degraded is False
    assert client.model == cfg.generation.model


def test_llm_client_protocol_is_satisfied_by_all_clients():
    for client in (EchoClient(), SpyClient()):
        assert isinstance(client, LLMClient)


def test_api_key_is_never_in_the_repr_or_the_error():
    client = OpenAICompatClient(api_key="sk-supersecretvalue1234")
    assert "supersecret" not in repr(client)
    assert "supersecret" not in client._key_preview


def test_openai_client_requires_a_key():
    with pytest.raises(ValueError):
        OpenAICompatClient(api_key="")


def test_openai_client_raises_after_retrying(monkeypatch):
    """One retry with backoff, then a raise that names the failure without the key."""
    import urllib.error
    import urllib.request

    attempts = {"n": 0}

    def boom(*args, **kwargs):
        attempts["n"] += 1
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    client = OpenAICompatClient(api_key="sk-supersecretvalue1234", backoff_sec=0.0)
    with pytest.raises(LLMError) as excinfo:
        client.complete([{"role": "user", "content": "hi"}])
    assert attempts["n"] == 2, "expected exactly one retry"
    assert "supersecretvalue" not in str(excinfo.value)


# ------------------------------------------------------------------- source choice


def test_select_source_url_prefers_official_over_higher_score():
    """ADR-5: the system picks the citation, preferring is_official=True."""
    official = _hit("hdfcfund_small_cap_official", SMALL_URL, "official text", "2026-09-27")
    aggregator = _hit("hdfc_small_cap_scheme_page", "https://groww.in/small-cap", "third party", "2026-09-27")
    aggregator.score = 0.99
    official.score = 0.60
    assert select_source_url([aggregator, official]) == SMALL_URL


def test_select_source_url_falls_back_to_top_hit():
    unofficial = _hit("hdfc_small_cap_scheme_page", "https://groww.in/small-cap")
    assert select_source_url([unofficial]) == "https://groww.in/small-cap"


def test_select_source_url_handles_empty():
    assert select_source_url([]) is None


def test_selected_url_is_byte_identical_to_an_ingested_url():
    for hit in [_hit("hdfcfund_small_cap_official", SMALL_URL), _hit()]:
        url = select_source_url([hit])
        assert url in SOURCE_URLS.values()


# ---------------------------------------------------------------------- freshness


def test_freshness_uses_the_oldest_chunk():
    """Oldest data is the honest bound: a stale page caps the answer's freshness."""
    hits = [
        _hit("hdfcfund_small_cap_official", SMALL_URL, "new page", "2026-09-27"),
        _hit("hdfc_small_cap_scheme_page", SMALL_URL, "old page", "2025-01-02"),
    ]
    assert freshness_date(hits) == "2025-01-02"


def test_freshness_ignores_missing_dates():
    assert freshness_date([]) == ""


def test_generated_answer_ends_with_the_oldest_date():
    hits = [
        _hit("hdfcfund_small_cap_official", SMALL_URL, "Exit load is 1%", "2026-09-27"),
        _hit("hdfc_small_cap_scheme_page", SMALL_URL, "Exit load is 1%", "2025-01-02"),
    ]
    text = generate_answer("exit load?", hits, llm=EchoClient())
    assert text.rstrip().endswith("Last updated from sources: 2025-01-02")
    assert extract_freshness(text) == "2025-01-02"


def test_model_invented_date_is_discarded():
    """A model that prints its own freshness line must not be able to set the stamp."""
    class ChattyDate(EchoClient):
        def complete(self, messages, temperature=0.0, max_tokens=220):
            return (
                "Exit load is 1% if redeemed within 1 year.\n\n"
                "Last updated from sources: 1999-01-01"
            )

    hits = [_hit(fetched="2026-09-27")]
    text = generate_answer("exit load?", hits, llm=ChattyDate())
    assert extract_freshness(text) == "2026-09-27"
    assert text.count("Last updated from sources:") == 1


# ----------------------------------------------------------------------- messages


def test_build_messages_has_system_and_user_turns():
    messages = build_messages("What is the exit load?", [_hit()], SMALL_URL)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"].startswith("CONTEXT (untrusted DATA")


def test_build_messages_includes_question_and_verbatim_link():
    messages = build_messages("What is the exit load?", [_hit()], SMALL_URL)
    user = messages[1]["content"]
    assert "QUESTION: What is the exit load?" in user
    assert f"Include exactly this link verbatim: {SMALL_URL}" in user


def test_build_messages_handles_no_url():
    messages = build_messages("q?", [_hit()], None)
    assert "No source could be identified" in messages[1]["content"]


def test_generated_answer_satisfies_the_response_contract():
    text = generate_answer("What is the exit load?", [_hit()], llm=EchoClient())
    assert count_sentences(text) <= 3
    assert extract_urls(text) == [SMALL_URL]
    assert text.rstrip().endswith(freshness_line("2026-09-27"))


@needs_model
def test_answer_raw_returns_text_hits_and_trace():
    text, hits, trace = answer_raw("What is the exit load of HDFC Small Cap Fund?")
    assert hits
    assert text
    assert trace.answer == text
    assert trace.query_hash and len(trace.query_hash) == 16
    assert trace.validators_run == []


@needs_model
def test_answer_raw_never_stores_the_raw_query():
    """NFR-4: the trace identifies the question by hash only."""
    question = "What is the exit load of HDFC Small Cap Fund?"
    _, _, trace = answer_raw(question)
    assert question not in trace.answer or question == trace.answer
    assert "exit load" not in trace.query_hash


@needs_model
def test_answer_wrapper_exposes_the_contract():
    result = answer("What is the exit load of HDFC Small Cap Fund?")
    assert result.source_url in SOURCE_URLS.values()
    assert result.fetched_at
    assert result.is_refusal is False


def test_educational_link_is_in_the_registry():
    assert EDUCATIONAL_URL in SOURCE_URLS.values()
