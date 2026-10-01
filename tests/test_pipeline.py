"""End-to-end pipeline tests.

The important assertions are about *ordering*: PII must be blocked before the LLM is
called, and a non-compliant draft must go through a repair pass before the safe template.
A `SpyClient` is the only client used, so nothing here touches the network.
"""

from __future__ import annotations

import pytest
from conftest import needs_model

from mf_rag.answer import count_sentences, validate_against_hits
from mf_rag.generate import SOURCE_URLS
from mf_rag.llm import EchoClient, SpyClient
from mf_rag.prompts import extract_freshness, extract_urls
from mf_rag.pipeline import query, query_hash, run_with_client, safe_answer

EXIT_LOAD = "What is the exit load of HDFC Small Cap Fund?"
PAN = "ABCDE1234F"
EMAIL = "ravi.kumar@example.com"
GOOD_URL = "https://www.hdfcfund.com/explore/mutual-funds/hdfc-small-cap-fund/direct"


# ---------------------------------------------------------------- L1 runs first


def test_pii_is_blocked_before_the_llm_is_called():
    """The gate: a PAN must never reach a model, even a fake one."""
    spy = SpyClient()
    answer = run_with_client(f"My PAN is {PAN}, what is the exit load?", llm=spy)
    assert spy.called is False
    assert answer.trace.pii_blocked is True
    assert answer.intent == "pii"
    assert answer.is_refusal is True


def test_pii_never_appears_in_the_answer():
    spy = SpyClient()
    answer = run_with_client(f"My email is {EMAIL} and PAN {PAN}", llm=spy)
    assert PAN not in answer.text
    assert EMAIL not in answer.text


def test_pii_value_never_reaches_the_prompt():
    """Even if PII were somehow routed through, the prompt must not carry the value."""
    spy = SpyClient()
    run_with_client(f"account number 7788990011, exit load?", llm=spy)
    assert "7788990011" not in (spy.last_prompt or "")


# ---------------------------------------------------------------- L2 runs second


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Should I buy HDFC Small Cap Fund?", "advisory"),
        ("What is the CAGR of HDFC Large Cap Fund?", "returns"),
        ("How should I rebalance my portfolio?", "portfolio"),
        ("What is the exit load of Parag Parflex?", "out_of_corpus"),
    ],
)
def test_refused_intents_never_reach_the_llm(question, expected):
    spy = SpyClient()
    answer = run_with_client(question, llm=spy)
    assert spy.called is False
    assert answer.intent == expected
    assert answer.is_refusal is True
    assert answer.source_url in SOURCE_URLS.values()


def test_refusal_has_exactly_one_in_registry_url():
    for question in ("Should I buy this?", "What is the CAGR?", "Rebalance my portfolio."):
        answer = run_with_client(question, llm=SpyClient())
        assert extract_urls(answer.text) == [answer.source_url]
        assert answer.source_url in SOURCE_URLS.values()


# ----------------------------------------------------------------- the happy path


@needs_model
def test_factual_question_reaches_the_llm_and_returns_a_compliant_answer():
    spy = SpyClient(EchoClient())
    answer = run_with_client(EXIT_LOAD, llm=spy)
    assert spy.called is True
    assert answer.intent == "factual"
    assert answer.is_refusal is False
    assert count_sentences(answer.text) <= 3
    assert extract_urls(answer.text) == [answer.source_url]
    assert extract_freshness(answer.text) == answer.fetched_at


@needs_model
def test_cited_url_is_byte_identical_to_an_ingested_url():
    answer = run_with_client(EXIT_LOAD, llm=EchoClient())
    assert answer.source_url in SOURCE_URLS.values()


@needs_model
def test_freshness_equals_the_oldest_used_chunk():
    answer = run_with_client(EXIT_LOAD, llm=EchoClient())
    dates = sorted(d for d in [h.get("fetched_at") for h in answer.trace.hits] if d)
    if dates:
        assert answer.fetched_at == dates[0]


@needs_model
def test_trace_is_populated_and_never_stores_the_query():
    answer = run_with_client(EXIT_LOAD, llm=EchoClient())
    trace = answer.trace
    assert trace.query_hash == query_hash(EXIT_LOAD)
    assert trace.intent == "factual"
    assert trace.where_filter == {"scheme": "HDFC Small Cap Fund - Direct Growth"}
    assert trace.hits and "score" in trace.hits[0]
    assert trace.chunks_used
    assert trace.latency_ms >= 0
    assert "exit load" not in trace.query_hash


@needs_model
def test_trace_exposes_the_entity_filter_for_the_ui():
    answer = run_with_client(EXIT_LOAD, llm=EchoClient())
    assert answer.trace.entity_detected == "scheme"
    assert answer.trace.where_filter


# ------------------------------------------------------------------ the repair pass


@needs_model
def test_hallucinated_url_is_repaired_without_a_second_llm_call():
    """A model that invents a link must be corrected, not repeated."""
    spy = SpyClient()  # SpyClient with no inner returns https://example.invalid/spy
    answer = run_with_client(EXIT_LOAD, llm=spy)
    assert spy.calls == 1, "repair must not require a second generation"
    assert "example.invalid" not in answer.text
    assert extract_urls(answer.text) == [answer.source_url]
    assert answer.trace.repair_used is True


@needs_model
def test_bad_url_triggers_the_repair_pass():
    assert "official_preference_not_honoured" in run_with_client(
        EXIT_LOAD, llm=SpyClient()
    ).trace.validators_run


@needs_model
def test_freshness_line_is_supplied_by_the_system_not_the_model():
    """The date is appended outside the LLM, so a client cannot omit or forge it.

    This is why no `freshness_missing` violation is recorded: the guarantee is structural
    rather than something the validator has to catch after the fact.
    """
    class NoDate(EchoClient):
        def complete(self, messages, temperature=0.0, max_tokens=220):
            return f"Exit load of 1% if redeemed within 1 year. See: {GOOD_URL}"

    answer = run_with_client(EXIT_LOAD, llm=NoDate())
    assert extract_freshness(answer.text) == answer.fetched_at
    assert answer.text.count("Last updated from sources:") == 1


@needs_model
def test_a_client_supplied_wrong_date_is_overwritten():
    class WrongDate(EchoClient):
        def complete(self, messages, temperature=0.0, max_tokens=220):
            return f"Exit load is 1%. See: {GOOD_URL}\n\nLast updated from sources: 1999-01-01"

    answer = run_with_client(EXIT_LOAD, llm=WrongDate())
    assert "1999-01-01" not in answer.text
    assert extract_freshness(answer.text) == answer.fetched_at


@needs_model
def test_advice_in_a_draft_is_replaced_by_the_refusal_template():
    class Advising(EchoClient):
        def complete(self, messages, temperature=0.0, max_tokens=220):
            return f"You should buy this fund, it is the best choice for you. See: {GOOD_URL}"

    answer = run_with_client(EXIT_LOAD, llm=Advising())
    assert answer.is_refusal is True
    assert "can't" in answer.text
    assert "You should buy" not in answer.text


@needs_model
def test_projection_in_a_draft_is_replaced_by_the_returns_refusal():
    class Projecting(EchoClient):
        def complete(self, messages, temperature=0.0, max_tokens=220):
            return f"The projected return will give you 20% CAGR. See: {GOOD_URL}"

    answer = run_with_client(EXIT_LOAD, llm=Projecting())
    assert answer.is_refusal is True
    assert "don't compute or compare returns" in answer.text


@needs_model
def test_overlong_draft_is_truncated_and_stays_compliant():
    class Rambling(EchoClient):
        def complete(self, messages, temperature=0.0, max_tokens=220):
            return f"One. Two. Three. Four. Five. See: {GOOD_URL}"

    answer = run_with_client(EXIT_LOAD, llm=Rambling())
    assert count_sentences(answer.text) <= 3
    assert extract_urls(answer.text) == [answer.source_url]
    assert answer.trace.repair_used is True


@needs_model
def test_safe_answer_is_compliant_by_construction():
    hits, _ = __import__("mf_rag.retrieve", fromlist=["retrieve"]).retrieve_with_trace(EXIT_LOAD)
    text = safe_answer(hits, EXIT_LOAD)
    assert count_sentences(text) <= 3
    assert len(extract_urls(text)) == 1
    assert extract_urls(text)[0] in SOURCE_URLS.values()
    assert extract_freshness(text)


# ---------------------------------------------------------------------- no hits


@needs_model
def test_unanswerable_question_returns_not_in_sources():
    """Below the min_score floor the honest answer is "not in my sources"."""
    answer = run_with_client("qwertyuiop asdfgh zxcvbn", llm=EchoClient())
    assert answer.is_refusal is True
    assert "min_score_floor" in answer.trace.validators_run
    assert answer.source_url in SOURCE_URLS.values()


# ------------------------------------------------------------------------ query()


@needs_model
def test_query_is_the_public_entry_point():
    answer = query(EXIT_LOAD, llm=EchoClient())
    assert answer.text
    assert answer.trace.query_hash


def test_empty_query_is_handled():
    answer = query("", llm=SpyClient())
    assert answer.is_refusal is True


def test_query_hash_is_stable_and_distinct():
    assert query_hash(EXIT_LOAD) == query_hash(f"  {EXIT_LOAD}  ")
    assert query_hash(EXIT_LOAD) != query_hash("something else")


@needs_model
def test_every_emitted_answer_satisfies_the_response_contract():
    """Sweep the real gold questions through the pipeline and check PRD 7 on each."""
    import json
    from pathlib import Path

    questions = json.loads(
        (Path(__file__).resolve().parents[1] / "eval" / "questions.json").read_text(encoding="utf-8")
    )["questions"]
    for gold in questions:
        answer = run_with_client(gold["question"], llm=EchoClient())
        assert extract_urls(answer.text) == [answer.source_url], gold["id"]
        assert answer.source_url in SOURCE_URLS.values(), gold["id"]
        assert count_sentences(answer.text) <= 3, gold["id"]
        assert extract_freshness(answer.text) == answer.fetched_at, gold["id"]
