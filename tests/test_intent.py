"""L2 intent classification tests.

Intent decides whether we answer at all, so the tests care about two failure modes:
missing a refusal (the dangerous direction) and over-refusing a plain factual question
(which makes the product useless).
"""

from __future__ import annotations

import pytest

from mf_rag.guardrails import (
    INTENTS,
    classify_intent,
    refusal_for,
    refusal_link,
    refusal_template,
)
from mf_rag.generate import SOURCE_URLS


@pytest.mark.parametrize(
    "text",
    [
        "Should I buy HDFC Small Cap Fund?",
        "Should we invest in HDFC ELSS?",
        "Which HDFC scheme do you recommend?",
        "Is HDFC ELSS worth buying?",
        "Which fund is best for long term growth?",
        "Is HDFC Large Cap a good fund to invest in?",
        "Help me choose a scheme",
        "What is a good fund to invest in?",
    ],
)
def test_advisory_is_detected(text):
    assert classify_intent(text) == "advisory"


@pytest.mark.parametrize(
    "text",
    [
        "How should I rebalance my portfolio?",
        "What percentage should I allocate to HDFC Balanced Advantage in my portfolio?",
        "Can you make a SIP plan for me?",
        "Review my portfolio allocation",
        "How should I split my money across these schemes?",
        "Should I add HDFC Flexi Cap to my holdings?",
    ],
)
def test_portfolio_is_detected(text):
    assert classify_intent(text) == "portfolio"


@pytest.mark.parametrize(
    "text",
    [
        "What is the CAGR of HDFC Large Cap Fund?",
        "How much will I get after 10 years?",
        "Which HDFC scheme has given the best returns?",
        "Show me the projected value in 5 years",
        "What is the expected return?",
        "Compare the performance of HDFC Small Cap and HDFC Large Cap",
    ],
)
def test_returns_is_detected(text):
    assert classify_intent(text) == "returns"


@pytest.mark.parametrize(
    "text",
    [
        "What is the exit load of Parag Parflex?",
        "Tell me about SBI Bluechip",
        "What does Mirae Asset Large Cap charge?",
    ],
)
def test_out_of_corpus_is_detected(text):
    assert classify_intent(text) == "out_of_corpus"


@pytest.mark.parametrize(
    "text",
    [
        "What is the exit load of HDFC Small Cap Fund?",
        "What is the ELSS lock-in period?",
        "What is the expense ratio of HDFC Large Cap Fund?",
        "How do I download the capital gains statement?",
        "What is the minimum SIP amount for HDFC ELSS?",
        "Who is the trustee of HDFC Mutual Fund?",
        "What is the riskometer level of HDFC Flexi Cap?",
    ],
)
def test_factual_questions_are_not_refused(text):
    assert classify_intent(text) == "factual"


def test_returns_wins_over_advisory():
    """'which fund gave the best returns' is a returns question, not an advisory one."""
    assert classify_intent("Which HDFC fund should I buy for the best returns?") == "returns"


def test_portfolio_wins_over_advisory():
    assert classify_intent("Should I rebalance my portfolio?") == "portfolio"


def test_every_returned_intent_is_declared():
    for text in ["should i buy", "my portfolio", "cagr", "Parag Parflex", "exit load"]:
        assert classify_intent(text) in INTENTS


@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_text_is_factual_not_a_crash(text):
    assert classify_intent(text or "") == "factual"


# ------------------------------------------------------------------- refusals


@pytest.mark.parametrize("intent", ["advisory", "portfolio", "returns", "out_of_corpus"])
def test_refusal_uses_the_right_template_and_one_registry_link(intent):
    question = "Should I buy HDFC Small Cap Fund?"
    answer = refusal_for(intent, question)
    assert answer.is_refusal is True
    assert answer.intent == intent
    assert answer.text == refusal_template(intent).format(link=answer.source_url)
    assert answer.source_url in SOURCE_URLS.values()
    assert "{link}" not in answer.text


def test_refusal_contains_exactly_one_url():
    from mf_rag.prompts import extract_urls

    for intent in ("advisory", "portfolio", "returns", "out_of_corpus"):
        assert len(extract_urls(refusal_for(intent, "Should I buy this?").text)) == 1


def test_returns_refusal_links_the_scheme_that_was_asked_about():
    url = refusal_link("returns", "What is the CAGR of HDFC Large Cap Fund?")
    assert "large-cap" in url


def test_returns_refusal_without_a_named_scheme_is_still_in_registry():
    assert refusal_link("returns", "What are the returns?") in SOURCE_URLS.values()


def test_out_of_corpus_points_at_the_regulator_directory():
    """The fund we do not have will not be on an HDFC page, so do not send the user there."""
    assert refusal_link("out_of_corpus", "What about Parag Parflex?") in SOURCE_URLS.values()


def test_advisory_refusal_never_advises():
    from mf_rag.answer import _lexicon_hits, ADVICE_LEXICON

    text = refusal_for("advisory", "Should I buy HDFC Small Cap?").text
    assert _lexicon_hits(ADVICE_LEXICON, text) == []
