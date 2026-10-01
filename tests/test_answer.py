"""L3 output-validator tests.

The interesting cases are the ones where naive checking gives a wrong answer: `1.5%` is one
sentence, not three, and a negated "I can't recommend" is not advice.
"""

from __future__ import annotations

import pytest

from mf_rag.answer import (
    ADVICE_LEXICON,
    RETURNS_LEXICON,
    ValidationResult,
    _lexicon_hits,
    count_sentences,
    has_substantive_content,
    substantive_content,
    truncate_to_sentences,
    validate,
    validate_against_hits,
)
from mf_rag.generate import SOURCE_URLS
from mf_rag.models import Chunk, RetrievalHit
from mf_rag.prompts import extract_urls, freshness_line

GOOD_URL = "https://www.hdfcfund.com/explore/mutual-funds/hdfc-small-cap-fund/direct"
GOOD_DATE = "2026-09-27"


def _hit(source_id="hdfcfund_small_cap_official", url=GOOD_URL, fetched=GOOD_DATE):
    return RetrievalHit(
        Chunk(
            chunk_id="c1",
            source_id=source_id,
            source_url=url,
            scheme="HDFC Small Cap Fund - Direct Growth",
            category="Small Cap",
            doc_type="scheme_page",
            section_heading="Exit load",
            chunk_index=0,
            text="Exit load of 1% if redeemed within 1 year",
            embed_text="x",
            token_count=9,
            fetched_at=fetched,
        ),
        0.7,
    )


# ------------------------------------------------------------- sentence counting


@pytest.mark.parametrize(
    "text,expected",
    [
        ("", 0),
        ("   ", 0),
        ("One sentence.", 1),
        ("One. Two. Three.", 3),
        ("The exit load is 1.5% for the direct plan.", 1),
        ("Return was 12.75% last year. Lock-in is 3 years.", 2),
        ("Ratio 1.03%. Lock-in 3 years.", 2),
        ("See e.g. the scheme factsheet for details.", 1),
        ("First line.\nSecond line.", 2),
        ("No trailing period", 1),
        # The freshness stamp is a required suffix, not prose: counting it would cap the
        # answer body at 2 sentences while the contract says 3.
        (f"Exit load is 1%. See: {GOOD_URL}\n\n{freshness_line(GOOD_DATE)}", 2),
        (f"One. Two.\n\n{freshness_line(GOOD_DATE)}", 2),
    ],
)
def test_count_sentences(text, expected):
    assert count_sentences(text) == expected


def test_count_sentences_does_not_split_a_decimal():
    """The naive `split('.')` bug that would force a spurious repair on every percentage."""
    assert count_sentences("The expense ratio is 1.03%.") == 1


def test_truncate_keeps_the_first_n_sentences():
    assert truncate_to_sentences("A one. B two. C three. D four.", 3) == "A one. B two. C three"


def test_truncate_drops_the_freshness_line():
    text = f"A one. B two. C three. D four.\n\n{freshness_line(GOOD_DATE)}"
    assert "Last updated" not in truncate_to_sentences(text, 3)


# ---------------------------------------------------------------------- lexicons


@pytest.mark.parametrize(
    "text",
    [
        "You should buy this fund.",
        "I recommend HDFC Small Cap.",
        "This is the best choice for you.",
        "This is ideal for you.",
        "You must buy now.",
        "This scheme is suitable for you.",
        "Worth buying for a 10 year horizon.",
    ],
)
def test_advice_lexicon_catches_advice(text):
    assert _lexicon_hits(ADVICE_LEXICON, text)


@pytest.mark.parametrize(
    "text",
    [
        "I only share factual information - I can't give investment advice or recommend a scheme.",
        "I don't compute or compare returns.",
        "I could not find that in the documents I have, so I won't guess.",
        "The exit load of the direct plan is 1% if redeemed within 1 year.",
    ],
)
def test_negated_mentions_are_not_flagged(text):
    """Otherwise the system would reject its own correct refusal template."""
    assert _lexicon_hits(ADVICE_LEXICON, text) == []


@pytest.mark.parametrize(
    "text",
    [
        "The fund will give you 20% returns.",
        "The expected return is high.",
        "Its CAGR of 14% is good.",
        "The projected value is X.",
        "That is a 12% return scheme.",
    ],
)
def test_returns_lexicon_catches_projections(text):
    assert _lexicon_hits(RETURNS_LEXICON, text)


def test_returns_lexicon_ignores_past_performance_facts():
    text = "As on 2026-09-27 the scheme's past performance is published in the factsheet."
    assert _lexicon_hits(RETURNS_LEXICON, text) == []


# --------------------------------------------------------------------- validation


def _good() -> str:
    return f"Exit load is 1% if redeemed within 1 year. See: {GOOD_URL}\n\n{freshness_line(GOOD_DATE)}"


def test_a_compliant_answer_passes():
    result = validate(_good(), expected_url=GOOD_URL, expected_date=GOOD_DATE)
    assert result.ok
    assert result.violations == []
    assert result.repaired is False


def test_too_many_sentences_is_repaired_by_truncation():
    draft = f"A one. B two. C three. D four. See: {GOOD_URL}\n\n{freshness_line(GOOD_DATE)}"
    result = validate(draft, expected_url=GOOD_URL, expected_date=GOOD_DATE)
    assert not result.ok
    assert "sentence_count>3" in result.violations
    assert count_sentences(result.repaired_text) <= 3


def test_invented_url_is_replaced_with_the_ingested_one():
    draft = f"Exit load is 1%. See: https://invented.example.com/fake\n\n{freshness_line(GOOD_DATE)}"
    result = validate(draft, expected_url=GOOD_URL, expected_date=GOOD_DATE)
    assert "url_not_in_ingested_sources" in result.violations
    assert "official_preference_not_honoured" in result.violations
    assert extract_urls(result.repaired_text) == [GOOD_URL]


def test_non_official_link_is_swapped_for_the_official_one():
    unofficial = "https://groww.in/mutual-funds/hdfc-small-cap"
    draft = f"Exit load is 1%. See: {unofficial}\n\n{freshness_line(GOOD_DATE)}"
    result = validate(draft, expected_url=GOOD_URL, expected_date=GOOD_DATE)
    assert extract_urls(result.repaired_text) == [GOOD_URL]
    assert "official_preference_not_honoured" in result.violations


def test_two_urls_collapses_to_one():
    other = "https://www.hdfcfund.com/services/faqs"
    draft = f"Exit load is 1%. See: {GOOD_URL} and {other}\n\n{freshness_line(GOOD_DATE)}"
    result = validate(draft, expected_url=GOOD_URL, expected_date=GOOD_DATE)
    assert extract_urls(result.repaired_text) == [GOOD_URL]
    assert any(v.startswith("url_count=") for v in result.violations)


def test_missing_freshness_line_is_appended():
    draft = f"Exit load is 1%. See: {GOOD_URL}"
    result = validate(draft, expected_url=GOOD_URL, expected_date=GOOD_DATE)
    assert "freshness_missing" in result.violations
    assert result.repaired_text.rstrip().endswith(freshness_line(GOOD_DATE))


def test_wrong_date_is_corrected_not_appended_again():
    draft = f"Exit load is 1%. See: {GOOD_URL}\n\n{freshness_line('1999-01-01')}"
    result = validate(draft, expected_url=GOOD_URL, expected_date=GOOD_DATE)
    assert "freshness_mismatch" in result.violations
    assert result.repaired_text.count("Last updated from sources:") == 1
    assert GOOD_DATE in result.repaired_text


def test_advice_language_is_reported_for_the_pipeline_to_substitute_a_template():
    draft = f"You should buy this fund. See: {GOOD_URL}\n\n{freshness_line(GOOD_DATE)}"
    result = validate(draft, expected_url=GOOD_URL, expected_date=GOOD_DATE)
    assert "advice_language" in result.violations


def test_returns_language_is_reported():
    draft = f"The fund will give 20% returns. See: {GOOD_URL}\n\n{freshness_line(GOOD_DATE)}"
    result = validate(draft, expected_url=GOOD_URL, expected_date=GOOD_DATE)
    assert "returns_language" in result.violations


def test_validate_against_hits_uses_the_oldest_date():
    hits = [_hit(fetched="2026-09-27"), _hit(source_id="other", fetched="2025-01-02")]
    result = validate_against_hits(f"Exit load is 1%. See: {GOOD_URL}", hits)
    assert result.repaired_text.rstrip().endswith("Last updated from sources: 2025-01-02")


def test_validate_against_hits_picks_the_official_url():
    hits = [
        _hit("hdfc_small_cap_scheme_page", "https://groww.in/small-cap"),
        _hit("hdfcfund_small_cap_official", GOOD_URL),
    ]
    result = validate_against_hits("Exit load is 1%. See: https://groww.in/small-cap", hits)
    assert extract_urls(result.repaired_text) == [GOOD_URL]


def test_validation_result_is_falsy_when_failing():
    assert not validate("no url no date", expected_url=GOOD_URL, expected_date=GOOD_DATE)


def test_validate_survives_empty_input():
    result = validate("", expected_url=GOOD_URL, expected_date=GOOD_DATE)
    assert not result.ok
    assert isinstance(result, ValidationResult)

# --- substantive content -------------------------------------------------
# Regression: a bare "See: <url>" plus the freshness line satisfies every structural
# rule (1 sentence, 1 ingested URL, no lexicon hits, date present) while telling the
# user nothing. `_replace_urls` manufactures exactly that when the model returns only
# whitespace, which happened with a Groq reasoning model that spent 218 of 220
# max_tokens on reasoning and returned empty content.

POINTER_ONLY = f"See: {GOOD_URL}\n\n{freshness_line(GOOD_DATE)}"
REAL_ANSWER = f"Exit load is 1% if redeemed within 1 year. {GOOD_URL}\n\n{freshness_line(GOOD_DATE)}"


def test_pointer_only_answer_is_not_substantive():
    assert not has_substantive_content(POINTER_ONLY)
    assert has_substantive_content(REAL_ANSWER)


@pytest.mark.parametrize(
    "text",
    [
        f"See: {GOOD_URL}",
        f"Source: {GOOD_URL}\n\n{freshness_line(GOOD_DATE)}",
        f"Read more at {GOOD_URL}",
        f"Refer {GOOD_URL} {freshness_line(GOOD_DATE)}",
    ],
)
def test_bare_pointers_are_rejected_regardless_of_wording(text):
    assert not has_substantive_content(text)


def test_validate_flags_an_answer_with_no_fact():
    result = validate(POINTER_ONLY, expected_url=GOOD_URL, expected_date=GOOD_DATE)
    assert not result.ok
    assert "no_substantive_content" in result.violations


def test_validate_accepts_a_real_answer():
    result = validate(REAL_ANSWER, expected_url=GOOD_URL, expected_date=GOOD_DATE)
    assert result.ok, result.violations


def test_substantive_content_keeps_the_figure():
    body = substantive_content(REAL_ANSWER)
    assert "1%" in body
    assert "hdfcfund.com" not in body
    assert "Last updated" not in body
