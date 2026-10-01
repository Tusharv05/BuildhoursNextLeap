"""Phase 5 entity-detection tests.

The critical property is not "does it match Small Cap" but "does it *only* match the right
thing". A filter that fires on the wrong scheme silently excludes the correct answer, so
the negative cases matter as much as the positive ones.
"""

from __future__ import annotations

import pytest

from mf_rag.entities import (
    CATEGORY_REGISTRY,
    SCHEME_REGISTRY,
    correct_domain_typos,
    describe,
    detect_entity,
    normalise,
    out_of_registry_schemes,
    where_for,
)

SMALL = "HDFC Small Cap Fund - Direct Growth"
LARGE = "HDFC Large Cap Fund - Direct Growth"
ELSS = "HDFC ELSS Tax Saver Fund - Direct Plan Growth"
FLEXI = "HDFC Equity (Flexi Cap) Fund - Direct Growth"
BAL = "HDFC Balanced Advantage Fund - Direct Growth"


@pytest.mark.parametrize(
    "query,expected_scheme",
    [
        ("What is the expense ratio of HDFC Small Cap Fund?", SMALL),
        ("Exit load of HDFC Small Cap Fund?", SMALL),
        ("smallcap charges", SMALL),
        ("small-cap minimum SIP", SMALL),
        ("HDFC Large Cap Fund exit load", LARGE),
        ("largecap expense ratio", LARGE),
        ("ELSS lock-in period", ELSS),
        ("80C deduction in ELSS", ELSS),
        ("tax saver fund lock in", ELSS),
        ("section 80c eligibility", ELSS),
        ("flexi cap strategy", FLEXI),
        ("hdfc equity fund", FLEXI),
        ("balanced advantage allocation", BAL),
    ],
)
def test_scheme_aliases_detected(query, expected_scheme):
    kind, where = detect_entity(query)
    assert kind == "scheme"
    assert where == {"scheme": expected_scheme}


def test_punctuation_and_case_do_not_matter():
    assert detect_entity("SMALL-CAP FUND?") == detect_entity("small cap fund")


def test_category_registry_agrees_with_scheme_registry():
    """Every category is reachable and the two registries never disagree on spelling."""
    for category, aliases in CATEGORY_REGISTRY.items():
        for alias in aliases:
            kind, where = detect_entity(alias)
            assert kind in {"scheme", "category"}
            assert where is not None
            value = where.get("scheme") or where.get("category")
            assert value in (category,) or value in SCHEME_REGISTRY


def test_bare_cap_does_not_match():
    """'Cap' is inside Small Cap, Large Cap and Flexi Cap - matching it would be wrong."""
    assert detect_entity("What is a cap?") == (None, None)
    assert detect_entity("market cap") == (None, None)


def test_corpus_wide_question_is_unfiltered():
    """No entity, so retrieval must run unfiltered rather than guess."""
    for query in (
        "How do I download the capital gains statement?",
        "What is the expense ratio?",
        "Who is the trustee?",
    ):
        assert detect_entity(query) == (None, None)


def test_empty_query_is_safe():
    assert detect_entity("") == (None, None)
    assert detect_entity("   ") == (None, None)


def test_where_for_respects_the_toggle():
    assert where_for("Exit load of HDFC Small Cap Fund?", enabled=False) == (None, None)
    assert where_for("Exit load of HDFC Small Cap Fund?", enabled=True)[1] == {"scheme": SMALL}


def test_longest_alias_wins_when_two_entities_appear():
    """'balanced advantage' is a more specific claim than a stray 'cap'."""
    _, where = detect_entity("Is HDFC Balanced Advantage Fund better than a large cap fund?")
    assert where == {"scheme": BAL}


def test_describe_is_human_readable():
    assert describe("scheme", {"scheme": SMALL}).startswith("scheme=")
    assert describe(None, None) == "none"


def test_out_of_registry_detection():
    assert "parag parflex" in out_of_registry_schemes("What is the NAV of Parag Parflex?")
    assert out_of_registry_schemes("Exit load of HDFC Small Cap Fund?") == []


def test_normalise_collapses_punctuation_runs():
    assert normalise("Small-Cap / Flexi (Cap)!") == "small cap flexi cap"


def test_filter_value_is_byte_identical_to_stored_metadata():
    """A misspelled filter value returns zero rows silently - this catches that."""
    from mf_rag.sources import SOURCES

    stored_schemes = {s["scheme"] for s in SOURCES if s["scheme"]}
    stored_categories = {s["category"] for s in SOURCES if s["category"]}
    assert set(SCHEME_REGISTRY) <= stored_schemes
    assert set(CATEGORY_REGISTRY) <= stored_categories


def test_known_domain_typos_are_corrected() -> None:
    assert correct_domain_typos("what is exist load?") == "what is exit load?"
    assert correct_domain_typos("expence ratio of large cap") == "expense ratio of large cap"
    assert correct_domain_typos("who is the fund manger") == "who is the fund manager"
    assert correct_domain_typos("sector allocaton of flexi cap") == "sector allocation of flexi cap"
    assert correct_domain_typos("turn over ratio") == "turnover ratio"


def test_typo_correction_is_case_insensitive() -> None:
    # Only the misspelled word is replaced. The rest of the query keeps its casing,
    # because this string also feeds the generation prompt and rewriting the whole of it
    # would be a change the user never asked for.
    assert correct_domain_typos("Expence Ratio") == "expense Ratio"
    assert correct_domain_typos("EXIST LOAD") == "exit load"


def test_typo_correction_does_not_corrupt_real_words() -> None:
    """The substitution is word-bounded, so these must survive untouched.

    "existence" contains "exist" and "operation" contains "ration"; a substring replace
    would mangle both, and these are ordinary English words a user may legitimately type.
    """
    assert correct_domain_typos("existence of the fund") == "existence of the fund"
    assert correct_domain_typos("operation ration") == "operation ratio"
    assert correct_domain_typos("") == ""


def test_typo_correction_preserves_unrelated_text_exactly() -> None:
    query = "What is the exit load and minimum SIP for HDFC Small Cap Fund?"
    assert correct_domain_typos(query) == query
