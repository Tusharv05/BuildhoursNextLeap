"""HTML cleaning: main-content isolation, boilerplate removal, fact rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from mf_rag.clean import (
    drop_noise_lines,
    extract_main_text,
    looks_useful,
    normalise_whitespace,
)
from mf_rag.structured import (
    extract_facts_section,
    extract_official_aum,
    render_scheme_facts,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def scheme_html() -> str:
    return (FIXTURES / "scheme_page.html").read_text(encoding="utf-8")


@pytest.fixture
def faq_html() -> str:
    return (FIXTURES / "faq_page.html").read_text(encoding="utf-8")


def test_extracts_title_and_body(scheme_html: str) -> None:
    title, text = extract_main_text(scheme_html)
    assert "HDFC Test Large Cap Fund" in title
    assert "large cap stocks" in text


def test_strips_navigation_header_footer(scheme_html: str) -> None:
    _, text = extract_main_text(scheme_html)
    assert "Related funds" not in text
    assert "Accept all cookies" not in text
    assert "site-footer" not in text


def test_strips_script_and_style(scheme_html: str) -> None:
    _, text = extract_main_text(scheme_html)
    assert "__NEXT_DATA__" not in text
    assert ".hidden" not in text


def _flat(text: str) -> str:
    """Collapse line wrapping so assertions are not defeated by source formatting."""
    return " ".join(text.split())


def test_keeps_faq_question_with_its_answer(scheme_html: str) -> None:
    _, text = extract_main_text(scheme_html)
    flat = _flat(text)
    assert "What is the lock-in period" in flat
    assert "no lock-in period for this scheme" in flat
    assert "capital gains statement" in flat
    assert "under the Reports section" in flat


def test_keeps_table_rows_intact(scheme_html: str) -> None:
    _, text = extract_main_text(scheme_html)
    assert "| Expense ratio | 1.50% |" in text
    assert "| Minimum SIP | INR 500 |" in text


def test_handles_empty_and_garbage_input() -> None:
    assert extract_main_text("") == ("", "")
    assert extract_main_text("   \n  ") == ("", "")
    # Not HTML at all: must not raise.
    title, text = extract_main_text("just some plain text, no markup at all")
    assert isinstance(title, str) and isinstance(text, str)


def test_normalise_whitespace_collapses_runs() -> None:
    assert normalise_whitespace("a  \t b\r\n\r\n\r\n\r\nc") == "a b\n\nc"


def test_drop_noise_lines_removes_menu_labels() -> None:
    text = "Log in\n\nReal content here.\n\nClose\n\nMore real content."
    assert drop_noise_lines(text) == "Real content here.\n\nMore real content."

def test_looks_useful_threshold() -> None:
    assert looks_useful("x" * 600)
    assert not looks_useful("x" * 100)


def test_faq_page_keeps_procedural_facts(faq_html: str) -> None:
    _, text = extract_main_text(faq_html)
    assert "capital gains statement" in text
    assert "registrar and transfer agent" in text


def test_renders_facts_from_embedded_payload(scheme_html: str) -> None:
    facts = extract_facts_section(scheme_html, "https://groww.in/x")
    assert "Expense ratio (annual %) (Direct plan): 1.50" in facts
    assert "Exit load (Direct plan): Nil" in facts
    assert "Minimum SIP amount (INR) (Direct plan): 500" in facts
    assert "Benchmark (Direct plan): Nifty 50 Total Return Index" in facts
    assert "Riskometer (Direct plan): Very High Riskometer" in facts
    assert "https://groww.in/x" in facts


def test_lock_in_absent_is_simply_omitted() -> None:
    facts = extract_facts_section((FIXTURES / "scheme_page.html").read_text(encoding="utf-8"), "u")
    assert "Lock-in period" not in facts  # all-null lock_in renders nothing


def test_historic_series_is_change_based_not_row_based(scheme_html: str) -> None:
    """Four daily records collapse to the two distinct expense ratios."""
    facts = extract_facts_section(scheme_html, "u")
    block = facts.split("Historic expense ratio changes:")[1]
    rows = [line for line in block.splitlines() if "expense_ratio=" in line]
    assert len(rows) == 2, rows
    assert "1.50" in rows[0]
    assert "1.55" in rows[1]
    assert "T00:00:00" not in facts  # dates trimmed


def test_no_payload_yields_no_facts(faq_html: str) -> None:
    assert extract_facts_section(faq_html, "u") == ""


def test_assignment_form_of_payload_is_supported() -> None:
    """Some pages assign window.__NEXT_DATA__ instead of emitting a JSON script tag."""
    html = (
        "<html><body><script>window.__NEXT_DATA__ = "
        '{"props":{"pageProps":{"mfServerSideData":{"expense_ratio":"0.65"}}}};'
        "</script></body></html>"
    )
    facts = extract_facts_section(html, "u")
    assert "Expense ratio (annual %): 0.65" in facts


def test_malformed_payload_does_not_raise() -> None:
    html = '<html><body><script id="__NEXT_DATA__" type="application/json">{not json</script></body></html>'
    assert extract_facts_section(html, "u") == ""


def test_render_scheme_facts_empty_when_no_known_fields() -> None:
    assert render_scheme_facts({"unrelated": "value"}, "u") == ""


def test_aum_is_the_fund_not_the_amc() -> None:
    """AUM must come from the scheme's own field, never the AMC's book.

    Groww's page carries three numbers: the fund's `aum` (113,606 Cr), the AMC's total
    `amc_info.aum` (986,237 Cr), and peer funds' AUMs. Its own prose blurs these and reads
    "The fund currently has an Asset Under Management(AUM) of Rs 9,86,237 Cr", which is the
    AMC figure - roughly 8.7x too large. Reading the structured field is what keeps this
    correct, and a prose regex would not.
    """
    facts = render_scheme_facts(
        {
            "plan_type": "Direct",
            "aum": 113606.46602051,
            "amc_info": {"aum": 986236.84},
            "peerComparison": [{"fund_name": "Bank of India Flexi Cap", "aum": 2953.0}],
        },
        "https://groww.in/x",
    )
    assert "Fund size / AUM (INR crore) (Direct plan): 113,606.47" in facts
    assert "986,236" not in facts and "986237" not in facts
    assert "2,953" not in facts


def test_aum_omitted_when_missing_or_nonsensical() -> None:
    for value in (None, 0, -5, "n/a"):
        facts = render_scheme_facts({"aum": value}, "https://groww.in/x")
        assert "Fund size" not in facts, f"aum={value!r} should not render"


def test_official_aum_is_read_from_the_amcs_own_payload() -> None:
    """hdfcfund.com publishes fund size as an adjacent aum/aumAsMonth key pair.

    Its visible markup is a bare <div>AUM<span>(31/08/2026)</span></div><div>113,606.47
    Cr.</div>, which main-text extraction drops, so the official page appeared to have no
    fund size and the answer had to be borrowed from an aggregator.
    """
    html = (
        '<script id="__NEXT_DATA__" type="application/json">'
        '{"props":{"data":{"aum":"113,606.47","aumAsMonth":"(31/08/2026)",'
        '"auminfo":"<div>Assets Under Management</div>"}}}'
        "</script>"
    )
    facts = extract_facts_section(html, "https://www.hdfcfund.com/x")
    assert "Fund size / AUM (INR crore): 113,606.47 (as of 31/08/2026)" in facts
    assert "Source page: https://www.hdfcfund.com/x" in facts


def test_official_aum_ignores_nonpositive_and_absent_values() -> None:
    assert "AUM" not in extract_official_aum('{"aum":"0"}', "u")
    assert "AUM" not in extract_official_aum('{"aum":""}', "u")
    assert extract_official_aum("<html>nothing here</html>", "u") == ""



def _scheme_payload(**overrides: object) -> str:
    import json as _json

    base = {
        "plan_type": "Direct",
        "expense_ratio": 1.03,
        "aum": 39933.37,
        "fund_house": "HDFC Mutual Fund",
        "launch_date": "01-Jan-2013",
        "registrar_agent": "CAMS",
        "portfolio_turnover": 18,
        "groww_rating": 4,
        "fund_manager": "Prashant Jain",
        "fund_manager_details": [
            {"person_name": "Rahul Baijal", "date_from": "2022-07-28T18:30:00.000Z"},
            {"person_name": "Dhruv Muchhal", "date_from": "2023-06-21T18:30:00.000Z"},
        ],
        "holdings": [
            {"company_name": "ICICI Bank Ltd", "sector_name": "Financial",
             "instrument_name": "Equity", "corpus_per": 10.05,
             "portfolio_date": "2026-08-30T18:30:00.000Z"},
            {"company_name": "Bharti Airtel Ltd", "sector_name": "Communication Services",
             "instrument_name": "Equity", "corpus_per": 6.23,
             "portfolio_date": "2026-08-30T18:30:00.000Z"},
            {"company_name": "Repo", "sector_name": "Financial",
             "instrument_name": "Repo", "corpus_per": 0.5,
             "portfolio_date": "2026-08-30T18:30:00.000Z"},
        ],
    }
    base.update(overrides)
    return (
        '<script id="__NEXT_DATA__" type="application/json">'
        + _json.dumps({"props": {"pageProps": {"mfServerSideData": base}}})
        + "</script>"
    )


def test_current_manager_wins_over_stale_scalar() -> None:
    """`fund_manager` is a stale label; the dated detail list is what is true now.

    For HDFC Large Cap the scalar still names Prashant Jain, a manager who left years
    ago, while the detail list carries the two managers actually in post. Rendering the
    scalar would confidently answer the question wrongly.
    """
    facts = extract_facts_section(_scheme_payload(), "u")
    assert "- Rahul Baijal (in role since 2022-07-28)" in facts
    assert "- Dhruv Muchhal (in role since 2023-06-21)" in facts
    assert "Prashant Jain" not in facts


def test_scalar_manager_used_when_there_are_no_dated_details() -> None:
    facts = extract_facts_section(_scheme_payload(fund_manager_details=[]), "u")
    assert "- Prashant Jain (as listed on the page; not dated)" in facts


def test_duplicate_manager_entries_are_collapsed() -> None:
    facts = extract_facts_section(
        _scheme_payload(
            fund_manager_details=[
                {"person_name": "Rahul Baijal", "date_from": "2022-07-28T18:30:00.000Z"},
                {"person_name": "Rahul Baijal", "date_from": "2022-07-28T18:30:00.000Z"},
            ]
        ),
        "u",
    )
    assert facts.count("Rahul Baijal") == 1


def test_sector_exposure_is_derived_and_grouped_from_holdings() -> None:
    facts = extract_facts_section(_scheme_payload(), "u")
    assert "Sector allocation and exposure" in facts
    assert "- Financial: 10.55%" in facts  # 10.05 equity + 0.50 repo, both Financial
    assert "- Communication Services: 6.23%" in facts
    assert "as of 2026-08-30" in facts


def test_asset_composition_comes_from_instrument_names() -> None:
    facts = extract_facts_section(_scheme_payload(), "u")
    assert "Asset allocation across equity, debt and cash" in facts
    assert "- Equity: 16.28%" in facts
    assert "- Repo: 0.50%" in facts


def test_allocation_headings_use_the_vocabulary_of_the_question() -> None:
    """The heading is the topical label, and recall depends on it matching the query.

    "equity debt cash allocation" and "equity debt cash split" are the two ways this gets
    asked. Cosine could not separate the sector block from the asset block for either -
    which one ranked first flipped with the wording - so both words are stated in the text.
    """
    facts = extract_facts_section(_scheme_payload(), "u").lower()
    for word in ("allocation", "split", "equity", "debt", "cash", "sector"):
        assert word in facts, f"{word!r} missing from the rendered facts"


def test_holdings_blocks_carry_their_own_headings() -> None:
    """Derived blocks must be headings so the chunker cannot fragment them.

    The sector, asset and holdings blocks each run a few hundred characters; glued into
    the facts block they would push each other past max_chars and be cut mid-list.
    """
    facts = extract_facts_section(_scheme_payload(), "u")
    assert "## Sector allocation and exposure" in facts
    assert "## Asset allocation across equity, debt and cash" in facts
    assert "## Fund manager" in facts


def test_top_holdings_excludes_non_equity_positions() -> None:
    """Repo is HDFC Small Cap's largest disclosed weight, and it is not a holding.

    Answering "the top holding is Repo" is defensible from the payload and useless to a
    reader; the debt and cash lines are already reported in the asset block.
    """
    facts = extract_facts_section(_scheme_payload(), "u")
    top = facts.split("## Top equity holdings")[1]
    assert "- ICICI Bank Ltd: 10.05%" in top
    assert "- Bharti Airtel Ltd: 6.23%" in top
    assert "- Repo" not in top


def test_scheme_level_facts_do_not_claim_to_be_plan_specific() -> None:
    """A "(Direct plan)" suffix on the fund house is nonsense."""
    facts = extract_facts_section(_scheme_payload(), "u")
    assert "- Fund house: HDFC Mutual Fund" in facts
    assert "- Registrar: CAMS" in facts
    assert "- Launch date: 01-Jan-2013" in facts
    assert "Fund house (Direct plan)" not in facts
    # Plan-specific figures do keep the suffix, which is what disambiguates them.
    assert "- Expense ratio (annual %) (Direct plan): 1.03" in facts


def test_null_ratings_are_not_rendered() -> None:
    facts = extract_facts_section(
        _scheme_payload(crisil_rating=None, groww_rating=None), "u"
    )
    assert "CRISIL rating" not in facts
    assert "Groww risk-reward rating" not in facts


def test_zero_weight_holdings_are_skipped() -> None:
    facts = extract_facts_section(
        _scheme_payload(
            holdings=[{"company_name": "X", "sector_name": "Tech",
                       "instrument_name": "Equity", "corpus_per": 0}]
        ),
        "u",
    )
    assert "Sector exposure" not in facts
