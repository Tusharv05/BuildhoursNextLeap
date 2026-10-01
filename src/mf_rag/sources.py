"""Source registry — the single source of truth for what the bot is allowed to read.

RAG stage: input to Stage 1 (Loading). No ad-hoc URL may enter the corpus except
through this module. Every host is checked against ALLOWED_HOSTS so that
"public sources only, no third-party blogs" is enforced structurally rather
than by convention.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

# Official AMC / industry / regulator domains, plus the Groww scheme pages that the
# brief mandates as seed sources. Anything outside this list is rejected.
# NOTE: HDFC Mutual Fund's official site is hdfcfund.com. hdfcmf.com is a parked
# domain for sale and is deliberately NOT allowed.
ALLOWED_HOSTS: frozenset[str] = frozenset(
    {
        "groww.in",
        "www.groww.in",
        "hdfcfund.com",
        "www.hdfcfund.com",
        "amfiindia.com",
        "www.amfiindia.com",
        "portal.amfiindia.com",
        "sebi.gov.in",
        "www.sebi.gov.in",
    }
)

# Schemes in scope: one AMC (HDFC Asset Management Company), five schemes.
# Aliases feed Phase 5 entity detection.
SCHEME_REGISTRY: dict[str, dict[str, Any]] = {
    "hdfc_large_cap": {
        "scheme": "HDFC Large Cap Fund - Direct Growth",
        "category": "Large Cap",
        "aliases": ["hdfc large cap", "large cap fund", "largecap"],
    },
    "hdfc_flexi_cap": {
        "scheme": "HDFC Equity (Flexi Cap) Fund - Direct Growth",
        "category": "Flexi Cap",
        "aliases": ["flexi cap", "flexicap", "hdfc equity", "hdfc flexi cap"],
    },
    "hdfc_elss": {
        "scheme": "HDFC ELSS Tax Saver Fund - Direct Plan Growth",
        "category": "ELSS",
        "aliases": ["elss", "tax saver", "tax-saver", "80c", "elss tax saver"],
    },
    "hdfc_small_cap": {
        "scheme": "HDFC Small Cap Fund - Direct Growth",
        "category": "Small Cap",
        "aliases": ["small cap", "smallcap"],
    },
    "hdfc_balanced_advantage": {
        "scheme": "HDFC Balanced Advantage Fund - Direct Growth",
        "category": "Balanced Advantage",
        "aliases": ["balanced advantage", "balanced advantage fund", "hybrid"],
    },
}

CATEGORY_REGISTRY: dict[str, str] = {
    "large cap": "Large Cap",
    "flexi cap": "Flexi Cap",
    "elss": "ELSS",
    "small cap": "Small Cap",
    "balanced advantage": "Balanced Advantage",
}

# The registry. verified=False marks a deep link whose exact path must be confirmed
# during Phase 1 ingestion; the ingest report will flag it and ingestion warns and
# skips rather than failing the build.
SOURCES: tuple[dict[str, Any], ...] = (
    {
        "source_id": "hdfc_large_cap_scheme_page",
        "source_url": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        "scheme": "HDFC Large Cap Fund - Direct Growth",
        "category": "Large Cap",
        "doc_type": "scheme_page",
        "title": "HDFC Large Cap Fund - Direct Growth",
        "is_official": False,
        "verified": True,
        "notes": "Seed URL from the brief.",
    },
    {
        "source_id": "hdfc_flexi_cap_scheme_page",
        "source_url": "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
        "scheme": "HDFC Equity (Flexi Cap) Fund - Direct Growth",
        "category": "Flexi Cap",
        "doc_type": "scheme_page",
        "title": "HDFC Equity (Flexi Cap) Fund - Direct Growth",
        "is_official": False,
        "verified": True,
        "notes": "Seed URL from the brief.",
    },
    {
        "source_id": "hdfc_elss_scheme_page",
        "source_url": "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
        "scheme": "HDFC ELSS Tax Saver Fund - Direct Plan Growth",
        "category": "ELSS",
        "doc_type": "scheme_page",
        "title": "HDFC ELSS Tax Saver Fund - Direct Plan Growth",
        "is_official": False,
        "verified": True,
        "notes": "Seed URL from the brief.",
    },
    {
        "source_id": "hdfc_small_cap_scheme_page",
        "source_url": "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
        "scheme": "HDFC Small Cap Fund - Direct Growth",
        "category": "Small Cap",
        "doc_type": "scheme_page",
        "title": "HDFC Small Cap Fund - Direct Growth",
        "is_official": False,
        "verified": True,
        "notes": "Seed URL from the brief.",
    },
    {
        "source_id": "hdfc_balanced_advantage_scheme_page",
        "source_url": "https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth",
        "scheme": "HDFC Balanced Advantage Fund - Direct Growth",
        "category": "Balanced Advantage",
        "doc_type": "scheme_page",
        "title": "HDFC Balanced Advantage Fund - Direct Growth",
        "is_official": False,
        "verified": True,
        "notes": "Seed URL from the brief.",
    },
    {
        "source_id": "hdfcfund_large_cap_official",
        "source_url": "https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct",
        "scheme": "HDFC Large Cap Fund - Direct Growth",
        "category": "Large Cap",
        "doc_type": "scheme_page",
        "title": "HDFC Large Cap Fund - Direct (official)",
        "is_official": True,
        "verified": True,
        "notes": "Official AMC page. Preferred citation target for Large Cap facts.",
    },
    {
        "source_id": "hdfcfund_flexi_cap_official",
        "source_url": "https://www.hdfcfund.com/explore/mutual-funds/hdfc-flexi-cap-fund/direct",
        "scheme": "HDFC Equity (Flexi Cap) Fund - Direct Growth",
        "category": "Flexi Cap",
        "doc_type": "scheme_page",
        "title": "HDFC Flexi Cap Fund - Direct (official)",
        "is_official": True,
        "verified": True,
        "notes": "Official AMC page. Preferred citation target for Flexi Cap facts.",
    },
    {
        "source_id": "hdfcfund_elss_official",
        "source_url": "https://www.hdfcfund.com/explore/mutual-funds/hdfc-elss-tax-saver-fund/direct",
        "scheme": "HDFC ELSS Tax Saver Fund - Direct Plan Growth",
        "category": "ELSS",
        "doc_type": "scheme_page",
        "title": "HDFC ELSS Tax Saver Fund - Direct (official)",
        "is_official": True,
        "verified": True,
        "notes": "Official AMC page. Preferred citation target for ELSS lock-in facts.",
    },
    {
        "source_id": "hdfcfund_small_cap_official",
        "source_url": "https://www.hdfcfund.com/explore/mutual-funds/hdfc-small-cap-fund/direct",
        "scheme": "HDFC Small Cap Fund - Direct Growth",
        "category": "Small Cap",
        "doc_type": "scheme_page",
        "title": "HDFC Small Cap Fund - Direct (official)",
        "is_official": True,
        "verified": True,
        "notes": "Official AMC page. Preferred citation target for Small Cap facts.",
    },
    {
        "source_id": "hdfcfund_balanced_advantage_official",
        "source_url": "https://www.hdfcfund.com/explore/mutual-funds/hdfc-balanced-advantage-fund/direct",
        "scheme": "HDFC Balanced Advantage Fund - Direct Growth",
        "category": "Balanced Advantage",
        "doc_type": "scheme_page",
        "title": "HDFC Balanced Advantage Fund - Direct (official)",
        "is_official": True,
        "verified": True,
        "notes": "Official AMC page. Preferred citation target for Balanced Advantage facts.",
    },
    {
        "source_id": "hdfcfund_investor_faq",
        "source_url": "https://www.hdfcfund.com/services/faqs",
        "scheme": None,
        "category": None,
        "doc_type": "faq",
        "title": "HDFC Mutual Fund - investor FAQs",
        "is_official": True,
        "verified": True,
        "notes": "Official AMC FAQ page. Source for procedural facts (statements, tax docs).",
    },
    {
        "source_id": "hdfcfund_consolidated_statement",
        "source_url": "https://www.hdfcfund.com/services/consolidated-account-statement",
        "scheme": None,
        "category": None,
        "doc_type": "guide",
        "title": "HDFC Mutual Fund - consolidated account statement (download)",
        "is_official": True,
        "verified": True,
        "notes": "Official how-to for downloading account statements. Source for the "
        "statement-download intent.",
    },
    {
        "source_id": "hdfcfund_fees_education",
        "source_url": "https://www.hdfcfund.com/learn/blog/what-expense-ratio-mutual-funds",
        "scheme": None,
        "category": None,
        "doc_type": "educational",
        "title": "What is expense ratio in mutual funds (HDFC MF)",
        "is_official": True,
        "verified": True,
        "notes": "AMC-hosted investor education. Educational link for refusals.",
    },
    {
        "source_id": "hdfcfund_ter_disclosure",
        "source_url": "https://www.hdfcfund.com/statutory-disclosure/total-expense-ratio-of-mutual-fund-schemes/reports",
        "scheme": None,
        "category": None,
        "doc_type": "fees",
        "title": "HDFC MF - total expense ratio disclosure reports",
        "is_official": True,
        "verified": True,
        "notes": "Statutory TER disclosure. Link target when asked about official figures.",
    },
    {
        "source_id": "amfi_hdfc_member",
        "source_url": "https://www.amfiindia.com/member/9",
        "scheme": None,
        "category": None,
        "doc_type": "guide",
        "title": "AMFI - HDFC Mutual Fund member details",
        "is_official": True,
        "verified": True,
        "notes": "AMFI registry entry: SEBI reg no, AMC/trustee/registrar details.",
    },
    {
        "source_id": "amfi_home",
        "source_url": "https://www.amfiindia.com/",
        "scheme": None,
        "category": None,
        "doc_type": "educational",
        "title": "AMFI - Association of Mutual Fund Investors",
        "is_official": True,
        "verified": True,
        "notes": "Industry body. General investor-education link for refusals.",
    },
    {
        "source_id": "sebi_home",
        "source_url": "https://www.sebi.gov.in/",
        "scheme": None,
        "category": None,
        "doc_type": "educational",
        "title": "SEBI - Securities and Exchange Board of India",
        "is_official": True,
        "verified": True,
        "notes": "Regulator. General investor-education link for refusals.",
    },
)

REQUIRED_FIELDS: tuple[str, ...] = (
    "source_id",
    "source_url",
    "scheme",
    "category",
    "doc_type",
    "title",
    "is_official",
)


def host_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if not host:
        raise ValueError(f"Could not parse a host from URL: {url!r}")
    return host


def validate_host(url: str) -> str:
    """Return the host if allowed, otherwise raise ValueError. Never logs the URL body."""
    host = host_of(url)
    if host not in ALLOWED_HOSTS:
        raise ValueError(
            f"Host '{host}' is not in the allowlist. "
            f"Public sources only: official AMC/AMFI/SEBI pages or the mandated "
            f"Groww scheme pages. Allowed hosts: {', '.join(sorted(ALLOWED_HOSTS))}"
        )
    return host


def validate_source(row: dict[str, Any]) -> dict[str, Any]:
    missing = [f for f in REQUIRED_FIELDS if f not in row]
    if missing:
        raise ValueError(f"Source row missing required field(s): {', '.join(missing)}")
    validate_host(row["source_url"])
    return row


def get_sources() -> list[dict[str, Any]]:
    return [dict(row) for row in SOURCES]


def get_source(source_id: str) -> dict[str, Any]:
    for row in SOURCES:
        if row["source_id"] == source_id:
            return dict(row)
    raise KeyError(f"Unknown source_id: {source_id}")


def allowed_urls() -> set[str]:
    """Every URL present in the ingested corpus. Phase 7 validates citations against this."""
    return {row["source_url"] for row in SOURCES}


def registry_for_category() -> dict[str, list[dict[str, Any]]]:
    return {key: [row for row in SOURCES if row["category"] == value] for key, value in CATEGORY_REGISTRY.items()}


def _validate_registry() -> None:
    seen: set[str] = set()
    for row in SOURCES:
        validate_source(row)
        if row["source_id"] in seen:
            raise ValueError(f"Duplicate source_id in registry: {row['source_id']}")
        seen.add(row["source_id"])


_validate_registry()
