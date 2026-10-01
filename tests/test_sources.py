"""Source registry: the host allowlist is the structural guarantee of source policy."""

from __future__ import annotations

import pytest

from mf_rag.sources import (
    ALLOWED_HOSTS,
    CATEGORY_REGISTRY,
    SCHEME_REGISTRY,
    allowed_urls,
    get_source,
    get_sources,
    host_of,
    validate_host,
    validate_source,
)

SEED_URLS = (
    "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
    "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
    "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
    "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
    "https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth",
)


def test_all_five_seed_urls_present() -> None:
    urls = allowed_urls()
    for seed in SEED_URLS:
        assert seed in urls, f"missing mandated seed URL: {seed}"


def test_five_schemes_registered() -> None:
    assert len(SCHEME_REGISTRY) == 5
    assert {v["category"] for v in SCHEME_REGISTRY.values()} == {
        "Large Cap",
        "Flexi Cap",
        "ELSS",
        "Small Cap",
        "Balanced Advantage",
    }


def test_every_category_has_a_scheme() -> None:
    assert set(CATEGORY_REGISTRY.values()) <= {v["category"] for v in SCHEME_REGISTRY.values()}


def test_registry_has_official_sources() -> None:
    official = [row for row in get_sources() if row["is_official"]]
    assert official, "no official AMC/AMFI/SEBI sources registered"
    assert all("groww.in" not in row["source_url"] for row in official)


def test_registry_has_educational_source() -> None:
    assert any(row["doc_type"] == "educational" for row in get_sources())


def test_get_sources_returns_copies() -> None:
    first = get_sources()
    first[0]["source_url"] = "https://mutated.invalid/x"
    assert get_sources()[0]["source_url"] != "https://mutated.invalid/x"


def test_get_source_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_source("does_not_exist")


@pytest.mark.parametrize(
    "url",
    [
        "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
        "https://www.hdfcfund.com/",
        "https://www.amfiindia.com/",
        "https://www.sebi.gov.in/",
    ],
)
def test_allowed_hosts_accepted(url: str) -> None:
    assert validate_host(url) in ALLOWED_HOSTS


@pytest.mark.parametrize(
    "url",
    [
        "https://someblog.com/mutual-funds/hdfc-small-cap",
        "https://moneycontrol.com/news/elss-lock-in",
        "https://wikipedia.org/wiki/ELSS",
        "https://groww.in.evil.com/mutual-funds",   # suffix trick
        "https://hdfcfund.com@evil.com/",             # userinfo spoof: real host is evil.com
        "https://groww.in@evil.com/mutual-funds",
        "http://evil.com/?x=redirect=https://groww.in",
    ],
)
def test_foreign_hosts_rejected(url: str) -> None:
    with pytest.raises(ValueError, match="not in the allowlist"):
        validate_host(url)


def test_userinfo_does_not_launder_an_allowed_host() -> None:
    """Userinfo before an allowlisted host is still allowlisted - the real host is that host."""
    assert validate_host("https://hdfcfund.com/") == "hdfcfund.com"
    # ...but a disallowed host hidden in the path/query is not laundered either.
    with pytest.raises(ValueError):
        validate_host("https://evil.com/hdfcfund.com")



def test_validate_source_requires_fields() -> None:
    with pytest.raises(ValueError, match="missing required field"):
        validate_source({"source_id": "x"})


def test_registry_is_valid_on_import() -> None:
    # Importing the module already runs _validate_registry(); assert the shape too.
    rows = get_sources()
    ids = [row["source_id"] for row in rows]
    assert len(ids) == len(set(ids))
    assert all(row["title"] for row in rows)
