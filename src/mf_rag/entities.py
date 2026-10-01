"""Entity detection for retrieval filtering (Phase 5).

HDFC's five schemes share almost identical boilerplate. Measured on the real index, the
query "Exit load of HDFC Small Cap Fund?" ranks a *Large Cap* chunk first (cos 0.7386)
and the correct *Small Cap* chunk second (cos 0.7231) - a gap of 0.015. Pure vector
similarity cannot separate them, so a `where` filter is not an optimisation here, it is
what makes the answer correct. This module decides that filter.

`detect_entity` returns `(kind, where_dict|None)` where `where_dict` is passed straight to
Chroma as `where=`. `(None, None)` means "no entity in the question, do not filter".
"""

from __future__ import annotations

import re

# Canonical values, byte-identical to the `scheme` / `category` values written into
# Chroma metadata by Stage 2/4. A filter that misspells these returns 0 rows silently.
SCHEME_REGISTRY: dict[str, tuple[str, ...]] = {
    "HDFC Balanced Advantage Fund - Direct Growth": ("balanced advantage",),
    "HDFC ELSS Tax Saver Fund - Direct Plan Growth": (
        "elss",
        "elss tax saver",
        "tax saver",
        "80c",
        "80 c",
        "section 80c",
    ),
    "HDFC Equity (Flexi Cap) Fund - Direct Growth": (
        "flexi cap",
        "flexicap",
        "flexi",
        "hdfc equity",
        "equity flexi cap",
    ),
    "HDFC Large Cap Fund - Direct Growth": ("large cap", "largecap"),
    "HDFC Small Cap Fund - Direct Growth": ("small cap", "smallcap"),
}

CATEGORY_REGISTRY: dict[str, tuple[str, ...]] = {
    "Balanced Advantage": ("balanced advantage",),
    "ELSS": ("elss", "elss tax saver", "tax saver", "80c", "80 c", "section 80c"),
    "Flexi Cap": ("flexi cap", "flexicap", "flexi", "hdfc equity", "equity flexi cap"),
    "Large Cap": ("large cap", "largecap"),
    "Small Cap": ("small cap", "smallcap"),
}

# Bare "cap" is deliberately absent: "Cap" appears inside Small Cap, Large Cap and
# Flexi Cap, so matching it would filter every capital-market question to one scheme.
_AMBIGUOUS = frozenset({"cap", "equity", "fund", "growth", "direct", "hdfc", "plan"})

# Misspellings of terms this corpus is actually full of. "what is exist load?" is a real
# query we saw reach retrieval unfixed, and an embedding for "exist" shares no vocabulary
# with the exit-load boilerplate it should be matching. The list is deliberately tiny and
# each entry is a term the corpus uses verbatim - this corrects known domain typos, it is
# not a general spell checker, so words outside these are left exactly as typed.
_TYPO_TERMS: dict[str, str] = {
    "exist load": "exit load",
    "allocaton": "allocation",
    "expence": "expense",
    "ration": "ratio",
    "manger": "manager",
    "nave": "nav",
    "mininum": "minimum",
    "secotr": "sector",
    "portfolo": "portfolio",
    "turn over": "turnover",
    "holdigns": "holdings",
    "expenss": "expense",
}
_TYPO_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in sorted(_TYPO_TERMS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def correct_domain_typos(text: str) -> str:
    """Rewrite known domain misspellings, leaving everything else untouched.

    Entries are mostly single words on purpose: correcting "allocaton" fixes both "asset
    allocaton" and "sector allocaton", whereas listing every phrase would not.
    """
    return _TYPO_PATTERN.sub(lambda m: _TYPO_TERMS[m.group(0).lower()], str(text))



def normalise(text: str) -> str:
    """Lowercase and collapse every non-alphanumeric run to a single space.

    This is what makes "Small-Cap", "smallcap" and "small cap" one query.
    """
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _match(query: str, registry: dict[str, tuple[str, ...]]) -> str | None:
    """First canonical key whose alias appears in the normalised query.

    Aliases are whole phrases, matched on space-padded boundaries, so "flexi" cannot fire
    on "flexibility" and "large cap" cannot fire on "large capacity". When a query names
    more than one entity ("compare ELSS and Flexi Cap") the longest alias wins, because
    "balanced advantage" is a more specific claim than a stray "cap".
    """
    padded = f" {normalise(query)} "
    best: tuple[int, str] | None = None
    for canonical, aliases in registry.items():
        for alias in aliases:
            phrase = normalise(alias)
            if not phrase or phrase in _AMBIGUOUS:
                continue
            if f" {phrase} " in padded:
                if best is None or len(phrase) > best[0]:
                    best = (len(phrase), canonical)
    return best[1] if best else None


def detect_entity(query: str) -> tuple[str | None, dict[str, str] | None]:
    """Detect a scheme or category in `query`.

    Returns `(kind, where)`:
        ("scheme",   {"scheme": "HDFC Small Cap Fund - Direct Growth"})
        ("category", {"category": "Small Cap"})
        (None, None) - nothing recognised, so retrieval runs unfiltered.

    A named scheme wins over a category: "HDFC ELSS Tax Saver Fund" narrows to one
    scheme page, while a bare "ELSS" legitimately spans every ELSS-tagged source.
    """
    if not query or not str(query).strip():
        return (None, None)

    scheme = _match(query, SCHEME_REGISTRY)
    if scheme is not None:
        return ("scheme", {"scheme": scheme})

    category = _match(query, CATEGORY_REGISTRY)
    if category is not None:
        return ("category", {"category": category})

    return (None, None)


def where_for(query: str, enabled: bool = True) -> tuple[str | None, dict[str, str] | None]:
    """`detect_entity` gated on `cfg.retrieval.filter_by_entity`."""
    if not enabled:
        return (None, None)
    return detect_entity(query)


def describe(kind: str | None, where: dict[str, str] | None) -> str:
    """Short human label for the UI trace, e.g. `scheme=HDFC Small Cap Fund - Direct Growth`."""
    if not where:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(where.items()))


def out_of_registry_schemes(query: str) -> list[str]:
    """Fund names in `query` that look like schemes but are not in the registry.

    Powers the `out_of_corpus` intent: "Should I buy Parag Parflex" is advisory *and*
    unscoped, and the honest answer is that this assistant has never read that fund.
    Matched against a conservative stem list, so only plausible fund names are returned.
    """
    padded = f" {normalise(query)} "
    found: list[str] = []
    for stem in ("parag parflex", "sbi bluechip", "axis long term", "mirae asset", "icici nifty", "kotak flexicap"):
        if f" {stem} " in padded and stem not in found:
            found.append(stem)
    return found
