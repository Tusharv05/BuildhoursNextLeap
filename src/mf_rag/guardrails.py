"""L1 input PII and L2 intent classification (Phase 7).

L1 runs before embedding and long before the LLM, so a PAN typed into the box is never
turned into a vector, never reaches a provider, and is never logged (NFR-4). `detect_pii`
returns pattern *names*, never the matched values, so a violation cannot be leaked by a
debug log line or a trace dict.

L2 is deliberately rule-based. A classifier that can be persuaded by the text it is
classifying would be a poor place to enforce "we don't give advice"; deterministic patterns
are demoable and testable (ARCHITECTURE 6.4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from mf_rag.models import Answer
from mf_rag.prompts import (
    NOT_IN_SOURCES,
    PII_WARNING,
    REFUSAL_ADVICE,
    REFUSAL_RETURNS,
    REFUSAL_TEMPLATES,
)

# --------------------------------------------------------------------------- L1 PII

# Each entry is (name, pattern). Names are what we report; values are never retained.
PII_PATTERNS: list[tuple[str, str]] = [
    ("pan", r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    ("aadhaar", r"\b[2-9]\d{11}\b"),
    ("phone", r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"),
    ("email", r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    ("account_number", r"\b(?:account|acc|folio|cusip|dp)\s*(?:no\.?|number|#)\s*[:\s]?\s*\d{4,}\b"),
    ("otp", r"\b(?:otp|one[\s-]?time\s+(?:password|code)|verification\s+code)\b\s*(?:is)?\s*[:\s]?\s*\d{4,6}\b"),
    ("ifsc", r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
    ("demat", r"\b(?:dp\s+id|dpid)\s*[:\s]?\s*\d{6,}\b"),
]

# Patterns whose keyword is spelled in prose, so they must match case-insensitively.
# `pan` and `ifsc` are deliberately left case-sensitive: a PAN is uppercase by definition,
# and matching case-insensitively would make `[A-Z]{5}` fire on ordinary words.
_CASELESS = frozenset({"email", "account_number", "otp", "demat"})

_COMPILED: list[tuple[str, re.Pattern[str]]] = [
    (name, re.compile(pattern, re.IGNORECASE if name in _CASELESS else 0))
    for name, pattern in PII_PATTERNS
]


def detect_pii(text: str) -> list[str]:
    """Names of the PII patterns present. Never the values (NFR-4)."""
    if not text:
        return []
    return [name for name, pattern in _COMPILED if pattern.search(text)]


@dataclass
class GuardResult:
    """L1 verdict. `matched_pattern_names` is safe to log; the text is not retained."""

    allowed: bool
    reason: str | None = None
    matched_pattern_names: list[str] = field(default_factory=list)


def guard_input(text: str, enabled: bool = True) -> GuardResult:
    """Block input containing personal identifiers.

    The matched substring is deliberately dropped before this returns: the caller gets a
    reason string and a list of names, and nothing else.
    """
    if not enabled:
        return GuardResult(allowed=True)
    hits = detect_pii(text)
    if not hits:
        return GuardResult(allowed=True)
    # NFR-4: never log the value. We report only which pattern fired.
    return GuardResult(
        allowed=False,
        reason=(
            "Blocked before retrieval and generation: the input contains a personal "
            f"identifier (pattern: {', '.join(sorted(hits))}). Nothing was stored or sent."
        ),
        matched_pattern_names=sorted(hits),
    )


# --------------------------------------------------------------------------- L2 intent

INTENTS = ("factual", "advisory", "portfolio", "returns", "out_of_corpus")

# Order matters: the first family that matches wins, and `returns`/`portfolio` are checked
# before `advisory` because "should I allocate to the fund that gave the best returns" is
# more usefully refused as a returns question.
_ADVISORY = re.compile(
    r"\b(should i|should we|would you (?:recommend|suggest)|do you (?:recommend|suggest)|"
    r"recommend|is it (?:good|safe|worth) (?:to|for)|good (?:fund|scheme|option) to|"
    r"which (?:fund|scheme) (?:should|is best|is better)|best (?:fund|scheme|option|choice) (?:for|to)|"
    r"worth (?:investing|buying)|good to invest|help me choose|pick a (?:fund|scheme))\b",
    re.IGNORECASE,
)
_PORTFOLIO = re.compile(
    r"\b(my (?:portfolio|allocation|asset allocation|holdings|sip plan)|"
    r"portfolio (?:allocation|rebalance|review|advice)|rebalance|"
    r"(?:how|what) (?:much|percentage) (?:should|do) i (?:allocate|invest|put)|"
    r"sip plan for me|split (?:my|across)|asset allocation|divert my|"
    r"i already hold|i currently hold|add to my)\b",
    re.IGNORECASE,
)
_RETURNS = re.compile(
    r"\b(returns?|cagr|projected?|projection|expected (?:growth|return)|"
    r"how much (?:will|would|could) (?:i )?(?:get|earn|make)|"
    r"after \d+ years?|in \d+ years? (?:how much|what will)|"
    r"% return|will (?:give|return|generate|earn|deliver)|performance (?:of|vs|versus)|"
    r"which (?:fund|scheme) (?:performed|gave|returns)|best performing)\b",
    re.IGNORECASE,
)


def classify_intent(text: str) -> str:
    """`factual` | `advisory` | `portfolio` | `returns` | `out_of_corpus`.

    `out_of_corpus` means the text names a fund outside the five-scheme registry: the
    assistant has never read that document, so the honest answer is "not in my sources",
    not a guess.
    """
    from mf_rag.entities import out_of_registry_schemes

    if not text or not text.strip():
        return "factual"
    if out_of_registry_schemes(text):
        return "out_of_corpus"
    if _RETURNS.search(text):
        return "returns"
    if _PORTFOLIO.search(text):
        return "portfolio"
    if _ADVISORY.search(text):
        return "advisory"
    return "factual"


# Educational links for refusals, drawn from the ingested set only. Never invented.
_EDUCATION_SOURCE = "hdfcfund_fees_education"
_FACTSHEET_SOURCE = "hdfcfund_investor_faq"
# For an out-of-corpus fund, the honest destination is the regulator's fund directory, not
# an HDFC page: the fund we do not have will not be there.
_DIRECTORY_SOURCE = "amfi_home"

# Factsheet-style official pages by scheme, for a returns refusal about a named scheme.
_SCHEME_FACTSHEET: dict[str, str] = {}


def _educational_url() -> str:
    from mf_rag.generate import SOURCE_URLS

    return SOURCE_URLS[_EDUCATION_SOURCE]


def _factsheet_url(question: str) -> str:
    """Official factsheet-style page for the scheme named in `question`."""
    from mf_rag.entities import detect_entity
    from mf_rag.generate import SOURCE_URLS

    _, where = detect_entity(question)
    if where:
        scheme = where.get("scheme")
        if scheme and scheme in _SCHEME_FACTSHEET:
            return SOURCE_URLS[_SCHEME_FACTSHEET[scheme]]
    return SOURCE_URLS[_FACTSHEET_SOURCE]


def _load_scheme_factsheets() -> None:
    """Populate `_SCHEME_FACTSHEET` from the ingested sources (called lazily, once)."""
    if _SCHEME_FACTSHEET:
        return
    from mf_rag.sources import SOURCES

    for source in SOURCES:
        scheme = source.get("scheme")
        if not scheme or scheme in _SCHEME_FACTSHEET:
            continue
        if source.get("is_official") and source["doc_type"] in {"scheme_page", "faq", "fees"}:
            _SCHEME_FACTSHEET[scheme] = source["source_id"]


def refusal_link(intent: str, question: str) -> str:
    """The single educational link for a refusal, always from the ingested set."""
    from mf_rag.generate import SOURCE_URLS

    if intent == "returns":
        _load_scheme_factsheets()
        return _factsheet_url(question)
    if intent == "out_of_corpus":
        return SOURCE_URLS[_DIRECTORY_SOURCE]
    return SOURCE_URLS[_EDUCATION_SOURCE]


def refusal_template(intent: str) -> str:
    return REFUSAL_TEMPLATES.get(intent, REFUSAL_ADVICE)


def refusal_for(intent: str, question: str) -> Answer:
    """Compose the refusal `Answer`: correct template + one in-registry educational link."""
    link = refusal_link(intent, question)
    text = refusal_template(intent).format(link=link)
    return Answer(
        text=text,
        source_url=link,
        fetched_at="",
        intent=intent,
        trace=_empty_trace(intent, text, link),
        is_refusal=True,
    )


def pii_answer() -> Answer:
    """The L1 refusal. No URL from retrieval, just the educational link."""
    from mf_rag.generate import _trace

    link = _educational_url()
    text = PII_WARNING.format(link=link)
    return Answer(
        text=text,
        source_url=link,
        fetched_at="",
        intent="pii",
        trace=_trace("", "pii", {}, [], link, text, 0, validators_run=["pii_guard"], pii_blocked=True),
        is_refusal=True,
    )


def _empty_trace(intent: str, text: str, url: str) -> "Any":  # type: ignore[name-defined]
    from mf_rag.models import Trace

    return Trace(
        query_hash="",
        intent=intent,
        pii_blocked=False,
        entity_detected=None,
        where_filter=None,
        hits=[],
        chunks_used=[],
        selected_source_url=url,
        validators_run=["refusal_template"],
        repair_used=False,
        latency_ms=0,
        answer=text,
    )


__all__ = [
    "INTENTS",
    "PII_PATTERNS",
    "GuardResult",
    "classify_intent",
    "detect_pii",
    "guard_input",
    "pii_answer",
    "refusal_for",
    "refusal_link",
]
