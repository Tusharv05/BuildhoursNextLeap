"""L3 output validators (Phase 7).

Everything here is a regex or a count. The point of the last guard layer is that
compliance must not depend on the model behaving:

| Validator          | Rule                                     | Repair              |
|--------------------|------------------------------------------|---------------------|
| sentence count     | <= 3                                     | truncate            |
| link count         | exactly 1                                | re-append           |
| link membership    | URL is in the ingested source set        | swap to top chunk   |
| official preference| if an official source exists, cite it    | swap                |
| advice lexicon     | no "you should" / "recommend"            | REFUSAL_ADVICE      |
| returns lexicon    | no "will give" / "CAGR of" / "projected" | REFUSAL_RETURNS     |
| freshness          | line present and matches computed date   | append the real one |

`validate` returns a `ValidationResult` rather than raising, because the pipeline needs to
know *which* rules fired in order to show them in the trace (FR-7.5) and decide whether a
repair pass is worth attempting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from mf_rag.generate import (
    EDUCATIONAL_URL,
    OFFICIAL_SOURCES,
    SOURCE_URLS,
    select_source_url,
)
from mf_rag.prompts import (
    FRESHNESS_PREFIX,
    URL_RE,
    extract_freshness,
    extract_urls,
    freshness_line,
    strip_freshness,
)

# Advice language. Matched case-insensitively. Phrased to catch the natural shapes of a
# recommendation without firing on factual statements like "the fund is suitable for
# investors who seek long-term growth" (which is a disclosure quote, not advice).
ADVICE_LEXICON: list[str] = [
    r"\byou should\b",
    r"\bi (?:would |'d )?recommend\b",
    r"\brecommend(?:ed|ation)?\b",
    r"\bbest choice\b",
    r"\bideal for you\b",
    r"\bmust buy\b",
    r"\bsuitable for you\b",
    r"\bgo for (?:it|this|the)\b",
    r"\bmy pick\b",
    r"\bworth (?:buying|investing)\b",
    r"\bconsider investing in\b",
]

# Returns language. These are the shapes a model uses when it starts projecting.
RETURNS_LEXICON: list[str] = [
    r"\bwill (?:give|return|generate|deliver|earn)\b",
    r"\bexpected return\b",
    r"\bcagr of\b",
    r"\bprojected\b",
    r"% return",
    r"\bexpected (?:cagr|growth)\b",
    r"\bwill outperform\b",
    r"\bguaranteed returns?\b",
    r"\bhow much will (?:i|we) (?:get|earn|make)\b",
]

# A sentence ends at . ! ? or newline, but never at a period between two digits ("1.5%")
# and never at an abbreviation dot we know about.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_DECIMAL_DOT = re.compile(r"(?<=\d)\.(?=\d)")
_ABBREV = re.compile(r"\b(?:e\.g|i\.e|etc|vs|Mr|Ms|Dr|No|St|approx)\.", re.IGNORECASE)


def count_sentences(text: str) -> int:
    """Sentence count that survives decimals, percentages and abbreviations.

    Two things are deliberately not counted as sentences:
      - a period between digits, so "The exit load is 1.5%." is one sentence, not three
      - the trailing `Last updated from sources:` line, which PRD 7 requires as a suffix
        but which is metadata rather than prose. Counting it would quietly cap the answer
        body at 2 sentences while the contract says 3.
    """
    if not text or not text.strip():
        return 0
    body = strip_freshness(text)
    if not body.strip():
        return 0
    parts = _split_sentences(body)
    return sum(1 for p in parts if p.strip())


def _split_sentences(text: str) -> list[str]:
    """Split into sentences, restoring protected decimal/abbreviation dots."""
    protected = _DECIMAL_DOT.sub("\x00", text)
    protected = _ABBREV.sub(lambda m: m.group(0).replace(".", "\x00"), protected)
    parts = _SENTENCE_SPLIT.split(protected)
    return [p.replace("\x00", ".").strip() for p in parts]


@dataclass
class ValidationResult:
    """Outcome of L3. `repaired_text` is what the pipeline should emit."""

    ok: bool
    violations: list[str] = field(default_factory=list)
    repaired_text: str = ""
    repaired: bool = False

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok


# A negation within this many characters before a lexicon hit means the model is declining
# rather than advising: "I can't give investment advice or recommend a scheme" is the
# refusal template, and flagging it would make the system reject its own correct output.
_NEGATION = re.compile(
    r"\b(?:can'?t|cannot|won'?t|do not|don'?t|does not|doesn'?t|not able to|never)\b",
    re.IGNORECASE,
)
_NEGATION_WINDOW = 45


def _lexicon_hits(patterns: Sequence[str], text: str) -> list[str]:
    """Patterns that fire on an *asserted* use, ignoring negated ones."""
    hits: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            prefix = text[max(0, match.start() - _NEGATION_WINDOW) : match.start()]
            if _NEGATION.search(prefix):
                continue
            hits.append(pattern)
            break
    return hits


def truncate_to_sentences(text: str, limit: int = 3) -> str:
    """First `limit` sentences, with the freshness line removed.

    The caller re-appends the freshness line, so dropping it here avoids ending up with two.
    """
    body = strip_freshness(text)
    kept = [s for s in _split_sentences(body) if s.strip()][:limit]
    return ". ".join(s.rstrip(".") for s in kept).strip()


def validate(
    answer_text: str,
    ctx: dict | None = None,
    max_sentences: int = 3,
    expected_url: str | None = None,
    expected_date: str | None = None,
) -> ValidationResult:
    """Check an answer against the PRD 7 response contract.

    `ctx` is optional context that widens the checks:
      - `official_url`: the URL the system would have preferred
      - `date`: the freshness date computed from chunk metadata
      - `min_score` / `intent`: carried for the pipeline trace only
    """
    ctx = ctx or {}
    text = answer_text or ""
    violations: list[str] = []
    repaired = text

    # 1. sentence count
    sentences = count_sentences(text)
    if sentences > max_sentences:
        violations.append(f"sentence_count>{max_sentences}")
        repaired = truncate_to_sentences(repaired, max_sentences)

    # 2. exactly one URL
    urls = extract_urls(repaired)
    target = expected_url if expected_url is not None else ctx.get("official_url")
    if len(urls) != 1:
        violations.append(f"url_count={len(urls)}")
        if target:
            repaired = strip_freshness(repaired)
            repaired = _replace_urls(repaired, [target])
        elif urls:
            repaired = _replace_urls(strip_freshness(repaired), [urls[0]])
    elif target and urls[0] != target:
        # 3. link membership + official preference
        if urls[0] not in SOURCE_URLS.values():
            violations.append("url_not_in_ingested_sources")
        violations.append("official_preference_not_honoured")
        repaired = _replace_urls(strip_freshness(repaired), [target])
    else:
        if urls[0] not in SOURCE_URLS.values():
            violations.append("url_not_in_ingested_sources")
            if target:
                repaired = _replace_urls(strip_freshness(repaired), [target])

    # 4. advice / returns lexicons
    advice = _lexicon_hits(ADVICE_LEXICON, repaired)
    if advice:
        violations.append("advice_language")
    returns = _lexicon_hits(RETURNS_LEXICON, repaired)
    if returns:
        violations.append("returns_language")

    # 5. freshness
    date = expected_date if expected_date is not None else ctx.get("date")
    found = extract_freshness(repaired)
    if not found:
        violations.append("freshness_missing")
        if date:
            repaired = f"{repaired.strip()}\n\n{freshness_line(date)}"
    elif date and found != date:
        violations.append("freshness_mismatch")
        repaired = strip_freshness(repaired)
        repaired = f"{repaired.strip()}\n\n{freshness_line(date)}"

    # 6. substance. Every check above is structural, and a bare "See: <url>" plus the
    # freshness line satisfies all of them: one sentence, one ingested URL, no advice
    # lexicon, date present. That is a compliant answer that tells the user nothing, and it
    # is exactly what `_replace_urls` manufactures when the model returns only whitespace.
    # This is not hypothetical: with a reasoning model on Groq, 218 of 220 max_tokens went
    # to reasoning, the model returned empty content, and the repair path produced
    # "See: <url>" which was then served as the final answer.
    if not has_substantive_content(repaired):
        violations.append("no_substantive_content")

    return ValidationResult(
        ok=not violations,
        violations=violations,
        repaired_text=repaired,
        repaired=repaired != text,
    )


def _replace_urls(text: str, urls: list[str]) -> str:
    """Drop every URL in `text`, then append `urls` at the end."""
    stripped = URL_RE.sub("", text)
    stripped = re.sub(r"\s{2,}", " ", stripped).strip()
    if not urls:
        return stripped
    tail = " See: " + " ".join(urls)
    return f"{stripped}{tail}" if stripped else tail.strip()


# Words that only point at the citation. An answer made only of these plus a URL carries
# no fact, so it must not be allowed to satisfy the response contract.
_POINTER_RE = re.compile(
    r"\b(see|refer|reference|source|sources|read more|read|more at|at|visit|link|"
    r"further details|details|more information|information|available)\b",
    re.IGNORECASE,
)

# The shortest useful answer in this corpus is a figure plus what it applies to, e.g.
# "Exit load (Direct plan): 1% if redeemed within 1 year." That is 8 words. Five is a
# deliberately low bar: the point is to reject "See: <url>", not to enforce prose quality.
_MIN_CONTENT_WORDS = 5


def substantive_content(text: str) -> str:
    """The part of an answer that actually says something.

    Strips the URL, the freshness suffix, and bare citation pointers. What remains should
    be the fact itself.
    """
    body = strip_freshness(URL_RE.sub(" ", text or ""))
    body = _POINTER_RE.sub(" ", body)
    body = re.sub(r"[^A-Za-z0-9%\.\-\s]", " ", body)
    return re.sub(r"\s{2,}", " ", body).strip()


def has_substantive_content(text: str) -> bool:
    """False for answers like "See: <url>", which are compliant but useless."""
    return len(substantive_content(text).split()) >= _MIN_CONTENT_WORDS


def validate_against_hits(
    answer_text: str, hits, max_sentences: int = 3
) -> ValidationResult:
    """Validate using the real retrieval context: preferred URL and oldest date."""
    dates = sorted(h.chunk.fetched_at for h in hits if h.chunk.fetched_at)
    return validate(
        answer_text,
        ctx={"official_url": select_source_url(hits)},
        max_sentences=max_sentences,
        expected_date=dates[0] if dates else None,
    )


__all__ = [
    "ADVICE_LEXICON",
    "FRESHNESS_PREFIX",
    "OFFICIAL_SOURCES",
    "RETURNS_LEXICON",
    "EDUCATIONAL_URL",
    "ValidationResult",
    "count_sentences",
    "has_substantive_content",
    "substantive_content",
    "truncate_to_sentences",
    "validate",
    "validate_against_hits",
]
