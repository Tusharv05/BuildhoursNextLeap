"""RAG Stage Group A / Stage 1: HTML -> main content extraction.

Strips navigation, cookie banners, and other boilerplate so the chunker only sees
facts. Prefers trafilatura (trained for main-content extraction) and falls back
to BeautifulSoup + a boilerplate blacklist.
"""

from __future__ import annotations

import re
import unicodedata

from bs4 import BeautifulSoup

# Tags whose entire subtree is boilerplate.
BOILERPLATE_TAGS: tuple[str, ...] = (
    "script",
    "style",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "noscript",
    "iframe",
    "svg",
    "template",
)

# Class/id fragments that reliably mark chrome rather than content.
BOILERPLATE_CLASS_HINTS: tuple[str, ...] = (
    "nav",
    "navbar",
    "menu",
    "sidebar",
    "footer",
    "header",
    "cookie",
    "consent",
    "popup",
    "modal",
    "banner",
    "breadcrumb",
    "pagination",
    "advert",
    "promo",
    "subscribe",
    "newsletter",
    "social",
    "share",
    "related",
    "comment",
    "search",
    "login",
    "signin",
)

_WS_RUN = re.compile(r"[ \t\u00a0]+")
_BLANK_RUN = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+\n")
_INLINE_JUNK = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad]")

# Lines that are pure menu/label noise even after tag stripping.
_NOISE_LINE = re.compile(
    r"^(?:"
    r"log\s*in|sign\s*in|sign\s*up|log\s*out|menu|close|search|"
    r"home|back|next|previous|prev|share|download\s*app|"
    r"accept\s*(?:all)?\s*cookies?|cookie\s*settings|"
    r"skip\s*(?:to\s*)?(?:main\s*)?content|"
    r"terms(?:\s*(?:of\s*use|&amp;|and)\s*(?:conditions|privacy))?|privacy\s*policy|"
    r"all\s*rights\s*reserved|"
    r"\d+\s*$"
    r")$",
    re.IGNORECASE,
)


def normalise_whitespace(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _INLINE_JUNK.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RUN.sub(" ", text)
    text = _TRAILING_WS.sub("\n", text)
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip()


def _attribute_markers(tag: object) -> str:
    """Lowercased class + id markers for a tag, tolerating multi-valued attributes.

    Tags already decomposed by the boilerplate pass raise on attribute access,
    so those are treated as marker-free and skipped by the caller.
    """
    markers: list[str] = []
    for attribute in ("class", "id"):
        try:
            value = tag.get(attribute)  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            return ""
        if value is None:
            continue
        if isinstance(value, str):
            markers.append(value)
        else:  # lxml/bs4 return a list for multi-valued attributes
            markers.extend(str(item) for item in value)
    return " ".join(markers).lower()


def _bs4_main_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(list(BOILERPLATE_TAGS)):
        tag.decompose()
    for tag in soup.find_all(True):
        if getattr(tag, "decomposed", False):
            continue
        if any(hint in _attribute_markers(tag) for hint in BOILERPLATE_CLASS_HINTS):
            tag.decompose()

    try:
        title = soup.title.get_text(strip=True) if soup.title else ""
    except (AttributeError, ValueError):
        title = ""
    root = soup.find("main") or soup.find("article") or soup.body or soup
    try:
        return title, root.get_text("\n", strip=True)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"Could not read main content: {exc}") from exc


def _trafilatura_main_text(html: str) -> tuple[str, str]:
    import trafilatura

    extracted = trafilatura.extract(
        html,
        output_format="txt",
        include_comments=False,
        include_tables=True,
        include_links=False,
        favor_precision=True,
        no_fallback=False,
    )
    if not extracted or len(extracted) < 200:
        return "", ""
    metadata = trafilatura.extract_metadata(html)
    title = (metadata.title or "") if metadata else ""
    return title.strip(), extracted


def drop_noise_lines(text: str) -> str:
    """Remove pure menu/label lines, along with the blank lines they leave behind."""
    kept = [line for line in text.split("\n") if not _NOISE_LINE.match(line.strip())]
    return _BLANK_RUN.sub("\n\n", "\n".join(kept)).strip()


def extract_main_text(html: str) -> tuple[str, str]:
    """Return (title, cleaned_text).

    Tries trafilatura first, falls back to BeautifulSoup. Returns empty strings
    if the page has no usable content, which the caller treats as a failure.
    """
    if not html or not html.strip():
        return "", ""

    title, text = ("", "")
    try:
        title, text = _trafilatura_main_text(html)
    except Exception:  # trafilatura is optional; a miss must not fail the build
        title, text = ("", "")

    if not text:
        title, text = _bs4_main_text(html)

    text = drop_noise_lines(normalise_whitespace(text))
    title = unicodedata.normalize("NFKC", title).strip()
    return title, text


def looks_useful(text: str, min_chars: int = 500) -> bool:
    """Cheap quality gate: a page with almost no text is not worth indexing."""
    return len(text) >= min_chars
