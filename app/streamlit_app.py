"""FundFacts - Streamlit UI for the HDFC Mutual Fund FAQ RAG assistant (Phase 8).

RAG stage: presentation. Every answer on screen comes from `mf_rag.pipeline.query`; this
module adds no retrieval, no generation and no post-editing of the answer text.

    streamlit run app/streamlit_app.py

The layout is a transcription of the Stitch "Objective Institutional Modern" design system
in `stitch_fundfacts_ui_design_system/`, screen by screen. Each render function below names
the screen it comes from, so a diff against the mockups stays readable:

    fundfacts_welcome_empty_state            -> render_verify_row (pill variant),
                                                 render_welcome, render_example_chips,
                                                 render_guardrail_notice
    fundfacts_loading_state                  -> render_loading, render_dock(busy=True)
    fundfacts_factual_answer_collapsed_trace -> render_index_row (dot variant),
                                                 render_user_turn, render_answer_card,
                                                 render_meta_tiles, render_dock
    fundfacts_logo_mark                      -> theme.LOGO_SVG (page icon + brand tile)

One centred 760px reading column, deep-institutional navy text on a muted teal anchor, pill
controls and white assistant cards with 1px `#E2E8F0` borders. Amber is reserved exclusively
for guardrails, PII blocks and scope refusals, per the design system's explicit prohibition on
green/red data cues.

Two rendering rules are load-bearing and easy to break:

1. A turn is drawn in exactly two halves - `render_user_turn` for the bubble, then
   `render_answer_stack` for the card and tiles. Never call a combined helper for the
   live turn as well, or the question appears twice.
2. A submitted question is picked up from `session_state["pending_query"]` on the next run and
   rendered as a "live" turn with no answer yet, so the bubble can be drawn above the loading
   card and the page keeps its top-to-bottom order within a single pass.

Exit gate (IMPLEMENTATION.md Phase 8):
    welcome line + 3 chips + disclaimer | clickable link | distinct refusal and PII
    rendering | runs with no API key via EchoClient behind a banner | nothing persisted but
    session_state
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

import theme
from theme import esc, esc_attr, icon
from mf_rag.answer import OFFICIAL_SOURCES
from mf_rag.config import AppConfig, get_config
from mf_rag.entities import detect_entity
from mf_rag.generate import SOURCE_URLS
from mf_rag.llm import describe_client, get_llm
from mf_rag.models import Answer
from mf_rag.pipeline import query, query_hash
from mf_rag.prompts import DISCLAIMER, extract_urls, strip_freshness
from mf_rag.sources import SOURCES
from mf_rag.store import verify_collection

APP_TITLE = "FundFacts - Mutual Fund FAQ Assistant"

# What the index row may honestly claim, given the registry that was actually ingested.
CORPUS_SCOPE = "HDFC AMC scheme pages \u00b7 AMFI \u00b7 SEBI"
# The provenance stamps on the top meta row, verbatim from the two screens.
SYNC_STAMP = "v2.4 Audit Sync"
INDEX_LABEL = "Audited Public Document Index"

# FR-7.2: exactly three example questions, one per intent shape. The tag, glyph and anchor
# are the design system's card chrome (`fundfacts_welcome_empty_state`, Card 1-3).
EXAMPLES: tuple[tuple[str, str, str, str], ...] = (
    (
        "Expense & Fees",
        "receipt_long",
        "What is the expense ratio of HDFC Large Cap Fund?",
        "KIM \u00b7 Sec 4(b)",
    ),
    (
        "Scheme Rules",
        "lock_clock",
        "What is the lock-in period for HDFC ELSS Tax Saver Fund?",
        "SID \u00b7 Sec 80C",
    ),
    (
        "Operations",
        "description",
        "How do I download my capital gains statement?",
        "Portal Procedure",
    ),
)

# The five schemes in scope (sources.SCHEME_REGISTRY), shown as scope pills.
SCHEME_LABELS: tuple[str, ...] = (
    "Large Cap",
    "Flexi Cap",
    "ELSS Tax Saver",
    "Small Cap",
    "Balanced Advantage",
)

# Compliance copy. FR-7.3: the dock caption is always visible, and both it and the footer
# open with the design system's required phrase "Facts-only. No investment advice."
DOCK_CAPTION = "Facts-only. No investment advice. Official SID / SAI / KIM references."
# The footer renders the canonical constant rather than a near-copy of it. FR-8.5 exports
# `prompts.DISCLAIMER` as a reusable file and the exit gate is that the exported string
# matches the UI's, which can only hold if the UI renders the constant itself - a
# hand-typed variant here would silently drift from the exported disclaimer.
FOOTER_TEXT = DISCLAIMER
SEARCH_HINT = "Click to preview lookup"

WELCOME_TITLE = "FundFacts \u2013 Mutual Fund FAQ Assistant"
WELCOME_LEDE = (
    "Hi, I&rsquo;m <strong>FundFacts</strong>, a facts-only assistant for HDFC Mutual Fund "
    "schemes: Large Cap, Flexi Cap, ELSS Tax Saver, Small Cap, and Balanced Advantage. "
    "I answer strictly from official public sources with exact citations. No investment "
    "advice or performance speculation."
)

GUARDRAIL_TITLE = "Strict Regulatory Guardrail Active"
GUARDRAIL_BODY = (
    "This workspace parses verified regulatory filings and operational handbooks. Questions "
    "containing requests for NAV projections, buy/sell recommendations, portfolio matching, "
    "or private investor identifiers (PAN / folio) will be rejected automatically."
)

LOADING_STAGES = (
    "Searching official sources\u2026",
    "Matching Scheme Information Document\u2026",
    "Extracting the matching disclosure\u2026",
)

DOC_TYPE_LABELS: dict[str, str] = {
    "scheme_page": "Official scheme page",
    "factsheet": "Monthly factsheet",
    "kim_sid": "KIM / SID extract",
    "faq": "Investor FAQ",
    "fees": "Fee disclosure",
    "riskometer": "Riskometer",
    "guide": "Investor guide",
    "educational": "Investor education",
}

REFUSAL_HEADLINES: dict[str, str] = {
    "advisory": "Advice request \u2014 refused by guardrail",
    "portfolio": "Portfolio request \u2014 refused by guardrail",
    "returns": "Return projection \u2014 refused by guardrail",
    "out_of_corpus": "Outside the indexed corpus \u2014 refused",
    "pii": "Personal identifier detected \u2014 input blocked",
}

_SPIN_RING = (
    '<svg class="ff-ring" width="20" height="20" viewBox="0 0 24 24" fill="none" '
    'aria-hidden="true">'
    '<circle class="ff-ring-track" cx="12" cy="12" r="9" stroke="currentColor" '
    'stroke-width="2.5" opacity="0.3"></circle>'
    '<path class="ff-run" fill="currentColor" opacity="0.9" '
    'd="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 '
    '1.135 5.824 3 7.938l3-2.647z"></path></svg>'
)


# --- cached runtime handles ------------------------------------------------


@st.cache_resource(show_spinner=False)
def _llm(cfg: AppConfig) -> tuple[object, bool]:
    """One LLM client per config. The LLM itself is never constructed in this module."""
    return get_llm(cfg)


@st.cache_data(show_spinner=False, ttl=30)
def _index_stats() -> dict[str, object]:
    """Chroma count + model for the index row. Re-read at most every 30s, never per render."""
    cfg = get_config()
    try:
        stats: dict[str, object] = dict(verify_collection(cfg))
    except Exception as exc:  # noqa: BLE001 - a missing index must not crash the UI
        return {"count": 0, "embedding_model": "index not built", "error": str(exc)}
    return stats


def _clock() -> str:
    """Wall-clock stamp for the user turn meta line, e.g. `10:42 AM`."""
    return datetime.now().strftime("%I:%M %p").lstrip("0")


# --- html helpers ----------------------------------------------------------


def _linkify(text: str) -> str:
    """Escape `text`, then turn every URL into an anchor. Never emits raw model text."""
    body = esc(text)
    for url in extract_urls(text):
        safe = esc_attr(url)
        body = body.replace(
            safe, f'<a href="{safe}" target="_blank" rel="noopener noreferrer">{safe}</a>'
        )
    return body


def _paragraphs(text: str) -> str:
    parts = [p.strip() for p in str(text or "").split("\n\n") if p.strip()]
    return "".join(f"<p>{_linkify(part)}</p>" for part in parts) or "<p></p>"


def _source_label(url: str | None, answer: Answer) -> tuple[str, bool]:
    """(human label, is_official) for the citation tag in the card header."""
    if not url:
        return "No citation", False
    source_id = next((sid for sid, known in SOURCE_URLS.items() if known == url), None)
    official = source_id in OFFICIAL_SOURCES if source_id else False
    hit = next((h for h in answer.trace.hits if h.get("source_url") == url), {})
    row = next((r for r in SOURCES if r.get("source_url") == url), {})
    doc_type = str(hit.get("doc_type") or row.get("doc_type") or "")
    if official:
        return DOC_TYPE_LABELS.get(doc_type, "Official public document"), True
    return DOC_TYPE_LABELS.get(doc_type, "Educational reference"), False


# --- top meta row ----------------------------------------------------------


def render_verify_row(stats: dict[str, object]) -> None:
    """Welcome state: the KNOWLEDGE BASE pill plus a mono sync stamp.

    `fundfacts_welcome_empty_state`, "Verified Corpus Meta Tag".
    """
    model = str(stats.get("embedding_model", "")).split("/")[-1]
    st.markdown(
        '<div class="ff-verify-row">'
        '<span class="ff-kb-pill">'
        '<span class="ff-dot ff-pulse"></span>'
        f"KNOWLEDGE BASE: {esc(CORPUS_SCOPE)}</span>"
        f'<span class="ff-mono" style="color:var(--ff-outline)">{esc(stats.get("count", 0))} '
        f"chunks &middot; {esc(SYNC_STAMP)} &middot; {esc(model)}</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_index_row(stats: dict[str, object]) -> None:
    """Conversation state: a pulsing dot, the index name, and a mono provenance stamp.

    `fundfacts_factual_answer_collapsed_trace`, top meta row.
    """
    model = str(stats.get("embedding_model", "")).split("/")[-1]
    st.markdown(
        '<div class="ff-index-row">'
        '<span class="ff-index-label">'
        '<span class="ff-dot ff-dot-lg ff-pulse"></span>'
        f"{esc(INDEX_LABEL)}</span>"
        f'<span class="ff-mono" style="color:var(--ff-secondary)">{esc(stats.get("count", 0))} '
        f"chunks &middot; {esc(model)}</span>"
        "</div>",
        unsafe_allow_html=True,
    )


# --- welcome block (fundfacts_welcome_empty_state) -------------------------


def render_welcome() -> None:
    st.markdown(
        '<div class="ff-brand-row">'
        f'<span class="ff-brand-tile">{theme.LOGO_SVG}</span>'
        f'<span class="ff-eyebrow">{esc("Statutory Repository Assistant")}</span>'
        "</div>"
        f'<h1 class="ff-title">{WELCOME_TITLE}</h1>'
        f'<p class="ff-lede">{WELCOME_LEDE}</p>',
        unsafe_allow_html=True,
    )
    pills = "".join(
        f'<span class="ff-pill">{icon("check_circle", 14, fill=True)} {esc(name)}</span>'
        for name in SCHEME_LABELS
    )
    st.markdown(f'<div class="ff-pill-row">{pills}</div>', unsafe_allow_html=True)


def render_example_chips() -> None:
    st.markdown(
        '<div class="ff-suggest-head">'
        f'<span class="ff-suggest-label">{esc("Try one of these queries")}</span>'
        f'<span class="ff-hint">{esc(SEARCH_HINT)}</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    for column, (tag, glyph, question, anchor) in zip(st.columns(3), EXAMPLES):
        with column:
            st.markdown(
                '<div class="ff-chip"></div>'
                '<div class="ff-chip-head">'
                f'<span class="ff-chip-badge">{esc(tag)}</span>'
                f'<span class="ff-chip-glyph">{icon(glyph, 18)}</span>'
                "</div>",
                unsafe_allow_html=True,
            )
            # FR-7.2: a real st.button that seeds session_state, so a chip click is a
            # first-class widget action rather than a JS shim.
            if st.button(question, key=f"example_{anchor}", width="stretch"):
                st.session_state["pending_query"] = question
                st.rerun()
            st.markdown(
                f'<div class="ff-chip-foot"><span>{esc(anchor)}</span>'
                f'<span class="ff-chip-arrow">{icon("arrow_forward", 16)}</span></div>',
                unsafe_allow_html=True,
            )


def render_guardrail_notice() -> None:
    st.markdown(
        '<div class="ff-guardrail">'
        f'<span>{icon("policy", 22)}</span>'
        f'<div><div class="ff-guardrail-title">{esc(GUARDRAIL_TITLE)}</div>'
        f'<div class="ff-guardrail-body">{esc(GUARDRAIL_BODY)}</div></div></div>',
        unsafe_allow_html=True,
    )


# --- user turn (fundfacts_factual_answer_collapsed_trace) ------------------


def render_user_turn(question: str, stamp: str) -> None:
    """The right-aligned teal bubble. Drawn exactly once per turn - see the module docstring."""
    st.markdown(
        '<div class="ff-turn"><div class="ff-user-wrap">'
        f'<div class="ff-user">{esc(question)}</div>'
        f'<div class="ff-user-meta">{esc(stamp)}</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )


# --- loading state (fundfacts_loading_state) -------------------------------


def render_loading(question: str, cfg: AppConfig) -> None:
    """Scope chip bar, loading card, retrieval skeleton, breadcrumb, target-scheme card.

    Every value shown is real: the scope and chunk depth come from the config, the entity
    from `detect_entity`, the hash from `query_hash`. Nothing is faked.
    """
    kind, where = detect_entity(question)
    target = str((where or {}).get("scheme") or (where or {}).get("category") or "")

    st.markdown(
        '<div class="ff-load-bar"><div class="ff-load-pills">'
        f'<span class="ff-scope-pill">{icon("folder_open", 14)} Scope: {esc(CORPUS_SCOPE)}</span>'
        '<span class="ff-scope-mono">'
        f"chunk_depth: {esc(cfg.retrieval.context_chunks)}"
        f" &middot; min_score: {esc(cfg.retrieval.min_score)}</span>"
        "</div>"
        '<div class="ff-load-live">'
        '<span class="ff-dot ff-dot-lg ff-ping"></span>Retrieving index node</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    stage = LOADING_STAGES[abs(hash(question)) % len(LOADING_STAGES)]
    st.markdown(
        '<div class="ff-loading">'
        '<div class="ff-loading-head">'
        f'<span class="ff-avatar ff-avatar-lg ff-avatar-fixed">{icon("verified", 18)}</span>'
        f"{_SPIN_RING}"
        '<div class="ff-loading-text">'
        f'<div class="ff-label ff-blink">{esc(stage)}</div>'
        f'<div class="ff-mono ff-truncate" style="color:var(--ff-secondary)">'
        f"query {esc(query_hash(question))}</div>"
        "</div>"
        f'<span class="ff-load-tag">entity: {esc(kind or "none")}</span>'
        "</div>"
        '<div class="ff-loading-panel">'
        '<div class="ff-loading-label">'
        f'<span>{icon("manage_search", 15)} <b>Statutory Vector Retrieval</b></span>'
        '<span class="ff-mono">cosine: &mdash;</span></div>'
        '<div class="ff-bar"><span></span></div>'
        '<div class="ff-skeleton w-56"></div>'
        '<div class="ff-skeleton w-34"></div>'
        "</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="ff-breadcrumb"><span class="ff-dot"></span>'
        "Accessing HDFC AMC scheme pages, AMFI disclosures and SEBI statutory filings "
        f"&mdash; {esc('entity-filtered' if where else 'global cosine search')}</div>"
        '<div class="ff-readonly"><div class="ff-readonly-who">'
        f'<span class="ff-avatar-sm">{icon("menu_book", 16)}</span>'
        "<div>"
        f'<div class="ff-label">Target scheme: {esc(target or "not named in the question")}</div>'
        '<div class="ff-caption" style="color:var(--ff-secondary)">Read-only retrieval '
        "&mdash; the model may only echo the source URL the system selects</div>"
        "</div></div>"
        f'<span class="ff-readonly-badge">{icon("lock", 13)} ReadOnly Verified Node</span>'
        "</div>",
        unsafe_allow_html=True,
    )


# --- answer card (fundfacts_factual_answer_collapsed_trace) ----------------


def render_citation_tag(answer: Answer) -> str:
    label, official = _source_label(answer.source_url, answer)
    cls = "ff-tag" if official else "ff-tag ff-tag-neutral"
    glyph = icon("verified" if official else "menu_book", 15, fill=official)
    return f'<span class="{cls}">{glyph} {esc(label)}</span>'


def render_source_pill(answer: Answer, label: str) -> str:
    if not answer.source_url:
        return ""
    url = esc_attr(answer.source_url)
    return (
        f'<a class="ff-source-pill" href="{url}" target="_blank" '
        f'rel="noopener noreferrer">{icon("link", 16)} Source: {esc(label)} '
        f"{icon('open_in_new', 14)}</a>"
    )


def render_meta_tiles(answer: Answer) -> str:
    trace = answer.trace
    if not trace.hits:
        return ""
    used = len(trace.chunks_used)
    top = trace.hits[0]
    doc_label = DOC_TYPE_LABELS.get(str(top.get("doc_type") or ""), "Official document")
    scope = str(top.get("scheme") or top.get("category") or "All indexed sources")
    tiles = (
        f'<div class="ff-meta-tile"><div><span class="ff-caption">Primary document</span>'
        f'<span class="ff-label">{esc(doc_label)}</span></div>'
        f'{icon("description", 20)}</div>'
        f'<div class="ff-meta-tile"><div><span class="ff-caption">Context used</span>'
        f'<span class="ff-label ff-num">{used} of {len(trace.hits)} chunks '
        f"&middot; {esc(scope)}</span></div>{icon('layers', 20)}</div>"
    )
    return f'<div class="ff-card-meta">{tiles}</div>'


def render_answer_card(answer: Answer) -> None:
    """One assistant turn. Refusals and PII blocks get the amber guardrail treatment."""
    trace = answer.trace
    blocked = answer.intent == "pii" or trace.pii_blocked
    refusal = answer.is_refusal or blocked
    card_cls = "ff-card ff-card-refusal" if refusal else "ff-card"

    if blocked:
        headline = REFUSAL_HEADLINES["pii"]
        tag = (
            '<span class="ff-tag ff-tag-amber">'
            f'{icon("lock", 15)} Input blocked before retrieval</span>'
        )
    elif refusal:
        headline = REFUSAL_HEADLINES.get(answer.intent, REFUSAL_HEADLINES["advisory"])
        tag = f'<span class="ff-tag ff-tag-amber">{icon("policy", 15)} Guardrail</span>'
    else:
        headline = "FundFacts"
        tag = render_citation_tag(answer)

    body_text = strip_freshness(answer.text)
    label, _ = _source_label(answer.source_url, answer)
    pill = render_source_pill(answer, label)
    source_row = f'<div class="ff-source-row">{pill}</div>' if pill else ""

    st.markdown(
        f'<div class="{card_cls}">'
        '<div class="ff-card-head"><div class="ff-card-who">'
        f'<span class="ff-avatar">{icon("shield", 17, fill=True)}</span>'
        f'<span class="ff-label">{esc(headline)}</span></div>{tag}</div>'
        f'<div class="ff-answer">{_paragraphs(body_text)}</div>'
        f"{source_row}"
        "</div>",
        unsafe_allow_html=True,
    )


# --- conversation ----------------------------------------------------------


def render_answer_stack(answer: Answer) -> None:
    """Everything below the user bubble: the card, then the metadata tiles.

    Split out from `render_turn` so the live path can draw the bubble once, show the loading
    state, then fill in the answer without the question being echoed twice.
    """
    render_answer_card(answer)
    tiles = render_meta_tiles(answer)
    if tiles:  # an empty markdown block still occupies a slot and adds phantom height
        st.markdown(tiles, unsafe_allow_html=True)


def render_turn(turn: dict) -> None:
    """One stored exchange, replayed verbatim on every rerun."""
    render_user_turn(turn["question"], turn["stamp"])
    render_answer_stack(turn["answer"])


# --- pinned dock -----------------------------------------------------------


def _on_send() -> None:
    """Form submit callback. Runs before the rerun, so the next pass sees the question."""
    text = str(st.session_state.get("ff_query", "")).strip()
    if text:
        st.session_state["pending_query"] = text


def render_dock(empty: bool) -> None:
    """The floating inquiry input, bounded inside the 760px column.

    `empty` draws the welcome-state shape (leading search disc, "Ask a factual question").
    Otherwise the follow-up shape: no disc, "Ask a follow-up factual question".

    There is deliberately no disabled variant here. `fundfacts_loading_state` shows the dock
    greyed out mid-flight, but Streamlit renders a whole pass at a time, so a "busy" dock
    would be committed in the same pass that already contains the answer and would leave the
    user with no input to type into. The loading screen's real content - the scope chip bar,
    the loading card, the retrieval skeleton, the breadcrumb and the target-scheme anchor -
    is rendered by `render_loading`, which is scoped to the conversation and disappears with it.
    """
    placeholder = (
        "Ask a factual question about a scheme\u2026"
        if empty
        else "Ask a follow-up factual question\u2026"
    )
    with st.form("ff_dock_form", clear_on_submit=True, enter_to_submit=True, border=False):
        if empty:
            lead, field, send = st.columns([0.6, 8.8, 0.6], vertical_alignment="center")
            with lead:
                st.markdown(
                    f'<span class="ff-dock-lead">{icon("search", 18)}</span>',
                    unsafe_allow_html=True,
                )
        else:
            field, send = st.columns([9.4, 0.6], vertical_alignment="center")

        with field:
            st.text_input(
                "Ask a question",
                key="ff_query",
                label_visibility="collapsed",
                placeholder=placeholder,
            )
        with send:
            st.form_submit_button(
                "Send", icon=":material/arrow_upward:", on_click=_on_send
            )
        st.markdown(
            f'<div class="ff-dock-caption">{icon("verified", 14, fill=True)} '
            f"{esc(DOCK_CAPTION)}</div>",
            unsafe_allow_html=True,
        )


def render_footer() -> None:
    st.markdown(
        f'<div class="ff-footer">{icon("verified_user", 16)} {esc(FOOTER_TEXT)}</div>',
        unsafe_allow_html=True,
    )


# --- app -------------------------------------------------------------------


def init_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("pending_query", "")


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=str(Path(__file__).resolve().parent / "assets" / "fundfacts_logo.svg"),
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    theme.inject()
    init_state()

    config = get_config()
    client, degraded = _llm(config)
    stats = _index_stats()

    history: list[dict] = st.session_state["history"]
    # A submitted question arrives here on the run after the dock, so it can be rendered in
    # page order with the bubble above the loading card.
    submitted = str(st.session_state.pop("pending_query", "") or "").strip()
    live: dict | None = None
    if submitted:
        live = {
            "question": submitted,
            "stamp": _clock(),
            "answer": None,
        }
    show_welcome = not history and live is None

    if show_welcome:
        render_verify_row(stats)
    else:
        render_index_row(stats)

    if stats.get("error"):
        st.error(
            "The vector index is not ready, so no question can be answered yet. "
            f"Build it with `python scripts/build_index.py --stage all`. ({stats['error']})"
        )
    if degraded:
        st.warning(
            f"Running without an LLM key ({describe_client(client)}). Answers are quoted "
            "verbatim from the retrieved source text instead of being written. "
            "Set LLM_API_KEY in .env for generated answers."
        )

    if show_welcome:
        render_welcome()

    for turn in history:
        render_turn(turn)

    if live is not None:
        render_user_turn(live["question"], live["stamp"])
        slot = st.empty()
        with slot.container():
            render_loading(live["question"], config)
            with st.spinner("Retrieving from the official HDFC index\u2026"):
                answer = query(live["question"], config)
        slot.empty()
        render_answer_stack(answer)
        history.append({**live, "answer": answer})

    if show_welcome:
        render_example_chips()
        render_guardrail_notice()

    render_dock(empty=show_welcome)
    render_footer()


if __name__ == "__main__":
    main()