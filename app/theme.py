"""Design tokens and global CSS for the FundFacts Streamlit UI (Phase 8).

RAG stage: presentation only. Nothing here reads the corpus, calls a model, or decides an
answer -- it only turns the design system into CSS so `streamlit_app.py` can stay readable.

Source of truth is `stitch_fundfacts_ui_design_system/`, screen by screen:
`objective_institutional_modern/DESIGN.md` for the prose rules, and the four rendered
`code.html` screens for the exact values:

    fundfacts_welcome_empty_state            -> the KNOWLEDGE BASE pill, brand block,
                                                 scheme pills, 3 suggestion chips, guardrail
    fundfacts_loading_state                  -> the scope chip bar, loading card, retrieval
                                                 skeleton, breadcrumb, target-scheme card,
                                                 the disabled dock
    fundfacts_factual_answer_collapsed_trace -> the user bubble, answer card, trace toggle,
                                                 trace panel, meta tiles, the pinned dock
    fundfacts_logo_mark                      -> LOGO_SVG

The Material colours are the tailwind-config `colors` block verbatim. Every radius, shadow
and spacing step is taken from the same config's `borderRadius` / `spacing` blocks, so the
Streamlit build cannot drift from the mockups:

    rounded      -> 4px    rounded-lg  -> 8px    rounded-xl -> 12px
    rounded-2xl  -> 16px   rounded-full-> 9999px rounded-tr-sm -> 2px

Editorial slates and the guardrail amber come from the DESIGN.md "Light Mode" section.
Amber is reserved exclusively for guardrails, PII blocks and scope refusals; no green or red
data cues anywhere, per the design system's explicit prohibition.
"""

from __future__ import annotations

import streamlit as st

# --- tokens ---------------------------------------------------------------

# Material 3 roles, verbatim from the Stitch tailwind-config.
PALETTE: dict[str, str] = {
    "canvas": "#f9f9ff",                      # background
    "surface": "#ffffff",                     # surface-container-lowest
    "surface_inset": "#f0f3ff",               # surface-container-low
    "surface_container": "#e7eeff",           # surface-container
    "surface_container_high": "#dee8ff",      # surface-container-high
    "surface_container_highest": "#d8e3fa",   # surface-container-highest
    "on_surface": "#111c2c",
    "on_surface_variant": "#3d4946",
    "outline": "#6d7a76",
    "outline_variant": "#bcc9c5",
    "primary": "#00685c",
    "primary_container": "#008375",
    "primary_fixed": "#81f6e3",
    "on_primary_fixed": "#00201c",
    "on_primary": "#ffffff",
    "secondary": "#485f82",
    "secondary_container": "#bdd6fe",
    "on_secondary_container": "#455d7f",
    "tertiary": "#805200",
    "error": "#ba1a1a",
    "error_container": "#ffdad6",
    "on_error_container": "#93000a",
}

# Editorial slates and the guardrail amber, from the DESIGN.md "Light Mode" prose.
EDITORIAL: dict[str, str] = {
    "border": "#e2e8f0",
    "border_strong": "#cbd5e1",
    "caption": "#718096",
    "accent": "#0e9f8e",          # logo mark + brand chip
    "warning": "#f5a524",          # guardrails / PII / scope refusals
    "warning_substrate": "#fffbeb",
}

RADIUS: dict[str, str] = {
    "card": "16px",      # rounded-2xl: user bubble, answer card, loading card, target card
    "panel": "12px",     # rounded-xl: suggestion chip, guardrail, trace, meta tile
    "hit": "8px",        # rounded-lg: one retrieved chunk inside the trace
    "badge": "4px",      # rounded: the category badge on a suggestion chip
    "anchor": "2px",     # rounded-tr-sm: the ground-anchor corner of a user bubble
    "control": "10px",
    "full": "9999px",    # pills, chips, avatars, the pinned input
}

# Spacing scale, verbatim from the tailwind-config `spacing` block.
SPACING: dict[str, str] = {
    "xs": "4px",     # space-xs
    "sm": "8px",     # space-sm
    "md": "16px",    # space-md, gutter, margin
    "lg": "24px",    # space-lg
    "xl": "32px",    # space-xl
}

# Elevation, from DESIGN.md "Elevation & Depth". Level 1 = cards, Level 2 = floating dock.
SHADOW_CARD = "0 1px 3px 0 rgba(15, 42, 74, 0.04), 0 1px 2px 0 rgba(15, 42, 74, 0.02)"
SHADOW_HOVER = "0 4px 12px 0 rgba(15, 42, 74, 0.06), 0 1px 3px 0 rgba(15, 42, 74, 0.04)"
SHADOW_FLOAT = "0 8px 24px -4px rgba(15, 42, 74, 0.08)"
SHADOW_FOCUS = "0 10px 28px -4px rgba(15, 42, 74, 0.12)"

# The reading column (DESIGN.md, Layout & Spacing) and the conversation rhythm (gap-6).
COLUMN = "760px"
TURN_GAP = "24px"

FONT_SANS = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
FONT_MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace"
FONT_ICON = "'Material Symbols Outlined'"

# The logo mark, reproduced verbatim from `fundfacts_logo_mark/code.html`.
LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40" width="40" height="40" '
    'fill="none" aria-hidden="true">'
    '<rect width="40" height="40" rx="10" fill="#0E9F8E"/>'
    '<path d="M20 9L12 13V20C12 25.5 15.4 30.7 20 32C24.6 30.7 28 25.5 28 20V13L20 9Z" '
    'stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>'
    '<path d="M17 20.5L19 22.5L23.5 17.5" stroke="#FFFFFF" stroke-width="2.2" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)


def icon(name: str, size: int = 18, fill: bool = False) -> str:
    """A Material Symbols glyph as inline HTML, so no icon font has to round-trip CSS."""
    variation = f"font-variation-settings: 'FILL' 1;" if fill else ""
    return (
        f'<span class="material-symbols-outlined" aria-hidden="true" '
        f'style="font-size:{size}px;{variation}">{name}</span>'
    )


def esc(text: object) -> str:
    """HTML-escape a pipeline value for a text node.

    Quotes are left alone: this only ever lands in text content, and escaping them would
    turn every apostrophe in an answer into a literal `&#x27;` in some Markdown pipelines.
    """
    import html

    return html.escape(str(text), quote=False)


def esc_attr(value: object) -> str:
    """HTML-escape a value used inside a double-quoted attribute, e.g. `href`."""
    import html

    return html.escape(str(value), quote=True)


# --- global stylesheet ----------------------------------------------------

_FONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Inter:wght@400;500;600;700"
    "&family=JetBrains+Mono:wght@400;500"
    "&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200"
    "&display=swap"
)


def _root_variables() -> str:
    lines = [f"  --ff-{name}: {value};" for name, value in PALETTE.items()]
    lines += [f"  --ff-{name}: {value};" for name, value in EDITORIAL.items()]
    lines += [f"  --ff-{name}: {value};" for name, value in RADIUS.items()]
    lines += [f"  --ff-space-{name}: {value};" for name, value in SPACING.items()]
    lines.append(f"  --ff-column: {COLUMN};")
    lines.append(f"  --ff-turn-gap: {TURN_GAP};")
    lines.append(f"  --ff-shadow-card: {SHADOW_CARD};")
    lines.append(f"  --ff-shadow-hover: {SHADOW_HOVER};")
    lines.append(f"  --ff-shadow-float: {SHADOW_FLOAT};")
    lines.append(f"  --ff-shadow-focus: {SHADOW_FOCUS};")
    return "\n".join(lines)


CSS = """@import url('__FONTS__');
:root {
__VARS__
}
html, body, [data-testid="stAppViewContainer"], .stApp,
button, input, textarea, select, [class*="css"] {
  font-family: __FONT_SANS__;
}
/* ---------- canvas & layout: one centred 760px reading column ---------- */
.stApp, [data-testid="stAppViewContainer"] {
  background-color: var(--ff-canvas);
  color: var(--ff-on_surface);
}
/* the architectural glow behind the reading column */
[data-testid="stAppViewContainer"]::before {
  content: "";
  position: fixed;
  top: -3rem;
  left: 50%;
  width: 24rem;
  height: 11rem;
  transform: translateX(-50%);
  background: rgba(222, 232, 255, 0.4);
  border-radius: 9999px;
  filter: blur(64px);
  pointer-events: none;
  z-index: 0;
}
[data-testid="stMainViewContainer"] > .main { background: transparent; padding-top: 0; }
[data-testid="stMainBlockContainer"], [data-testid="stMainViewContainer"] .block-container {
  position: relative;
  z-index: 1;
  max-width: var(--ff-column) !important;
  padding: 1.5rem 1.5rem 1rem !important;
}
/* One vertical rhythm for the whole column. Components own the space inside themselves
   and lean on this gap for the space between them, so a turn reads as one group. */
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] { gap: var(--ff-space-md); }
[data-testid="stMainBlockContainer"] [data-testid="stElementContainer"] { margin-bottom: 0; }
[data-testid="stMarkdownContainer"] > *:last-child { margin-bottom: 0; }
/* the bottom dock needs room so the last turn can scroll clear of it */
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"]:has(.ff-dock-plate) {
  margin-bottom: 0;
}
/* ---------- drop Streamlit's own chrome ---------- */
#MainMenu, footer, [data-testid="stStatusWidget"], [data-testid="stDecoration"],
[data-testid="stToolbar"], [data-testid="stHeader"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }
header[data-testid="stHeader"]:empty { display: none; }

/* ---------- typography primitives (DESIGN.md `typography`) ---------- */
.ff-mono { font-family: __FONT_MONO__; font-size: 12px; line-height: 18px; font-weight: 400; }
.ff-caption { font-size: 12px; line-height: 16px; font-weight: 500; }
.ff-label { font-size: 13px; line-height: 18px; font-weight: 600; letter-spacing: 0.01em; }
.ff-title {
  font-size: 32px; line-height: 40px; font-weight: 700; letter-spacing: -0.02em;
  color: var(--ff-on_surface); margin: 0;
}
.ff-num { font-variant-numeric: tabular-nums; }
.ff-truncate { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0; }

/* ---------- top meta row ---------- */
/* welcome variant: the KNOWLEDGE BASE pill plus a mono sync stamp */
.ff-verify-row { display: flex; align-items: center; gap: var(--ff-space-sm); margin-bottom: 56px !important; }
.ff-kb-pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 12px; border-radius: var(--ff-full);
  background: var(--ff-surface_container); color: var(--ff-on_surface_variant);
  font-family: __FONT_MONO__; font-size: 12px; line-height: 16px;
  box-shadow: var(--ff-shadow-card);
}
.ff-dot { width: 6px; height: 6px; border-radius: 9999px; background: var(--ff-primary); display: inline-block; }
.ff-dot-lg { width: 8px; height: 8px; }
.ff-pulse { animation: ff-pulse 1.8s ease-in-out infinite; }
@keyframes ff-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
.ff-ping { animation: ff-ping 1.4s ease-out infinite; }
@keyframes ff-ping {
  0% { box-shadow: 0 0 0 0 rgba(0, 104, 92, 0.45); }
  70% { box-shadow: 0 0 0 7px rgba(0, 104, 92, 0); }
  100% { box-shadow: 0 0 0 0 rgba(0, 104, 92, 0); }
}
/* conversation variant: a pulsing dot, the index name, and a mono provenance stamp */
.ff-index-row {
  display: flex; align-items: center; justify-content: space-between; gap: var(--ff-space-sm);
  padding: 0 var(--ff-space-sm); margin-bottom: 32px;
  font-size: 12px; line-height: 16px; font-weight: 500; color: var(--ff-on_surface_variant);
}
.ff-index-label { display: inline-flex; align-items: center; gap: 6px; color: var(--ff-on_surface); }

/* ---------- welcome block ---------- */
.ff-brand-row { display: flex; align-items: center; gap: var(--ff-space-sm); margin-top: 20px !important; margin-bottom: var(--ff-space-sm); }
.ff-brand-tile {
  width: 36px; height: 36px; border-radius: var(--ff-panel);
  display: flex; align-items: center; justify-content: center;
  color: var(--ff-primary); background: var(--ff-surface_container);
  box-shadow: var(--ff-shadow-card);
}
.ff-brand-tile svg { width: 26px; height: 26px; }
.ff-eyebrow {
  font-size: 13px; line-height: 18px; font-weight: 600;
  letter-spacing: 0.06em; text-transform: uppercase; color: var(--ff-secondary);
}
.ff-lede {
  font-size: 16px; line-height: 28px; color: var(--ff-on_surface);
  margin: var(--ff-space-md) 0 32px 0; max-width: 44rem;
}
.ff-lede strong { color: var(--ff-primary); font-weight: 600; }

/* ---------- scheme scope pills ---------- */
.ff-pill-row {
  display: flex; flex-wrap: wrap; gap: 12px;
  margin-top: 4px; margin-bottom: 48px !important;
}
.ff-pill {
  display: inline-flex; align-items: center; gap: 8px;
  height: 38px; padding: 0 16px; border-radius: var(--ff-full);
  border: 1px solid var(--ff-border); background: var(--ff-surface);
  color: var(--ff-secondary); font-size: 13px; line-height: 16px; font-weight: 500;
}
.ff-pill .material-symbols-outlined { color: var(--ff-primary); }

/* ---------- suggestion section head ---------- */
.ff-suggest-head {
  display: flex; align-items: center; justify-content: space-between;
  gap: var(--ff-space-sm); margin-top: 36px !important; margin-bottom: 20px !important;
}
.ff-suggest-label {
  font-size: 12px; line-height: 16px; font-weight: 600;
  letter-spacing: 0.06em; text-transform: uppercase; color: var(--ff-secondary);
}
.ff-hint { font-family: __FONT_MONO__; font-size: 12px; line-height: 18px; color: var(--ff-outline); }

/* ---------- suggestion cards: 3-up grid (fundfacts_welcome_empty_state) ---------- */
/* The column is the card. `.ff-chip` is a zero-height marker so `:has()` can target only
   the chip columns, and the question itself stays a real `st.button`. */
.ff-chip { display: none; }
[data-testid="stHorizontalBlock"]:has(.ff-chip) { gap: 24px !important; }
[data-testid="stColumn"]:has(.ff-chip) {
  position: relative;
  background: var(--ff-surface);
  border: 1px solid var(--ff-border);
  border-radius: var(--ff-panel);
  box-shadow: var(--ff-shadow-card);
  padding: 22px 20px;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
/* the 4px accent rule down the leading edge */
[data-testid="stColumn"]:has(.ff-chip):before {
  content: "";
  position: absolute;
  top: -1px; left: -1px; bottom: -1px;
  width: 4px;
  border-radius: var(--ff-panel) 0 0 var(--ff-panel);
  background: var(--ff-surface_container_highest);
  transition: background 0.2s ease;
}
[data-testid="stColumn"]:has(.ff-chip):hover {
  border-color: var(--ff-border_strong);
  box-shadow: var(--ff-shadow-hover);
}
[data-testid="stColumn"]:has(.ff-chip):hover:before { background: var(--ff-primary); }
.ff-chip-head {
  display: flex; align-items: center; justify-content: space-between;
  gap: var(--ff-space-sm); margin-bottom: 14px; padding-left: var(--ff-space-xs);
}
.ff-chip-badge {
  display: inline-block; padding: 2px 8px; border-radius: var(--ff-badge);
  background: var(--ff-surface_container); color: var(--ff-secondary);
  font-size: 12px; line-height: 16px; font-weight: 500;
}
.ff-chip-glyph { display: inline-flex; color: var(--ff-outline); transition: color 0.2s ease; }
[data-testid="stColumn"]:has(.ff-chip):hover .ff-chip-glyph { color: var(--ff-primary); }
/* the question line is the button itself: borderless, left-aligned, full width */
[data-testid="stColumn"]:has(.ff-chip) .stButton > button,
[data-testid="stColumn"]:has(.ff-chip) [data-testid="stBaseButton-secondary"] {
  height: auto; min-height: 0; width: 100%; padding: 0;
  border: 0 !important; border-radius: var(--ff-badge);
  background: transparent !important; box-shadow: none !important;
  text-align: left !important; white-space: normal !important;
  color: var(--ff-on_surface) !important;
  font-family: __FONT_SANS__ !important;
  font-size: 14px !important; font-weight: 500 !important; line-height: 22px !important;
  transition: color 0.2s ease;
}
[data-testid="stColumn"]:has(.ff-chip) .stButton > button p,
[data-testid="stColumn"]:has(.ff-chip) [data-testid="stBaseButton-secondary"] p {
  margin: 0; padding-left: var(--ff-space-xs);
}
[data-testid="stColumn"]:has(.ff-chip) .stButton > button:hover,
[data-testid="stColumn"]:has(.ff-chip) [data-testid="stBaseButton-secondary"]:hover {
  background: transparent !important; color: var(--ff-primary) !important;
}
[data-testid="stColumn"]:has(.ff-chip) .stButton > button:focus-visible,
[data-testid="stColumn"]:has(.ff-chip) [data-testid="stBaseButton-secondary"]:focus-visible {
  outline: 2px solid var(--ff-primary) !important; outline-offset: 4px;
}
.ff-chip-foot {
  display: flex; align-items: center; justify-content: space-between;
  gap: var(--ff-space-sm); margin-top: 22px; padding-top: var(--ff-space-xs);
  padding-left: var(--ff-space-xs);
  font-family: __FONT_MONO__; font-size: 12px; line-height: 18px; color: var(--ff-outline);
}
.ff-chip-arrow {
  display: inline-flex; color: var(--ff-primary);
  opacity: 0; transform: translateX(-4px);
  transition: opacity 0.2s ease, transform 0.2s ease;
}
[data-testid="stColumn"]:has(.ff-chip):hover .ff-chip-arrow { opacity: 1; transform: none; }

/* ---------- guardrail notice ---------- */
.ff-guardrail {
  display: flex; align-items: flex-start; gap: var(--ff-space-sm);
  padding: var(--ff-space-md); margin-bottom: var(--ff-space-lg);
  border-radius: var(--ff-panel);
  background: var(--ff-surface_inset); box-shadow: var(--ff-shadow-card);
  color: var(--ff-on_surface);
}
.ff-guardrail > .material-symbols-outlined { color: var(--ff-primary); flex: 0 0 auto; margin-top: 2px; }
.ff-guardrail-title { font-size: 13px; line-height: 18px; font-weight: 600; color: var(--ff-on_surface); }
.ff-guardrail-body {
  font-size: 12px; line-height: 16px; font-weight: 500;
  color: var(--ff-on_surface_variant); margin-top: 2px;
}

/* ---------- conversation rhythm ---------- */
.ff-turn { display: flex; width: 100%; }
.ff-user-wrap {
  display: flex; flex-direction: column; align-items: flex-end;
  max-width: 32rem; margin-left: auto;
}
/* rounded-2xl with a 2px rounded-tr-sm anchor corner */
.ff-user {
  background: var(--ff-primary); color: var(--ff-on_primary);
  padding: 14px 20px;
  border-radius: var(--ff-card) var(--ff-anchor) var(--ff-card) var(--ff-card);
  box-shadow: var(--ff-shadow-card);
  font-size: 16px; line-height: 26px; overflow-wrap: anywhere;
}
.ff-user-meta {
  display: flex; align-items: center; gap: 6px; margin: 4px 4px 0 0;
  font-size: 12px; line-height: 16px; font-weight: 500; color: var(--ff-secondary);
}

/* ---------- assistant answer card ---------- */
.ff-card {
  width: 100%; background: var(--ff-surface);
  border: 1px solid var(--ff-border); border-radius: var(--ff-card);
  padding: var(--ff-space-lg); box-shadow: var(--ff-shadow-card);
}
.ff-card-head {
  display: flex; align-items: center; justify-content: space-between;
  gap: 10px; padding-bottom: var(--ff-space-md);
}
.ff-card-who { display: flex; align-items: center; gap: 10px; min-width: 0; }
.ff-avatar {
  width: 28px; height: 28px; border-radius: 9999px;
  background: var(--ff-primary); color: var(--ff-on_primary);
  display: flex; align-items: center; justify-content: center;
  box-shadow: var(--ff-shadow-card); flex: 0 0 auto;
}
.ff-avatar-lg { width: 32px; height: 32px; }
.ff-avatar-fixed { background: var(--ff-primary_fixed); color: var(--ff-on_primary_fixed); }
.ff-tag {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 12px; border-radius: var(--ff-full);
  background: rgba(0, 131, 117, 0.15); color: var(--ff-primary);
  font-size: 12px; line-height: 16px; font-weight: 500; white-space: nowrap;
}
.ff-tag-neutral { background: var(--ff-surface_container); color: var(--ff-secondary); }
.ff-tag-amber { background: rgba(245, 165, 36, 0.16); color: var(--ff-tertiary); }
.ff-answer {
  padding: var(--ff-space-sm) 0;
  font-size: 16px; line-height: 26px; color: var(--ff-on_surface); overflow-wrap: anywhere;
}
.ff-answer p { margin: 0 0 16px 0; line-height: 1.65; }
.ff-answer p:last-child { margin-bottom: 0; }
.ff-answer a { color: var(--ff-primary); overflow-wrap: anywhere; }
.ff-source-row {
  display: flex; flex-wrap: wrap; align-items: center; gap: 12px;
  padding-top: var(--ff-space-md);
}
.ff-source-pill {
  display: inline-flex; align-items: center; gap: var(--ff-space-sm);
  padding: 6px 14px; border-radius: var(--ff-full);
  background: var(--ff-surface_inset); color: var(--ff-primary);
  font-size: 13px; line-height: 18px; font-weight: 600; letter-spacing: 0.01em;
  text-decoration: none; max-width: 100%; overflow-wrap: anywhere;
  transition: background 0.2s ease;
}
.ff-source-pill:hover { background: var(--ff-surface_container); text-decoration: none; }
.ff-source-pill .material-symbols-outlined:last-child { color: var(--ff-secondary); }
.ff-fresh {
  display: flex; align-items: center; gap: 6px;
  margin-top: var(--ff-space-sm); padding-top: var(--ff-space-md);
  font-size: 12px; line-height: 16px; font-weight: 500; color: var(--ff-secondary);
}
/* guardrail answer card: soft amber border over an ultra-light amber fill */
.ff-card-refusal { border-color: var(--ff-warning); background: var(--ff-warning_substrate); }
.ff-card-refusal .ff-avatar { background: var(--ff-warning); color: #3b2600; }
.ff-card-refusal .ff-fresh { color: var(--ff-tertiary); }

/* ---------- loading state (fundfacts_loading_state) ---------- */
.ff-load-bar {
  display: flex; align-items: center; justify-content: space-between;
  gap: var(--ff-space-sm); flex-wrap: wrap;
  padding: 0 var(--ff-space-xs); margin-bottom: var(--ff-space-lg);
}
.ff-load-pills { display: flex; align-items: center; gap: var(--ff-space-xs); flex-wrap: wrap; }
.ff-scope-pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 12px; border-radius: var(--ff-full);
  background: var(--ff-surface_container_high); color: var(--ff-on_surface_variant);
  font-size: 12px; line-height: 16px; font-weight: 500; box-shadow: var(--ff-shadow-card);
}
.ff-scope-pill .material-symbols-outlined { color: var(--ff-primary); }
.ff-scope-mono {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 4px 10px; border-radius: var(--ff-full);
  background: var(--ff-surface_container); color: var(--ff-secondary);
  font-family: __FONT_MONO__; font-size: 12px; line-height: 16px;
}
.ff-load-live {
  display: inline-flex; align-items: center; gap: var(--ff-space-xs);
  font-size: 12px; line-height: 16px; font-weight: 500; color: var(--ff-secondary);
}
.ff-loading {
  width: 100%; background: var(--ff-surface); border: 1px solid var(--ff-border);
  border-radius: var(--ff-card); padding: var(--ff-space-md);
  box-shadow: var(--ff-shadow-card);
  display: flex; flex-direction: column; gap: var(--ff-space-md);
}
.ff-loading-head { display: flex; align-items: center; gap: 14px; }
.ff-loading-text { display: flex; flex-direction: column; flex: 1 1 auto; min-width: 0; }
.ff-blink { animation: ff-blink 1.6s ease-in-out infinite; }
@keyframes ff-blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.45; } }
.ff-ring { animation: ff-spin 0.9s linear infinite; flex: 0 0 auto; }
.ff-ring .ff-ring-track { color: var(--ff-surface_container_highest); }
.ff-ring .ff-run { color: var(--ff-primary); }
@keyframes ff-spin { to { transform: rotate(360deg); } }
.ff-load-tag {
  display: inline-flex; align-items: center; gap: 4px; flex: 0 0 auto;
  padding: 2px 8px; border-radius: var(--ff-badge);
  background: var(--ff-surface_container); color: var(--ff-secondary);
  font-family: __FONT_MONO__; font-size: 12px; line-height: 16px; white-space: nowrap;
}
/* the retrieval inspector skeleton */
.ff-loading-panel {
  width: 100%; background: var(--ff-surface_inset);
  border-radius: var(--ff-panel); padding: 14px;
  display: flex; flex-direction: column; gap: 10px;
}
.ff-loading-label { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.ff-loading-label > span:first-child {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 12px; line-height: 16px; font-weight: 500; color: var(--ff-on_surface_variant);
}
.ff-loading-label .material-symbols-outlined { color: var(--ff-primary); }
.ff-bar {
  width: 100%; height: 6px; border-radius: 9999px;
  background: var(--ff-surface_container); overflow: hidden;
}
.ff-bar span {
  display: block; height: 100%; width: 68%; border-radius: 9999px;
  background: var(--ff-primary); animation: ff-slide 1.5s ease-in-out infinite;
}
@keyframes ff-slide { 0% { transform: translateX(-60%); } 100% { transform: translateX(120%); } }
.ff-skeleton {
  height: 10px; border-radius: 9999px; background: var(--ff-surface_container_high);
  animation: ff-pulse 1.6s ease-in-out infinite;
}
.ff-skeleton.w-56 { width: 83.33%; }
.ff-skeleton.w-34 { width: 75%; }
.ff-breadcrumb {
  display: flex; align-items: center; gap: var(--ff-space-sm);
  margin: var(--ff-space-sm) 0 0 var(--ff-space-sm);
  font-size: 12px; line-height: 16px; font-weight: 500; color: var(--ff-on_surface_variant);
}
.ff-breadcrumb .ff-dot { background: var(--ff-primary); flex: 0 0 auto; }
/* the target-scheme anchor card */
.ff-readonly {
  width: 100%; margin-top: var(--ff-space-md);
  padding: var(--ff-space-md); background: var(--ff-surface_inset);
  border-radius: var(--ff-card); box-shadow: var(--ff-shadow-card);
  display: flex; align-items: center; justify-content: space-between;
  gap: var(--ff-space-sm); flex-wrap: wrap;
}
.ff-readonly-who { display: flex; align-items: center; gap: var(--ff-space-sm); min-width: 0; }
.ff-avatar-sm {
  width: 28px; height: 28px; border-radius: 9999px; flex: 0 0 auto;
  background: var(--ff-surface_container_high); color: var(--ff-primary);
  display: flex; align-items: center; justify-content: center;
}
.ff-readonly .ff-label { color: var(--ff-on_surface); }
.ff-readonly-badge {
  display: inline-flex; align-items: center; gap: 6px; flex: 0 0 auto;
  padding: 4px 10px; border-radius: var(--ff-full);
  background: var(--ff-surface_container); color: var(--ff-secondary);
  font-family: __FONT_MONO__; font-size: 12px; line-height: 16px; white-space: nowrap;
}
.ff-readonly-badge .material-symbols-outlined { color: var(--ff-primary); }

/* ---------- retrieval trace (fundfacts_factual_answer_collapsed_trace) ---------- */
/* A native st.expander, restyled into the design's full-width secondary bar so it keeps
   keyboard and ARIA behaviour. */
details[data-testid="stExpander"] {
  width: 100%; background: var(--ff-surface_inset);
  border: 1px solid var(--ff-border); border-radius: var(--ff-panel);
  box-shadow: var(--ff-shadow-card); overflow: hidden;
}
details[data-testid="stExpander"] > summary {
  list-style: none; cursor: pointer;
  padding: 12px var(--ff-space-md);
  display: flex; align-items: center; gap: 10px;
  font-family: __FONT_MONO__; font-size: 12px; line-height: 18px;
  font-weight: 500; color: var(--ff-on_surface);
}
details[data-testid="stExpander"] > summary::-webkit-details-marker { display: none; }
details[data-testid="stExpander"] > summary:hover { background: var(--ff-surface_container); }
details[data-testid="stExpander"] > summary p { margin: 0; }
details[data-testid="stExpander"] > summary::before {
  content: "terminal";
  font-family: __FONT_ICON__; font-size: 18px; color: var(--ff-secondary);
  flex: 0 0 auto; line-height: 1;
}
details[data-testid="stExpander"] [data-testid="stExpanderIcon"] { display: none; }
details[data-testid="stExpander"] > summary::after {
  content: "expand_more";
  font-family: __FONT_ICON__; font-size: 18px; color: var(--ff-secondary);
  margin-left: auto; transition: transform 0.2s ease; line-height: 1; flex: 0 0 auto;
}
details[data-testid="stExpander"][open] > summary::after { transform: rotate(180deg); }
details[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
  padding: 0 var(--ff-space-md) var(--ff-space-md) var(--ff-space-md);
}
/* the collapsed bar's right-hand hint, mirrored inside the panel head */
.ff-trace-head-right { display: inline-flex; align-items: center; gap: var(--ff-space-sm); }
/* the intent pill sits inside the panel, above the log */
.ff-trace-intent {
  display: inline-flex; align-items: center; padding: 2px 8px; border-radius: var(--ff-full);
  background: var(--ff-surface_container_high); color: var(--ff-on_secondary_container);
  font-size: 12px; line-height: 16px; font-weight: 500; white-space: nowrap;
  text-transform: none; letter-spacing: 0.01em;
}
/* the expanded panel */
.ff-trace-panel {
  width: 100%; padding: var(--ff-space-md); background: var(--ff-surface);
  border: 1px solid var(--ff-border); border-radius: var(--ff-panel);
  box-shadow: var(--ff-shadow-card);
  display: flex; flex-direction: column; gap: 12px;
}
.ff-trace-head {
  display: flex; align-items: center; justify-content: space-between;
  gap: var(--ff-space-sm); padding-bottom: var(--ff-space-sm);
  border-bottom: 1px solid var(--ff-border);
  font-family: __FONT_MONO__; font-size: 11px; line-height: 16px;
  letter-spacing: 0.08em; text-transform: uppercase; color: var(--ff-secondary);
}
.ff-trace-head .ff-latency {
  font-size: 12px; line-height: 18px; letter-spacing: 0;
  text-transform: none; color: var(--ff-primary); font-weight: 500;
}
.ff-kv {
  display: grid; grid-template-columns: minmax(104px, auto) 1fr;
  gap: 4px var(--ff-space-sm);
  font-family: __FONT_MONO__; font-size: 12px; line-height: 18px;
}
.ff-kv dt { color: var(--ff-secondary); }
.ff-kv dd { margin: 0; color: var(--ff-on_surface); overflow-wrap: anywhere; }
/* one retrieved chunk */
.ff-trace-hit {
  padding: 12px; border-radius: var(--ff-hit); background: var(--ff-surface_inset);
  display: flex; flex-direction: column; gap: var(--ff-space-sm);
}
.ff-trace-hit-row {
  display: flex; align-items: center; justify-content: space-between;
  gap: 10px; font-family: __FONT_MONO__; font-size: 12px; line-height: 18px;
  color: var(--ff-secondary);
}
.ff-trace-score { color: var(--ff-primary); font-weight: 500; white-space: nowrap; }
.ff-trace-quote {
  margin: 0; font-family: __FONT_MONO__; font-size: 12px; line-height: 18px;
  color: var(--ff-on_surface_variant); overflow-wrap: anywhere;
}
.ff-trace-url { color: var(--ff-outline); }
.ff-trace-empty {
  padding: 12px; border-radius: var(--ff-hit); background: var(--ff-surface_inset);
  font-family: __FONT_MONO__; font-size: 12px; line-height: 18px;
  color: var(--ff-on_surface_variant);
}
.ff-trace-foot {
  display: flex; align-items: center; justify-content: space-between;
  gap: var(--ff-space-sm); flex-wrap: wrap; padding-top: var(--ff-space-xs);
  font-size: 12px; line-height: 16px; font-weight: 500; color: var(--ff-secondary);
}
.ff-trace-foot > span:first-child { display: inline-flex; align-items: center; gap: var(--ff-space-xs); }
.ff-trace-foot .material-symbols-outlined { color: var(--ff-primary); }
.ff-trace-chips { display: flex; flex-wrap: wrap; gap: 6px; }

/* ---------- meta tiles ---------- */
.ff-card-meta {
  display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding-top: var(--ff-space-xs);
}
.ff-meta-tile {
  padding: 14px; border-radius: var(--ff-panel);
  background: var(--ff-surface); border: 1px solid var(--ff-border);
  box-shadow: var(--ff-shadow-card);
  display: flex; align-items: center; justify-content: space-between; gap: var(--ff-space-sm);
}
.ff-meta-tile .ff-caption { color: var(--ff-secondary); display: block; margin-bottom: 2px; }
.ff-meta-tile .ff-label { color: var(--ff-on_surface); overflow-wrap: anywhere; }
.ff-meta-tile .material-symbols-outlined { color: var(--ff-secondary); flex: 0 0 auto; }

/* ---------- pinned dock: the floating inquiry input ---------- */
/* The app has exactly one form, so the dock selectors do not need a class of their own.
   The form itself is the sticky plate: gradient fade + the pill + the caption underneath. */
.ff-dock-plate,
[data-testid="stForm"] {
  position: sticky; bottom: 0; z-index: 30;
  padding: var(--ff-space-lg) 0 var(--ff-space-md) 0;
  background: linear-gradient(to top, var(--ff-canvas) 62%, rgba(249, 249, 255, 0));
}
/* the pill itself: Level 2 elevation, 1px border, floating shadow */
[data-testid="stForm"] [data-testid="stHorizontalBlock"] {
  background: var(--ff-surface);
  border: 1px solid var(--ff-border_strong);
  border-radius: var(--ff-full);
  box-shadow: var(--ff-shadow-float);
  padding: var(--ff-space-sm) var(--ff-space-sm) var(--ff-space-sm) var(--ff-space-md);
  align-items: center;
  gap: var(--ff-space-sm) !important;
  transition: box-shadow 0.2s ease;
}
[data-testid="stForm"] [data-testid="stHorizontalBlock"]:focus-within {
  box-shadow: var(--ff-shadow-focus);
}
/* the leading search affordance on the empty state */
.ff-dock-lead {
  width: 32px; height: 32px; border-radius: 9999px;
  background: var(--ff-surface_container); color: var(--ff-secondary);
  display: flex; align-items: center; justify-content: center;
}
[data-testid="stForm"] [data-testid="stTextInput"],
[data-testid="stForm"] [data-testid="stTextInput"] > div,
[data-testid="stForm"] [data-testid="stTextInput"] > div > div {
  background: transparent !important; border: 0 !important; box-shadow: none !important;
}
[data-testid="stForm"] [data-testid="stTextInput"] input {
  background: transparent !important; border: 0 !important; box-shadow: none !important;
  font-family: __FONT_SANS__ !important; font-size: 14px !important;
  font-weight: 400 !important; color: var(--ff-on_surface) !important;
  padding: 10px 0 !important; height: auto !important; min-height: 0 !important;
}
[data-testid="stForm"] [data-testid="stTextInput"] input:focus {
  outline: none !important; box-shadow: none !important;
}
[data-testid="stForm"] [data-testid="stTextInput"] input::placeholder {
  color: var(--ff-secondary); opacity: 1;
}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button {
  width: 40px !important; height: 40px !important; min-height: 40px !important;
  border-radius: 9999px !important; background: var(--ff-primary) !important;
  color: var(--ff-on_primary) !important; border: 0 !important;
  padding: 0 !important;
  display: flex; align-items: center; justify-content: center;
  transition: background 0.2s ease, transform 0.15s ease;
}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button:hover {
  background: var(--ff-primary_container) !important;
}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button:active { transform: scale(0.95); }
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button p { display: none; }
[data-testid="stForm"] [data-testid="stFormSubmitButton"] svg {
  fill: var(--ff-on_primary); stroke: var(--ff-on_primary);
}
/* the caption directly under the input (FR-7.3, always visible) */
.ff-dock-caption {
  display: flex; align-items: center; justify-content: center; gap: 6px;
  margin-top: 10px;
  font-size: 12px; line-height: 16px; font-weight: 500; color: var(--ff-secondary);
}
.ff-dock-caption .material-symbols-outlined { color: var(--ff-primary); }

/* ---------- footer ---------- */
.ff-footer {
  display: flex; align-items: center; justify-content: center; gap: var(--ff-space-xs);
  padding: var(--ff-space-md) 0 0 0;
  font-size: 12px; line-height: 16px; font-weight: 500; color: var(--ff-on_surface_variant);
}
.ff-footer .material-symbols-outlined { color: var(--ff-primary); flex: 0 0 auto; }

/* ---------- degraded banner ---------- */
.stAlert[data-testid="stAlert"] {
  border-radius: var(--ff-panel); border: 1px solid var(--ff-warning);
  background: var(--ff-warning_substrate); color: var(--ff-on_surface);
}
/* ---------- spinner ---------- */
[data-testid="stSpinner"] svg { color: var(--ff-primary); }
[data-testid="stSpinner"] { color: var(--ff-secondary); font-size: 13px; }

/* ---------- misc ---------- */
hr { border-color: var(--ff-border); }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: var(--ff-outline_variant); border-radius: 9999px; }

/* ---------- responsive: 390px and below ---------- */
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"] { padding-left: var(--ff-space-md) !important; padding-right: var(--ff-space-md) !important; }
  .ff-title { font-size: 24px; line-height: 32px; letter-spacing: -0.01em; }
  .ff-lede { font-size: 14px; line-height: 22px; }
  .ff-card { padding: var(--ff-space-md); }
  .ff-card-meta { grid-template-columns: 1fr; }
  .ff-index-row, .ff-verify-row { flex-wrap: wrap; }
  [data-testid="stHorizontalBlock"]:has(.ff-chip) { flex-wrap: wrap; }
  [data-testid="stColumn"]:has(.ff-chip) { width: 100%; flex: 1 1 100% !important; }
  .ff-load-live { display: none; }
  .ff-kv { grid-template-columns: 1fr; gap: 2px; }
  .ff-kv dt { margin-top: 6px; }
  .ff-user-wrap { max-width: 90%; }
}
@media (prefers-reduced-motion: reduce) {
  .ff-pulse, .ff-ping, .ff-blink, .ff-ring, .ff-bar span, .ff-skeleton { animation: none; }
}
"""


def stylesheet() -> str:
    """The full stylesheet with the token block and font stacks substituted in."""
    return (
        CSS.replace("__FONTS__", _FONTS_URL)
        .replace("__VARS__", _root_variables())
        .replace("__FONT_SANS__", FONT_SANS)
        .replace("__FONT_MONO__", FONT_MONO)
        .replace("__FONT_ICON__", FONT_ICON)
    )


def inject() -> None:
    """Push the design system into the page. Call once, immediately after set_page_config."""
    st.markdown(f"<style>{stylesheet()}</style>", unsafe_allow_html=True)


__all__ = [
    "COLUMN",
    "CSS",
    "EDITORIAL",
    "LOGO_SVG",
    "PALETTE",
    "RADIUS",
    "SHADOW_CARD",
    "SHADOW_FLOAT",
    "SPACING",
    "TURN_GAP",
    "esc",
    "esc_attr",
    "icon",
    "inject",
    "stylesheet",
]