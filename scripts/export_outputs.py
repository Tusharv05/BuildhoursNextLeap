"""Phase 9 - Packaging. Generate every PRD FR-8 deliverable from real data.

    python scripts/export_outputs.py                    # all four artefacts
    python scripts/export_outputs.py --only sources      # source_list.csv + .md
    python scripts/export_outputs.py --only qa           # sample_qa.md (live pipeline)
    python scripts/export_outputs.py --only disclaimer   # disclaimer.md
    python scripts/export_outputs.py --check             # is the committed copy current?
    python scripts/export_outputs.py --client echo       # keyless, offline, deterministic

Writes `outputs/source_list.csv`, `outputs/source_list.md`, `outputs/sample_qa.md` and
`outputs/disclaimer.md`.

Nothing here is hand-written. The source list is projected from `data/sources.csv` (the
Stage 1 output), the disclaimer is imported from `mf_rag.prompts`, and the sample answers
come from real `mf_rag.pipeline.query()` calls. A hand-typed answer could not be trusted to
satisfy the response contract; a generated one is re-checked against it before it is
written, and the check result is printed next to the entry.

`--delay` exists because the hosted provider most people will point this at (Groq, free
tier) caps input tokens per minute at 7,000 and one question costs ~1,580 - about 4.4
questions per minute, so eight questions back to back is an HTTP 429. The two refusals cost
nothing: L2 decides them before any model call, so the wait is only applied before the
cases that actually reach the LLM.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mf_rag.answer import count_sentences, has_substantive_content, validate  # noqa: E402
from mf_rag.config import AppConfig, get_config  # noqa: E402
from mf_rag.generate import SOURCE_URLS  # noqa: E402
from mf_rag.guardrails import REFUSAL_TEMPLATES  # noqa: E402
from mf_rag.llm import EchoClient, SpyClient, describe_client, get_llm  # noqa: E402
from mf_rag.pipeline import query  # noqa: E402
from mf_rag.prompts import (  # noqa: E402
    DISCLAIMER,
    PII_WARNING,
    extract_freshness,
    extract_urls,
)

# FR-8.2 column order. Exactly these eight, in this order.
SOURCE_LIST_COLUMNS: tuple[str, ...] = (
    "source_id",
    "scheme",
    "category",
    "doc_type",
    "is_official",
    "source_url",
    "fetched_at",
    "status",
)

# The L3 rules, in the order PRD 7 lists them. Named here rather than imported from
# pipeline.VALIDATOR_ORDER because that tuple is the order the pipeline *records*
# violations in; here the order is presentation.
VALIDATOR_LABELS: tuple[str, ...] = (
    "sentence_count <= 3",
    "url_count == 1",
    "url_in_ingested_sources",
    "official_preference",
    "advice_language",
    "returns_language",
    "freshness",
)


@dataclass(frozen=True)
class SampleCase:
    """One question in the deliverable, plus what the pipeline must do with it."""

    id: str
    role: str
    question: str
    expect_intent: str
    expect_refusal: bool


# FR-8.4 asks for 5-10 queries; IMPLEMENTATION Phase 9 fixes it at 8 and the exit gate
# counts 8 entries. The coverage list has nine slots (3 numeric, ELSS, riskometer,
# benchmark, statement, advisory, returns), so the ELSS lock-in does double duty: it is
# both the ELSS-specific question and the third numeric fact, because the answer is a
# number ("3 years"). Three numeric facts therefore land on three different schemes -
# Large Cap, Small Cap, ELSS - and the file has exactly eight entries.
SAMPLE_CASES: tuple[SampleCase, ...] = (
    SampleCase(
        id="s01_expense_ratio_large_cap",
        role="numeric fact - HDFC Large Cap (expense ratio)",
        question="What is the expense ratio of HDFC Large Cap Fund?",
        expect_intent="factual",
        expect_refusal=False,
    ),
    SampleCase(
        id="s02_exit_load_small_cap",
        role="numeric fact - HDFC Small Cap (exit load)",
        question="What is the exit load of HDFC Small Cap Fund?",
        expect_intent="factual",
        expect_refusal=False,
    ),
    SampleCase(
        id="s03_elss_lock_in",
        role="ELSS-specific - lock-in period (also the 3rd numeric fact)",
        question="What is the lock-in period for HDFC ELSS Tax Saver Fund?",
        expect_intent="factual",
        expect_refusal=False,
    ),
    SampleCase(
        id="s04_riskometer_flexi_cap",
        role="riskometer",
        question="What is the riskometer level of HDFC Flexi Cap Fund?",
        expect_intent="factual",
        expect_refusal=False,
    ),
    SampleCase(
        id="s05_benchmark_balanced_advantage",
        role="benchmark",
        question="What is the benchmark of HDFC Balanced Advantage Fund?",
        expect_intent="factual",
        expect_refusal=False,
    ),
    SampleCase(
        id="s06_capital_gains_statement",
        role="statement download",
        question="How do I download my capital gains statement?",
        expect_intent="factual",
        expect_refusal=False,
    ),
    SampleCase(
        id="s07_advisory_refusal",
        role="advisory - must refuse",
        question="Should I buy HDFC Small Cap Fund?",
        expect_intent="advisory",
        expect_refusal=True,
    ),
    SampleCase(
        id="s08_returns_refusal",
        role="returns - must refuse",
        question="What is the CAGR of HDFC Large Cap Fund?",
        expect_intent="returns",
        expect_refusal=True,
    ),
)


@dataclass
class QAResult:
    """One sample Q&A: what was asked, what came back, and whether the contract held."""

    case: SampleCase
    answer_text: str
    source_url: str | None
    fetched_at: str
    intent: str
    is_refusal: bool
    checks: list[tuple[str, bool]] = field(default_factory=list)
    latency_ms: int = 0
    repair_used: bool = False
    llm_calls: int = 0
    top_hit: str = ""
    hits_returned: int = 0

    @property
    def ok(self) -> bool:
        """Intent landed as expected and every re-check of the emitted text passed."""
        return self.intent == self.case.expect_intent and all(ok for _, ok in self.checks)


# -------------------------------------------------------------------------------- sources


def load_source_rows(cfg: AppConfig | None = None) -> list[dict[str, str]]:
    """Every registered source, read from the Stage 1 output. Fails loudly if it is absent."""
    config = cfg or get_config()
    path = config.sources_csv
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/run_ingest.py --reset` (Stage 1) first - "
            f"the source list is projected from its output, never typed by hand."
        )
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has a header but no rows; nothing to export.")
    return rows


def _project(row: dict[str, str]) -> list[str]:
    return [(row.get(col) or "").strip() for col in SOURCE_LIST_COLUMNS]


def render_source_csv(rows: Sequence[dict[str, str]]) -> str:
    """FR-8.2 CSV. Line endings are pinned so the file is byte-identical on Windows."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(SOURCE_LIST_COLUMNS)
    for row in rows:
        writer.writerow(_project(row))
    return buffer.getvalue()


def render_source_md(rows: Sequence[dict[str, str]]) -> str:
    """FR-8.2 MD: the same eight columns, plus what a reviewer needs to trust them."""
    ingested = [row for row in rows if _project(row)[7] == "ok"]
    skipped = [row for row in rows if _project(row)[7] != "ok"]
    dates = sorted(_project(row)[6] for row in ingested if _project(row)[6])

    out: list[str] = [
        "# Source list",
        "",
        f"{len(rows)} registered sources, {len(ingested)} ingested, {len(skipped)} not "
        f"ingested. Generated by `scripts/export_outputs.py` from `data/sources.csv` "
        f"(PRD FR-8.2).",
        "",
        "| source_id | scheme | category | doc_type | is_official | fetched_at | status |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        source_id, scheme, category, doc_type, official, _url, fetched_at, status = _project(row)
        out.append(
            f"| `{source_id}` | {scheme or '-'} | {category or '-'} | {doc_type} | "
            f"{official} | {fetched_at or '-'} | {status} |"
        )

    out += ["", "## URLs", ""]
    for row in rows:
        source_id, _s, _c, _d, official, url, fetched_at, status = _project(row)
        out.append(
            f"- `{source_id}` ({status}, official={official}, "
            f"fetched {fetched_at or 'n/a'}): {url}"
        )

    out += [
        "",
        "## How to read this list",
        "",
        "- **Allowed hosts only.** `mf_rag.sources.ALLOWED_HOSTS` rejects any URL outside "
        "`hdfcfund.com`, `amfiindia.com`, `sebi.gov.in` and the mandated `groww.in` scheme "
        "pages, so no third-party blog can enter the corpus.",
        "- **`is_official`** marks an AMC or regulator page. Citations prefer it over the "
        "aggregator page carrying the same fact (ADR-5, ADR-10).",
        "- **`status`** is the Stage 1 outcome. A page that is `skipped` was fetched but "
        "yielded nothing indexable (a JS-rendered shell, or an empty extraction). "
        "Ingestion warns and continues, so one dead page cannot block the build.",
        "- A `skipped` source is still listed on purpose: a reviewer needs to see which "
        "official pages were *attempted*, not only the ones that worked.",
    ]
    if skipped:
        out += ["", "Not ingested on the last run:", ""]
        out += [
            f"- `{_project(row)[0]}` - {_project(row)[7]} "
            f"({row.get('error') or 'see outputs/ingest_report.json'})"
            for row in skipped
        ]
    if dates:
        out += [
            "",
            f"Newest successful fetch: **{dates[-1]}**.",
            "",
            "Answers carry `Last updated from sources:` stamped from the **oldest** "
            "`fetched_at` among the chunks they were built from, so a mixed-age context can "
            "never be stamped with the newest page's date.",
        ]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------- disclaimer


def render_disclaimer() -> str:
    """FR-8.5. The UI string is imported, never retyped, so the two cannot drift."""
    templates = "\n".join(
        f"- **{name}**: \"{template}\"" for name, template in REFUSAL_TEMPLATES.items()
    )
    return (
        "# Disclaimer\n"
        "\n"
        "Exported by `scripts/export_outputs.py` from `mf_rag.prompts.DISCLAIMER`, which is "
        "the same constant the Streamlit UI renders (PRD FR-8.5). Copy the paragraph below "
        "verbatim into any other surface.\n"
        "\n"
        f"{DISCLAIMER}\n"
        "\n"
        "## Refusal wording\n"
        "\n"
        "Refusals are constants too (`mf_rag.prompts.REFUSAL_TEMPLATES`), so the exact words "
        "are reviewable in one file and identical on every run. Each takes a `{link}` "
        "placeholder, filled at runtime with an ingested URL:\n"
        "\n"
        f"{templates}\n"
        "\n"
        f"`REFUSAL_TEMPLATES['pii']` is `{PII_WARNING.split('.')[0].strip()}.` - the same "
        "constant the L1 PII guard returns verbatim.\n"
        "\n"
        "Generated file - do not hand-edit; rerun `python scripts/export_outputs.py`.\n"
    )


# ----------------------------------------------------------------------------- sample Q&A


def _is_known_refusal(text: str, url: str | None) -> bool:
    """True when the text is byte-identical to a PRD 7 template with this link filled in."""
    link = url or ""
    candidates = list(REFUSAL_TEMPLATES.values()) + [PII_WARNING]
    return any(text == template.format(link=link) for template in candidates)


def _official_preference_check(answer) -> tuple[str, bool]:
    """Did the citation go to an official page whenever one was in the context?

    Checked against the retrieved hits rather than the validator's word, because the
    validator can only compare the emitted URL with the URL the system would have preferred
    *given those hits*. When retrieval brought no official page - which happens on terse
    questions - the honest result is "nothing official was available", not "preference
    honoured".
    """
    from mf_rag.generate import OFFICIAL_SOURCES

    official = [h for h in (answer.trace.hits or []) if h.get("source_id") in OFFICIAL_SOURCES]
    if not official:
        return ("official_preference (no official page retrieved to prefer)", True)
    expected = official[0].get("source_url")
    return (
        f"official_preference (cited the official page: {expected})",
        answer.source_url == expected,
    )


def contract_checks(answer, max_sentences: int = 3) -> list[tuple[str, bool]]:
    """Re-check the *emitted* text against the PRD 7 contract, one row per rule.

    The pipeline validates internally, but `Trace.validators_run` holds the violations it
    found and repaired, not a pass list, and after a repair the delivered string is not the
    string that was checked. So this re-runs the validators over exactly the bytes the
    reader gets. It is what makes `sample_qa.md` a deliverable rather than a transcript.

    A refusal never reaches generation, so the L3 list is not what proved it. For those the
    proof is different and is stated instead: the text is a template constant byte for byte,
    carrying an ingested link.
    """
    text = answer.text
    url = answer.source_url or ""
    ingested = set(SOURCE_URLS.values())
    urls = extract_urls(text)

    if answer.is_refusal:
        return [
            ("L1 pii_guard ran first", not answer.trace.pii_blocked),
            (
                "L2 intent routed to a refusal",
                answer.intent in {"advisory", "portfolio", "returns", "out_of_corpus", "pii"},
            ),
            ("L3 not applicable - refused before generation", True),
            ("text is a PRD 7 template, byte-exact", _is_known_refusal(text, url)),
            ("link is an ingested source", url in ingested),
            (
                f"sentence_count <= {max_sentences} (found {count_sentences(text)})",
                count_sentences(text) <= max_sentences,
            ),
        ]

    outcome = validate(
        text,
        ctx={"official_url": url},
        max_sentences=max_sentences,
        expected_date=answer.fetched_at or None,
    )
    fired = set(outcome.violations)

    def _fired(prefix: str) -> bool:
        return any(violation.split("=")[0] == prefix for violation in fired)

    official_label, official_ok = _official_preference_check(answer)
    return [
        (
            f"sentence_count <= {max_sentences} (found {count_sentences(text)})",
            not _fired("sentence_count"),
        ),
        (f"url_count == 1 (found {len(urls)})", not _fired("url_count")),
        ("url_in_ingested_sources", all(u in ingested for u in urls)),
        (official_label, official_ok),
        ("advice_language", not _fired("advice_language")),
        ("returns_language", not _fired("returns_language")),
        (
            f"freshness stamp == {answer.fetched_at or 'n/a'}",
            extract_freshness(text) == (answer.fetched_at or None),
        ),
        ("answer carries a fact, not just a link", has_substantive_content(text)),
    ]


def run_sample(cfg: AppConfig, client: Any, delay: float = 0.0) -> list[QAResult]:
    """Run every sample question through the real pipeline. No answer is composed here.

    `client` is wrapped in a `SpyClient` so the file can state per question whether the
    model was called at all. That is the difference between "the bot refused" and "the
    model happened to be cautious", and the guardrail claims only hold if it is the former.
    """
    results: list[QAResult] = []
    for position, case in enumerate(SAMPLE_CASES):
        spy = SpyClient(client)
        started = time.perf_counter()
        answer = query(case.question, cfg=cfg, llm=spy)
        latency = int((time.perf_counter() - started) * 1000)

        hits = answer.trace.hits or []
        top_hit = ""
        if hits:
            top_hit = f"{hits[0].get('chunk_id')} @ {hits[0].get('score')}"

        result = QAResult(
            case=case,
            answer_text=answer.text,
            source_url=answer.source_url,
            fetched_at=answer.fetched_at,
            intent=answer.intent,
            is_refusal=answer.is_refusal,
            checks=contract_checks(answer, cfg.generation.max_sentences),
            latency_ms=latency,
            repair_used=answer.trace.repair_used,
            llm_calls=spy.calls,
            top_hit=top_hit,
            hits_returned=len(hits),
        )
        results.append(result)

        detail = " | ".join(name for name, ok in result.checks if not ok) or "all checks pass"
        print(
            f"  [{'ok' if result.ok else 'FAIL':<4}] {case.id:<36} "
            f"intent={answer.intent:<10} refusal={str(answer.is_refusal):<5} "
            f"llm={spy.calls} {latency:6d}ms"
        )
        if not result.ok:
            print(f"         {detail}")

        nxt = SAMPLE_CASES[position + 1] if position + 1 < len(SAMPLE_CASES) else None
        if delay > 0 and nxt is not None and not nxt.expect_refusal:
            time.sleep(delay)
    return results


def _date_sentence(dates: Sequence[str]) -> str:
    """One honest sentence about how old the pages behind these answers are."""
    if not dates:
        return "No source dates were available for these answers."
    unique = sorted(set(dates))
    if len(unique) == 1:
        return f"Every page behind these answers was fetched on **{unique[0]}**."
    return (
        f"The pages behind these answers were fetched between **{unique[0]}** and "
        f"**{unique[-1]}**."
    )


def render_sample_qa(
    results: Sequence[QAResult], client_name: str, cfg: AppConfig
) -> str:
    """FR-8.4: Question / Answer / Source link / Intent / Validators passed, per entry."""
    passed = sum(1 for result in results if result.ok)
    dates = sorted(result.fetched_at for result in results if result.fetched_at)
    refusals = [r for r in results if r.is_refusal]

    out: list[str] = [
        "# Sample Q&A",
        "",
        "Every answer below was produced by calling `mf_rag.pipeline.query()` - none is "
        f"hand-written. Generation client: `{client_name}`. Retrieval settings at generation "
        f"time: `top_k={cfg.retrieval.top_k}`, `min_score={cfg.retrieval.min_score}`, "
        f"chunking `{cfg.chunking.strategy}`, max {cfg.generation.max_sentences} sentences.",
        "",
        f"**{len(results)} questions, {passed}/{len(results)} satisfying the response "
        f"contract** (<= 3 sentences, exactly one ingested link, advice and returns refused).",
        "",
        "| # | Coverage | Intent | Refusal | LLM calls | Contract |",
        "|---|---|---|---|---|---|",
    ]
    for index, result in enumerate(results, start=1):
        out.append(
            f"| {index} | {result.case.role} | `{result.intent}` | "
            f"{'yes' if result.is_refusal else 'no'} | {result.llm_calls} | "
            f"{'pass' if result.ok else '**FAIL**'} |"
        )

    for index, result in enumerate(results, start=1):
        out += [
            "",
            "---",
            "",
            f"### {index}. {result.case.role}",
            "",
            f"**Question:** {result.case.question}",
            "",
            "**Answer:**",
            "",
        ]
        out += [f"> {line}" if line.strip() else ">" for line in result.answer_text.splitlines()]
        out += [
            "",
            f"**Source link:** {result.source_url or '(none)'}",
            "",
            f"**Intent:** `{result.intent}`"
            f"{' (refusal)' if result.is_refusal else ''}, expected `{result.case.expect_intent}`",
            "",
            "**Validators passed:**",
            "",
        ]
        out += [f"- {'PASS' if ok else 'FAIL'} - {name}" for name, ok in result.checks]
        out += [
            "",
            f"<sub>retrieved {result.hits_returned} chunk(s); top hit "
            f"`{result.top_hit or 'n/a'}`; latency {result.latency_ms} ms; repair used: "
            f"{'yes' if result.repair_used else 'no'}; LLM calls: {result.llm_calls}; "
            f"freshness date: {result.fetched_at or 'n/a'}</sub>",
        ]

    out += [
        "",
        "---",
        "",
        "## How to read these",
        "",
        f"- {len(refusals)} of {len(results)} questions are refusals and show "
        f"`LLM calls: 0`. Intent routing (L2) decides them from the question text alone, "
        "so there is no prompt in which a model could be talked past them.",
        "- Every generated answer ends with `Last updated from sources: <date>`, stamped by "
        "the system from chunk metadata, and carries exactly one URL chosen by "
        "`select_source_url()` rather than authored by the model. Both are structural "
        "(ADR-5), not prompt instructions.",
        f"- {_date_sentence(dates)} A figure can change on the AMC's site after that date; "
        "verify there before acting.",
        "- Regenerate with `python scripts/export_outputs.py --only qa`. With a hosted LLM "
        "the wording can move between runs even at temperature 0, which is why `--check` "
        "does not byte-compare this file; the per-question validator rows are what must hold "
        "every time, and a FAIL there means the file should not be shipped as-is.",
        "",
    ]
    return "\n".join(out)


# --------------------------------------------------------------------------------- driver


def artefact_paths(cfg: AppConfig) -> dict[str, Path]:
    return {
        "sources_csv": cfg.outputs_dir / "source_list.csv",
        "sources_md": cfg.outputs_dir / "source_list.md",
        "qa": cfg.outputs_dir / "sample_qa.md",
        "disclaimer": cfg.outputs_dir / "disclaimer.md",
    }


def build_deterministic(cfg: AppConfig) -> dict[Path, str]:
    """The artefacts that are pure functions of files already on disk."""
    rows = load_source_rows(cfg)
    return {
        artefact_paths(cfg)["sources_csv"]: render_source_csv(rows),
        artefact_paths(cfg)["sources_md"]: render_source_md(rows),
        artefact_paths(cfg)["disclaimer"]: render_disclaimer(),
    }


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def check(deterministic: dict[Path, str], cfg: AppConfig) -> int:
    """Exit non-zero when a committed artefact is missing or has drifted.

    `source_list.*` and `disclaimer.md` are byte-comparable: same inputs, same bytes. So is
    the *shape* of `sample_qa.md` - entry count - but not its wording, since it is model
    output. That asymmetry is why this function checks the first three by diff and the
    fourth by structure.
    """
    drifted = 0
    for path, content in sorted(deterministic.items()):
        name = path.relative_to(PROJECT_ROOT)
        if not path.is_file():
            print(f"  MISSING  {name}")
            drifted += 1
        elif path.read_text(encoding="utf-8") != content:
            print(f"  DRIFTED  {name}")
            drifted += 1
        else:
            print(f"  current  {name}")

    sample = artefact_paths(cfg)["qa"]
    name = sample.relative_to(PROJECT_ROOT)
    if not sample.is_file():
        print(f"  MISSING  {name}")
        drifted += 1
    else:
        entries = sample.read_text(encoding="utf-8").count("**Question:**")
        if entries == len(SAMPLE_CASES):
            print(f"  current  {name} ({entries} entries; wording is model output, not diffed)")
        else:
            print(f"  STALE    {name} ({entries} entries, expected {len(SAMPLE_CASES)})")
            drifted += 1

    if drifted:
        print(
            f"\nFAIL: {drifted} artefact(s) out of date. "
            f"Rerun `python scripts/export_outputs.py`."
        )
    else:
        print("\nPASS: every committed artefact matches what this script generates now.")
    return 1 if drifted else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 9 - generate the PRD FR-8 deliverables")
    parser.add_argument(
        "--only",
        choices=["sources", "qa", "disclaimer", "all"],
        default="all",
        help="which artefact(s) to write (default: all)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed artefacts are current; write nothing",
    )
    parser.add_argument(
        "--client",
        choices=["config", "echo"],
        default="config",
        help="config = the configured provider (needs LLM_API_KEY); echo = keyless extractive",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="seconds to wait before each question that reaches the LLM "
             "(default: 16 for a hosted provider, 0 for echo)",
    )
    args = parser.parse_args()

    cfg = get_config()
    paths = artefact_paths(cfg)
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)

    if args.client == "echo":
        client, degraded = EchoClient(cfg), True
    else:
        client, degraded = get_llm(cfg)

    print(f"Phase 9 - export outputs  (only={args.only}, client={describe_client(client)})")
    if degraded:
        print("  degraded=True: answers are extractive (EchoClient), not generated prose.")

    needs_index = args.only in ("qa", "all")
    if needs_index or args.check:
        try:
            from mf_rag.store import verify_collection

            verify_collection(cfg)
        except Exception as exc:  # noqa: BLE001 - the message is the actionable part
            print(f"\nERROR: the vector index is not usable ({exc})")
            print("Run `python scripts/build_index.py --reset` (Stages 3-4) first.")
            return 1

    deterministic = build_deterministic(cfg)
    written: list[Path] = []

    if args.check:
        print("\nCHECK")
        return check(deterministic, cfg)

    if args.only in ("sources", "all"):
        for path in (paths["sources_csv"], paths["sources_md"]):
            write(path, deterministic[path])
            written.append(path)
        print(f"\nsource list: {len(load_source_rows(cfg))} rows from {cfg.sources_csv.name}")

    if args.only in ("disclaimer", "all"):
        write(paths["disclaimer"], deterministic[paths["disclaimer"]])
        written.append(paths["disclaimer"])
        print("disclaimer: imported from mf_rag.prompts.DISCLAIMER")

    if needs_index:
        delay = args.delay if args.delay is not None else (0.0 if degraded else 16.0)
        print(f"\nsample Q&A: {len(SAMPLE_CASES)} live queries (delay {delay:g}s)")
        results = run_sample(cfg, client, delay=delay)
        write(paths["qa"], render_sample_qa(results, describe_client(client), cfg))
        written.append(paths["qa"])
        failed = [result.case.id for result in results if not result.ok]
        if failed:
            print(f"  WARNING: response contract failed for {failed}")

    for path in written:
        print(f"wrote {path.relative_to(PROJECT_ROOT)}")
    print(f"\n{len(written)} file(s) written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())