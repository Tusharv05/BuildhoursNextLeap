"""Phase 9 deliverables: the artefacts must be generated, and must stay in step with the code.

Three kinds of drift are guarded here, because each one ships a wrong document without
raising anything:

1. The exported source list drifting from `data/sources.csv` (a source added, a page that
   started failing, a URL changed).
2. `disclaimer.md` drifting from `mf_rag.prompts.DISCLAIMER`, which is the exact string the
   UI renders - the whole point of FR-8.5 is that the two cannot diverge.
3. `sample_qa.md` drifting in shape: the wrong number of entries, or an entry that no longer
   satisfies the PRD 7 response contract.

None of these tests call a hosted LLM. The pipeline runs with `EchoClient`, injected
explicitly, so the suite is hermetic (see conftest's `_no_ambient_llm_credentials`).
"""

from __future__ import annotations

import csv
import importlib.util
import io
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "export_outputs.py"

from conftest import needs_model  # noqa: E402

from mf_rag.prompts import DISCLAIMER, REFUSAL_TEMPLATES, extract_urls  # noqa: E402
from mf_rag.sources import allowed_urls  # noqa: E402

SOURCE_LIST_COLUMNS = (
    "source_id",
    "scheme",
    "category",
    "doc_type",
    "is_official",
    "source_url",
    "fetched_at",
    "status",
)


def _load_module():
    """Import the script as a module.

    It is registered in `sys.modules` before execution, not after: the module defines
    dataclasses and uses `from __future__ import annotations`, so `dataclasses` resolves the
    field annotations through `sys.modules[cls.__module__]` and fails on `None` if the module
    was never registered.
    """
    spec = importlib.util.spec_from_file_location("export_outputs_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exporter():
    return _load_module()


# ------------------------------------------------------------------- the script itself


def test_help_runs_without_an_index_or_a_key() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=180
    )
    assert proc.returncode == 0, proc.stderr
    assert "--only" in proc.stdout
    assert "--check" in proc.stdout


def test_import_is_side_effect_free(exporter) -> None:
    """Importing must not write files or make network calls."""
    assert callable(exporter.main)
    assert callable(exporter.render_source_csv)
    assert callable(exporter.render_disclaimer)


def test_bad_flag_fails_loudly() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--only", "not-a-target"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode != 0


def test_missing_sources_csv_says_how_to_fix_it(exporter, tmp_path, cfg) -> None:
    import dataclasses

    empty = dataclasses.replace(cfg, root=tmp_path)
    with pytest.raises(FileNotFoundError, match="run_ingest.py"):
        exporter.load_source_rows(empty)


# ------------------------------------------------------------------------ source list


def test_csv_has_exactly_the_fr_8_2_columns(exporter, cfg) -> None:
    rows = exporter.load_source_rows(cfg)
    parsed = list(csv.reader(io.StringIO(exporter.render_source_csv(rows))))
    assert tuple(parsed[0]) == SOURCE_LIST_COLUMNS
    assert len(parsed) == len(rows) + 1


def test_csv_covers_every_registered_source(exporter, cfg) -> None:
    """Every URL in the registry must appear, including ones that failed to fetch."""
    rows = exporter.load_source_rows(cfg)
    exported = {row["source_url"] for row in rows}
    assert exported == allowed_urls()


def test_csv_is_byte_identical_on_a_second_render(exporter, cfg) -> None:
    """Reproducibility: the CSV must be a pure function of data/sources.csv."""
    rows = exporter.load_source_rows(cfg)
    assert exporter.render_source_csv(rows) == exporter.render_source_csv(rows)
    assert "\r\n" not in exporter.render_source_csv(rows)


def test_every_exported_url_is_allowlisted(exporter, cfg) -> None:
    from mf_rag.sources import validate_host

    for row in exporter.load_source_rows(cfg):
        validate_host(row["source_url"])


def test_markdown_lists_every_source_including_the_skipped_ones(exporter, cfg) -> None:
    rows = exporter.load_source_rows(cfg)
    body = exporter.render_source_md(rows)
    for row in rows:
        assert row["source_id"] in body, f"{row['source_id']} missing from source_list.md"
    skipped = [r for r in rows if r["status"] != "ok"]
    if skipped:
        assert "Not ingested on the last run" in body


def test_committed_source_list_is_current(exporter, cfg) -> None:
    """`--check` must pass on a clean tree, or the shipped artefacts are stale."""
    for path, content in exporter.build_deterministic(cfg).items():
        assert path.is_file(), f"{path} is missing; run scripts/export_outputs.py"
        assert path.read_text(encoding="utf-8") == content, f"{path} has drifted"


def test_committed_csv_matches_the_committed_markdown(exporter, cfg) -> None:
    """The two FR-8.2 formats must describe the same corpus, in the same order."""
    paths = exporter.artefact_paths(cfg)
    csv_rows = list(csv.DictReader(io.StringIO(paths["sources_csv"].read_text(encoding="utf-8"))))
    md_body = paths["sources_md"].read_text(encoding="utf-8")
    for row in csv_rows:
        assert f"`{row['source_id']}`" in md_body
        assert row["source_url"] in md_body
    # The first table is the source table; the one under "## URLs" is a bulleted list.
    md_order = [
        line.split("`")[1]
        for line in md_body.split("## URLs")[0].splitlines()
        if line.startswith("| `")
    ]
    assert md_order == [row["source_id"] for row in csv_rows]


# --------------------------------------------------------------------------- disclaimer


def test_disclaimer_file_contains_the_exact_ui_string() -> None:
    body = (PROJECT_ROOT / "outputs" / "disclaimer.md").read_text(encoding="utf-8")
    assert DISCLAIMER in body


def test_disclaimer_is_a_paragraph_of_its_own() -> None:
    """Not wrapped in quotes or prefixed - it has to be pasteable as-is."""
    body = (PROJECT_ROOT / "outputs" / "disclaimer.md").read_text(encoding="utf-8")
    assert DISCLAIMER in [line.strip() for line in body.splitlines()]


def test_disclaimer_file_carries_the_refusal_templates(exporter) -> None:
    body = exporter.render_disclaimer()
    for name, template in REFUSAL_TEMPLATES.items():
        assert name in body
        assert template.format(link="{link}") in body


def test_committed_disclaimer_is_current(exporter, cfg) -> None:
    path = exporter.artefact_paths(cfg)["disclaimer"]
    assert path.read_text(encoding="utf-8") == exporter.render_disclaimer()


def test_the_ui_renders_the_exported_disclaimer_itself() -> None:
    """FR-8.5's claim is that the exported string is the UI's string.

    Checked against the app source rather than by importing it: importing `streamlit_app`
    would need a Streamlit runtime, and the guarantee is about which constant is bound.
    """
    app_source = (PROJECT_ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert "FOOTER_TEXT = DISCLAIMER" in app_source, (
        "the UI must render prompts.DISCLAIMER, not a retyped copy of it, or the exported "
        "disclaimer.md can drift from what the user sees"
    )
    assert "DOCK_CAPTION = \"Facts-only. No investment advice." in app_source, (
        "FR-7.3 requires the always-visible dock caption to carry the design system's "
        "exact 'Facts-only. No investment advice.' phrase"
    )


# --------------------------------------------------------------------------- sample Q&A


def test_sample_qa_has_the_required_coverage(exporter) -> None:
    """FR-8.4 asks for 5-10; Phase 9 fixes 8, spanning these categories."""
    body = (PROJECT_ROOT / "outputs" / "sample_qa.md").read_text(encoding="utf-8")
    assert body.count("**Question:**") == 8
    for case in exporter.SAMPLE_CASES:
        assert case.question in body, f"{case.id} is missing from sample_qa.md"
    for role in ("riskometer", "benchmark", "statement download", "refusal"):
        assert role in body


def test_sample_case_list_covers_what_phase_9_requires(exporter) -> None:
    """3 numeric facts on 3 different schemes, ELSS, riskometer, benchmark, statement, 2 refusals."""
    cases = exporter.SAMPLE_CASES
    assert len(cases) == 8
    schemes = {
        "Large Cap" if "Large Cap" in case.question else
        "Small Cap" if "Small Cap" in case.question else
        "ELSS" if "ELSS" in case.question else
        "Flexi Cap" if "Flexi Cap" in case.question else
        "Balanced Advantage" if "Balanced Advantage" in case.question else
        "none"
        for case in cases
    }
    assert {"Large Cap", "Small Cap", "ELSS"} <= schemes
    assert sum(1 for case in cases if case.expect_refusal) == 2
    assert {case.expect_intent for case in cases if case.expect_refusal} == {"advisory", "returns"}


def test_sample_qa_never_shows_a_failed_check() -> None:
    body = (PROJECT_ROOT / "outputs" / "sample_qa.md").read_text(encoding="utf-8")
    assert "- FAIL -" not in body
    assert "**FAIL**" not in body


def test_sample_qa_answers_are_within_the_contract() -> None:
    """Re-check the shipped answers rather than trusting the generator's own report."""
    from mf_rag.answer import count_sentences

    body = (PROJECT_ROOT / "outputs" / "sample_qa.md").read_text(encoding="utf-8")
    blocks = body.split("\n### ")[1:]
    assert len(blocks) == 8
    ingested = allowed_urls()
    for block in blocks:
        quoted = "\n".join(
            line[2:] for line in block.splitlines() if line.startswith("> ")
        )
        answer = quoted.split("**Source link:**")[0]
        urls = extract_urls(answer)
        assert len(urls) == 1, f"expected exactly one link in:\n{answer}"
        assert urls[0] in ingested, f"{urls[0]} is not an ingested source"
        assert count_sentences(answer) <= 3, f"too many sentences:\n{answer}"


def test_both_required_refusals_appear_verbatim() -> None:
    """A refusal must be the PRD 7 constant, not something the model phrased."""
    body = (PROJECT_ROOT / "outputs" / "sample_qa.md").read_text(encoding="utf-8")
    assert REFUSAL_TEMPLATES["advisory"].split("{link}")[0].strip() in body
    assert REFUSAL_TEMPLATES["returns"].split("{link}")[0].strip() in body
    assert "LLM calls: 0" in body


@needs_model
def test_sample_runs_end_to_end_with_the_keyless_client(exporter, cfg) -> None:
    """The deliverable must be regenerable without an API key, or a reviewer cannot rebuild it."""
    from mf_rag.llm import EchoClient

    results = exporter.run_sample(cfg, EchoClient(cfg), delay=0.0)
    assert len(results) == len(exporter.SAMPLE_CASES)
    for result in results:
        assert result.intent == result.case.expect_intent, result.case.id
        assert result.llm_calls == (0 if result.case.expect_refusal else 1), result.case.id
    body = exporter.render_sample_qa(results, "EchoClient", cfg)
    assert body.count("**Question:**") == len(exporter.SAMPLE_CASES)


@needs_model
def test_contract_checks_catch_a_non_compliant_answer(exporter, cfg) -> None:
    """A checker that cannot fail proves nothing.

    The Answer is built by hand rather than by the pipeline because the pipeline repairs
    non-compliant drafts before returning, so a rogue client would come back compliant and
    the check would have nothing to catch.
    """
    from types import SimpleNamespace

    trace = SimpleNamespace(
        pii_blocked=False,
        hits=[
            {
                "source_id": "hdfcfund_large_cap_official",
                "source_url": "https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct",
                "score": 0.83,
            }
        ],
    )
    bad = SimpleNamespace(
        text=(
            "You should buy this fund today. "
            "https://invented.example.com/fake\n\nLast updated from sources: 2026-09-28"
        ),
        source_url="https://invented.example.com/fake",
        fetched_at="2026-09-28",
        intent="factual",
        is_refusal=False,
        trace=trace,
    )
    checks = dict(exporter.contract_checks(bad))
    assert checks["advice_language"] is False
    assert checks["url_in_ingested_sources"] is False
    assert checks["official_preference (cited the official page: https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct)"] is False

    # ...and a compliant answer against the same retrieved hits passes all of them.
    bad.source_url = "https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct"
    bad.text = (
        f"The expense ratio is 1.03%. {bad.source_url}\n\nLast updated from sources: 2026-09-28"
    )
    assert all(ok for _, ok in exporter.contract_checks(bad))


def test_refusal_checks_reject_a_rephrased_refusal(exporter) -> None:
    """A refusal the model wrote in its own words is not the PRD 7 template."""
    from types import SimpleNamespace

    trace = SimpleNamespace(pii_blocked=False, hits=[])
    almost = SimpleNamespace(
        text="I am not able to suggest which fund you should pick.",
        source_url="https://www.hdfcfund.com/learn/blog/what-expense-ratio-mutual-funds",
        fetched_at="",
        intent="advisory",
        is_refusal=True,
        trace=trace,
    )
    checks = dict(exporter.contract_checks(almost))
    assert checks["text is a PRD 7 template, byte-exact"] is False