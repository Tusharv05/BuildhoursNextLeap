"""Stage 1 ingestion: fetch policy, warn-and-skip, snapshots, report. No network."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from mf_rag.config import AppConfig
from mf_rag.ingest import (
    IngestReport,
    SourceFetchError,
    SourceResult,
    build_source_doc,
    fetch,
    ingest_all,
    load_source_docs,
    load_source_rows,
)
from mf_rag.sources import SOURCES

FIXTURES = Path(__file__).parent / "fixtures"
SCHEME_HTML = (FIXTURES / "scheme_page.html").read_text(encoding="utf-8")
FAQ_HTML = (FIXTURES / "faq_page.html").read_text(encoding="utf-8")

ROW = {
    "source_id": "test_scheme",
    "source_url": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
    "scheme": "HDFC Large Cap Fund - Direct Growth",
    "category": "Large Cap",
    "doc_type": "scheme_page",
    "title": "HDFC Large Cap Fund - Direct Growth",
    "is_official": False,
    "notes": "",
}


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_build_source_doc_includes_rendered_facts() -> None:
    doc, error, _ = build_source_doc(ROW, SCHEME_HTML, "2026-09-27")
    assert error == ""
    assert doc is not None
    assert "Expense ratio (annual %) (Direct plan): 1.50" in doc.text
    assert "large cap stocks" in doc.text
    assert doc.fetched_at == "2026-09-27"
    assert doc.is_official is False


def test_build_source_doc_rejects_thin_page() -> None:
    doc, error, _ = build_source_doc(ROW, "<html><body><p>hi</p></body></html>", "2026-09-27")
    assert doc is None
    assert "too little text" in error


def test_build_source_doc_rejects_empty_html() -> None:
    doc, error, _ = build_source_doc(ROW, "", "2026-09-27")
    assert doc is None
    assert "no main content" in error


def test_fetch_returns_html_and_final_url(cfg: AppConfig) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=SCHEME_HTML)

    with _client(handler) as client:
        html, status, final = fetch(ROW["source_url"], cfg, client=client)
    assert status == 200
    assert "large cap stocks" in html
    assert final == ROW["source_url"]


def test_fetch_rejects_cross_host_redirect(cfg: AppConfig) -> None:
    """A redirect to a non-allowlisted host must not enter the corpus."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "groww.in":
            return httpx.Response(302, headers={"Location": "https://someblog.com/hdfc"})
        return httpx.Response(200, text=SCHEME_HTML)  # would be the smuggled content

    with _client(handler) as client:
        with pytest.raises(SourceFetchError, match="not in the allowlist"):
            fetch(ROW["source_url"], cfg, client=client)


def test_fetch_does_not_retry_404(cfg: AppConfig) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(404, text="nope")

    with _client(handler) as client:
        with pytest.raises(SourceFetchError):
            fetch(ROW["source_url"], cfg, client=client)
    assert len(calls) == 1, "404 is not transient; it must not be retried"


def test_fetch_retries_server_error(cfg: AppConfig) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(503, text="down")

    with _client(handler) as client:
        with pytest.raises(SourceFetchError):
            fetch(ROW["source_url"], cfg, client=client)
    assert len(calls) == cfg.ingest.max_retries


def test_ingest_warns_and_skips_a_dead_source(cfg: AppConfig, tmp_path: Path, monkeypatch) -> None:
    """One bad URL must not take down the build (NFR-9)."""
    local = replace(cfg, root=tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if "dead" in str(request.url):
            return httpx.Response(404, text="gone")
        return httpx.Response(200, text=SCHEME_HTML)

    dead_row = dict(ROW, source_id="dead_source", source_url="https://groww.in/dead")
    good_row = dict(ROW, source_id="good_source")
    monkeypatch.setattr("mf_rag.ingest.SOURCES", (dead_row, good_row))

    with _client(handler) as client:
        report = ingest_all(local, reset=True, client=client)

    totals = report.totals
    assert totals["total"] == 2
    assert totals["failed"] == 1
    assert totals["ok"] == 1
    assert (tmp_path / "data" / "raw" / "good_source.txt").is_file()
    assert not (tmp_path / "data" / "raw" / "dead_source.txt").exists()
    assert (tmp_path / "data" / "sources.csv").is_file()
    assert (tmp_path / "outputs" / "ingest_report.json").is_file()

    report_json = json.loads((tmp_path / "outputs" / "ingest_report.json").read_text("utf-8"))
    assert report_json["totals"]["failed"] == 1

    # The surviving source is rehydratable for the next stage.
    rows = load_source_rows(local)
    assert [r["source_id"] for r in rows] == ["good_source"]
    docs = load_source_docs(local)
    assert len(docs) == 1
    assert "Expense ratio" in docs[0].text


def test_ingest_snapshot_has_metadata_header(cfg: AppConfig, tmp_path: Path, monkeypatch) -> None:
    local = replace(cfg, root=tmp_path)
    monkeypatch.setattr("mf_rag.ingest.SOURCES", (dict(ROW),))
    with _client(lambda r: httpx.Response(200, text=FAQ_HTML)) as client:
        ingest_all(local, reset=True, client=client)

    snapshot = (tmp_path / "data" / "raw" / "test_scheme.txt").read_text(encoding="utf-8")
    assert snapshot.startswith("# source_url: https://groww.in/")
    assert "# doc_type: scheme_page" in snapshot
    assert "# fetched_at: " in snapshot


def test_ingest_report_totals() -> None:
    report = IngestReport(started_at="a", finished_at="b")
    report.sources = [
        SourceResult("a", "u", None, None, "faq", "t", "d", True, True, "", status="ok", chars=10),
        SourceResult("b", "u", None, None, "faq", "t", "d", True, True, "", status="skipped"),
        SourceResult("c", "u", None, None, "faq", "t", "d", True, True, "", status="failed"),
    ]
    totals = report.totals
    assert totals == {"total": 3, "ok": 1, "skipped": 1, "failed": 1, "total_chars": 10}
    assert "sources" in report.to_dict()


def test_dry_run_makes_no_network_calls(cfg: AppConfig) -> None:
    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("dry run must not fetch")

    original = httpx.Client
    try:
        import mf_rag.ingest as ingest_module

        ingest_module.httpx.Client = explode  # type: ignore[assignment]
        report = ingest_all(cfg, dry_run=True)
    finally:
        import mf_rag.ingest as ingest_module

        ingest_module.httpx.Client = original  # type: ignore[assignment]

    assert report.totals["total"] == len(SOURCES)
    assert all(s.status == "skipped" and s.error == "dry run" for s in report.sources)


def test_load_source_rows_requires_ingest_first(cfg: AppConfig, tmp_path: Path) -> None:
    local = replace(cfg, root=tmp_path)
    with pytest.raises(FileNotFoundError, match="run_ingest"):
        load_source_rows(local)


def test_load_source_rows_drops_non_ok(cfg: AppConfig, tmp_path: Path) -> None:
    local = replace(cfg, root=tmp_path)
    csv_path = local.sources_csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        "source_id,source_url,scheme,category,doc_type,title,fetched_at,is_official,"
        "verified,status,http_status,chars,error,notes\n"
        "ok_one,https://groww.in/a,Scheme A,Large Cap,scheme_page,T,2026-09-27,False,True,"
        "ok,200,900,,n\n"
        "dead_one,https://groww.in/b,Scheme B,ELSS,scheme_page,T,2026-09-27,False,True,"
        "failed,404,0,boom,n\n",
        encoding="utf-8",
    )
    rows = load_source_rows(local)
    assert [r["source_id"] for r in rows] == ["ok_one"]
    assert rows[0]["scheme"] == "Scheme A"


def test_load_source_docs_rehydrates_snapshots(cfg: AppConfig, tmp_path: Path) -> None:
    local = replace(cfg, root=tmp_path)
    local.raw_dir.mkdir(parents=True, exist_ok=True)
    local.sources_csv.write_text(
        "source_id,source_url,scheme,category,doc_type,title,fetched_at,is_official,"
        "verified,status,http_status,chars,error,notes\n"
        "ok_one,https://groww.in/a,Scheme A,Large Cap,scheme_page,Title,2026-09-27,True,True,"
        "ok,200,900,,n\n",
        encoding="utf-8",
    )
    (local.raw_dir / "ok_one.txt").write_text(
        "# source_url: https://groww.in/a\n"
        "# doc_type: scheme_page\n"
        "# fetched_at: 2026-09-27\n"
        "# title: Title\n\nBody text with facts.",
        encoding="utf-8",
    )

    docs = load_source_docs(local)
    assert len(docs) == 1
    assert docs[0].text == "Body text with facts."
    assert docs[0].is_official is True
    assert docs[0].fetched_at == "2026-09-27"


def test_report_json_is_serialisable() -> None:
    report = IngestReport(started_at="a", finished_at="b")
    report.sources = [SourceResult("a", "u", None, None, "faq", "t", "d", True, True, "")]
    assert json.dumps(report.to_dict())
