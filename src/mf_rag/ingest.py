"""RAG Stage Group A / Stage 1: Loading.

Fetches every source in the registry, cleans it into main content, stamps
metadata, and writes raw snapshots + a CSV + a JSON report.

Failure policy (NFR-9): a dead or blocked URL produces a warning row and the
build continues. One bad link must never take down ingestion.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx

from mf_rag.clean import extract_main_text, looks_useful
from mf_rag.config import AppConfig, get_config
from mf_rag.models import SourceDoc
from mf_rag.sources import SOURCES, validate_host
from mf_rag.structured import extract_facts_section

CSV_FIELDS: tuple[str, ...] = (
    "source_id",
    "source_url",
    "scheme",
    "category",
    "doc_type",
    "title",
    "fetched_at",
    "is_official",
    "verified",
    "status",
    "http_status",
    "chars",
    "error",
    "notes",
)

MIN_USABLE_CHARS = 500


@dataclass
class SourceResult:
    source_id: str
    source_url: str
    scheme: str | None
    category: str | None
    doc_type: str
    title: str
    fetched_at: str
    is_official: bool
    verified: bool
    notes: str
    status: str = "pending"  # ok | skipped | failed
    http_status: int | None = None
    chars: int = 0
    error: str = ""

    def to_doc_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IngestReport:
    started_at: str
    finished_at: str
    sources: list[SourceResult] = field(default_factory=list)

    @property
    def totals(self) -> dict[str, Any]:
        ok = [s for s in self.sources if s.status == "ok"]
        return {
            "total": len(self.sources),
            "ok": len(ok),
            "skipped": sum(1 for s in self.sources if s.status == "skipped"),
            "failed": sum(1 for s in self.sources if s.status == "failed"),
            "total_chars": sum(s.chars for s in self.sources),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "totals": self.totals,
            "sources": [asdict(s) for s in self.sources],
        }


class SourceFetchError(RuntimeError):
    pass


def today() -> str:
    return date.today().isoformat()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def fetch(
    url: str,
    cfg: AppConfig,
    client: httpx.Client | None = None,
) -> tuple[str, int, str]:
    """Fetch a URL. Returns (html, http_status, final_url).

    The final URL is re-checked against the host allowlist so a redirect to a
    third-party site cannot smuggle content into the corpus.
    """
    own_client = client is None
    client = client or httpx.Client(
        timeout=cfg.ingest.timeout_sec,
        follow_redirects=True,
        headers={
            "User-Agent": cfg.ingest.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-IN,en;q=0.9",
        },
    )
    try:
        last_error: Exception | None = None
        for attempt in range(1, cfg.ingest.max_retries + 1):
            try:
                response = client.get(url)
                response.raise_for_status()
                final_url = str(response.url)
                validate_host(final_url)  # reject cross-host redirects
                return response.text, response.status_code, final_url
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code in (404, 403, 401, 410):
                    break  # not transient; do not retry
            except Exception as exc:  # network, timeout, redirect-policy, etc.
                last_error = exc
            if attempt < cfg.ingest.max_retries:
                time.sleep(min(2 ** attempt, 8))
        raise SourceFetchError(f"{type(last_error).__name__}: {last_error}") from last_error
    finally:
        if own_client:
            client.close()


def build_source_doc(
    row: dict[str, Any],
    html: str,
    fetched_at: str,
) -> tuple[SourceDoc | None, str, int]:
    """Clean HTML into a SourceDoc. Returns (doc_or_None, error, http_status_placeholder).

    For client-rendered pages the published scheme data is also rendered as labelled
    fact lines and appended, so the numeric facts the brief asks about (expense
    ratio, exit load, lock-in, minimum SIP, benchmark, riskometer) are retrievable.
    """
    title, text = extract_main_text(html)
    if not text:
        return None, "no main content extracted", 0
    if not looks_useful(text, MIN_USABLE_CHARS):
        return None, f"too little text ({len(text)} chars < {MIN_USABLE_CHARS})", 0

    facts = ""
    try:
        facts = extract_facts_section(html, row["source_url"])
    except Exception:  # payload parsing must never fail the build
        facts = ""
    if facts:
        text = f"{text}\n\n{facts}"

    doc = SourceDoc(
        source_id=row["source_id"],
        source_url=row["source_url"],
        scheme=row.get("scheme"),
        category=row.get("category"),
        doc_type=row["doc_type"],
        title=title or row["title"],
        text=text,
        fetched_at=fetched_at,
        is_official=bool(row.get("is_official", False)),
    )
    return doc, "", 0


def _write_csv(results: Iterable[SourceResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.to_doc_row().get(key) for key in CSV_FIELDS})


def _build_client(cfg: AppConfig) -> httpx.Client:
    return httpx.Client(
        timeout=cfg.ingest.timeout_sec,
        follow_redirects=True,
        headers={
            "User-Agent": cfg.ingest.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-IN,en;q=0.9",
        },
    )


def ingest_all(
    cfg: AppConfig | None = None,
    reset: bool = False,
    only: str | None = None,
    dry_run: bool = False,
    client: httpx.Client | None = None,
) -> IngestReport:
    """Fetch and clean every registry source.

    `client` may be injected (tests use a MockTransport); one is built from config
    and closed here when not supplied.
    """
    cfg = cfg or get_config()
    rows = [row for row in SOURCES if only is None or row["source_id"] == only]
    if only is not None and not rows:
        raise KeyError(f"Unknown source_id: {only}")

    report = IngestReport(started_at=now_iso(), finished_at="")
    if dry_run:
        report.finished_at = now_iso()
        for row in rows:
            report.sources.append(
                SourceResult(
                    source_id=row["source_id"],
                    source_url=row["source_url"],
                    scheme=row.get("scheme"),
                    category=row.get("category"),
                    doc_type=row["doc_type"],
                    title=row["title"],
                    fetched_at="",
                    is_official=bool(row.get("is_official", False)),
                    verified=bool(row.get("verified", False)),
                    notes=row.get("notes", ""),
                    status="skipped",
                    error="dry run",
                )
            )
        return report

    raw_dir = cfg.raw_dir
    if reset and raw_dir.exists():
        for path in raw_dir.glob("*.txt"):
            path.unlink()
    raw_dir.mkdir(parents=True, exist_ok=True)

    owned = client is None
    active = client or _build_client(cfg)
    try:
        for index, row in enumerate(rows):
            source_id = row["source_id"]
            result = SourceResult(
                source_id=source_id,
                source_url=row["source_url"],
                scheme=row.get("scheme"),
                category=row.get("category"),
                doc_type=row["doc_type"],
                title=row["title"],
                fetched_at=today(),
                is_official=bool(row.get("is_official", False)),
                verified=bool(row.get("verified", False)),
                notes=row.get("notes", ""),
            )
            try:
                html, status, _final = fetch(row["source_url"], cfg, client=active)
                result.http_status = status
                doc, error, _ = build_source_doc(row, html, result.fetched_at)
                if doc is None:
                    result.status = "skipped"
                    result.error = error
                    print(f"  SKIP  {source_id}: {error}")
                else:
                    result.status = "ok"
                    result.title = doc.title
                    result.chars = len(doc.text)
                    (raw_dir / f"{source_id}.txt").write_text(
                        f"# source_url: {doc.source_url}\n"
                        f"# doc_type: {doc.doc_type}\n"
                        f"# fetched_at: {doc.fetched_at}\n"
                        f"# title: {doc.title}\n\n{doc.text}",
                        encoding="utf-8",
                    )
                    print(f"  OK    {source_id}: {result.chars} chars")
            except Exception as exc:
                result.status = "failed"
                result.error = str(exc)[:300]
                print(f"  FAIL  {source_id}: {result.error}")
                if cfg.ingest.fail_fast:
                    raise

            report.sources.append(result)
            if index < len(rows) - 1 and cfg.ingest.request_delay_sec > 0:
                time.sleep(cfg.ingest.request_delay_sec)
    finally:
        if owned:
            active.close()

    _write_csv(report.sources, cfg.sources_csv)
    report.finished_at = now_iso()
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    (cfg.outputs_dir / "ingest_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def load_source_rows(cfg: AppConfig | None = None) -> list[dict[str, Any]]:
    """Re-read data/sources.csv written by ingest_all, dropping failed sources."""
    cfg = cfg or get_config()
    if not cfg.sources_csv.is_file():
        raise FileNotFoundError(
            f"{cfg.sources_csv} not found - run 'python scripts/run_ingest.py' first (Phase 1)"
        )
    rows: list[dict[str, Any]] = []
    with cfg.sources_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            rows.append(
                {
                    "source_id": row["source_id"],
                    "source_url": row["source_url"],
                    "scheme": row["scheme"] or None,
                    "category": row["category"] or None,
                    "doc_type": row["doc_type"],
                    "title": row["title"],
                    "is_official": row["is_official"].strip().lower() == "true",
                    "fetched_at": row["fetched_at"],
                }
            )
    return rows


def load_source_docs(cfg: AppConfig | None = None) -> list[SourceDoc]:
    """Rehydrate SourceDoc objects from the raw snapshots on disk."""
    cfg = cfg or get_config()
    docs: list[SourceDoc] = []
    for row in load_source_rows(cfg):
        path = cfg.raw_dir / f"{row['source_id']}.txt"
        if not path.is_file():
            continue
        header, _, body = path.read_text(encoding="utf-8").partition("\n\n")
        meta: dict[str, str] = {}
        for line in header.splitlines():
            if line.startswith("# ") and ": " in line:
                key, _, value = line[2:].partition(": ")
                meta[key.strip()] = value.strip()
        docs.append(
            SourceDoc(
                source_id=row["source_id"],
                source_url=row["source_url"],
                scheme=row["scheme"],
                category=row["category"],
                doc_type=row["doc_type"],
                title=row["title"],
                text=body.strip(),
                fetched_at=row["fetched_at"],
                is_official=row["is_official"],
            )
        )
    return docs
