# Implementation Guide — Mutual Fund FAQ Assistant (RAG Chatbot)

| Field | Value |
|---|---|
| Document type | Phase-wise Implementation Guide |
| Version | 1.0 |
| Purpose | Drive implementation in Cursor, one phase at a time |
| Upstream | [`ARCHITECTURE.md`](./ARCHITECTURE.md) → [`PRD.md`](./PRD.md) |
| Prereq | Python 3.10+, `git`, a virtualenv, (optional) one LLM API key |

---

## 0. How to Use This Document with Cursor

**Golden rule: one phase per Cursor session/agent task. Do not batch phases.**

Each phase below has this shape:

| Block | Meaning |
|---|---|
| **Goal** | One sentence — what exists at the end of the phase |
| **Depends on** | Phases that must be complete and committed first |
| **Files** | Exact paths to create/modify (matches ARCHITECTURE §4) |
| **Build steps** | Ordered tasks |
| **Contracts** | Dataclass / function signatures Cursor must honour (copy verbatim) |
| **Verify** | Commands + the exact expected output that proves the phase |
| **Exit gate** | The checkbox list that must be fully ticked before moving on |
| **Cursor prompt** | A ready-to-paste block |

**Operating rules**

1. At the start of every phase, read `ARCHITECTURE.md` § for that phase and `PRD.md` FR refs — do not re-derive the design.
2. Implement **only** the current phase. No speculative extra features.
3. Every phase ends with a **runnable command and visible output**. If you can't show output, the phase isn't done.
4. Commit after each phase: `feat(phase-N): <summary>`.
5. Never commit `.env`, `data/raw`, `data/chroma`, or `__pycache__`.
6. If a phase's exit gate fails, fix it in that phase — do not carry debt forward.

**Definition of Done (global, applies to every phase)**

- [ ] Code runs from a clean virtualenv with `pip install -r requirements.txt`
- [ ] No secrets in code; all keys via `.env`
- [ ] Type hints on public functions
- [ ] Module docstring naming the RAG stage it implements
- [ ] Errors handled per ARCHITECTURE §9 (no bare `except: pass`)
- [ ] `python -m pytest tests/ -q` passes
- [ ] Phase's `scripts/` command runs and prints the expected report

---

## Phase Overview

| Phase | Stage (PRD) | Goal | Est. |
|---|---|---|---|
| 0 | — | Repo scaffold, config, models, tests, env | S |
| 1 | **Loading** | Official pages fetched, cleaned, stamped, snapshotted | M |
| 2 | **Chunking** | Corpus chunked; strategy chosen with measured evidence | L |
| 3 | **Embedding** | All chunks embedded with `all-MiniLM-L6-v2` | S |
| 4 | **Store Vector Data** | ChromaDB `mf_faq` persisted + verified | S |
| 5 | Retrieval | Top-k retrieval with entity filters + threshold | M |
| 6 | Generation | Strict-prompt LLM answers, ≤3 sentences, 1 link | M |
| 7 | Guardrails | PII block, refusals, output validators | M |
| 8 | UI | Streamlit app with 3 examples + trace panel | M |
| 9 | Packaging | README, source list, sample Q&A, disclaimer | S |
| 10 | Demo | ≤3-min video / hosted link, narrated pipeline | S |

**Critical path:** Phases 1 → 2 → 3 → 4 are strictly sequential (each consumes the previous artifact). Phases 5 → 6 → 7 → 8 are sequential. Phase 9 needs 1–8. Phase 10 needs everything.

---

## Phase 0 — Scaffold, Config, Contracts

**Goal:** A runnable skeleton with config loading, dataclasses, source registry, and a green test suite — no network, no models.

**Depends on:** nothing

**Files**

```
config.yaml
.env.example
.gitignore
requirements.txt
src/mf_rag/__init__.py
src/mf_rag/config.py
src/mf_rag/models.py
src/mf_rag/sources.py
tests/test_config.py
tests/test_models.py
tests/conftest.py
scripts/run_ingest.py        (stub that prints "not implemented yet")
```

**Build steps**

1. Create the full directory tree from ARCHITECTURE §4 (empty packages with `__init__.py`).
2. `requirements.txt` — pin: `httpx`, `beautifulsoup4`, `lxml`, `trafilatura`, `tiktoken`, `sentence-transformers`, `chromadb`, `streamlit`, `pyyaml`, `python-dotenv`, `pytest`.
3. `config.yaml` — copy the **entire** block from ARCHITECTURE §8 verbatim, including comments.
4. `config.py` — dataclass-per-section loader + `get_config(path=None)` cached with `functools.lru_cache`. Unknown YAML keys must raise (catches typos early). Env overrides for `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`.
5. `models.py` — the dataclasses below, verbatim.
6. `sources.py` — the source registry: 5 seed scheme URLs from PRD §1 plus placeholder official entries (HDFC AMC factsheet/FAQ/fees, AMFI, SEBI). Each row: `source_id, source_url, scheme, category, doc_type, title, is_official`. **Enforce a host allowlist** here (ARCHITECTURE §10): `groww.in`, `hdfc mutual fund official domain(s)`, `amfiindia.com`, `sebi.gov.in`. Any other host → `ValueError`.
7. `.gitignore` — `.env`, `data/raw/`, `data/chroma/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `outputs/results/`.
8. `.env.example` — `LLM_API_KEY=`, `LLM_BASE_URL=`, `LLM_MODEL=`.
9. `tests/` — config loads; unknown key raises; `SourceDoc` round-trips; registry host allowlist rejects a foreign host.

**Contracts — `src/mf_rag/models.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

DocType = Literal["scheme_page","factsheet","kim_sid","faq","fees",
                  "riskometer","guide","educational"]

@dataclass
class SourceDoc:
    source_id: str
    source_url: str
    scheme: str | None
    category: str | None
    doc_type: str
    title: str
    text: str
    fetched_at: str          # ISO-8601 date (YYYY-MM-DD)
    is_official: bool

@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    source_url: str
    scheme: str | None
    category: str | None
    doc_type: str
    section_heading: str
    chunk_index: int
    text: str
    embed_text: str
    token_count: int
    fetched_at: str

    def to_metadata(self) -> dict[str, Any]:
        """Flat, Chroma-safe metadata (str/int/bool only)."""
        return {
            "source_id": self.source_id,
            "source_url": self.source_url,
            "scheme": self.scheme,
            "category": self.category,
            "doc_type": self.doc_type,
            "section_heading": self.section_heading,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "fetched_at": self.fetched_at,
        }

@dataclass
class RetrievalHit:
    chunk: Chunk
    score: float               # cosine similarity, 0..1

@dataclass
class Trace:
    query_hash: str
    intent: str
    pii_blocked: bool
    entity_detected: str | None
    where_filter: dict[str, Any] | None
    hits: list[dict[str, Any]]
    chunks_used: list[str]
    selected_source_url: str | None
    validators_run: list[str]
    repair_used: bool
    latency_ms: int
    answer: str

@dataclass
class Answer:
    text: str
    source_url: str | None
    fetched_at: str
    intent: str
    trace: Trace
    is_refusal: bool = False
```

**Verify**

```powershell
python -c "from mf_rag.config import get_config; print(get_config())"
python -m pytest tests/ -q
python scripts/run_ingest.py          # must print "not implemented yet" and exit 0
```

**Exit gate**

- [ ] Directory tree matches ARCHITECTURE §4
- [ ] `config.yaml` is byte-identical to ARCHITECTURE §8 block
- [ ] All 7 dataclasses exist with the exact fields above
- [ ] Registry has all 5 scheme URLs + official placeholder rows
- [ ] Foreign host in registry raises `ValueError` (unit-tested)
- [ ] `pytest` green; `.env` is git-ignored and absent from git

<details><summary><b>Cursor prompt — Phase 0</b></summary>

```
Implement Phase 0 of the Mutual Fund FAQ RAG assistant.

Read first: ARCHITECTURE.md (§3, §4, §8, §10) and PRD.md §16.

Create the full directory tree from ARCHITECTURE.md §4 with __init__.py in every
package. Then:

1. requirements.txt — pin httpx, beautifulsoup4, lxml, trafilatura, tiktoken,
   sentence-transformers, chromadb, streamlit, pyyaml, python-dotenv, pytest.
2. config.yaml — reproduce the YAML block in ARCHITECTURE.md §8 verbatim,
   comments included.
3. .env.example with LLM_API_KEY, LLM_BASE_URL, LLM_MODEL.
4. .gitignore — ignore .env, data/raw/, data/chroma/, __pycache__, .pytest_cache.
5. src/mf_rag/config.py — frozen dataclasses per YAML section, get_config() with
   lru_cache, raise on unknown keys, env overrides for the three LLM vars.
6. src/mf_rag/models.py — exactly the dataclasses in the implementation guide,
   including Chunk.to_metadata().
7. src/mf_rag/sources.py — the source registry as a module-level list of dicts:
   the 5 seed scheme URLs from PRD.md §1, plus official placeholder rows
   (HDFC AMC factsheet / scheme FAQ / fee-charges pages, AMFI, SEBI) with
   is_official=True. Include SCHEME_REGISTRY and CATEGORY_REGISTRY maps for the
   5 schemes. Add ALLOWED_HOSTS and a validate_source() that raises ValueError
   for any host not in the allowlist.
8. tests/test_config.py, tests/test_models.py, tests/test_sources.py,
   tests/conftest.py.
9. scripts/run_ingest.py stub that prints "Phase 1 not implemented yet" and exits 0.

No network calls, no model downloads in this phase.
Run: python -m pytest tests/ -q  (must be green)
```
</details>

---

## Phase 1 — Stage Group A / Stage 1: Loading

**Goal:** `python scripts/run_ingest.py --reset` fetches every registry source, cleans it, writes `data/raw/*.txt` + `data/sources.csv` + `outputs/ingest_report.json` — and survives a dead URL.

**Depends on:** Phase 0

**Files**

```
src/mf_rag/ingest.py
src/mf_rag/clean.py            # HTML → main content
tests/test_clean.py
tests/test_ingest.py
scripts/run_ingest.py
```

**Build steps**

1. `clean.py`
   - `extract_main_text(html: str) -> tuple[str, str]` → `(title, text)`.
   - Priority: `trafilatura` if installed, else BeautifulSoup + `lxml` with a
     boilerplate blacklist (`script, style, nav, header, footer, aside, form, noscript`).
   - Normalise: collapse 3+ newlines → 2, strip trailing spaces, normalise
     Unicode (`unicodedata.normalize("NFKC")`), keep ₹ and %.
2. `ingest.py`
   - `fetch(url) -> str` — `httpx` with `timeout`, `max_retries`, `follow_redirects`,
     custom UA, `request_delay_sec` sleep between URLs.
   - **Failure policy (FR-1.5 / NFR-9):** catch, log a warning entry, continue. Never crash the build.
   - `build_source_doc(source_row, html, fetched_at) -> SourceDoc`.
   - `ingest_all(cfg, reset=False) -> IngestReport` — writes:
     - `data/raw/<source_id>.txt`
     - `data/sources.csv` (all SourceDoc fields + `status`, `http_status`, `chars`, `error`)
   - `IngestReport`: per-source `{source_id, url, ok, http_status, chars, error, fetched_at}` + totals.
3. `scripts/run_ingest.py` — CLI: `--reset`, `--only <source_id>`, `--dry-run`. Prints a table: source_id, status, chars, date. Writes `outputs/ingest_report.json`.
4. Tests with **saved HTML fixtures** (no network in tests): cleaner strips nav/footer; fetch failure yields a warn-and-skip report entry; report totals are correct.

**Verify**

```powershell
python scripts/run_ingest.py --reset
# expect: table of ~10+ rows, 5 scheme pages OK, data/sources.csv written,
#         outputs/ingest_report.json written, no traceback
python -c "import json;r=json.load(open('outputs/ingest_report.json'));print(r['totals'])"
(Get-ChildItem data/raw/*.txt).Count
```

**Exit gate**

- [ ] ≥5 scheme pages ingested with `chars > 500` each
- [ ] `data/sources.csv` has one row per registry source with a `status`
- [ ] `outputs/ingest_report.json` has totals + per-source entries
- [ ] A deliberately bad URL produces a warning row, not a crash
- [ ] Raw text contains real fee/expense/benchmark content (spot-check 2 schemes)
- [ ] `pytest` green, no network in tests
- [ ] `data/raw/` is git-ignored

<details><summary><b>Cursor prompt — Phase 1</b></summary>

```
Implement Phase 1 (RAG Stage Group A / Stage 1: Loading) of the Mutual Fund FAQ
RAG assistant.

Read first: ARCHITECTURE.md §5.1 and PRD.md FR-1.1 to FR-1.5, NFR-9.

Phase 0 is complete — do not modify config.yaml, models.py, or sources.py
except to add fields the ingestion report needs.

Create:
1. src/mf_rag/clean.py — extract_main_text(html) -> (title, text).
   Use trafilatura if available, else BeautifulSoup+lxml. Strip
   script/style/nav/header/footer/aside/form/noscript. Normalise whitespace
   and Unicode. Keep the rupee sign and percent signs.
2. src/mf_rag/ingest.py —
   - fetch(url) using httpx with cfg.ingest.timeout_sec, max_retries,
     follow_redirects, custom User-Agent.
   - ingest_all(cfg, reset=False) -> IngestReport
   - On any per-source failure: record it, print a warning, CONTINUE.
     Never raise out of ingest_all for a single bad URL.
   - Write data/raw/<source_id>.txt, data/sources.csv (every SourceDoc field
     plus status, http_status, chars, error), and outputs/ingest_report.json
     with {totals, sources:[{source_id, url, ok, http_status, chars, error,
     fetched_at}]}.
   - fetched_at is the download date in YYYY-MM-DD.
3. scripts/run_ingest.py — argparse with --reset, --only, --dry-run.
   Print a formatted table: source_id, status, chars, date. Exit 0 even if
   some sources failed.
4. tests/fixtures/*.html — save 2-3 small representative HTML fixtures.
5. tests/test_clean.py and tests/test_ingest.py — these must run with NO
   network. Use the fixtures and monkeypatched httpx.

Run: python scripts/run_ingest.py --reset
Then: python -m pytest tests/ -q
```
</details>

---

## Phase 2 — Stage Group A / Stage 2: Chunking

**Goal:** `data/chunks.jsonl` exists, produced by a **chosen** strategy justified by measured comparison of all three candidates.

**Depends on:** Phase 1 (needs `data/sources.csv` + `data/raw/*.txt`)

**Files**

```
src/mf_rag/chunking/base.py
src/mf_rag/chunking/fixed.py
src/mf_rag/chunking/section.py
src/mf_rag/chunking/parent_child.py
src/mf_rag/chunking/__init__.py     # get_chunker(name, cfg) registry
src/mf_rag/chunk.py                 # orchestrator -> chunks.jsonl
eval/questions.json                 # gold set (10 questions + expected fact)
scripts/run_chunk_eval.py           # compares the 3 strategies
tests/test_chunking.py
```

**Build steps**

1. `base.py` — `Chunker` Protocol: `name: str`, `split(doc: SourceDoc) -> list[Chunk]`. Plus shared helpers used by all strategies:
   - `build_context_header(doc, heading) -> str` → `"{scheme} — {doc_type} — {heading}"`
   - `make_chunk(doc, text, heading, index) -> Chunk` — computes `chunk_id = sha1(source_id + index)`, `embed_text = header + "\n\n" + text`, `token_count` via `tiktoken`.
2. `fixed.py` — target/max chars + `overlap_chars`, sliding window. **Records an orphan flag** when a chunk boundary lands inside a line starting with `|`.
3. `section.py` **(default)** — split on markdown/heading lines, then on table blocks and FAQ `Q:`/`A:` blocks; never split a table row; keep a heading breadcrumb; merge sections below `min_chars`; hard-split above `max_chars`.
4. `parent_child.py` — split into small embed chunks (~300 chars) but map each to its `parent_section`; `Chunk.section_heading` holds the parent; selection in Phase 5 returns the parent text.
5. `__init__.py` — `get_chunker(name, cfg)` with a `REGISTRY = {"fixed":..., "section":..., "parent_child":...}`; unknown name → `ValueError`.
6. `chunk.py` — `build_chunks(cfg, strategy) -> list[Chunk]`, idempotent: same inputs → identical `chunks.jsonl`. Writes JSONL sorted by `source_id, chunk_index`, plus a stats block (`chunks/sources.py`: per-doc counts, avg/median token count, orphan-table-row count).
7. **`eval/questions.json`** — 10 gold questions covering the in-scope intents (FR-2 / PRD §4.2): expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer, benchmark, capital-gains statement download, objective, direct vs regular. Each entry: `{"id", "question", "scheme", "expected_keywords": [...], "doc_type"}`. `expected_keywords` are the tokens that must appear in the retrieved chunk (e.g. `["1.5", "expense"]`).
8. **`scripts/run_chunk_eval.py`** — for each strategy, build chunks and report: chunk count, avg tokens, **answer-bearing hit@5 on the gold set** (lexical match of `expected_keywords` against chunks — no embeddings needed yet), and orphan-row count. Prints a comparison table; writes `outputs/chunk_eval.json`.

**Why this phase is heavy (D7):** the winning strategy must be *evidence-based*. Do not hardcode `section` — run the eval and let the numbers decide. Then record the winner + numbers in `README.md` (Phase 9) and the demo narration.

**Verify**

```powershell
python -m mf_rag.chunk --strategy section
python scripts/run_chunk_eval.py
# expect: 3-row comparison table; report written; a clear winner
python -c "import json;[print(json.loads(l)['chunk_id'], json.loads(l)['token_count']) for l in open('data/chunks.jsonl')][:5]"
```

**Exit gate**

- [ ] `data/chunks.jsonl` produced; re-running produces an identical file (idempotent)
- [ ] All 3 strategies run without error
- [ ] `run_chunk_eval.py` prints a comparison table with a measurable winner
- [ ] Winner recorded in `outputs/chunk_eval.json` **and** copied into `README.md`
- [ ] No table row split in the `section` output (tested)
- [ ] No FAQ question separated from its answer (tested)
- [ ] Every chunk has all metadata fields; avg tokens within 300–800 (PRD §6.3 criterion 3)
- [ ] `chunk_id` stable across runs

<details><summary><b>Cursor prompt — Phase 2</b></summary>

```
Implement Phase 2 (RAG Stage Group A / Stage 2: Chunking) of the Mutual Fund FAQ
RAG assistant.

Read first: ARCHITECTURE.md §5.2 (all five assembly rules) and PRD.md
FR-2.1 to FR-2.4, §6.3.

Phase 1 is complete: data/sources.csv and data/raw/*.txt exist.

Create:
1. src/mf_rag/chunking/base.py — Chunker Protocol (name, split(doc)->list[Chunk])
   plus shared helpers build_context_header(doc, heading) and
   make_chunk(doc, text, heading, index). make_chunk must compute
   chunk_id = sha1(f"{source_id}:{index}"), build embed_text as
   header + "\n\n" + text, and count tokens with tiktoken.
2. src/mf_rag/chunking/fixed.py — sliding window, cfg.chunking.target_chars /
   max_chars / overlap_chars. Track how many cut points land inside a line
   starting with "|" (orphan table rows) and expose it on the chunker instance.
3. src/mf_rag/chunking/section.py — the structure-aware strategy. Split on
   headings, table blocks, and FAQ Q/A blocks. NEVER split a table row.
   NEVER separate a FAQ question from its answer. Keep a heading breadcrumb
   in section_heading. Merge sections smaller than min_chars, hard-split
   anything larger than max_chars.
4. src/mf_rag/chunking/parent_child.py — small embed chunks (~300 chars) that
   each record their parent section in section_heading.
5. src/mf_rag/chunking/__init__.py — get_chunker(name, cfg) with a REGISTRY
   dict; unknown name raises ValueError.
6. src/mf_rag/chunk.py — build_chunks(cfg, strategy) -> list[Chunk].
   Writes data/chunks.jsonl sorted by (source_id, chunk_index), plus
   data/chunk_stats.json with per-source counts, avg/median token_count, and
   orphan table-row count. Must be idempotent.
   Add a CLI: python -m mf_rag.chunk --strategy section
7. eval/questions.json — 10 gold questions, one per in-scope intent
   (expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer, benchmark,
   capital-gains statement download, objective, direct vs regular, tax docs).
   Each: {"id","question","scheme","expected_keywords":[...],"doc_type"}.
8. scripts/run_chunk_eval.py — for EACH strategy build chunks and report:
   chunk count, avg tokens, orphan-row count, and answer-bearing hit@5
   (does any chunk contain all expected_keywords? do this lexically, no
   embeddings yet). Print a comparison table, write outputs/chunk_eval.json.
9. tests/test_chunking.py — assert: table rows never split; FAQ Q/A stay
   together; chunk_id stable; idempotent re-run; get_chunker unknown -> ValueError.

Then run the eval and report which strategy wins and on what numbers.
Do not hardcode the winner before seeing the measurements.
```
</details>

---

## Phase 3 — Stage Group A / Stage 3: Embedding

**Goal:** every chunk has a 384-dim MiniLM vector, computed once and disk-cached.

**Depends on:** Phase 2

**Files**

```
src/mf_rag/embedder.py          # get_embedder() singleton
src/mf_rag/embed_store.py       # Stage 3 (embed) + Stage 4 (store)
tests/test_embedder.py
scripts/build_index.py
```

**Build steps**

1. `embedder.py`
   - `get_embedder(model_name=None, cfg=None)` — `@lru_cache` singleton returning a
     `SentenceTransformer(cfg.embedding.model)`.
   - `embed_texts(texts: list[str]) -> np.ndarray` — `batch_size`,
     `normalize_embeddings=cfg.embedding.normalize`, converts to plain float32 list.
   - `embed_model_name() -> str` — the **exact** model string currently loaded,
     used by the Phase 4 identity assertion.
2. `embed_store.py` (Stage 3 portion)
   - `embedding_cache_key(model, text) -> str` = `sha1(f"{model}:{text}")`.
   - `embed_chunks(chunks, cfg) -> dict[chunk_id, vector]`
     - Load `data/embeddings.jsonl` cache; embed only misses; append and save.
   - Assert all chunks embedded; return count + cache-hit ratio for the report.
3. `scripts/build_index.py --stage embed` — CLI: `--reset-cache` to ignore the cache.

**Verify**

```powershell
python scripts/build_index.py --stage embed
# expect: "embedded N chunks (cache hits: M, misses: K)" and 384-dim shape
python -c "import numpy as np, json; v=[json.loads(l) for l in open('data/embeddings.jsonl')]; print(len(v), len(v[0]['vector']))"
```

**Exit gate**

- [ ] `len(embeddings.jsonl) == len(chunks.jsonl)`
- [ ] Every vector is 384-dim float32
- [ ] Second run reports 100% cache hits and produces identical vectors
- [ ] `get_embedder()` is a true singleton (tested)
- [ ] `pytest` green

<details><summary><b>Cursor prompt — Phase 3</b></summary>

```
Implement Phase 3 (RAG Stage Group A / Stage 3: Embedding).

Read first: ARCHITECTURE.md §5.3, PRD.md FR-3.1, ADR-2.

Phase 2 is complete: data/chunks.jsonl exists with fields per models.Chunk.

Create:
1. src/mf_rag/embedder.py
   - get_embedder(cfg=None) as an lru_cache singleton that loads
     SentenceTransformer(cfg.embedding.model).
   - embed_texts(texts) -> np.ndarray using cfg.embedding.batch_size and
     normalize_embeddings=cfg.embedding.normalize. Return shape (n, 384).
   - embed_query(text) -> list[float] (same model, same normalisation).
   - embed_model_name() -> str  (the exact model string in use).
   IMPORTANT: the index-time and query-time paths MUST both go through this
   module. Do not instantiate SentenceTransformer anywhere else.
2. src/mf_rag/embed_store.py — Stage 3 half only for now.
   - embedding_cache_key(model, text) = sha1(f"{model}:{text}")
   - embed_chunks(chunks, cfg) -> (dict[chunk_id, vector], stats)
     Loads data/embeddings.jsonl, embeds only cache misses, appends new rows,
     returns stats {total, cache_hits, cache_misses}.
   - Raise if any chunk ends up without a vector.
3. scripts/build_index.py with --stage embed and --reset-cache.
   Prints "embedded N chunks (cache hits: M, misses: K)".
4. tests/test_embedder.py — singleton identity, output shape, cache hit
   behaviour. Mark the model-loading test so it can be skipped offline.

Do NOT add ChromaDB yet — that is Phase 4.
Run: python scripts/build_index.py --stage embed
```
</details>

---

## Phase 4 — Stage Group A / Stage 4: Store Vector Data

**Goal:** a persistent ChromaDB collection `mf_faq` that survives a restart, supports metadata filters, and passes the count assertion.

**Depends on:** Phase 3

**Files**

```
src/mf_rag/store.py             # get_client, get_collection, build_collection, verify_collection
tests/test_store.py
scripts/build_index.py          # add --stage store / full pipeline
```

**Build steps**

1. `store.py`
   - `get_client(cfg)` → `chromadb.PersistentClient(path="data/chroma")`
   - `get_or_create_collection(cfg)` → name `mf_faq`, metadata
     `{"hnsw:space": "cosine", "embedding_model": <model name>}`.
     **If the collection exists with a different model name, raise** — silent
     model drift is the #1 RAG bug (ARCHITECTURE §5.3).
   - `build_collection(chunks, vectors, cfg, reset=False)` — upsert with
     `ids=chunk_id`, `documents=embed_text`, `metadatas=chunk.to_metadata()`
     (all values str/int/bool — no None; substitute `""`/`-1`).
   - `verify_collection(cfg, expected_count)` — **hard assertions**:
     1. `collection.count() == expected_count`
     2. `collection.peek()` metadata round-trips
     3. a `where={"category": "ELSS"}` filter returns >0 docs
     Raise a clear error naming the failing check.
2. `scripts/build_index.py --stage all` — embed + store + verify, print the report.

**Verify**

```powershell
python scripts/build_index.py --stage all
# expect: "collection mf_faq: N/N chunks stored", "metadata filter OK"
# then in a BRAND NEW process:
python -c "from mf_rag.store import get_client,get_or_create_collection;print(get_or_create_collection().count())"
# expect: same N  → persistence proven
```

**Exit gate**

- [ ] `count() == len(chunks.jsonl)` in a **fresh process**
- [ ] `where={"category":"ELSS"}` returns results
- [ ] Model-mismatch guard raises on a renamed model
- [ ] No `None` values in stored metadata
- [ ] `pytest` green

<details><summary><b>Cursor prompt — Phase 4</b></summary>

```
Implement Phase 4 (RAG Stage Group A / Stage 4: Store Vector Data).

Read first: ARCHITECTURE.md §5.4 (Chroma schema table), PRD.md FR-3.2/3.3/3.4.

Phases 2-3 are complete: data/chunks.jsonl and data/embeddings.jsonl exist.

Create:
1. src/mf_rag/store.py
   - get_client(cfg) -> chromadb.PersistentClient(path="data/chroma")
   - get_or_create_collection(cfg) — name "mf_faq", metadata
     {"hnsw:space":"cosine","embedding_model":<embed_model_name()>}.
     If an existing collection's stored embedding_model differs from the
     current one, RAISE a clear error (do not silently reuse).
   - build_collection(chunks, vectors, cfg, reset=False) — upsert with
     ids=chunk_id, documents=embed_text, metadatas=chunk.to_metadata().
     Chroma rejects None metadata values: coerce None -> "" for strings and
     -> -1 for chunk_index.
   - verify_collection(cfg, expected_count) — HARD assertions, each with a
     distinct error message:
       a. collection.count() == expected_count
       b. peek(1) metadata round-trips
       c. collection.get(where={"category":"ELSS"}) returns >0 docs
2. Extend scripts/build_index.py with --stage store and --stage all.
   --stage all runs embed -> store -> verify and prints a short report.
   Add --reset to drop and recreate the collection.
3. tests/test_store.py using a tmp_path chroma dir: count, filter, and
   the model-mismatch raise.

Then PROVE persistence: run --stage all, then in a separate python process
print the collection count. It must be unchanged.
```
</details>

---

## Phase 5 — Stage Group B: Retrieval

**Goal:** `retrieve("Exit load of HDFC Small Cap?")` returns ranked chunks from the *Small Cap* source with an explainable trace.

**Depends on:** Phase 4

**Files**

```
src/mf_rag/entities.py          # scheme/category detection from the query
src/mf_rag/retrieve.py
eval/run_eval.py                # retrieval-only mode
tests/test_entities.py
tests/test_retrieve.py
```

**Build steps**

1. `entities.py`
   - `detect_entity(query) -> tuple[str|None, dict|None]` → `(kind, {"scheme"|"category": value})`
   - Match against `SCHEME_REGISTRY` / `CATEGORY_REGISTRY` with **aliases and
     fuzzy token overlap** (e.g. `"elss"` → ELSS, `"tax saver"` → ELSS,
     `"smallcap"`/`"small cap"` → Small Cap, `"flexi"`/`"hdfc equity"` → Flexi Cap,
     `"balanced advantage"` → Balanced Advantage).
   - No match → `(None, None)`.
2. `retrieve.py`
   - `retrieve(query, k=None, min_score=None, entity_filter=True) -> list[RetrievalHit]`
   - Order of operations (ARCHITECTURE §6.2):
     1. `embed_query(query)` (must call `mf_rag.embedder`)
     2. `detect_entity(query)` → build `where`
     3. `collection.query(query_embeddings=[q], n_results=k, where=where)`
     4. If filtered query returns 0 hits **and** a filter was applied → retry
        unfiltered (`fallback_unfiltered: true`)
     5. Convert distances to cosine scores (`score = 1 - distance`), drop
        anything `< min_score`
     6. Dedupe by `source_id` keeping the best chunk, cap at
        `cfg.retrieval.context_chunks`
   - `retrieve_with_trace(...) -> tuple[list[RetrievalHit], dict]` where the
     trace dict has `entity_detected`, `where_filter`, `hits` (chunk_id, score,
     source_url, section_heading), `top_score`.
3. `eval/run_eval.py --mode retrieval` — for each gold question in
   `eval/questions.json`, run retrieval and report whether an answer-bearing
   chunk made top-k. Prints per-question pass/fail + a hit rate, writes
   `outputs/retrieval_eval.json`.
4. Unit tests: entity detection cases; `where` filter applied for a Small Cap
   question; fallback when the filter yields nothing; `min_score` drops weak hits.

**Verify**

```powershell
python -c "from mf_rag.retrieve import retrieve_with_trace; h,t=retrieve_with_trace('Exit load of HDFC Small Cap Fund?'); print(t['where_filter']); [print(round(x.score,3), x.chunk.source_url) for x in h]"
python eval/run_eval.py --mode retrieval
# expect: hit rate printed; tune min_score only with evidence (OQ-5)
```

**Exit gate**

- [ ] A scheme-named question returns a `where` filter and hits only that scheme
- [ ] Gold-set hit@5 recorded in `outputs/retrieval_eval.json`
- [ ] `min_score` chosen from the eval, documented in README
- [ ] Unfiltered fallback works
- [ ] `pytest` green

<details><summary><b>Cursor prompt — Phase 5</b></summary>

```
Implement Phase 5 (Retrieval) of the Mutual Fund FAQ RAG assistant.

Read first: ARCHITECTURE.md §6.2, PRD.md FR-4.1 to FR-4.5.

Phase 4 is complete: the persistent Chroma collection "mf_faq" exists.

Create:
1. src/mf_rag/entities.py — detect_entity(query) -> (kind, where_dict|None).
   Match against SCHEME_REGISTRY and CATEGORY_REGISTRY with aliases and
   token-overlap fuzzy matching. Required aliases: elss, tax saver, tax-saver,
   80c -> ELSS; smallcap, small cap -> Small Cap; flexi, hdfc equity ->
   Flexi Cap; balanced advantage -> Balanced Advantage; large cap, largecap.
   No match -> (None, None).
2. src/mf_rag/retrieve.py
   - retrieve(query, k=None, min_score=None, entity_filter=True) -> list[RetrievalHit]
     1. embed_query via mf_rag.embedder.embed_query (never load the model here)
     2. detect_entity -> where filter
     3. collection.query(n_results=k, where=where)
     4. if the filtered query returns 0 hits and a filter was applied,
        retry unfiltered (cfg.retrieval.fallback_unfiltered)
     5. score = 1 - distance; drop scores < min_score
     6. dedupe by source_id keeping the best chunk; cap at
        cfg.retrieval.context_chunks
   - retrieve_with_trace(...) -> (hits, trace) where trace contains
     entity_detected, where_filter, hits[{chunk_id, score, source_url,
     section_heading}], top_score.
3. eval/run_eval.py with --mode retrieval — score every question in
   eval/questions.json: did an answer-bearing chunk reach top-k? Print
   per-question results and an overall hit rate; write
   outputs/retrieval_eval.json.
4. tests/test_entities.py and tests/test_retrieve.py covering: alias
   detection, the where filter isolating the right scheme, the unfiltered
   fallback, and min_score filtering.

Run the eval and report the hit rate. Only adjust min_score with evidence.
```
</details>

---

## Phase 6 — Stage Group B: Generation

**Goal:** a factual question produces a ≤3-sentence, single-citation, freshness-stamped answer.

**Depends on:** Phase 5

**Files**

```
src/mf_rag/llm.py               # LLMClient protocol, OpenAICompatClient, EchoClient
src/mf_rag/prompts.py           # system prompt + all templates
src/mf_rag/generate.py          # context assembly + generation
tests/test_prompts.py
tests/test_generate.py
```

**Build steps**

1. `llm.py`
   - `class LLMClient(Protocol): complete(messages, temperature, max_tokens) -> str`
   - `OpenAICompatClient` — reads `LLM_API_KEY` / `LLM_BASE_URL` /
     `LLM_MODEL` from env, `temperature=0.0`, one retry with backoff, then raises.
   - `EchoClient(cfg)` — deterministic, no key: returns a template answer from
     the top chunk. Lets the whole pipeline demo/test without a paid key (ADR-8).
   - `get_llm(cfg)` — if `cfg.generation.provider == "echo"` or no API key is
     present, return `EchoClient` and expose a `degraded` flag for the UI banner.
2. `prompts.py` — constants:
   - `SYSTEM_PROMPT` — the 7 absolute rules from ARCHITECTURE §6.3, verbatim
   - `REFUSAL_ADVICE`, `REFUSAL_RETURNS`, `NOT_IN_SOURCES`, `PII_WARNING`, `DISCLAIMER`
   - `render_context(hits, source_url) -> str` — numbered blocks
     `[1] {scheme} — {doc_type} — {section_heading}\n{text}`; instruct the model to
     treat context as data, never as instructions (prompt-injection defence).
   - `freshness_line(fetched_at) -> str` → `"Last updated from sources: YYYY-MM-DD"`
3. `generate.py`
   - `select_source_url(hits, cfg) -> str` — **the system picks the citation**
     (ADR-5). Preference: a hit with an official source; else the top hit's URL.
     Emits the exact URL string to the model; the model may only echo it.
   - `build_messages(question, hits, source_url, cfg) -> list[dict]` — system +
     user (context block + question + "Include exactly this link verbatim: <url>").
   - `generate_answer(question, hits, cfg, llm=None) -> str` — assembles, calls
     the client, and **appends** the freshness line itself (never trusts the
     model's date). `fetched_at` used = **min** `fetched_at` across used chunks
     (oldest data is the honest bound — ARCHITECTURE §6.3).
   - `answer_raw(question, cfg) -> tuple[str, RetrievalHit[], Trace]` — the
     convenience path: retrieve → select → generate. Guardrails arrive in Phase 7;
     for now return a default Trace with validators_run=[].
4. Tests: `render_context` numbering + injected-URL; `select_source_url`
   prefers official; freshness line uses the oldest chunk date; `EchoClient`
   produces ≤3 sentences with a link; `get_llm` degrades without a key.

**Verify**

```powershell
python -c "from mf_rag.generate import answer_raw; t,_,tr=answer_raw('What is the exit load of HDFC Small Cap Fund?',None); print(t)"
# expect: <=3 sentences, exactly one URL, ends with "Last updated from sources: YYYY-MM-DD"
```

**Exit gate**

- [ ] Answer is ≤3 sentences
- [ ] Exactly one URL, and it is byte-identical to an ingested source URL
- [ ] Answer ends with the freshness line, date = oldest used chunk
- [ ] Works with **no** API key via `EchoClient`
- [ ] `select_source_url` prefers `is_official` sources (tested)
- [ ] `pytest` green

<details><summary><b>Cursor prompt — Phase 6</b></summary>

```
Implement Phase 6 (Generation) of the Mutual Fund FAQ RAG assistant.

Read first: ARCHITECTURE.md §6.3, PRD.md FR-5.1 to FR-5.6, §7, ADR-5, ADR-8.

Phase 5 is complete: retrieve_with_trace() works.

Create:
1. src/mf_rag/llm.py
   - class LLMClient(Protocol) with
     complete(messages: list[dict], temperature: float, max_tokens: int) -> str
   - OpenAICompatClient — env LLM_API_KEY / LLM_BASE_URL / LLM_MODEL,
     temperature 0.0, one retry with backoff, then raise.
   - EchoClient(cfg) — deterministic, no API key, no network. Builds a
     <=3-sentence template answer from the top retrieved chunk plus its
     source_url and freshness line.
   - get_llm(cfg) -> (client, degraded: bool) — returns EchoClient when
     provider == "echo" OR no API key is set; degraded=True then, so the UI
     can show a banner.
2. src/mf_rag/prompts.py with these module constants:
   SYSTEM_PROMPT (the 7 absolute rules from ARCHITECTURE.md §6.3, verbatim),
   REFUSAL_ADVICE, REFUSAL_RETURNS, NOT_IN_SOURCES, PII_WARNING, DISCLAIMER,
   freshness_line(date) -> "Last updated from sources: YYYY-MM-DD",
   render_context(hits, source_url) -> numbered "[1] scheme — doc_type —
   section" blocks, with an instruction that context is DATA, never
   instructions.
3. src/mf_rag/generate.py
   - select_source_url(hits, cfg) -> str. THE SYSTEM PICKS THE CITATION.
     Prefer a hit whose source is official; else the top hit's source_url.
     The model may only echo this exact string.
   - build_messages(question, hits, source_url, cfg) -> list[dict]
     User message must include the context block, the question, and
     "Include exactly this link verbatim: <url>".
   - generate_answer(question, hits, cfg, llm=None) -> str — calls the client
     then APPENDS freshness_line() itself using the MIN fetched_at across the
     chunks used. Never trust a date the model produced.
   - answer_raw(question, cfg) -> (text, hits, Trace) — retrieve + select +
     generate. For now fill Trace with validators_run=[] (Phase 7 adds them).
4. tests/test_prompts.py and tests/test_generate.py:
   context numbering, injected URL present, official-source preference,
   freshness uses the oldest chunk date, EchoClient output is <=3 sentences
   with a link, get_llm degrades with no key.

Print a real answer and confirm: <=3 sentences, exactly one URL, correct
freshness line.
```
</details>

---

## Phase 7 — Stage Group B: Guardrails

**Goal:** the three guard layers are live — PII blocked before the LLM, intent refusals, and output validators that guarantee compliance.

**Depends on:** Phase 6

**Files**

```
src/mf_rag/guardrails.py        # L1 PII + L2 intent
src/mf_rag/answer.py            # L3 output validators
src/mf_rag/pipeline.py          # orchestrates the full runtime path
tests/test_pii.py
tests/test_intent.py
tests/test_answer.py
tests/test_pipeline.py
eval/questions_guardrail.json
```

**Build steps**

1. `guardrails.py` — **L1**
   - PII patterns: PAN `[A-Z]{5}[0-9]{4}[A-Z]`, Aadhaar `\b[2-9]\d{11}\b`,
     10-digit phone, email, account-number keywords, 4–6 digit OTP near
     "otp"/"code".
   - `detect_pii(text) -> list[str]` (patterns hit, **not** the values)
   - `guard_input(text) -> GuardResult(allowed, reason, matched_pattern_names)`
   - Never log the matched text (NFR-4).
2. `guardrails.py` — **L2**
   - `classify_intent(text) -> str` ∈ `{"factual","advisory","portfolio",
     "returns","out_of_corpus"}`
   - Deterministic pattern rules first (should I buy / recommend / best /
     suitable for / allocate / portfolio / SIP plan for me → `advisory` or
     `portfolio`; return/CAGR/projected/after 10 years → `returns`), plus a
     scheme-name check → `out_of_corpus` when the text names a fund outside
     the registry.
   - `refusal_for(intent, question) -> Answer` — returns the right template +
     one relevant **educational** link drawn from the registry (never invented).
3. `answer.py` — **L3** validators
   - `count_sentences(text) -> int` (handles `1.5%` decimals — do not split on a
     period between digits)
   - `extract_urls(text) -> list[str]`
   - `validate(answer_text, ctx) -> ValidationResult(ok, violations[], repaired_text)`
   - Checks: ≤3 sentences · exactly 1 URL · URL ∈ ingested source set ·
     official preference honoured · no advice lexicon · no returns lexicon ·
     freshness line present and correct.
   - `ADVICE_LEXICON` and `RETURNS_LEXICON` regex lists live here.
4. `pipeline.py`
   - `query(text, cfg) -> Answer` — the runtime entry point:
     ```
     L1 PII  -> blocked? -> PII_WARNING Answer
     L2 intent -> advisory/portfolio/returns? -> refusal Answer
     F5 retrieve -> no hits above threshold? -> NOT_IN_SOURCES Answer
     F6 generate -> draft
     L3 validate -> pass -> Answer
                  -> 1 repair pass -> revalidate -> still bad -> safe template Answer
     ```
   - Always return a populated `Trace` (query hashed, never raw).
5. `eval/questions_guardrail.json` — ≥3 advisory, ≥2 portfolio, ≥2 returns,
   ≥2 PII, ≥2 out-of-corpus inputs with `expected_intent`.
6. `eval/run_eval.py --mode guardrails` — assert every input yields the expected
   intent, that advice/returns lexicons have **zero** hits in all outputs, and
   that no PII value ever reaches the LLM client (assert via a spy client).

**Verify**

```powershell
python -c "from mf_rag.pipeline import query; print(query('Should I buy HDFC Small Cap Fund?', None).text)"
python -c "from mf_rag.pipeline import query; print(query('My PAN is ABCDE1234F', None).text)"
python -c "from mf_rag.pipeline import query; print(query('What is the ELSS lock-in period?', None).text)"
python eval/run_eval.py --mode guardrails
```

**Exit gate**

- [ ] Every PII test input blocked **before** the LLM (proved with a spy client)
- [ ] ≥7 guardrail inputs classified correctly
- [ ] Refusals contain the correct template + one in-registry educational link
- [ ] All outputs pass L3 validation: ≤3 sentences, 1 valid URL, no advice, no returns
- [ ] Repair pass exercised at least once in a test
- [ ] `pytest` green

<details><summary><b>Cursor prompt — Phase 7</b></summary>

```
Implement Phase 7 (Guardrails) of the Mutual Fund FAQ RAG assistant.

Read first: ARCHITECTURE.md §6.4 (all three layers + the validator table) and
§10, PRD.md FR-6.1 to FR-6.4, FR-5.1 to FR-5.4, §7 (templates).

Phase 6 is complete: answer_raw() works, prompts.py has the templates.

Create:
1. src/mf_rag/guardrails.py
   L1 (input, runs BEFORE embedding and before any LLM call):
   - detect_pii(text) -> list[str] of matched PATTERN NAMES only
     (never return the matched value)
   - patterns: PAN [A-Z]{5}[0-9]{4}[A-Z], Aadhaar \b[2-9]\d{11}\b, 10-digit
     phone, email, account number, OTP (4-6 digits near "otp"/"code")
   - guard_input(text) -> GuardResult(allowed, reason, patterns)
   L2 (intent):
   - classify_intent(text) -> "factual" | "advisory" | "portfolio" |
     "returns" | "out_of_corpus"
   - deterministic regex rules first (PR-6.1). A fund name outside the
     registry -> "out_of_corpus".
   - refusal_for(intent, question) -> Answer using the prompts.py template
     plus ONE educational link pulled from the source registry
     (doc_type "educational" or an official factsheet). Never invent a URL.
2. src/mf_rag/answer.py — L3 output validators
   - count_sentences(text) — must NOT split on the period inside "1.5"
   - extract_urls(text)
   - ADVICE_LEXICON and RETURNS_LEXICON regexes (you should, recommend,
     best choice, ideal for you, must buy, suitable for / will give,
     expected return, CAGR of, projected, % return)
   - validate(answer_text, ctx) -> ValidationResult(ok, violations, repaired_text)
     Checks: <=3 sentences, exactly 1 URL, URL is in the ingested source set,
     official preference honoured, no advice, no returns, freshness line
     present and equal to the expected date.
   - repair(answer_text, ctx) -> str — one corrective call.
3. src/mf_rag/pipeline.py — query(text, cfg) -> Answer. Full path:
   L1 PII -> PII_WARNING | L2 intent -> refusal | retrieve -> no hits ->
   NOT_IN_SOURCES | generate -> L3 validate -> repair once -> revalidate ->
   safe template fallback. Always return a populated Trace with a hashed
   query (never the raw text), entity_detected, where_filter, hits, latency_ms.
4. eval/questions_guardrail.json — at least 3 advisory, 2 portfolio,
   2 returns, 2 PII, 2 out-of-corpus entries with expected_intent.
5. eval/run_eval.py --mode guardrails — verify expected intents, assert ZERO
   advice/returns lexicon hits across all outputs, and assert (with a spy
   LLMClient) that no PII value ever reaches the LLM.
6. tests/test_pii.py, test_intent.py, test_answer.py, test_pipeline.py.

Demonstrate: an advisory question, a returns question, a PII input, and a
normal factual question. Show all four outputs.
```
</details>

---

## Phase 8 — UI

**Goal:** a Streamlit app where a stranger can ask a question, see a compliant cited answer, and inspect the retrieval trace.

**Depends on:** Phase 7

**Files**

```
app/streamlit_app.py
```

**Build steps**

1. Welcome line — identifies the assistant, its 5-scheme scope, and the facts-only framing.
2. **Exactly 3 example question chips** (`FR-7.2`), one per intent type, and they
   must be `st.button` → `st.session_state`:
   - "What is the expense ratio of HDFC Large Cap Fund?"
   - "What is the lock-in period for HDFC ELSS Tax Saver Fund?"
   - "How do I download my capital gains statement?"
3. Disclaimer `st.caption("Facts-only. No investment advice.")` — always visible (`FR-7.3`).
4. `st.chat_input` (or a form) → `pipeline.query(text, cfg)`.
5. Answer card: text, clickable `source_url`, freshness line. Refusals get a visually distinct style; PII blocks get a warning style.
6. `st.expander("Retrieval trace")` — intent, PII status, entity + `where` filter, each hit with cosine score + heading + URL, which chunk was used, selected citation, validators run, repair used, latency (`FR-7.5`).
7. `st.spinner` while querying (`FR-7.6`); warn banner when `get_llm` reports `degraded`.
8. No persistence beyond `session_state` (`NFR-4`).

**Verify**

```powershell
streamlit run app/streamlit_app.py
# manual: 3 chips work, answer shows 1 link + freshness, trace expands,
#         advisory question is refused, PII input is blocked, disclaimer visible
```

**Exit gate**

- [ ] Welcome line, 3 chips, disclaimer all present
- [ ] Answers show a clickable link + `Last updated from sources:`
- [ ] Trace expander shows chunks + scores
- [ ] Refusal and PII paths render distinctly
- [ ] Works with no API key (EchoClient + banner)
- [ ] No chat persistence

<details><summary><b>Cursor prompt — Phase 8</b></summary>

```
Implement Phase 8 (UI) of the Mutual Fund FAQ RAG assistant.

Read first: ARCHITECTURE.md §6.5, PRD.md FR-7.1 to FR-7.6, NFR-4.

Phases 1-7 are complete: mf_rag.pipeline.query(text, cfg) -> Answer is the
runtime entry point.

Create app/streamlit_app.py:
1. st.set_page_config; welcome line naming the assistant, the HDFC AMC scope,
   the 5 schemes, and the facts-only framing.
2. EXACTLY 3 example question chips, each st.button that writes its question
   into st.session_state and reruns:
   - "What is the expense ratio of HDFC Large Cap Fund?"
   - "What is the lock-in period for HDFC ELSS Tax Saver Fund?"
   - "How do I download my capital gains statement?"
3. st.caption("Facts-only. No investment advice.") — always visible.
4. Input via st.chat_input. Call pipeline.query() inside st.spinner.
5. Render the Answer: text, the source_url as a clickable link, and the
   "Last updated from sources:" line. Style refusals and PII blocks
   differently from normal answers.
6. st.expander("Retrieval trace"): intent, pii_blocked, entity_detected,
   where_filter, every hit with cosine score + section_heading + source_url,
   which chunk was used, selected citation, validators_run, repair_used,
   latency_ms.
7. If get_llm(cfg) reports degraded=True, show a st.warning banner that the
   assistant is running without an LLM and answers are template-only.
8. Cache get_config() and the embedder; do NOT persist chat state beyond
   st.session_state.

Run it and show the output for all three chips plus one advisory question
and one PII input.
```
</details>

---

## Phase 9 — Packaging & Deliverables

**Goal:** every artefact in PRD FR-8.1–8.5 exists, generated by a script, not by hand.

**Depends on:** Phases 1–8

**Files**

```
scripts/export_outputs.py
outputs/source_list.csv
outputs/source_list.md
outputs/sample_qa.md
outputs/disclaimer.md
README.md
```

**Build steps**

1. `export_outputs.py`
   - `source_list.csv/.md` from `data/sources.csv` — `source_id, scheme,
     category, doc_type, is_official, source_url, fetched_at, status` (FR-8.2)
   - `sample_qa.md` — run **8** questions spanning: 3 numeric facts across
     different schemes, 1 ELSS-specific, 1 riskometer, 1 benchmark, 1
     statement-download, 1 advisory (refusal), 1 returns (refusal). Each entry:
     **Question / Answer / Source link / Intent / Validators passed**
     (FR-8.4). Run them live — do not hand-write answers.
   - `disclaimer.md` — the exact UI disclaimer string (FR-8.5), imported from
     `prompts.DISCLAIMER` so the two cannot drift.
2. `README.md` (FR-8.3) must contain:
   - What it is / non-goals (facts-only, no advice)
   - Scope: AMC = HDFC Asset Management, the 5 schemes
   - **Setup:** venv, `pip install -r requirements.txt`, `.env`, then the
     exact run sequence
   - **The RAG pipeline**, one line per stage, with the real command
   - **Chunking strategy + the measured numbers** from `outputs/chunk_eval.json`
   - `min_score` value and its justification
   - **Sources used** (table)
   - **Sample Q&A** (the generated file)
   - **Known limits** — copy PRD §14
   - Deliverables index
3. Regenerate everything with one command and diff-check reproducibility.

**Verify**

```powershell
python scripts/export_outputs.py
# expect: 4 files written; README has no TODO placeholders
python -m pytest tests/ -q
```

**Exit gate**

- [ ] `source_list.csv` **and** `.md` generated from real data
- [ ] `sample_qa.md` has 8 entries, every answer ≤3 sentences with 1 link
- [ ] `disclaimer.md` matches the UI string exactly
- [ ] README has setup, scope, chunking rationale, sources, known limits
- [ ] Full clean-clone run succeeds following only the README

<details><summary><b>Cursor prompt — Phase 9</b></summary>

```
Implement Phase 9 (Packaging & Deliverables).

Read first: PRD.md FR-8.1 to FR-8.5 and §14, ARCHITECTURE.md §13.

Phases 1-8 are complete.

Create:
1. scripts/export_outputs.py with subcommands (or flags) that generate:
   - outputs/source_list.csv and outputs/source_list.md from data/sources.csv
     (source_id, scheme, category, doc_type, is_official, source_url,
     fetched_at, status)
   - outputs/sample_qa.md by ACTUALLY RUNNING 8 questions through
     mf_rag.pipeline.query: three numeric facts on different schemes, one
     ELSS lock-in, one riskometer, one benchmark, one capital-gains
     statement download, one advisory (refusal), one returns (refusal).
     Each entry: Question / Answer / Source link / Intent / Validators passed.
     Do NOT hand-write the answers.
   - outputs/disclaimer.md containing the exact prompts.DISCLAIMER string
     (import it, do not retype it).
2. README.md containing: overview and non-goals; scope (HDFC AMC + the 5
   schemes); setup (venv, pip install, .env, exact run sequence); the RAG
   pipeline with one command per stage; the CHUNKING STRATEGY DECISION with the
   real numbers from outputs/chunk_eval.json; the min_score value and why;
   the sources table; the sample Q&A; known limits copied from PRD §14; and a
   deliverables index.

Run: python scripts/export_outputs.py, then verify README has no TODO or
placeholder text left.
```
</details>

---

## Phase 10 — Demo

**Goal:** a ≤3-minute recording that walks the RAG pipeline, not just the chat.

**Depends on:** everything

**Build steps**

1. Re-run `scripts/run_ingest.py --reset` and `scripts/build_index.py --stage all` so freshness dates are current.
2. Record in this order (≤3 min total):
   - **0:00–0:20** — Problem, scope, non-goals, the disclaimer
   - **0:20–0:45** — The pipeline diagram; show `run_ingest.py` output (Stage 1), `chunk_eval.py` (Stage 2 + the strategy decision), `build_index.py` (Stages 3–4, with the persistence re-check in a fresh process)
   - **0:45–1:30** — Three good questions: expense ratio, ELSS lock-in, statement download. Expand the **trace** panel for one so the grader sees the chunks and scores
   - **1:30–2:15** — Two refusals: "Should I buy/sell?" and a returns question; show the PII block
   - **2:15–2:45** — `sample_qa.md` and `source_list.csv` on screen
   - **2:45–3:00** — Known limits
3. Save to `outputs/demo.mp4`; also push the app and note the link.
4. Rehearse against the PRD §15 acceptance checklist line by line.

**Exit gate**

- [ ] Video ≤3 minutes
- [ ] Every PRD §15 acceptance box demonstrably satisfied
- [ ] All 4 RAG stage groups shown running
- [ ] Both refusal types and the PII block shown
- [ ] `outputs/demo.mp4` committed; link recorded in README

<details><summary><b>Cursor prompt — Phase 10</b></summary>

```
Support Phase 10 (Demo).

1. Re-run the pipeline end to end so all fetched_at dates are current:
   python scripts/run_ingest.py --reset
   python scripts/build_index.py --stage all
   python scripts/export_outputs.py
2. Print a fresh ingest_report.json and chunk_eval.json summary.
3. Walk PRD.md section 15 (Acceptance Criteria) item by item and report for
   each: SATISFIED / NOT SATISFIED with the evidence (command output, file
   path, or line). Do not claim satisfaction without showing the evidence.
4. Draft outputs/demo_script.md — a shot-by-shot <=3-minute script following
   the timing in the implementation guide Phase 10, listing exactly which
   screen/terminal output appears in each shot.
5. List any NOT SATISFIED acceptance criteria with the fix needed and which
   phase owns it.
```
</details>

---

## Cross-Phase Reference

### FR → Phase map

| FR | Requirement | Phase |
|---|---|---|
| FR-1 | Corpus ingestion | 1 |
| FR-2 | Chunking | 2 |
| FR-3 | Embedding + store | 3, 4 |
| FR-4 | Retrieval | 5 |
| FR-5 | Answer generation | 6, 7 |
| FR-6 | Refusal & safety | 7 |
| FR-7 | UI | 8 |
| FR-8 | Deliverables | 9, 10 |

### NFR → Phase map

| NFR | Phase |
|---|---|
| NFR-1 latency ≤8s | 6, 8 (verify in eval) |
| NFR-2 reproducibility | 2, 3, 4, 9 |
| NFR-3 transparency | 5, 8, 11 trace |
| NFR-4 privacy/secrets | 0, 7, 8 |
| NFR-5 offline-first | 3, 4 |
| NFR-6 portability | 0, 9 |
| NFR-7 maintainability | all |
| NFR-8 cost | 6 (EchoClient) |
| NFR-9 graceful failure | 1, 9 |

### Known traps to avoid

| Trap | Where | Prevention |
|---|---|---|
| Embedding model drift between index and query | 3, 4, 5 | All access via `mf_rag.embedder`; model name stored in collection metadata; test asserts identity |
| Silent partial index write | 4 | `verify_collection()` hard assertions |
| Model-invented citation URL | 6, 7 | System picks the URL; validator checks it against the source set |
| Filtering away the right chunk | 5 | Unfiltered fallback before concluding "not in sources" |
| Sentence counter splitting "1.5%" | 7 | Digit-aware `count_sentences` |
| Advice language leaking into answers | 7 | L3 lexicon + replacement template |
| Over-filtering kills recall | 5, 9 | Measure hit@5 in `eval/run_eval.py`; only tune with evidence |
| PII persisted in logs | 7, 8 | Never log raw query; `log_pii: false` |
| Third-party blog sneaking into the corpus | 0, 1 | Host allowlist enforced in `sources.py` |

### Suggested git history

```
chore(phase-0): scaffold, config, dataclasses, source registry
feat(phase-1): stage 1 loading — fetch, clean, snapshot, report
feat(phase-2): stage 2 chunking — 3 strategies + measured evaluation
feat(phase-3): stage 3 embedding — MiniLM with disk cache
feat(phase-4): stage 4 store vector data — persistent Chroma + verification
feat(phase-5): retrieval — entity filters, threshold, trace
feat(phase-6): generation — strict prompt, system-selected citation, EchoClient
feat(phase-7): guardrails — PII, intent refusals, output validators
feat(phase-8): ui — streamlit with trace panel
docs(phase-9): readme, source list, sample qa, disclaimer
chore(phase-10): demo script and recording
```
