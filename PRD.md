# PRD — Mutual Fund FAQ Assistant (RAG Chatbot)

| Field | Value |
|---|---|
| Document type | Product Requirements Document |
| Version | 1.0 |
| Status | Draft for class demo |
| Owner | Buildhours team |
| Source | `Docs/requirements.txt.txt` |
| Target | Working prototype (app or notebook) + ≤3-min demo video |

---

## 1. Overview

We are building a **Retrieval-Augmented Generation (RAG) chatbot** that answers factual questions about a small, well-defined corpus of mutual fund (MF) schemes. The bot answers **only** from ingested official public pages, always cites one source link, never gives investment advice, and refuses opinionated or portfolio questions.

The product is a **facts-only FAQ assistant**, not a financial advisor.

**Corpus scope:** one AMC (HDFC Asset Management Company) and its 5 schemes, as listed in the requirements.

| # | Category | Scheme | URL |
|---|---|---|---|
| 1 | Large Cap | HDFC Large Cap Fund – Direct Growth | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| 2 | Flexi Cap | HDFC Equity (Flexi Cap) Fund – Direct Growth | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth |
| 3 | ELSS | HDFC ELSS Tax Saver Fund – Direct Plan Growth | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-growth |
| 4 | Small Cap | HDFC Small Cap Fund – Direct Growth | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth |
| 5 | Balanced Advantage (Hybrid) | HDFC Balanced Advantage Fund – Direct Growth | https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth |

> Note: the URLs above are the seed pages mandated by the requirements. Where a seed page does not contain a fact (e.g. official factsheet PDF, statement-download guide), the assistant must link to the corresponding official HDFC AMC / AMFI / SEBI page — **no third-party blogs**.

---

## 2. Problem Statement

Retail users comparing mutual fund schemes repeatedly ask the same factual questions: expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer level, benchmark, and how to download statements. These answers already exist in official documents (factsheets, KIM/SID, scheme FAQ pages, fee/charge pages, AMFI/SEBI pages) but are scattered, hard to compare across schemes, and often buried in PDFs.

Support and content teams answer these questions manually, repeatedly, with the risk of stale or incorrect figures.

We need a small, auditable assistant that surfaces the fact, in ≤3 sentences, with a link the user can verify.

---

## 3. Goals & Non-Goals

### 3.1 Goals

| ID | Goal |
|---|---|
| G1 | Answer factual MF questions for the 5 scoped HDFC schemes using only the ingested official corpus |
| G2 | Cite exactly one verifiable source link in every substantive answer |
| G3 | Refuse opinionated / portfolio questions politely, with a facts-only message + one educational link |
| G4 | Enforce hard safety constraints: no PII intake/storage, no performance or return calculations, ≤3 sentences per answer, visible freshness stamp |
| G5 | Demonstrate a full, explainable RAG pipeline: Loading → Chunking → Embedding → Vector store → Retrieval → Generation |
| G6 | Ship a tiny UI: welcome line, 3 example questions, disclaimer note |

### 3.2 Non-Goals

- Not a financial advisor, not a recommendation engine (explicitly out of scope).
- No return/performance computation, comparison, ranking, or projection of any kind.
- No live NAV, portfolio tracking, transaction execution, or account management.
- No multi-AMC coverage in v1 (one AMC only).
- No user accounts, login, persistence of chat history, or analytics dashboards.
- No scraping of third-party blogs or review sites.
- No mobile app — web demo only.

---

## 4. Users & Use Cases

### 4.1 Personas

| Persona | Need | Typical friction |
|---|---|---|
| **P1 — Retail investor (primary)** | Compare schemes on fees, load, SIP, lock-in, riskometer | Facts spread across factsheets; doesn't know which doc to trust |
| **P2 — Support / content team (secondary)** | Answer repetitive MF questions quickly and consistently | Manual search; inconsistent answers; no citations |
| **P3 — Demo evaluator / instructor** | See the RAG working end-to-end with traceable sources | Needs visible citations and a short video |

### 4.2 In-Scope Questions (functional intents)

1. Expense ratio / TER of a scheme
2. Exit load
3. Minimum SIP amount
4. ELSS lock-in period
5. Riskometer level / risk category
6. Benchmark
7. How to download statements (capital-gains statement, account statement, tax documents)
8. Fund objective / category basics
9. Direct vs Regular plan distinction (factual only)

### 4.3 Out-of-Scope Questions (must be refused)

- "Should I buy/sell HDFC Small Cap?"
- "Is now a good time to enter?"
- "Which of these 5 is best for me?"
- "How much will I get after 10 years?"
- "Compare returns of A vs B."
- Any request for PAN, Aadhaar, account number, OTP, email, phone number.

---

## 5. Functional Requirements

### FR-1 — Corpus Ingestion (Offline)

| ID | Requirement |
|---|---|
| FR-1.1 | Fetch and store the 5 seed scheme pages locally as raw HTML/text snapshots. |
| FR-1.2 | Also ingest the supporting official pages: HDFC AMC factsheet pages, KIM/SID documents, scheme FAQ pages, fee & charge pages, riskometer/benchmark notes, and statement/tax-doc download guides. |
| FR-1.3 | Every ingested document must be stamped with `source_url`, `scheme`, `category`, `doc_type` (e.g. `factsheet`, `faq`, `fees`, `riskometer`, `guide`), and `fetched_at`. |
| FR-1.4 | The full source list must be exportable as CSV and MD (deliverable). |
| FR-1.5 | Ingestion must be re-runnable (idempotent) so the corpus can be refreshed before a demo. |

### FR-2 — Chunking (Offline)

| ID | Requirement |
|---|---|
| FR-2.1 | Apply an explicit, documented chunking strategy decided after inspecting the real data (see §6.3). |
| FR-2.2 | Each chunk must retain metadata: `source_url`, `scheme`, `category`, `doc_type`, `section_heading`, `chunk_index`, `fetched_at`. |
| FR-2.3 | Numeric facts (expense ratio %, exit load %, SIP ₹) must not be split away from the sentence/table row that gives them meaning. |
| FR-2.4 | Chunk count and average size must be reported after ingestion and printed for the demo. |

### FR-3 — Embedding + Vector Store (Offline)

| ID | Requirement |
|---|---|
| FR-3.1 | Embedding model: `sentence-transformers/all-MiniLM-L6-v2`. |
| FR-3.2 | Vector store: **ChromaDB**, persistent, local. |
| FR-3.3 | Metadata filters must be supported at query time (e.g. restrict to `category = "ELSS"` or `scheme = "..."`). |
| FR-3.4 | Collection must be rebuildable from the chunked corpus with one command. |

### FR-4 — Retrieval (Online)

| ID | Requirement |
|---|---|
| FR-4.1 | Embed the user query with the same model used at index time. |
| FR-4.2 | Retrieve top-k candidates (default k = 5). |
| FR-4.3 | If `scheme` / `category` is detected in the query, apply the metadata filter before ranking. |
| FR-4.4 | If top score is below a similarity threshold, the bot must say it does not have the information rather than guess. |
| FR-4.5 | Retrieval metadata (chunks, scores, sources) must be inspectable for the demo/debug view. |

### FR-5 — Answer Generation (Online)

| ID | Requirement |
|---|---|
| FR-5.1 | Answers **factual queries only**, in **≤ 3 sentences**. |
| FR-5.2 | Every substantive answer must include **exactly one clear citation link** to an official source. |
| FR-5.3 | Answers must **not** contain performance claims, return calculations, comparisons, or projections. If asked about returns, the bot must decline to compute and link to the official factsheet. |
| FR-5.4 | Every answer must include a freshness line: `Last updated from sources: <YYYY-MM-DD>`. |
| FR-5.5 | Answers must be grounded in the retrieved context only; no external knowledge. |
| FR-5.6 | If the context is insufficient, the bot must say so and point to the scheme's official page. |

### FR-6 — Refusal & Safety

| ID | Requirement |
|---|---|
| FR-6.1 | Opinionated / advisory / portfolio questions must be refused with a **polite, facts-only message** plus one **relevant educational link**. |
| FR-6.2 | The system must not accept or store PAN, Aadhaar, account numbers, OTPs, emails, or phone numbers. Input matching these patterns must trigger a PII warning and must never be logged. |
| FR-6.3 | No investment advice language may appear in any output; a fixed disclaimer must be present in the UI. |
| FR-6.4 | Refusal behaviour must be demonstrated in the sample Q&A file. |

### FR-7 — UI (Web, tiny)

| ID | Requirement |
|---|---|
| FR-7.1 | Welcome line identifying the assistant, its scope, and the facts-only nature. |
| FR-7.2 | Exactly **3 example questions** clickable to populate the input. |
| FR-7.3 | Visible note: **"Facts-only. No investment advice."** |
| FR-7.4 | Each answer renders the citation as a clickable link. |
| FR-7.5 | A debug/trace toggle showing retrieved chunks + scores (for demo and evaluation). |
| FR-7.6 | Loading indicator while retrieval + generation runs. |

### FR-8 — Deliverables / Outputs

| ID | Requirement |
|---|---|
| FR-8.1 | Working prototype (app or notebook) or a ≤3-minute demo video. |
| FR-8.2 | Source list of the URLs used, as **CSV and MD**. |
| FR-8.3 | README: setup steps, scope (AMC + schemes), known limits. |
| FR-8.4 | Sample Q&A file: 5–10 queries with the assistant's answers and links. |
| FR-8.5 | Disclaimer snippet used in the UI, exported as a reusable text/markdown file. |

---

## 6. System Design — The RAG Architecture

The pipeline must follow **both** RAG stage groups explicitly, in this order, and each stage must be visible/explainable in the demo.

```
STAGE GROUP A — DATA INGESTION (offline)
  Loading → Chunking → Embedding → Store Vector Data

STAGE GROUP B — RETRIEVAL & GENERATION (online)
  Query → Embed → Search (+filter) → Select context → Prompt → LLM → Cited answer
```

### 6.1 Stage Group A

| Stage | Decision |
|---|---|
| **Loading** | Fetch the 5 seed pages + supporting official pages; strip nav/boilerplate; keep main content; normalise whitespace; store raw snapshots. |
| **Chunking** | Strategy decided after inspecting real data — see §6.3. Must be structure-aware and metadata-enriched. |
| **Embedding** | `sentence-transformers/all-MiniLM-L6-v2` (384-dim), run locally via `sentence-transformers`. |
| **Store Vector Data** | ChromaDB persistent collection `mf_faq`, cosine space, metadata per FR-2.2. |

### 6.2 Stage Group B

| Step | Behaviour |
|---|---|
| Guard | PII scan + intent classification (factual / advisory / out-of-scope) |
| Embed query | Same MiniLM model |
| Search | Chroma similarity top-k = 5, optional metadata filter |
| Score check | Below threshold → "I don't have that in my sources" |
| Select context | Top 3–5 chunks, de-duplicated, ordered |
| Prompt | Strict system prompt: ≤3 sentences, exactly one link, no advice, no returns, no PII, use only context, append freshness line |
| Generate | LLM call (provider TBD, swappable behind one interface) |
| Post-process | Link check (must be one of the ingested URLs), sentence-count check, advice-language check |
| Return | Answer + citation + freshness + disclaimer |

### 6.3 Chunking Strategy (to be finalised after data inspection)

Requirements ask that the chunking strategy be decided from the actual data. Working hypothesis to validate in Stage 2:

- **Candidate A — Fixed-size overlap:** ~500 chars, ~80 overlap. Simple, but risks cutting numeric fact rows.
- **Candidate B — Semantic/Markdown section split:** split on `##` headings, tables, and FAQ question boundaries, then merge/split to a target size. Preserves meaning for FAQs and fee tables.
- **Candidate C — Sentence/paragraph with parent-child index:** embed small chunks, return the parent section for more context.

**Selection criteria:** (1) no numeric fact orphaned from its label, (2) ≥90% of sample questions retrieve a chunk containing the answer, (3) average chunk size 300–800 chars, (4) each chunk traceable to a single source URL.

**Decision to be recorded** in `README.md` and the demo narration, with the measured numbers.

### 6.4 Architecture Diagram (logical)

```
[Official HDFC/AMFI/SEBI pages]
            │  (1) Loading / fetch + clean
            ▼
      Raw documents (per-source snapshots)
            │  (2) Chunking (structure-aware + metadata)
            ▼
      Chunk store (JSON)  ──► source_list.csv / source_list.md
            │  (3) Embedding: all-MiniLM-L6-v2
            ▼
      ChromaDB (mf_faq)  ◄───────── [User Query]
                                   (4) guard → embed → search → filter → select
                                            │
                                            ▼
                                   (5) Prompt + LLM → answer
                                            │
                                            ▼
                                   (6) post-process → UI (answer + 1 link + freshness)
```

---

## 7. Guardrails & Response Contract

Every assistant response must follow this contract:

| Field | Rule |
|---|---|
| Length | ≤ 3 sentences |
| Citations | Exactly 1 link, must exist in the ingested source list |
| Freshness | Must end with `Last updated from sources: <date>` |
| Advice | Prohibited — any "should", "recommend", "best for you" style output is a bug |
| Returns | Prohibited — no computing, comparing, or ranking returns; link to official factsheet instead |
| PII | Never requested, accepted, echoed, or stored |
| Unknown | Must state it does not have the information, not guess |

**Refusal template (advisory/opinion question):**
> I only share factual information from official scheme documents — I can't give investment advice or recommend a scheme. For general investor-education guidance, see: `<one educational link>`.

**Return-related template:**
> I don't compute or compare returns. The official factsheet has the fund's reported performance: `<factsheet link>`.

**Disclaimer snippet (shown in UI):**
> Facts-only. No investment advice. Information is taken from official public scheme pages and may change. Verify details on the AMC/AMFI site before acting.

---

## 8. Non-Functional Requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-1 | Performance | End-to-end answer ≤ 8 seconds for a typical question (local embedding, LLM latency dominant). |
| NFR-2 | Reproducibility | One command builds the corpus; one command runs the app. Seeded/deterministic where possible. |
| NFR-3 | Transparency | Every answer traceable to a chunk and a URL; debug view available. |
| NFR-4 | Privacy | No PII stored; no chat logging by default; API keys via env vars only, never committed. |
| NFR-5 | Offline-first | Ingestion, chunking, embedding, and vector store run locally. |
| NFR-6 | Portability | Runs on Windows/macOS/Linux; dependencies pinned in `requirements.txt`. |
| NFR-7 | Maintainability | Stage modules separated: `ingest`, `chunk`, `embed_store`, `retrieve`, `generate`, `guardrails`, `ui`. |
| NFR-8 | Cost | Free-tier friendly; local sentence-transformers preferred over paid embedding APIs. |
| NFR-9 | Robustness | Graceful failure if a source URL is unreachable — skip with a warning, do not crash the build. |

---

## 9. Tech Stack (Proposed)

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.10+ | Ecosystem fit for RAG; required by `sentence-transformers` |
| Ingestion | `requests` / `httpx` + `beautifulsoup4` (+ `trafilatura` if needed) | Clean main-content extraction |
| Chunking | Custom structure-aware splitter (+ `tiktoken` for counting) | Requirements ask for a data-driven decision |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Mandated; local, free, fast |
| Vector DB | `chromadb` | Mandated; local, persistent, metadata filters |
| LLM | Pluggable provider (OpenAI-compatible or a free/local model) | Keep swappable for demo cost |
| Orchestration | LangChain or plain Python | Plain Python preferred for transparency in a class demo |
| UI | Streamlit (or Gradio) | Fastest path to a tiny web UI with a debug trace panel |
| Config | `.env` + `config.yaml` | API keys out of git; chunk params tunable |
| Outputs | CSV + MD for source list and sample Q&A | Deliverable format |

---

## 10. Data & Artifacts

| Artifact | Path (proposed) | Purpose |
|---|---|---|
| Raw snapshots | `data/raw/<source_id>.txt` | Auditable ingestion output |
| Chunk store | `data/chunks.jsonl` | Chunked corpus with metadata |
| Chroma DB | `data/chroma/` | Persistent vector store |
| Source list | `outputs/source_list.csv`, `outputs/source_list.md` | Deliverable FR-8.2 |
| Sample Q&A | `outputs/sample_qa.md` | Deliverable FR-8.4 |
| Disclaimer | `outputs/disclaimer.md` | Deliverable FR-8.5 |
| Demo video | `outputs/demo.mp4` (≤3 min) | Deliverable FR-8.1 |
| Logs | `outputs/ingest_report.json` | Chunk counts, per-source stats |

---

## 11. Success Metrics

| Metric | Target |
|---|---|
| Grounded answer rate on the 5–10 sample questions | 100% with correct citation |
| Answers containing exactly one valid in-corpus link | 100% |
| Answers exceeding 3 sentences | 0 |
| Opinionated questions correctly refused | 100% |
| Advisory/return language violations | 0 |
| PII accepted or stored | 0 |
| Questions answered from outside the corpus (hallucination) | 0 |
| Ingestion reproducibility | 100% (same sources → same chunk count) |

---

## 12. Milestones & Build Stages

| Stage | Name | Deliverable | Exit criteria |
|---|---|---|---|
| 0 | Scope & PRD | This PRD + confirmed scheme list | PRD agreed |
| 1 | **Loading** | `ingest.py`, raw snapshots, ingest report | 5 schemes + supporting official pages saved with metadata |
| 2 | **Chunking** | `chunk.py`, `chunks.jsonl`, written strategy rationale | Strategy chosen with measured justification |
| 3 | **Embedding** | `embed_store.py` | MiniLM vectors built, all chunks embedded |
| 4 | **Store Vector Data** | ChromaDB collection `mf_faq` | Persistence verified across restarts; filters work |
| 5 | Retrieval | `retrieve.py` + relevance test set | k=5 returns answer-bearing chunks for sample questions |
| 6 | Generation | `generate.py` + strict prompt | ≤3 sentences, one link, freshness stamp |
| 7 | Guardrails | `guardrails.py` | Refusals + PII block + no-returns behaviour verified |
| 8 | UI | Streamlit app | Welcome + 3 examples + disclaimer + trace panel |
| 9 | Packaging | README, source list, sample Q&A, disclaimer | All deliverables present |
| 10 | Demo | ≤3-min video or hosted link | Recorded end-to-end |

Each RAG stage is a separate, demonstrable step — the demo will walk the pipeline rather than jump from question to answer.

---

## 13. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Groww/AMC pages change layout or block scraping | Ingestion breaks | Cache raw snapshots; prefer official PDF/HTML fallbacks; warn-and-skip on failure |
| Facts go stale (expense ratio changes) | Wrong answers | Show `Last updated from sources`; re-run ingestion before each demo |
| MiniLM is weak on long/numeric financial tables | Missed or garbled facts | Structure-aware chunking; metadata filters; optional bigger embedding model as fallback |
| Retrieval pulls chunks from the wrong scheme | Wrong-scheme answers | Entity detection → hard metadata filter before ranking |
| LLM drifts into advice or computes returns | Requirement violation | Strict system prompt + post-generation validators (advice regex, sentence count, link check) |
| Rate limits on source fetching | Partial corpus | Sequential fetch with delay, retries, caching |
| LLM API key missing / cost | Demo failure | Pluggable provider + local model fallback; key from env only |
| Scope creep into "financial advisor" territory | Fails requirements | Non-goals section; refusal path is a first-class feature |

---

## 14. Known Limits (to be disclosed in README)

1. **Tiny corpus** — 5 schemes of one AMC only; not representative of the MF industry.
2. **Snapshot freshness** — answers reflect pages as of `fetched_at`; values change (TER, exit load).
3. **No numeric computation** — deliberately cannot calculate returns, corpus plans, or SIP projections.
4. **No personalisation** — no user profile, no goal-based guidance, by design.
5. **Chunking trade-offs** — fixed-size chunking can cut tables; a table/FAQ-aware splitter mitigates but does not eliminate this.
6. **MiniLM retrieval ceiling** — short embeddings may miss deeply worded official language; mitigated by filters and larger k, not solved.
7. **English only, no OCR** — scanned PDF factsheets are out of scope unless text-extractable.
8. **Not a financial product** — informational demo; users must verify on official AMC/AMFI pages.

---

## 15. Acceptance Criteria

The submission is accepted when all of the following are true:

- [ ] All Stage Group A stages (Loading → Chunking → Embedding → Store Vector Data) are implemented and demonstrable.
- [ ] Retrieval + generation (Stage Group B) run end-to-end in the prototype.
- [ ] `all-MiniLM-L6-v2` and `ChromaDB` are used as specified.
- [ ] A written, measured chunking-strategy rationale exists.
- [ ] Every answer in the sample Q&A is ≤3 sentences and has exactly one official link.
- [ ] `Last updated from sources:` appears in every answer.
- [ ] An opinionated question is refused with a facts-only message + educational link.
- [ ] A returns/performance question is declined with a factsheet link.
- [ ] PII (PAN/Aadhaar/account/OTP/email/phone) is neither requested nor stored.
- [ ] UI shows a welcome line, 3 example questions, and "Facts-only. No investment advice."
- [ ] Source list delivered as CSV **and** MD.
- [ ] README contains setup steps, scope (AMC + schemes), and known limits.
- [ ] Disclaimer snippet file delivered.
- [ ] Working prototype link or ≤3-minute demo video delivered.

---

## 16. Open Questions

| # | Question | Owner | Needed by |
|---|---|---|---|
| OQ-1 | Which LLM provider/model for generation (hosted vs local)? | Team | Stage 6 |
| OQ-2 | Streamlit or Gradio for the UI? | Team | Stage 8 |
| OQ-3 | Final chunking strategy after data inspection — Section or Parent-Child? | Team | Stage 2 |
| OQ-4 | Include official HDFC AMC factsheet PDFs in addition to the 5 seed URLs? | Team | Stage 1 |
| OQ-5 | Similarity threshold value for the "not in my sources" path? | Team | Stage 5 |
| OQ-6 | Demo video or hosted app link as primary submission? | Team | Stage 10 |
