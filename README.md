# HDFC Mutual Fund RAG Assistant

Facts-only FAQ assistant over a public-source corpus of HDFC AMC scheme pages and
AMC/regulator education material. Class demo: every answer carries a citation to the
page it came from, and the bot refuses anything it cannot source.

## What this is, and what it is not

**It is** a retrieval-augmented chatbot that answers factual questions about five HDFC
Mutual Fund schemes using only pages it fetched itself, cites the exact page for every
sentence, and declines — with a link to investor-education material — anything that would
be advice or a performance claim.

**It is not** a financial product, an adviser, or a returns calculator. It cannot compute
returns, rank funds, or tell you what to buy; those paths are refused on purpose, and the
refusals are constants in one file so they read identically on every run. See
[Known limits](#known-limits) for the honest caveats.

| | |
|---|---|
| Corpus | 17 registered official/aggregator pages, 14 indexed, **fetched once at build time** |
| Models | `sentence-transformers/all-MiniLM-L6-v2` (local) + any OpenAI-compatible chat model |
| Store | ChromaDB, persistent, cosine space |
| Prompts & refusals | `src/mf_rag/prompts.py` — every user-visible string |
| Guards | PII → intent → score floor → L3 output validation, in that order |

## Scope

**AMC: HDFC Asset Management Company Limited** (HDFC Mutual Fund), five schemes, direct
growth plans:

| # | Scheme | Category |
|---|---|---|
| 1 | HDFC Large Cap Fund - Direct Growth | Large Cap |
| 2 | HDFC Equity (Flexi Cap) Fund - Direct Growth | Flexi Cap |
| 3 | HDFC ELSS Tax Saver Fund - Direct Plan Growth | ELSS |
| 4 | HDFC Small Cap Fund - Direct Growth | Small Cap |
| 5 | HDFC Balanced Advantage Fund - Direct Growth | Balanced Advantage |

Plus non-scheme official pages needed for procedural facts: investor FAQs, the consolidated
account-statement download guide, the TER disclosure, an expense-ratio explainer, and the
AMFI/SEBI registry entries. Only `hdfcfund.com`, `amfiindia.com`, `sebi.gov.in` and the
mandated `groww.in` scheme pages are allowed — enforced in `src/mf_rag/sources.py`, not by
convention. The full list is generated: [`outputs/source_list.csv`](outputs/source_list.csv)
/ [`outputs/source_list.md`](outputs/source_list.md).

Out of scope by design: any other AMC, any other scheme, portfolio advice, return
calculations, Hindi or regional-language sources, and scanned PDFs that are not
text-extractable.

## Setup

Windows, PowerShell, Python 3.11 (any OS works — the commands below are PowerShell):

```powershell
git clone <repo-url> && cd Buildhours
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env      # then paste LLM_API_KEY (optional - see below)
```

**`.env` is optional.** With no key the app runs on `EchoClient`, which extracts a factual
sentence from the retrieved chunk instead of generating prose, behind a visible "degraded
mode" banner. Everything still works: retrieval, guardrails, citations, freshness stamp.
Add a key only for demo-grade prose:

```powershell
notepad .env
.\.venv\Scripts\python.exe scripts\check_llm.py --live
```

### Exact run sequence

One command per RAG stage. From a clean clone, in order:

```powershell
# Stage 1 - Loading          (fetch + clean 17 pages -> data/raw, data/sources.csv)
.\.venv\Scripts\python.exe scripts\run_ingest.py --reset

# Stage 2 - Chunking         (strategy from config.yaml; writes data/chunks.jsonl)
.\.venv\Scripts\python.exe scripts\run_ingest.py --chunk-only
.\.venv\Scripts\python.exe scripts\run_chunk_eval.py --rank embed   # the decision run

# Stages 3 & 4 - Embedding + Store Vector Data  (embeddings.jsonl + data/chroma)
.\.venv\Scripts\python.exe scripts\build_index.py --reset

# Evaluation (retrieval + guardrails) and the Phase 5/6/7/9 exit gates
.\.venv\Scripts\python.exe eval\run_eval.py
.\.venv\Scripts\python.exe eval\check_exit_gates.py

# Stage 8 - UI
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py

# Stage 9 - Deliverables (source list, sample Q&A, disclaimer)
.\.venv\Scripts\python.exe scripts\export_outputs.py
```

Then, at any time:

```powershell
.\.venv\Scripts\python.exe scripts\ask.py "What is the ELSS lock-in period?"
.\.venv\Scripts\python.exe scripts\search.py "What is the expense ratio of HDFC Large Cap?"
.\.venv\Scripts\python.exe -m pytest
```

Note that Stage 1 needs the network and takes ~1.5 s per source by design
(`ingest.request_delay_sec`). Everything after it is offline except the LLM call.

## The RAG pipeline

Stage groups A and B exactly as PRD §6 requires, one command each:

| Stage group | Stage | What it does | Command |
|---|---|---|---|
| A | 1 Loading | Fetch 17 pages, strip nav, extract main content + structured fields | `python scripts/run_ingest.py --reset` |
| A | 2 Chunking | Section-aware split, tables and Q/A kept atomic, metadata per chunk | `python -m mf_rag.chunk` |
| A | 3 Embedding | MiniLM-L6-v2, 384-dim, cached by `sha1(model + text)` | `python scripts/build_index.py --stage embed` |
| A | 4 Store Vector Data | Chroma `mf_faq`, cosine, metadata round-trip verified | `python scripts/build_index.py --stage all --reset` |
| B | 5 Retrieval | PII guard → intent → entity `where` filter → search → score floor → dedupe | `python scripts/search.py "<question>"` |
| B | 6 Generation | Context assembly, one LLM call, system-stamped freshness line | `python scripts/build_index.py --stage all` (index) then `python scripts/ask.py "<question>"` |
| B | 7 Guardrails | L3 validators, one repair pass, then a compliant system-composed answer | same as 6; `python eval/run_eval.py --mode guardrails` |
| B | 8 UI | Streamlit chat with a live retrieval-trace expander | `python -m streamlit run app/streamlit_app.py` |
| — | 9 Packaging | Source list, sample Q&A, disclaimer — all generated | `python scripts/export_outputs.py` |

## Why these source pages

Every URL was fetched and inspected before being registered. `hdfcmf.com` turned out to
be a parked domain, so the official AMC pages come from `hdfcfund.com`.

| Domain | Role | `is_official` |
|---|---|---|
| `hdfcfund.com` | HDFC AMC: 5 scheme pages + education/FAQ/statement guides | `True` |
| `amfiindia.com` | Regulator: HDFC mutual fund member details | `True` |
| `sebi.gov.in` | Regulator homepage | `True` |
| `groww.in` | Scheme pages (mandated); facts from the public `__NEXT_DATA__` payload | `False` |

Ingestion is intentionally fail-soft: a source that times out, is too thin to be useful,
or returns an error is recorded in `data/sources.csv` with a reason and skipped, so one
dead page never blocks the build.

## Phase 1 - Loading

```powershell
.\.venv\Scripts\python.exe scripts\run_ingest.py --reset
.\.venv\Scripts\python.exe -m mf_rag.chunk
.\.venv\Scripts\python.exe scripts\run_chunk_eval.py --rank embed
.\.venv\Scripts\python.exe scripts\build_index.py --stage all
```

Latest run: **17 registered sources, 14 OK, 3 skipped, 70,697 characters.**

- Skipped as too thin to index (JS-rendered shells with under 500 characters of main text):
  `hdfcfund_ter_disclosure`, `amfi_home`, `sebi_home`. Non-blocking by design — a fetch
  failure or a timeout is recorded with its reason and skipped rather than aborting the
  build.

Outputs: `data/raw/*.txt` snapshots, `data/sources.csv`, `outputs/ingest_report.json`.
The corpus is small on purpose, so historic NAV/DCA series are stored deduplicated
(re-fetched daily, not one row per run) to stop the corpus bloating by ~700 KB per run.

## Phase 2 - Chunking

```powershell
.\.venv\Scripts\python.exe scripts\run_chunk_eval.py    # compares the three strategies
.\.venv\Scripts\python.exe -m mf_rag.chunk              # writes the winning chunks
```

**The chosen strategy is `section`, decided by measurement.** All three candidates were run
over the real corpus (14 documents, 727 pipe-table rows, 4 question/answer pairs) by
`scripts/run_chunk_eval.py --rank embed`, which ranks with the same MiniLM cosine Stage 5
uses. Numbers are from [`outputs/chunk_eval_embed.json`](outputs/chunk_eval_embed.json),
the run that decided:

| strategy | chunks | avg chars | broken rows | split Q/A | hit@1 | hit@5 | MRR | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **section** | 103 | 685 | 0 | 0 | 0.75 | 1.00 | **0.833** | PASS |
| fixed | 142 | 570 | 4 | 0 | 0.67 | 1.00 | 0.806 | **FAIL** |
| parent_child | 274 | 257 | 0 | 0 | 0.67 | 1.00 | 0.787 | PASS |

Two-stage selection, exactly as `ARCHITECTURE` 5.2 specifies:

1. **Gates (pass/fail).** A strategy is rejected outright if it separates a number from
   its label, or a question from its answer. `fixed` fails: 4 of 727 table rows were cut
   in half, so a chunk could hold "1.21%" with no indication of which fee it belonged to.
2. **Ranking (MRR, then hit@1, then hit@5).** Among the survivors, `section` wins.

`section` is the only candidate meeting **all four** PRD §6.3 criteria: no orphaned
numbers, 100% of gold questions answerable, 685 average chars (inside the 300–800 band),
and every chunk traceable to a single source URL.

The cheap lexical proxy now agrees, on a wider question set (19 questions, the full gold
set plus the retrieval eval set) — see
[`outputs/chunk_eval.json`](outputs/chunk_eval.json): `section` MRR 0.565 / hit@5 1.000 vs
`parent_child` 0.496 / 0.789 and `fixed` 0.398 / 0.737, with the same two broken table rows
for `fixed`. Both ranking modes pick the same winner; only the absolute numbers move,
because the proxy has no embeddings.

### How this decision was wrong twice before it was right

Worth recording, because the wrong answers were both plausible:

1. **First run:** all three strategies scored 0.92 and the tiebreak picked `fixed`. The
   pass/fail check only looked for `expected_keywords` and never verified that a table row
   stayed intact. Adding the row-integrity gate is what exposed `fixed`'s real defect.
2. **Second run (lexical ranking):** `parent_child` won with MRR 0.604 vs 0.531, and was
   written into `config.yaml`. That was an **artifact of the ranking proxy** — an IDF
   bag-of-words scorer with no synonym matching, which cannot tell that "exit load" and
   "Exit load of 1% if redeemed within 1 year" mean the same thing. It also under-counted
   `section`, whose larger chunks carry more surrounding context.
3. **Third run (`--rank embed`, real MiniLM cosine):** `section` wins at MRR 0.833, and
   all three strategies reach hit@5 = 1.00. The config was switched back to `section` and
   the index rebuilt.

The lesson is the one D7 exists to enforce: **measure with the retriever you will actually
ship.** A cheap proxy is fine for the pass/fail gates, which are exact, but not for
ranking.

Two real bugs were found and fixed along the way, both invisible to the proxy:

- `parent_child` returned *unbounded* parents. Unheaded pages arrive as one block, so the
  "parent" was often the whole 28,000-character document — unusable in a prompt, and it
  let a hit match an answer nowhere near the child. Expansion is now bounded by `max_chars`.
- Parents were keyed `(source_id, heading)`, but every block of an unheaded page shares
  heading `''`, so each block overwrote the last one's parent. Keys are now per block.

## Phase 3 - Embedding

```powershell
.\.venv\Scripts\python.exe scripts\build_index.py --stage embed
```

Every chunk gets a 384-dim MiniLM-L6-v2 vector, computed once and cached on disk.

```
embedded 103 chunks (cache hits: 103, misses: 0, hit ratio: 100.0%)
```

The cache is keyed on `sha1(model + text)`, so:

- An unchanged corpus re-run costs **zero** model calls (14.2s → 0.1s).
- Editing one document re-embeds only that document.
- Renaming the model invalidates every entry automatically, rather than silently mixing
  vectors from two models.

The cache is **pruned to the live chunk set** each run. Without that it only grows:
switching chunking strategy left the previous run's 270 vectors behind, so
`embeddings.jsonl` grew to 373 entries against 103 chunks and the
`len(embeddings) == len(chunks)` guarantee quietly stopped holding. Covered by a
regression test.

`embedder.get_embedder()` is a `lru_cache` singleton keyed on the **model name**, not the
whole config — the model object is fully determined by its name, so this is a true
singleton per model while `batch_size` and `normalize` stay per-call settings.

## Phase 4 - Store Vector Data

```powershell
.\.venv\Scripts\python.exe scripts\build_index.py --stage all --reset
```

```
collection mf_faq: 103/103 chunks stored
count check : 103/103 OK
metadata    : round-trip OK
filter check: where={'category': 'ELSS'} -> 10 docs OK
```

Persistent ChromaDB at `data/chroma`, cosine space, 103 records surviving process exit
(verified from a brand-new interpreter).

Two things this stage does beyond writing vectors:

- **Model-drift guard.** The collection stores `embedding_model` in its metadata. If the
  config names a different model, `get_or_create_collection()` raises
  `ModelMismatchError` naming both models and pointing at `--reset`. This is the single
  most common RAG bug: reuse the collection, embed queries with the new model, and
  everything ranks badly with no error anywhere.
- **`verify_collection()`** asserts count, metadata round-trip, and that a
  `where={"category": "ELSS"}` filter returns rows — each failure with a distinct message,
  so a broken index fails at build time rather than mid-demo.

Chroma rejects `None` metadata, so `_chroma_safe()` coerces `scheme`/`category` to `""`
and `chunk_index` to `-1`. 38 of 103 chunks have no scheme, so this is load-bearing.

### A concrete finding for Phase 5

Querying the built index directly for *"Exit load of HDFC Small Cap Fund?"* returns
**Large Cap** chunks at ranks 1 and 3; the correct Small Cap fact is rank 2:

```
dist=0.2614  hdfcfund_large_cap_official
dist=0.2769  hdfc_small_cap_scheme_page     <- correct
dist=0.2814  hdfc_large_cap_scheme_page
```

Unfiltered MiniLM similarity is not enough to keep schemes apart. Stage 5's entity
detection plus a `where` filter is what makes this correct, and it is why the Phase 2 eval
scores a filtered ranking — otherwise every strategy looks worse than it will be.

## Phase 5 - Retrieval

```powershell
.\.venv\Scripts\python.exe eval\run_eval.py --mode retrieval
```

```
hit@5 = 1.000 (19/19)  hit@1 = 1.000  MRR = 1.000  slowest = 18526ms
```

All 19 gold questions retrieve an answer-bearing chunk at **rank 1**.
`outputs/retrieval_eval.json` records per-question rank, score, active filter and latency.

`entities.py` maps a question to a Chroma `where` filter via aliases and whole-phrase token
matching (`elss` / `tax saver` / `80c` → ELSS, `smallcap` → Small Cap, `flexi` / `hdfc equity`
→ Flexi Cap, `balanced advantage` → Balanced Advantage, `largecap` → Large Cap). A named
scheme wins over a category. Bare `cap` is deliberately *not* an alias — it appears inside
Small Cap, Large Cap and Flexi Cap, so matching it would pin every capital-market question
to one scheme.

`retrieve_with_trace()` follows ARCHITECTURE §6.2 in order: `embed_query` → `detect_entity` →
filtered query → unfiltered retry if the filter matched nothing → `score = 1 - distance` →
`min_score` floor → dedupe by `source_id` → cap at `context_chunks`. The trace records the
entity, the active filter, whether the fallback fired, and every hit with its score.

Dedupe by source is not a nicety: five chunks from one scheme page would crowd out every
other source and leave the generator five near-identical passages.

### min_score = 0.25, chosen by measurement (OQ-5)

Absolute MiniLM cosines are low, so the floor has to come from the corpus. 22 answer-bearing
chunks across the gold set span **0.4839–0.8201** (median 0.7035); the highest non-answer
chunk in the same runs is **0.3206**. Terse real questions sit lower even when they route to
the right source — *"What is the ELSS lock-in period?"* (the UI's own example question) scores
0.3025, and its **official** hdfcfund.com alternative only **0.2471**.

Sweeping the floor over the gold set plus 8 terse phrasings (measured when the gold set was
12 questions; it has since grown to 19):

| floor | gold recall | terse recall | official citation available (gold) | (terse) |
|-------|-------------|--------------|-----------------------------------|---------|
| 0.20  | 12/12       | 7/8          | 12/12                             | 7/8     |
| **0.25** | **12/12** | **6/8**      | **12/12**                         | **6/8** |
| 0.30  | 12/12       | 6/8          | 12/12                             | 5/8     |
| 0.35  | 12/12       | 4/8          | 12/12                             | 4/8     |

Re-measured on the current 19-question gold set, recall is **19/19 at every floor from 0.20
to 0.35** (`python eval/run_eval.py --mode retrieval --min-score 0.20|0.25|0.30|0.35`). So
gold recall no longer discriminates between floors at all, and the choice rests entirely on
the two things the floor actually trades off: terse real questions, and degenerate inputs
that should honestly return `NOT_IN_SOURCES`.

0.25 dominates 0.30 on both — one more terse question keeps an *official* source in context
so `select_source_url` can cite hdfcfund.com instead of the groww.in aggregator, while it
still rejects the degenerate inputs ("what is 80C" 0.1776, bare "lock-in?" 0.2213).

At 0.25, *"What is the ELSS lock-in period?"* cites
`hdfcfund.com/.../hdfc-elss-tax-saver-fund/direct`. At 0.30 it fell back to groww.in,
because the official chunk was below the floor — a traceability regression (FR-2.3) that the
first sweep would have missed. That same failure mode is still reachable: it is what happens
to the ELSS lock-in question in
[`outputs/sample_qa.md`](outputs/sample_qa.md), whose phrasing is terser than the sweep's.

## Asking it questions

Once a key is in `.env`, this is the thing to actually use:

```powershell
.\.venv\Scripts\python.exe scripts\ask.py
```

```
  HDFC Mutual Fund FAQ assistant
  103 chunks | sentence-transformers/all-MiniLM-L6-v2
  LLM: OpenAICompatClient (qwen/qwen3.8-27b)
  Type a question, or 'exit'. Try:
    What is the exit load of HDFC Small Cap Fund?
    Should I invest in HDFC Large Cap?     (refused on purpose)
    My PAN is ABCDE1234F                    (blocked on purpose)

  you> What is the ELSS lock-in period?

  The lock-in period for the HDFC ELSS Tax Saver Fund Direct Plan Growth is 3
  years. Source: https://www.hdfcfund.com/.../hdfc-elss-tax-saver-fund/direct
  Last updated from sources: 2026-09-27

  source  https://www.hdfcfund.com/.../hdfc-elss-tax-saver-fund/direct
  as of   2026-09-27
```

It runs the whole pipeline per question: guardrails, entity-filtered retrieval, Groq, then
L3 validation with repair. Guardrail hits are labelled, so you can see the safety layers
work rather than having to take their word for it:

```
  you> Should I invest in HDFC Large Cap?

  [refused by a guardrail - working as intended]

  I only share factual information from official scheme documents - I can't give
  investment advice or recommend a scheme. ...
```

Useful flags: `--debug` (full trace: filter, per-hit scores, validators, repair, latency),
`--timing`, `--no-color` (for piping to a file), or pass a question as an argument to ask
one question and exit.

### Just the answer, nothing else

```powershell
.\.venv\Scripts\python.exe scripts\ask.py "What is the ELSS lock-in period?" --bare
```

```
The lock-in period for the HDFC ELSS Tax Saver Fund Direct Plan Growth is 3 years.
Source: https://www.hdfcfund.com/explore/mutual-funds/hdfc-elss-tax-saver-fund/direct

Last updated from sources: 2026-09-27
```

`--bare` drops the header, the question echo, the separate source line, the refusal banner
and the trace. The answer still ends with its URL and as-of date, because the response
contract requires both — that is part of the answer, not decoration.

Combined with the interactive form, the `you> ` prompt moves to stderr, so a batch of
questions gives one clean answer per line:

```powershell
"What is the exit load of HDFC Small Cap Fund?`nWhat is the ELSS lock-in period?`nexit`n" |
    .\.venv\Scripts\python.exe scripts\ask.py --bare > answers.txt
```

This needed `input()` to be called with no argument: `input(prompt)` writes the prompt to
**stdout**, which put `you> ` in the middle of the piped answer text. The prompt is now
printed to stderr explicitly.

`scripts/search.py` is the other half: same pipeline but stopping at retrieval, so you see
the raw chunks and scores rather than a written answer.

## Testing retrieval

Two levels: a scored eval for "is it working", and `scripts/search.py` for "why did it do
that".

### 1. Scored eval (the regression gate)

```powershell
.\.venv\Scripts\python.exe eval\run_eval.py --mode retrieval
```

```
[PASS] q03_elss_lock_in     rank=1  top=0.644525 n=2   17ms
...
hit@5 = 1.000 (19/19)  hit@1 = 1.000  MRR = 1.000  slowest = 18526ms
wrote outputs\retrieval_eval.json
```

`eval/questions.json` holds 19 gold questions, each with the `source_id`s that count as a
hit. Run it after any change to chunking, embeddings, the entity filter or `min_score` — it
is the only thing here that will catch a silent ranking regression. Useful flags:
`--top-k`, `--min-score`, `--no-entity-filter` (to see what the embeddings alone rank).

The first query in a fresh process pays ~13s to load the sentence-transformers model; every
query after that is ~17ms. Only `slowest` reflects that one-time cost.

### 2. Interactive search (the debugging tool)

```powershell
# one-shot
.\.venv\Scripts\python.exe scripts\search.py "What is the ELSS lock-in period?"

# or a REPL: type questions until you want out
.\.venv\Scripts\python.exe scripts\search.py
```

```
QUERY   What is the ELSS lock-in period?
------------------------------------------------------------
entity        scheme
where filter  scheme=HDFC ELSS Tax Saver Fund - Direct Plan Growth
fallback      no
min_score     0.25    top_k 5
candidates    4 -> 2 after per-source dedupe

  [1] 0.3025  hdfc_elss_scheme_page
      official=no
      fetched_at=2026-09-27
      https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth
      ...chunk text...

  [2] 0.2545  hdfcfund_elss_official
      official=yes
      https://www.hdfcfund.com/explore/mutual-funds/hdfc-elss-tax-saver-fund/direct
```

It prints the decisions, not just the hits: which entity was detected, the exact `where`
filter applied, whether the unfiltered fallback fired, how many candidates survived dedupe,
and per hit the score, `source_id`, scheme, official/aggregator, `fetched_at`, URL and the
raw chunk text. Every one of those lines exists to answer a specific "why is this wrong?":

| Symptom | Look at |
|---|---|
| Wrong scheme's answer | `where filter` — did entity detection fire, and on what? |
| An unrelated answer | `fallback: YES` means the filter matched nothing and everything was fair game |
| `NO HITS above the floor` | expected for out-of-corpus questions; `--min-score 0.15` to see what was just below |
| Only one source represented | `candidates -> after dedupe`; dedupe is per `source_id` by design |
| Aggregator cited instead of HDFC | `official=` column; the official chunk may simply be below the floor |

The most useful flag when you suspect the filter is the problem:

```powershell
.\.venv\Scripts\python.exe scripts\search.py "Exit load of HDFC Small Cap Fund?" --no-entity-filter
```

```
  [1] 0.7386  hdfcfund_large_cap_official      <- wrong scheme wins unfiltered
  [2] 0.7231  hdfc_small_cap_scheme_page       <- correct
  [3] 0.7186  hdfc_large_cap_scheme_page
```

With the filter on, that same question puts Small Cap at rank 1. That comparison is the
entire justification for Phase 5, and `--no-entity-filter` reproduces it on demand.

`--json` dumps the raw trace for piping into other tools.

### 3. Other checks

```powershell
.\.venv\Scripts\python.exe eval\check_persistence.py   # is data on disk, is it intact
.\.venv\Scripts\python.exe eval\check_exit_gates.py    # all 19 Phase 5/6/7 spec gates
.\.venv\Scripts\python.exe scripts\inspect_embeddings.py  # full vector/neighbour dump
```

### Rate limits: expect about four questions a minute

This is the single most likely reason the assistant "suddenly stops answering". Groq
enforces **7,000 input tokens per minute** for `qwen/qwen3.8-27b` on the `on_demand` tier,
and one question costs ~1,580 tokens (245 system + ~1,335 context for 4 chunks). That is
**4.4 questions per minute**. A REPL session run faster than that returns
`HTTP 429 Too Many Requests ... on input tokens per minute (ITPM): Limit 7000`.

Ask questions at human speed and this never comes up. Paste six at once and it will.

Two things make it worse than it needs to be, and both are now handled:

- The client retries 429 with backoff, then raises `LLMError` naming the limit and the
  amount used, so the cause is visible instead of mysterious.
- `ask.py` reports it as an `ERROR` line and keeps the session alive. It does **not** fall
  back to a "the provided context does not contain that" refusal, because a rate-limited
  request says nothing about what the corpus contains.

Context is trimmed where it was pure repetition: the `source_url` and `fetched_at` lines
are per page, so they are printed once per source rather than once per chunk. Worth ~74
tokens a question. The remaining cost is real passage text, and cutting `context_chunks`
below 3 would cost answer quality - the fund-size question needs three chunks from one page
(see below). If you want more throughput, switch model; every model id in your account drops
in via `LLM_MODEL` with no code change.

## Adding an LLM API key

The app runs with **no** key at all: `get_llm()` returns `EchoClient` and sets
`degraded=True`, so you get extractive answers behind a "degraded mode" banner instead of a
crash. Add a key only when you want real generated prose.

```powershell
notepad .env
```

Three variables, already wired (`config.py:167` loads the file, `llm.py:254-259` reads them):

| Variable | Needed? | Notes |
|---|---|---|
| `LLM_API_KEY` | yes, to leave degraded mode | your provider's key |
| `LLM_BASE_URL` | only for non-OpenAI providers | **leave blank for OpenAI** — it defaults to `https://api.openai.com/v1` |
| `LLM_MODEL` | no | blank uses `gpt-4o-mini` from `config.yaml`; must match your provider's model id exactly |

Then check it without pasting the key anywhere visible:

```powershell
.\.venv\Scripts\python.exe scripts\check_llm.py          # local config only, safe offline
.\.venv\Scripts\python.exe scripts\check_llm.py --live   # one real call, confirms the key works
```

It prints the key's shape (`sk-t...90 (24 chars)`), never the key, and on failure names the
likely cause: wrong key, wrong `LLM_BASE_URL` path, wrong model id, or no credit.

Two things to know:

- **`provider` in `config.yaml` must stay `openai_compatible`.** If it is `echo`, a valid key
  is ignored and you stay in degraded mode.
- **A real environment variable beats `.env`**, because `load_dotenv` is called with
  `override=False`. Handy for a one-off test:
  `$env:LLM_API_KEY="sk-..."; .\.venv\Scripts\python.exe scripts\check_llm.py`

`.env` is git-ignored (`.gitignore:2`); `.env.example` is the committed, always-blank
template. Never paste a real key into `.env.example`.

**The test suite ignores `.env` entirely.** `tests/conftest.py` stubs
`mf_rag.config.load_dotenv` for the whole session, so a populated `.env` cannot turn unit
tests into live billable API calls. This was a real bug, not a hypothetical: three tests in
`test_generate.py` called `answer_raw()` without injecting a client, so they issued genuine
requests to `api.openai.com` and failed with `LLMError: ... HTTPError` the moment a key was
present. `test_a_key_in_the_env_file_cannot_reach_the_test_suite` guards it.

## Phase 6 - Generation

```powershell
.\.venv\Scripts\python.exe -c "from mf_rag.generate import answer_raw; print(answer_raw('What is the exit load of HDFC Small Cap Fund?',None)[0])"
```

```
Exit load (Direct plan): Exit load of 1% if redeemed within 1 year See:
https://www.hdfcfund.com/explore/mutual-funds/hdfc-small-cap-fund/direct

Last updated from sources: 2026-09-27
```

Two structural choices, not prompt discipline:

- **The system picks the citation** (ADR-5). `select_source_url()` chooses the URL from
  retrieved metadata, preferring `is_official=True` over a higher-scoring aggregator, and
  passes it to the model as a fixed token to echo. The model never authors a URL, so a
  hallucinated link is structurally impossible rather than merely unlikely.
- **The system stamps the date** (rule 7). The freshness line is appended in `generate_answer`
  from chunk metadata after any model-invented date is stripped, so the date can only ever
  come from `fetched_at`.

`ARCHITECTURE.md` §6.3 says freshness is the **max** `fetched_at` in one sentence and
"oldest data is the honest bound" in the next; `IMPLEMENTATION.md` Phase 6 says **min**. The
second reading is the intent, so min is implemented: an answer assembled from a page fetched
today and one fetched last year is only as current as the older page.

`get_llm()` returns `(client, degraded)`. With no `LLM_API_KEY`, or `provider: echo`, it
returns `EchoClient` and sets `degraded=True` so the UI can banner the fact. `EchoClient` is
deterministic, offline and cannot hallucinate — every word it emits came from a retrieved
chunk. It picks its sentence by scoring context lines against the question, and when no line
shares vocabulary with the question it says it could not find a fact rather than emitting an
unrelated sentence. Its *sentence selection* is a heuristic, not semantic matching: on the 12
gold questions it reproduces the right figure for roughly 8, and picks a wrong-but-real line
for a few (riskometer, benchmark). Demo-grade prose needs a real model via
`LLM_API_KEY`; the structural guarantees (≤3 sentences, one ingested URL, correct freshness)
hold either way.

## Phase 7 - Guardrails

```powershell
.\.venv\Scripts\python.exe eval\run_eval.py --mode guardrails
```

```
pass rate = 1.000 (18/18)  intent accuracy = 1.000  lexicon leaks = 0  PII reaching LLM = 0
```

`pipeline.query()` runs the three layers in an order that is itself the security property:

```
L1 PII       blocked?                -> PII_WARNING   (before embed_query, before the LLM)
L2 intent    advisory/portfolio/
             returns/out_of_corpus?  -> refusal template
F5 retrieve  nothing above the floor? -> NOT_IN_SOURCES
F6 generate                            -> draft
L3 validate  pass                     -> Answer
             one repair pass          -> revalidate
             still failing            -> safe system-composed answer
```

- **L1** covers PAN, Aadhaar, 10-digit phone, email, account number, OTP, IFSC and DP id.
  `detect_pii()` returns pattern **names, never values**, so a trace or log line cannot leak
  a PAN (NFR-4). Tests assert the value appears in no returned object.
- **L2** is rule-based on purpose: a classifier that can be persuaded by the text it
  classifies is a poor place to enforce "we don't give advice". `returns` and `portfolio` are
  checked before `advisory`, and a fund named outside the five-scheme registry is
  `out_of_corpus` — the honest answer is "not in my sources", not a guess.
- **L3** checks sentence count, exactly-one-URL, URL ∈ ingested set, official preference,
  the advice and returns lexicons, and the freshness line.

Three findings worth recording:

- `count_sentences` does not split on a period between digits, so *"The expense ratio is
  1.03%."* is one sentence. It also **excludes the freshness line**, which is a required
  suffix rather than prose — counting it would have silently capped the answer body at 2
  sentences while the contract says 3.
- The advice lexicon ignores **negated** matches. `REFUSAL_ADVICE` says "I can't give
  investment advice or **recommend** a scheme"; without the negation guard the system would
  reject its own correct refusal. The guardrail eval found this.
- Advice and returns violations are **not repaired in place** — the whole answer is replaced
  by the refusal template, because a factual sentence with advice spliced into it is still
  advice.

Every path returns a populated `Trace` holding a SHA-256 hash of the query, never the query
text.

### Notes on how this was measured

The retrieval scorer in `scripts/run_chunk_eval.py` is a lexical stand-in, not the real
retriever. It deliberately models the two behaviours the architecture specifies, because
omitting them changes the answer:

- **Parent expansion** for `parent_child` (ARCHITECTURE 5.2, Candidate C).
- **Entity filtering before ranking** (ARCHITECTURE 6.2, ADR-7), so a question about one
  scheme cannot be answered from another scheme's page.

It is still a proxy: it has no embeddings and no synonym matching, so absolute numbers move
once Stage 5 lands. What it is reliable for is the *ranking* of the three strategies, which
is what the decision needed.

> **Resolved in Phase 5.** The proxy's own numbers were superseded: the winner was re-scored
> with real MiniLM cosine (`--rank embed`) and `section` still won. The proxy is retained
> only for cheap strategy comparison, and its ranking role is now taken by
> `eval/run_eval.py --mode retrieval`.

Two earlier runs produced the wrong winner and both were discarded rather than reported:

- The first comparison scored all three at 0.92 hit rate and picked `fixed`, because the
  original pass/fail keyword check did not verify that a table row stayed intact. Adding
  the row-integrity gate is what exposed `fixed`'s real defect.
- The `capital gains statement` question could not be retrieved because HDFC's FAQ page
  renders only a list of FAQ *titles* in HTML - the answers are JS-rendered. Rather than
  loosen the question until it passed, the real official page
  (`/services/consolidated-account-statement`) was found and added to the registry, and
  the gold question was rewritten to match what that page actually says.

### Fund size (AUM), and why the prose lies about it

"What is the fund size of HDFC Flexi Cap?" used to be answered with *the provided context
does not contain the fund size*, even though the corpus had the number. Three separate
defects had to be fixed, and the first two were not the interesting ones.

**1. Extraction.** AUM never reached the stored text. Groww carries it in the scheme
payload, and hdfcfund.com carries it in its own page payload as an adjacent key pair,
`"aum":"113,606.47","aumAsMonth":"(31/08/2026)"` - visible on the official page, but
flattened away by main-text extraction, which only saw a bare
`<div>AUM<span>(31/08/2026)</span></div><div>113,606.47 Cr.</div>`. Both are now read from
their structured fields, so the official page is authoritative for fund size instead of the
answer being borrowed from an aggregator.

> **The trap.** Groww's own prose reads *"The fund currently has an Asset Under
> Management(AUM) of Rs 9,86,237 Cr"*. That is the **AMC's** whole book, not the fund's.
> The fund's AUM is Rs 1,13,606.47 Cr - about 8.7x smaller. A regex over the prose would
> have produced a confident, wrong answer, so the extractor reads the structured `aum` field
> only, where the fund / AMC / peer distinction is already made by key. All five official
> pages now report their own figure, and each matches the aggregator, which is a useful
> cross-check.

**2. Chunking.** The peer-comparison table was cut after its header, leaving a chunk of bare
unlabelled numbers:

```
| | HDFC Flexi Cap Direct Plan Growth | -0.32% | +15.42% | 1,13,606.47 |
```

Nothing in that line says `1,13,606.47` is a fund size, so the model was right to refuse.
`hard_split` now gives a piece that begins mid-table its own table's header back, reserving
room for it so the piece still fits `max_chars`. It tracks the header governing *each* line,
not just the block's first two - the block that mattered held a returns table *and* a peer
table, and the break fell inside the second one.

**3. Retrieval, and the real cause.** `_dedupe_by_source` kept only the single best chunk per
page ("one page, one voice"), on the assumption that a page's chunks are near-identical.
They are not: a scheme page holds distinct sections - labelled facts, returns, holdings,
peers. The chunk stating `Fund size / AUM (INR crore): 113,606.47` *was* retrieved, at
0.7367, above the 0.25 threshold, and then discarded because a useless peer-table tail
scored 0.7917. The generator never saw the answer. Dedupe now judges redundancy from the text
(3-gram overlap) and lets a page contribute up to 3 distinct chunks.

That last fix also cleared a separate long-standing bug: multi-part questions such as *"the
minimum SIP **and** the exit load for HDFC Flexi Cap"* had been answering "SIP amount is not
available" while INR 100 sat in the corpus.

`eval/questions.json` now carries `q13_fund_size_flexi_cap` so this cannot regress quietly.

## Phase 8 - UI

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

`app/streamlit_app.py` is a single chat surface with the compliance chrome the design
system specifies, and nothing else: a welcome state naming the five schemes, three example
chips, the always-visible `Facts-only. No investment advice.` caption, and a footer that
renders `prompts.DISCLAIMER` itself rather than a retyped copy of it (FR-8.5 depends on
that: the exported disclaimer has to be the string the UI shows).

- **Retrieval trace expander** (FR-7.5): intent, whether PII blocked, the entity that was
  detected, the `where` filter, every hit with its cosine score and section heading, which
  chunk was used, the selected citation, the validators that ran, whether a repair was
  needed, and the latency.
- **Refusals and PII blocks get their own treatment** — amber guardrail card and a headline
  that names the layer which stopped it, so a refusal is visibly a guardrail rather than a
  failure to answer.
- **Degraded-mode banner** when `get_llm()` returns `degraded=True`, i.e. there is no API
  key and answers are extractive.
- `get_config()` and the embedder are cached; chat state lives only in
  `st.session_state`. Nothing typed is written to disk.

## Phase 9 - Deliverables

### Deliverables index

| Requirement | Delivered as |
|---|---|
| FR-8.1 working prototype | [`app/streamlit_app.py`](app/streamlit_app.py) (Stage 8), or a CLI equivalent in [`scripts/ask.py`](scripts/ask.py) |
| FR-8.2 source list, CSV **and** MD | [`outputs/source_list.csv`](outputs/source_list.csv), [`outputs/source_list.md`](outputs/source_list.md) |
| FR-8.3 README with setup, scope, known limits | this file — [Setup](#setup), [Scope](#scope), [Known limits](#known-limits) |
| FR-8.4 sample Q&A, 5–10 queries with answers and links | [`outputs/sample_qa.md`](outputs/sample_qa.md) — 8 entries, generated |
| FR-8.5 disclaimer snippet as a reusable file | [`outputs/disclaimer.md`](outputs/disclaimer.md) |

```powershell
.\.venv\Scripts\python.exe scripts\export_outputs.py           # write all four
.\.venv\Scripts\python.exe scripts\export_outputs.py --check   # are the committed ones current?
```

All four are **generated, never typed**. The source list is projected from
`data/sources.csv`; the disclaimer is imported from `mf_rag.prompts.DISCLAIMER`; the sample
answers come from live `mf_rag.pipeline.query()` calls.

`--check` is the reproducibility gate: it regenerates `source_list.csv`, `source_list.md`
and `disclaimer.md` in memory and byte-compares them against the committed copies, and
verifies `sample_qa.md` still has 8 entries. It exits non-zero on drift, so a stale
deliverable cannot pass unnoticed. `sample_qa.md` is the one artefact `--check` does *not*
byte-compare — it is model output, so its wording can move between runs even at
temperature 0. What must hold every time is the per-question validator row, and a `FAIL`
there means the file should not be shipped as-is.

> Hosted providers rate-limit by input tokens: one question costs ~1,580 of Groq's
> 7,000-per-minute budget, so the script sleeps `--delay` (default 16 s) between the
> questions that reach the model. The two refusals cost nothing — the guardrail decides
> them before any LLM call — and are not delayed. Use `--client echo` for a keyless,
> offline, deterministic rebuild.

### Sample Q&A

8 questions spanning the coverage Phase 9 asks for: three numeric facts on three different
schemes (Large Cap expense ratio, Small Cap exit load, ELSS lock-in — which is both the
ELSS-specific question and a numeric one), a riskometer, a benchmark, a capital-gains
statement download, an advisory question and a returns question. Full answers with
validator rows are in [`outputs/sample_qa.md`](outputs/sample_qa.md); the summary:

| # | Question | Answer (abridged) | Link |
|---|---|---|---|
| 1 | Expense ratio of HDFC Large Cap? | 1.03% | [hdfcfund.com](https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct) |
| 2 | Exit load of HDFC Small Cap? | 1% if redeemed within 1 year; nil after | [hdfcfund.com](https://www.hdfcfund.com/explore/mutual-funds/hdfc-small-cap-fund/direct) |
| 3 | ELSS lock-in period? | 3 years | groww.in scheme page |
| 4 | Riskometer level of HDFC Flexi Cap? | Moderately High | [hdfcfund.com](https://www.hdfcfund.com/explore/mutual-funds/hdfc-flexi-cap-fund/direct) |
| 5 | Benchmark of HDFC Balanced Advantage? | NIFTY 50 Hybrid Composite Debt 50:50 Index | [hdfcfund.com](https://www.hdfcfund.com/explore/mutual-funds/hdfc-balanced-advantage-fund/direct) |
| 6 | How do I download my capital gains statement? | The indexed guide covers the consolidated account statement | [hdfcfund.com](https://www.hdfcfund.com/services/consolidated-account-statement) |
| 7 | Should I buy HDFC Small Cap? | **Refused** — facts only, no advice | [expense-ratio explainer](https://www.hdfcfund.com/learn/blog/what-expense-ratio-mutual-funds) |
| 8 | CAGR of HDFC Large Cap? | **Refused** — does not compute returns | [hdfcfund.com](https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct) |

Two things worth reading off that table rather than taking on trust:

- **Both refusals show `LLM calls: 0`.** Intent routing decides them from the question text
  before retrieval, so there is no prompt in which a model could be talked past them. A
  refusal is a rule, not a hope.
- **Question 3 cites groww.in, not the AMC page.** That is not a bug in citation selection:
  `select_source_url()` always prefers an official source, but for that terse phrasing
  retrieval returned no official ELSS chunk at all (it scores 0.2471, below the 0.25 floor
  — see `min_score` below). `sample_qa.md` states this explicitly per question rather than
  reporting a bare PASS, so the limitation is visible in the deliverable.

## Known limits

Copied from PRD §14, and the first thing a reviewer should read:

1. **Tiny corpus** — 5 schemes of one AMC only; not representative of the MF industry.
2. **Snapshot freshness** — answers reflect pages as of `fetched_at`; values change (TER,
   exit load). Every answer states its date.
3. **No numeric computation** — deliberately cannot calculate returns, corpus plans, or SIP
   projections.
4. **No personalisation** — no user profile, no goal-based guidance, by design.
5. **Chunking trade-offs** — a table/FAQ-aware splitter mitigates, but does not eliminate,
   the risk of splitting a table across chunks.
6. **MiniLM retrieval ceiling** — short embeddings may miss deeply worded official
   language; mitigated by filters and a larger k, not solved. Question 3 above is an
   instance.
7. **English only, no OCR** — scanned PDF factsheets are out of scope unless
   text-extractable.
8. **Not a financial product** — informational demo; users must verify on official
   AMC/AMFI pages.

Implementation-specific limits found while building (not in the PRD list):

- `hdfcfund.com/services/faqs` yields FAQ titles with no answers; real Q/A prose only
  arrives from the education blog pages. The Q/A gate therefore rests on 4 real pairs.
  A corpus with more FAQ content would make that gate much more meaningful.
- Plain-text pages have no headings, so they arrive as one large block. The chunkers keep
  question/answer pairs and table rows atomic, but a very long unheaded page still yields
  weaker heading breadcrumbs.
- Two JS-rendered sources (TER disclosure, AMFI homepage) and the SEBI homepage are still
  not indexed: each yields under 500 characters of main text.
- The gold set is 19 questions over 5 schemes. That is enough to choose a chunking strategy
  and to set `min_score`, but a hit rate of 1.000 on 19 questions has a wide confidence
  interval; it should not be read as a claim about unseen questions.
- AUM is read from each page's structured payload. A source that publishes fund size only as
  prose, or only in a client-rendered API call, would still yield no AUM. The two sources here
  happen to embed it, which is not something the pipeline can assume in general.
- `hard_split` can repeat a table header up to 3 extra chunks' worth of tokens per document.
  That is cheap here, but a table-dense corpus would want the repetition budget made explicit.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

448 passing. Model-dependent tests skip themselves when the sentence-transformers model is
unavailable, so the suite still runs offline:

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not model"   # 405 passed, 43 deselected
```

Coverage includes config strictness, source-registry shape, redirect allowlisting, cleaning,
structured fact extraction, all three chunkers' invariants, the embedding singleton and cache,
the Chroma model-drift guard, entity alias detection and its negative cases, the retrieval
order of operations (filter, fallback, floor, dedupe, cap) against both a fake collection and
the real index, prompt-template pinning, `select_source_url` official preference, oldest-date
freshness, `get_llm` degradation, PII detection and non-retention, intent classification,
every L3 validator including the repair paths, an end-to-end sweep asserting the PRD §7
contract holds on all 19 gold questions, and the Phase 9 deliverables — that
`source_list.csv`/`.md` still match `data/sources.csv`, that `disclaimer.md` still matches
the UI string, and that `sample_qa.md` still has 8 entries each satisfying the response
contract.

