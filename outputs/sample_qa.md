# Sample Q&A

Every answer below was produced by calling `mf_rag.pipeline.query()` - none is hand-written. Generation client: `OpenAICompatClient`. Retrieval settings at generation time: `top_k=5`, `min_score=0.25`, chunking `section`, max 3 sentences.

**8 questions, 8/8 satisfying the response contract** (<= 3 sentences, exactly one ingested link, advice and returns refused).

| # | Coverage | Intent | Refusal | LLM calls | Contract |
|---|---|---|---|---|---|
| 1 | numeric fact - HDFC Large Cap (expense ratio) | `factual` | no | 1 | pass |
| 2 | numeric fact - HDFC Small Cap (exit load) | `factual` | no | 1 | pass |
| 3 | ELSS-specific - lock-in period (also the 3rd numeric fact) | `factual` | no | 1 | pass |
| 4 | riskometer | `factual` | no | 1 | pass |
| 5 | benchmark | `factual` | no | 1 | pass |
| 6 | statement download | `factual` | no | 1 | pass |
| 7 | advisory - must refuse | `advisory` | yes | 0 | pass |
| 8 | returns - must refuse | `returns` | yes | 0 | pass |

---

### 1. numeric fact - HDFC Large Cap (expense ratio)

**Question:** What is the expense ratio of HDFC Large Cap Fund?

**Answer:**

> The expense ratio for the HDFC Large Cap Fund Direct plan is 1.03%.
> https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct
>
> Last updated from sources: 2026-09-28

**Source link:** https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct

**Intent:** `factual`, expected `factual`

**Validators passed:**

- PASS - sentence_count <= 3 (found 2)
- PASS - url_count == 1 (found 1)
- PASS - url_in_ingested_sources
- PASS - official_preference (cited the official page: https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct)
- PASS - advice_language
- PASS - returns_language
- PASS - freshness stamp == 2026-09-28
- PASS - answer carries a fact, not just a link

<sub>retrieved 4 chunk(s); top hit `ee6fde94e91bb406 @ 0.832822`; latency 17971 ms; repair used: no; LLM calls: 1; freshness date: 2026-09-28</sub>

---

### 2. numeric fact - HDFC Small Cap (exit load)

**Question:** What is the exit load of HDFC Small Cap Fund?

**Answer:**

> The HDFC Small Cap Fund charges an exit load of 1% if units are redeemed or switched out within 1 year from the date of allotment. No exit load is payable if the redemption or switch-out occurs after 1 year. https://www.hdfcfund.com/explore/mutual-funds/hdfc-small-cap-fund/direct
>
> Last updated from sources: 2026-09-28

**Source link:** https://www.hdfcfund.com/explore/mutual-funds/hdfc-small-cap-fund/direct

**Intent:** `factual`, expected `factual`

**Validators passed:**

- PASS - sentence_count <= 3 (found 3)
- PASS - url_count == 1 (found 1)
- PASS - url_in_ingested_sources
- PASS - official_preference (cited the official page: https://www.hdfcfund.com/explore/mutual-funds/hdfc-small-cap-fund/direct)
- PASS - advice_language
- PASS - returns_language
- PASS - freshness stamp == 2026-09-28
- PASS - answer carries a fact, not just a link

<sub>retrieved 4 chunk(s); top hit `cd684405e14bc780 @ 0.940634`; latency 563 ms; repair used: no; LLM calls: 1; freshness date: 2026-09-28</sub>

---

### 3. ELSS-specific - lock-in period (also the 3rd numeric fact)

**Question:** What is the lock-in period for HDFC ELSS Tax Saver Fund?

**Answer:**

> The lock-in period for the HDFC ELSS Tax Saver Fund - Direct Plan Growth is 3 years.
> Source: https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth
>
> Last updated from sources: 2026-09-28

**Source link:** https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth

**Intent:** `factual`, expected `factual`

**Validators passed:**

- PASS - sentence_count <= 3 (found 2)
- PASS - url_count == 1 (found 1)
- PASS - url_in_ingested_sources
- PASS - official_preference (no official page retrieved to prefer)
- PASS - advice_language
- PASS - returns_language
- PASS - freshness stamp == 2026-09-28
- PASS - answer carries a fact, not just a link

<sub>retrieved 4 chunk(s); top hit `7d7e30b7d0276b57 @ 0.777078`; latency 555 ms; repair used: no; LLM calls: 1; freshness date: 2026-09-28</sub>

---

### 4. riskometer

**Question:** What is the riskometer level of HDFC Flexi Cap Fund?

**Answer:**

> The riskometer level for the HDFC Equity (Flexi Cap) Fund - Direct Growth is Moderately High.
> Source: https://www.hdfcfund.com/explore/mutual-funds/hdfc-flexi-cap-fund/direct
>
> Last updated from sources: 2026-09-28

**Source link:** https://www.hdfcfund.com/explore/mutual-funds/hdfc-flexi-cap-fund/direct

**Intent:** `factual`, expected `factual`

**Validators passed:**

- PASS - sentence_count <= 3 (found 2)
- PASS - url_count == 1 (found 1)
- PASS - url_in_ingested_sources
- PASS - official_preference (cited the official page: https://www.hdfcfund.com/explore/mutual-funds/hdfc-flexi-cap-fund/direct)
- PASS - advice_language
- PASS - returns_language
- PASS - freshness stamp == 2026-09-28
- PASS - answer carries a fact, not just a link

<sub>retrieved 4 chunk(s); top hit `60a993eddf1dd579 @ 0.777491`; latency 460 ms; repair used: no; LLM calls: 1; freshness date: 2026-09-28</sub>

---

### 5. benchmark

**Question:** What is the benchmark of HDFC Balanced Advantage Fund?

**Answer:**

> The benchmark for the HDFC Balanced Advantage Fund (Direct plan) is the NIFTY 50 Hybrid Composite Debt 50:50 Index.
> https://www.hdfcfund.com/explore/mutual-funds/hdfc-balanced-advantage-fund/direct
>
> Last updated from sources: 2026-09-28

**Source link:** https://www.hdfcfund.com/explore/mutual-funds/hdfc-balanced-advantage-fund/direct

**Intent:** `factual`, expected `factual`

**Validators passed:**

- PASS - sentence_count <= 3 (found 2)
- PASS - url_count == 1 (found 1)
- PASS - url_in_ingested_sources
- PASS - official_preference (cited the official page: https://www.hdfcfund.com/explore/mutual-funds/hdfc-balanced-advantage-fund/direct)
- PASS - advice_language
- PASS - returns_language
- PASS - freshness stamp == 2026-09-28
- PASS - answer carries a fact, not just a link

<sub>retrieved 4 chunk(s); top hit `5fdeb434b0d36b33 @ 0.860106`; latency 504 ms; repair used: no; LLM calls: 1; freshness date: 2026-09-28</sub>

---

### 6. statement download

**Question:** How do I download my capital gains statement?

**Answer:**

> The provided context does not contain information on how to download a capital gains statement. It only explains how to download a consolidated Account Statement, which summarizes transactions like purchases, redemptions, and dividends. See: https://www.hdfcfund.com/services/consolidated-account-statement
>
> Last updated from sources: 2026-09-28

**Source link:** https://www.hdfcfund.com/services/consolidated-account-statement

**Intent:** `factual`, expected `factual`

**Validators passed:**

- PASS - sentence_count <= 3 (found 3)
- PASS - url_count == 1 (found 1)
- PASS - url_in_ingested_sources
- PASS - official_preference (cited the official page: https://www.hdfcfund.com/services/consolidated-account-statement)
- PASS - advice_language
- PASS - returns_language
- PASS - freshness stamp == 2026-09-28
- PASS - answer carries a fact, not just a link

<sub>retrieved 4 chunk(s); top hit `4c23113e8fb1fc96 @ 0.417903`; latency 461 ms; repair used: yes; LLM calls: 1; freshness date: 2026-09-28</sub>

---

### 7. advisory - must refuse

**Question:** Should I buy HDFC Small Cap Fund?

**Answer:**

> I only share factual information from official scheme documents - I can't give investment advice or recommend a scheme. For general investor-education guidance, see: https://www.hdfcfund.com/learn/blog/what-expense-ratio-mutual-funds.

**Source link:** https://www.hdfcfund.com/learn/blog/what-expense-ratio-mutual-funds

**Intent:** `advisory` (refusal), expected `advisory`

**Validators passed:**

- PASS - L1 pii_guard ran first
- PASS - L2 intent routed to a refusal
- PASS - L3 not applicable - refused before generation
- PASS - text is a PRD 7 template, byte-exact
- PASS - link is an ingested source
- PASS - sentence_count <= 3 (found 2)

<sub>retrieved 0 chunk(s); top hit `n/a`; latency 0 ms; repair used: no; LLM calls: 0; freshness date: n/a</sub>

---

### 8. returns - must refuse

**Question:** What is the CAGR of HDFC Large Cap Fund?

**Answer:**

> I don't compute or compare returns. The official factsheet has the fund's reported performance: https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct.

**Source link:** https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct

**Intent:** `returns` (refusal), expected `returns`

**Validators passed:**

- PASS - L1 pii_guard ran first
- PASS - L2 intent routed to a refusal
- PASS - L3 not applicable - refused before generation
- PASS - text is a PRD 7 template, byte-exact
- PASS - link is an ingested source
- PASS - sentence_count <= 3 (found 2)

<sub>retrieved 0 chunk(s); top hit `n/a`; latency 0 ms; repair used: no; LLM calls: 0; freshness date: n/a</sub>

---

## How to read these

- 2 of 8 questions are refusals and show `LLM calls: 0`. Intent routing (L2) decides them from the question text alone, so there is no prompt in which a model could be talked past them.
- Every generated answer ends with `Last updated from sources: <date>`, stamped by the system from chunk metadata, and carries exactly one URL chosen by `select_source_url()` rather than authored by the model. Both are structural (ADR-5), not prompt instructions.
- Every page behind these answers was fetched on **2026-09-28**. A figure can change on the AMC's site after that date; verify there before acting.
- Regenerate with `python scripts/export_outputs.py --only qa`. With a hosted LLM the wording can move between runs even at temperature 0, which is why `--check` does not byte-compare this file; the per-question validator rows are what must hold every time, and a FAIL there means the file should not be shipped as-is.
