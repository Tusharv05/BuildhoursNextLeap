"""Phase 2 strategy comparison: build the corpus with each chunker and measure it.

The winner is decided by evidence, not assumption (D7). Two kinds of measurement:

  GATES (pass/fail, from ARCHITECTURE 5.2 assembly rules)
    - every pipe-table row must survive intact inside a single chunk (FR-2.3)
    - a FAQ question must stay with its answer

  RANKING (lexical stand-in for retrieval, no embeddings needed yet)
    - hit@1, MRR, hit@5 over the gold questions, using IDF-weighted overlap
    - the naive fixed strategy ties on a pass/fail keyword check but loses here,
      because it emits 40% more chunks and more distractor noise

Usage: python scripts/run_chunk_eval.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mf_rag.chunk import build_chunks  # noqa: E402
from mf_rag.chunking import available_strategies  # noqa: E402
from mf_rag.config import AppConfig, get_config  # noqa: E402
from mf_rag.ingest import load_source_docs  # noqa: E402
from mf_rag.models import Chunk, SourceDoc  # noqa: E402

QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.json"
TOP_K = 5
ROW = re.compile(r"^\s*\|.*\|\s*$")
_SEP = re.compile(r"^[\s|:\-]+$")
QUESTION = re.compile(r"^\s*(?:Q[:.)]|\d+[.)])\s+\S", re.IGNORECASE)
HEADING_LINE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
HEADING_QUESTION = re.compile(r"^\s{0,3}#{1,6}\s+(?P<q>[^#]*\?)\s*$")
# A short standalone line ending in '?' - a section heading posing a question.
BARE_QUESTION = re.compile(r"^\s{0,3}(?!#|\||\d+[.)])\S[^\n]{6,80}\?\s*$")
_WORD = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from", "how",
    "i", "in", "is", "it", "its", "of", "on", "or", "the", "to", "what", "which", "with",
    "my", "me", "if", "s", "fund", "hdfc", "mutual",
}


def load_questions() -> list[dict[str, Any]]:
    if not QUESTIONS_PATH.is_file():
        raise FileNotFoundError(f"{QUESTIONS_PATH} not found")
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]


def tokenize(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


# ---------------------------------------------------------------- gates
def table_rows(doc: SourceDoc) -> list[str]:
    return [
        line
        for line in doc.text.split("\n")
        if ROW.match(line) and not _SEP.match(line)
    ]


def broken_table_rows(docs: Iterable[SourceDoc], chunks: list[Chunk]) -> tuple[int, int]:
    """(broken, total) pipe-table rows across the corpus.

    A row is broken when it does not appear intact in any single chunk of its
    document - i.e. the strategy cut the number away from its label.
    """
    broken = 0
    total = 0
    by_source: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.source_id, []).append(chunk)
    for doc in docs:
        source_chunks = by_source.get(doc.source_id, [])
        for row in table_rows(doc):
            total += 1
            needle = row.strip()
            if not any(needle in c.text for c in source_chunks):
                broken += 1
    return broken, total


def question_pairs(doc: SourceDoc) -> list[tuple[str, str]]:
    """Every (question, answer) pair in a document, in any shape the corpus uses.

    Trafilatura renders Q&A in three different ways depending on the page, and all
    three occur in this corpus:
      1. `## <question>` markdown heading followed by the answer prose
      2. `Q: <question>` / `A: <answer>` adjacent lines
      3. a bare line ending in `?` followed by the answer prose (blog/education pages)
    """
    lines = doc.text.split("\n")
    pairs: list[tuple[str, str]] = []
    for index, line in enumerate(lines):
        heading = HEADING_QUESTION.match(line)
        inline = QUESTION.match(line)
        bare = BARE_QUESTION.match(line)
        if heading:
            question = heading.group("q").strip()
        elif inline:
            question = re.sub(r"^\s*(?:Q[:.)]|\d+[.)])\s+", "", line, flags=re.I).strip()
        elif bare:
            question = line.strip()
        else:
            continue
        if question.lower().startswith("title:"):
            continue
        answer = ""
        for follower in lines[index + 1 : index + 6]:
            stripped = follower.strip()
            if not stripped:
                continue
            if HEADING_LINE.match(follower) or QUESTION.match(follower):
                break
            if BARE_QUESTION.match(follower):
                break
            answer = stripped
            break
        if answer and not answer.endswith("?") and not ROW.match(answer):
            pairs.append((question, answer))
    return pairs


def split_qa_pairs(docs: Iterable[SourceDoc], chunks: list[Chunk]) -> tuple[int, int]:
    """(broken, total) question/answer pairs whose two halves landed apart."""
    broken = 0
    total = 0
    by_source: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.source_id, []).append(chunk)
    for doc in docs:
        source_chunks = by_source.get(doc.source_id, [])
        for question, answer in question_pairs(doc):
            total += 1
            q = " ".join(question.split())[:40]
            a = " ".join(answer.split())[:40]
            if not any(q in c.text and a in c.text for c in source_chunks):
                broken += 1
    return broken, total


# ---------------------------------------------------------------- ranking
class EmbeddingIndex:
    """The real retriever: MiniLM vectors, cosine similarity.

    This is the ranking Stage 5 will actually use. Chunks are embedded from
    `embed_text` (the context-prefixed form that goes into Chroma) and the query is
    embedded by the same module, so query and index are guaranteed comparable. The
    metadata pre-filter and the parent expansion are identical to `LexicalIndex` and to
    the production path, so only the scoring function differs between the two modes.
    """

    def __init__(self, chunks: list[Chunk], cfg, expand=None) -> None:
        import numpy as np

        from mf_rag.embedder import embed_texts

        self.np = np
        self.chunks = chunks
        self.expand = expand
        self.vectors = embed_texts([c.embed_text for c in chunks], cfg)

    def rank(
        self,
        query: str,
        top_k: int = TOP_K,
        scheme: str | None = None,
        category: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        from mf_rag.embedder import embed_texts

        np = self.np
        pool = list(range(len(self.chunks)))
        filtered = False
        if scheme or category:
            key, value = ("scheme", scheme) if scheme else ("category", category)
            narrowed = [
                i for i in pool if (getattr(self.chunks[i], key) or "") == value
            ]
            if narrowed:
                pool = narrowed
                filtered = True

        def score(indices: list[int]) -> list[tuple[Chunk, float]]:
            if not indices:
                return []
            query_vec = embed_texts([query], None)[0]
            sims = self.vectors[indices] @ query_vec
            ranked = [
                (self.chunks[i], float(sims[j]))
                for j, i in enumerate(indices)
                if sims[j] > 0
            ]
            ranked.sort(key=lambda pair: pair[1], reverse=True)
            return ranked[:top_k]

        scored = score(pool)
        if not scored and filtered:
            scored = score(list(range(len(self.chunks))))
        return scored

    def context_for(self, chunk: Chunk) -> str:
        return self.expand(chunk) if self.expand else chunk.text

    def is_answer_bearing(self, chunk: Chunk, keywords: list[str]) -> bool:
        haystack = " ".join(self.context_for(chunk).lower().split())
        return all(" ".join(k.lower().split()) in haystack for k in keywords if k)


class LexicalIndex:
    """IDF-weighted bag-of-words scorer standing in for embedding retrieval.

    For the parent_child strategy the ranked unit stays the small child chunk, but
    relevance is judged on the parent section the child belongs to - that is the
    text Phase 5 actually puts in the prompt (ARCHITECTURE 5.2, Candidate C).
    """

    def __init__(self, chunks: list[Chunk], expand=None) -> None:
        self.chunks = chunks
        self.expand = expand
        self.tokens = [tokenize(f"{c.section_heading} {c.text}") for c in chunks]
        doc_freq: Counter[str] = Counter()
        for tokens in self.tokens:
            doc_freq.update(set(tokens))
        total = max(1, len(chunks))
        self.idf = {
            term: math.log((total + 1) / (count + 1)) + 1.0 for term, count in doc_freq.items()
        }

    def rank(
        self,
        query: str,
        top_k: int = TOP_K,
        scheme: str | None = None,
        category: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Rank chunks, applying the same metadata pre-filter Phase 5 applies
        (ARCHITECTURE 6.2 / ADR-7): an entity-derived `where` filter narrows the
        candidate set *before* ranking, so a question about one scheme can never be
        answered from another scheme's page. Falls back to the unfiltered set when
        the filter leaves nothing."""
        terms = tokenize(query)
        if not terms:
            return []

        pool = self.chunks
        filtered = False
        if scheme or category:
            key, value = ("scheme", scheme) if scheme else ("category", category)
            narrowed = [c for c in self.chunks if getattr(c, key) == value]
            if narrowed:
                pool = narrowed
                filtered = True
        scored = self._score(pool, terms)
        if not scored and filtered:
            scored = self._score(self.chunks, terms)  # fallback_unfiltered
        return scored[:top_k]

    def _score(self, pool: list[Chunk], terms: list[str]) -> list[tuple[Chunk, float]]:
        scored: list[tuple[Chunk, float]] = []
        for chunk in pool:
            toks = tokenize(f"{chunk.section_heading} {chunk.text}")
            if not toks:
                continue
            counts = Counter(toks)
            length = len(toks)
            score = 0.0
            for term in terms:
                if term in counts:
                    score += self.idf.get(term, 1.0) * (counts[term] / length) ** 0.5
            if score:
                scored.append((chunk, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

    def context_for(self, chunk: Chunk) -> str:
        """The text the LLM would see for this hit."""
        return self.expand(chunk) if self.expand else chunk.text

    def is_answer_bearing(self, chunk: Chunk, keywords: list[str]) -> bool:
        haystack = " ".join(self.context_for(chunk).lower().split())
        return all(" ".join(k.lower().split()) in haystack for k in keywords if k)


def rank_metrics(index: LexicalIndex, questions: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for question in questions:
        ranked = index.rank(
            question["question"],
            scheme=question.get("scheme"),
            category=question.get("category"),
        )
        rank = next(
            (
                i
                for i, (chunk, _) in enumerate(ranked, start=1)
                if index.is_answer_bearing(chunk, question["expected_keywords"])
            ),
            0,
        )
        rows.append({"id": question["id"], "intent": question["intent"], "rank": rank})
    hit1 = sum(1 for r in rows if r["rank"] == 1)
    hitk = sum(1 for r in rows if 1 <= r["rank"] <= TOP_K)
    mrr = sum(1 / r["rank"] for r in rows if r["rank"]) / max(1, len(rows))
    return {
        "hit_at_1": round(hit1 / len(rows), 3),
        "hit_at_5": round(hitk / len(rows), 3),
        "mrr": round(mrr, 3),
        "unanswered": [r["id"] for r in rows if not r["rank"]],
        "per_question": rows,
    }


def score_strategy(docs: list[SourceDoc], cfg: AppConfig, name: str, questions, mode: str) -> dict[str, Any]:
    chunks, chunker = build_chunks(docs, cfg, name)
    broken, total_rows = broken_table_rows(docs, chunks)
    split_qa, total_qa = split_qa_pairs(docs, chunks)
    # parent_child retrieves a child but answers from its parent section.
    expand = getattr(chunker, "parent_for", None)
    index = (
        EmbeddingIndex(chunks, cfg, expand=expand)
        if mode == "embed"
        else LexicalIndex(chunks, expand=expand)
    )
    ranking = rank_metrics(index, questions)
    avg_chars = round(sum(len(c.text) for c in chunks) / max(1, len(chunks)))
    return {
        "strategy": chunker.name,
        "chunks": len(chunks),
        "avg_tokens": round(sum(c.token_count for c in chunks) / max(1, len(chunks)), 1),
        "avg_chars": avg_chars,
        "broken_table_rows": broken,
        "table_rows": total_rows,
        "split_qa_pairs": split_qa,
        "qa_pairs": total_qa,
        "expands_to_parent": bool(expand),
        **ranking,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 chunking strategy comparison")
    parser.add_argument(
        "--rank",
        choices=("lexical", "embed"),
        default="lexical",
        help="lexical: fast IDF proxy, no model. embed: real MiniLM cosine, as Stage 5 uses.",
    )
    args = parser.parse_args()
    mode = args.rank

    cfg = get_config()
    if not cfg.sources_csv.is_file():
        print("ERROR: no corpus. Run 'python scripts/run_ingest.py' first (Phase 1).")
        return 1
    docs = load_source_docs(cfg)
    questions = load_questions()
    results = [score_strategy(docs, cfg, name, questions, mode) for name in available_strategies()]

    def gates_pass(result: dict[str, Any]) -> bool:
        return result["broken_table_rows"] == 0 and result["split_qa_pairs"] == 0

    eligible = [r for r in results if gates_pass(r)]
    pool = eligible or results
    winner = sorted(pool, key=lambda r: (-r["mrr"], -r["hit_at_1"], -r["hit_at_5"]))[0]

    print(f"Phase 2 - chunking strategy comparison (rank={mode})")
    print("=" * 96)
    print(
        f"{'strategy':<14} {'chunks':>7} {'avg_char':>9} {'rows':>6} {'broken':>8} "
        f"{'splitQA':>8} {'qa':>4} {'hit@1':>7} {'hit@5':>7} {'MRR':>6}  gates"
    )
    print("-" * 96)
    for result in sorted(results, key=lambda r: -r["mrr"]):
        mark = "PASS" if gates_pass(result) else "FAIL"
        flag = "  <-- WINNER" if result is winner else ""
        print(
            f"{result['strategy']:<14} {result['chunks']:>7} {result['avg_chars']:>9} "
            f"{result['table_rows']:>6} {result['broken_table_rows']:>8} "
            f"{result['split_qa_pairs']:>8} {result['qa_pairs']:>4} "
            f"{result['hit_at_1']:>7.2f} {result['hit_at_5']:>7.2f} "
            f"{result['mrr']:>6.3f}  {mark}{flag}"
        )
    print("-" * 96)
    print("GATE  : zero broken table rows and zero split question/answer pairs (FR-2.3)")
    print("RANK  : highest MRR, then hit@1, then hit@5")
    qa_total = results[0]["qa_pairs"]
    if not qa_total:
        print("NOTE  : the corpus contains 0 question/answer pairs, so the Q/A gate is")
        print("       vacuous here. It is enforced by unit tests instead (tests/test_chunking.py).")
    print(f"WINNER: {winner['strategy']}  MRR={winner['mrr']}  hit@1={winner['hit_at_1']}  "
          f"hit@5={winner['hit_at_5']}  chunks={winner['chunks']}  avg_chars={winner['avg_chars']}")
    if winner["unanswered"]:
        print(f"  not retrieved in top-{TOP_K}: {', '.join(winner['unanswered'])}")
    print("=" * 96)

    payload = {
        "questions": len(questions),
        "rank_mode": mode,
        "winner": winner["strategy"],
        "decision_rule": "gates (0 broken table rows, 0 split Q/A pairs) then MRR, hit@1, hit@5",
        "corpus": {
            "documents": len(docs),
            "table_rows": results[0]["table_rows"],
            "qa_pairs": qa_total,
            "note": "0 qa_pairs means the Q/A gate could not be measured on this corpus",
        },
        "results": sorted(results, key=lambda r: -r["mrr"]),
    }
    out = cfg.outputs_dir / f"chunk_eval{'' if mode == 'lexical' else '_embed'}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
