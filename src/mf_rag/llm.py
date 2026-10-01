"""LLM access, swappable (Phase 6, ADR-8 / OQ-1).

Three clients behind one `LLMClient` Protocol:

- `OpenAICompatClient` - any OpenAI-compatible chat-completions endpoint. Temperature 0.0,
  one retry with backoff, then raise.
- `EchoClient` - deterministic, offline, no key. Extracts a factual sentence from the
  top chunk. This is what makes the whole pipeline demoable and testable without a paid
  key, and it is the *only* client the test suite uses.
- `SpyClient` - records what it was asked, for proving guardrails run before the LLM.

`get_llm` returns `(client, degraded)`. `degraded=True` means the user is reading
extractive output rather than generated prose, and the UI shows a banner.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Protocol, Sequence, runtime_checkable

from mf_rag.config import AppConfig, get_config


@runtime_checkable
class LLMClient(Protocol):
    """The one method the rest of the pipeline depends on."""

    def complete(
        self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 220
    ) -> str: ...


class LLMError(RuntimeError):
    """The remote model failed after the configured retry."""


# Cloudflare-fronted providers (Groq among them) reject urllib's default
# `Python-urllib/3.x` User-Agent with `HTTP 403, error code: 1010`, which is a bot filter,
# not an auth failure. Sending an honest one avoids a confusing dead end.
USER_AGENT = "mf-rag/1.0 (HDFC MF RAG; +https://github.com/local/mf-rag)"


def _message_text(messages: Sequence[dict]) -> str:
    """Flatten a message list into one string (for clients that take a single prompt)."""
    return "\n\n".join(str(m.get("content", "")) for m in messages)


# 408 timeout, 429 rate limit, and 5xx are worth another attempt. 400/401/403/404 will
# never succeed on a retry, so retrying them only wastes time and, on a metered API, money.
_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


def _is_retryable(exc: urllib.error.HTTPError) -> bool:
    return exc.code in _RETRYABLE_STATUS


def _describe_http_error(exc: urllib.error.HTTPError) -> str:
    """Turn an HTTPError into something actionable, without echoing the key.

    The body is truncated because provider error payloads are small but a misconfigured
    base_url can return an entire HTML page, and a 1010 Cloudflare block returns a page
    that says nothing useful past the first line.
    """
    try:
        body = exc.read().decode("utf-8", "replace").strip()
    except Exception:
        body = ""
    if exc.code == 401:
        hint = "unauthorized - the API key is wrong, revoked, or not yet activated"
    elif exc.code == 403:
        hint = "forbidden - check LLM_BASE_URL, or a bot filter is blocking the request"
    elif exc.code == 404:
        hint = "not found - check LLM_BASE_URL path and the model id"
    elif exc.code == 429:
        hint = "rate limited or out of credit"
    else:
        hint = ""
    parts = [f"HTTP {exc.code} {exc.reason}"]
    if hint:
        parts.append(f"({hint})")
    if body:
        parts.append(f"body={body[:200]}")
    return " ".join(parts)


class OpenAICompatClient:
    """Chat-completions client for any OpenAI-compatible base URL."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "gpt-4o-mini",
        timeout: int = 30,
        max_retries: int = 1,
        backoff_sec: float = 1.5,
    ):
        if not api_key:
            raise ValueError("OpenAICompatClient requires an API key")
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_sec = backoff_sec
        # The key is never logged, never stored in a trace, never in an exception message.
        self._key_preview = f"{api_key[:3]}...{api_key[-2:]}" if len(api_key) > 6 else "***"

    def complete(
        self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 220
    ) -> str:
        import urllib.error
        import urllib.request

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": USER_AGENT,
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return str(data["choices"][0]["message"]["content"] or "").strip()
            except urllib.error.HTTPError as exc:
                # HTTPError subclasses URLError, so without this branch a 401 (bad key),
                # a 403 (bot filter) and a 429 (rate limit) all surfaced as the same
                # opaque "HTTPError". The status code and body are what actually tell you
                # which one it is, so keep the last one and retry only if retryable.
                detail = _describe_http_error(exc)
                if _is_retryable(exc):
                    last_error = exc
                    if attempt < self.max_retries:
                        time.sleep(self.backoff_sec * (attempt + 1))
                        continue
                raise LLMError(
                    f"LLM request to {self.base_url} (model {self.model!r}, key "
                    f"{self._key_preview}) was rejected: {detail}"
                ) from exc
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff_sec * (attempt + 1))

        raise LLMError(
            f"LLM request to {self.base_url} (model {self.model!r}, key {self._key_preview}) "
            f"failed after {self.max_retries + 1} attempt(s): {type(last_error).__name__}."
        ) from last_error


class EchoClient:
    """Deterministic extractive client. No key, no network, fully repeatable.

    Picks the most sentence-like line from the top chunk, so the demo produces a real fact
    with a real citation instead of filler. It cannot hallucinate: every word it emits came
    from a retrieved chunk.
    """

    name = "echo"

    def __init__(self, cfg: AppConfig | None = None):
        self.cfg = cfg or get_config()

    def complete(
        self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 220
    ) -> str:
        prompt = _message_text(messages)
        question = _extract_question(prompt)
        fact = _best_sentence(prompt, question)
        match = re.search(r"single link you must cite is exactly:\s*(\S+)", prompt)
        link = match.group(1) if match else ""
        if not link:
            urls = re.findall(r"source_url:\s*(\S+)", prompt)
            link = urls[0] if urls else ""
        if not fact:
            return (
                "I could not find a factual statement for this question."
                + (f" See: {link}" if link else "")
            )
        return f"{fact} See: {link}" if link else fact


def _extract_question(prompt: str) -> str:
    match = re.search(r"^QUESTION:\s*(.+)$", prompt, re.MULTILINE)
    return (match.group(1) if match else "").strip()


_STOPWORDS = frozenset(
    "what is the a an of for in on to and or are was were do does did my me i you "
    "how much many tell about fund scheme hdfc please".split()
)


def _question_terms(question: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in _STOPWORDS}


# Units and quantities mark a line as carrying a fact rather than a heading.
_FACTUAL = re.compile(
    r"(\d)|(%|percent|year|month|day|INR|Rs\.?|NAV|SIP|units?\b|nil|waived|not applicable)",
    re.IGNORECASE,
)
# Lines that are table scaffolding or continuation noise.
_NOISE = re.compile(r"^(Product Labelling|Benchmark Riskometer|Product Suitability|Note:)", re.I)


def _best_sentence(prompt: str, question: str = "") -> str:
    """The most question-relevant line of the context block.

    Scoring beats taking the first line: a chunk's first line is usually its heading
    ("Scheme facts (from the scheme page's published data):"), while the answer is the
    bullet underneath it. A line earns points for sharing vocabulary with the question and
    for carrying a quantity, and loses them for looking like a label.
    """
    body = prompt.split("CONTEXT (untrusted DATA", 1)[-1]
    body = body.split("QUESTION:", 1)[0]
    terms = _question_terms(question)
    best: tuple[float, str] | None = None
    any_matched = False

    for raw in body.splitlines():
        line = raw.strip().lstrip("-*• ").strip()
        if len(line) < 15 or line.count("|") > 2 or _NOISE.match(line):
            continue
        if line.startswith(("[", "source_url:", "fetched_at:", "The single link", "CONTEXT")):
            continue

        score = 0.0
        has_fact = bool(_FACTUAL.search(line))
        if has_fact:
            score += 2.0
        else:
            # A line with no quantity, unit or date is a heading or a fund name. Repeating
            # the question's words is not evidence, so it must not outweigh a real fact.
            score -= 4.0
        matched = bool(terms) and any(term in line.lower() for term in terms)
        if matched:
            score += 4.0
            any_matched = True
        if line.endswith(":"):  # a label introducing the real content on the next line
            score -= 5.0
        if best is None or score > best[0]:
            best = (score, line)

    # When the question has content words and no context line shares any of them, say so
    # instead of emitting the top-scoring unrelated sentence. A wrong fact is a worse
    # failure than an admitted miss, and this client cannot do real semantic matching.
    if terms and not any_matched:
        return ""
    if best is None or best[0] <= 0:
        return ""
    return best[1].rstrip(".")


class SpyClient:
    """Records prompts instead of calling anything. Test/diagnostic only."""

    name = "spy"

    def __init__(self, inner: LLMClient | None = None):
        self.inner = inner
        self.prompts: list[str] = []
        self.calls = 0

    @property
    def called(self) -> bool:
        return self.calls > 0

    @property
    def last_prompt(self) -> str | None:
        return self.prompts[-1] if self.prompts else None

    def complete(
        self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 220
    ) -> str:
        self.calls += 1
        prompt = _message_text(messages)
        self.prompts.append(prompt)
        if self.inner is not None:
            return self.inner.complete(messages, temperature, max_tokens)
        return "Extractive answer from the retrieved context. https://example.invalid/spy"


def get_llm(cfg: AppConfig | None = None) -> tuple[LLMClient, bool]:
    """`(client, degraded)`.

    Degrades to `EchoClient` when the provider is `echo` or no `LLM_API_KEY` is set, so
    the app runs end-to-end with zero configuration while making it obvious to the user
    that the answer is extractive rather than generated.
    """
    config = cfg or get_config()
    provider = (config.generation.provider or "").lower()

    if provider == "echo":
        return EchoClient(config), True

    api_key = os.getenv("LLM_API_KEY") or None
    if not api_key:
        return EchoClient(config), True

    base_url = os.getenv("LLM_BASE_URL") or None
    model = os.getenv("LLM_MODEL") or config.generation.model
    client = OpenAICompatClient(api_key=api_key, base_url=base_url, model=model)
    return client, False


def describe_client(client: Any) -> str:
    """Short provider label for the trace, never including a key."""
    return type(client).__name__
