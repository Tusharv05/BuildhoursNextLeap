"""Typed configuration loader for the RAG pipeline (Phase 0).

Loads config.yaml into frozen dataclasses. Unknown keys raise, so typos in the
YAML surface immediately instead of being silently ignored.
"""

from __future__ import annotations

import os
from dataclasses import MISSING, dataclass, fields, is_dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

T = TypeVar("T")


@dataclass(frozen=True)
class IngestConfig:
    user_agent: str
    request_delay_sec: float
    max_retries: int
    timeout_sec: int
    fail_fast: bool


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str
    target_chars: int
    max_chars: int
    min_chars: int
    overlap_chars: int
    keep_tables_atomic: bool
    add_context_header: bool


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    batch_size: int
    normalize: bool
    cache: bool


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int
    context_chunks: int
    min_score: float
    distance: str
    filter_by_entity: bool
    fallback_unfiltered: bool


@dataclass(frozen=True)
class GenerationConfig:
    provider: str
    model: str
    temperature: float
    max_tokens: int
    max_sentences: int
    repair_attempts: int


@dataclass(frozen=True)
class GuardrailsConfig:
    pii_block: bool
    log_pii: bool
    intent_rules_only: bool


@dataclass(frozen=True)
class AppConfig:
    ingest: IngestConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    guardrails: GuardrailsConfig
    # Project root; every path below derives from it. Overridable in tests via
    # dataclasses.replace(cfg, root=tmp_path) - the dataclass is frozen.
    root: Path = PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "chunks.jsonl"

    @property
    def embeddings_path(self) -> Path:
        return self.data_dir / "embeddings.jsonl"

    @property
    def sources_csv(self) -> Path:
        return self.data_dir / "sources.csv"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.raw_dir, self.outputs_dir):
            path.mkdir(parents=True, exist_ok=True)


_SECTIONS: dict[str, type] = {
    "ingest": IngestConfig,
    "chunking": ChunkingConfig,
    "embedding": EmbeddingConfig,
    "retrieval": RetrievalConfig,
    "generation": GenerationConfig,
    "guardrails": GuardrailsConfig,
}


def _build(cls: type[T], values: dict[str, Any], path: str) -> T:
    if not is_dataclass(cls):
        raise TypeError(f"{path}: expected a dataclass type, got {cls!r}")
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ValueError(
            f"Unknown config key(s) in '{path}': {', '.join(unknown)}. Allowed: {', '.join(sorted(known))}"
        )
    # Fields with defaults (only AppConfig.root) may be omitted.
    missing = sorted(
        name
        for name, f in known.items()
        if name not in values and f.default is MISSING
    )
    if missing:
        raise ValueError(f"Missing required key(s) in '{path}': {', '.join(missing)}")
    return cls(**values)  # type: ignore[arg-type]


def _apply_env_overrides(cfg: AppConfig) -> AppConfig:
    """LLM settings may be supplied via environment; .env is loaded for convenience."""
    model = os.getenv("LLM_MODEL")
    if not model:
        return cfg
    return replace(cfg, generation=replace(cfg.generation, model=model))


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path}: top level must be a mapping, got {type(raw).__name__}")

    unknown_sections = sorted(set(raw) - set(_SECTIONS))
    if unknown_sections:
        raise ValueError(
            f"Unknown config section(s): {', '.join(unknown_sections)}. "
            f"Allowed: {', '.join(sorted(_SECTIONS))}"
        )
    missing_sections = sorted(set(_SECTIONS) - set(raw))
    if missing_sections:
        raise ValueError(f"Missing config section(s): {', '.join(missing_sections)}")

    section_args = {
        name: _build(cls, raw[name] or {}, name) for name, cls in _SECTIONS.items()
    }
    return _apply_env_overrides(AppConfig(**section_args))


@lru_cache(maxsize=4)
def get_config(path: str | None = None) -> AppConfig:
    """Cached accessor. Pass an explicit path string in tests to get a fresh load."""
    return load_config(path)


def llm_credentials() -> tuple[str | None, str | None]:
    """Returns (api_key, base_url) from the environment; neither is ever logged."""
    return os.getenv("LLM_API_KEY") or None, os.getenv("LLM_BASE_URL") or None
