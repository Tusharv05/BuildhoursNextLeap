"""Config loading: default load, unknown/missing key handling, env overrides."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mf_rag.config import (
    AppConfig,
    GenerationConfig,
    get_config,
    llm_credentials,
    load_config,
)


def test_loads_every_section(cfg: AppConfig) -> None:
    assert cfg.ingest.user_agent
    assert cfg.chunking.strategy in {"fixed", "section", "parent_child"}
    assert cfg.embedding.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.retrieval.top_k == 5
    assert cfg.generation.max_sentences == 3
    assert cfg.guardrails.pii_block is True
    assert cfg.guardrails.log_pii is False


def test_unknown_key_raises(project_root: Path, tmp_path: Path) -> None:
    raw = yaml.safe_load((project_root / "config.yaml").read_text(encoding="utf-8"))
    raw["retrieval"]["top_K"] = 5  # typo: capital K

    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown config key"):
        load_config(path)


def test_unknown_section_raises(project_root: Path, tmp_path: Path) -> None:
    raw = yaml.safe_load((project_root / "config.yaml").read_text(encoding="utf-8"))
    raw["retrival"] = raw.pop("retrieval")  # typo: missing 'e'

    path = tmp_path / "bad_section.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown config section"):
        load_config(path)


def test_missing_section_raises(project_root: Path, tmp_path: Path) -> None:
    raw = yaml.safe_load((project_root / "config.yaml").read_text(encoding="utf-8"))
    raw.pop("guardrails")

    path = tmp_path / "missing_section.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing config section"):
        load_config(path)


def test_missing_key_in_section_raises(project_root: Path, tmp_path: Path) -> None:
    raw = yaml.safe_load((project_root / "config.yaml").read_text(encoding="utf-8"))
    raw["embedding"].pop("normalize")

    path = tmp_path / "missing_key.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required key"):
        load_config(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_env_overrides_generation_model(project_root: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "local-llm")
    cfg = load_config(project_root / "config.yaml")
    assert isinstance(cfg.generation, GenerationConfig)
    assert cfg.generation.model == "local-llm"
    # Everything else is untouched.
    assert cfg.generation.max_sentences == 3
    assert cfg.retrieval.top_k == 5


def test_llm_credentials_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    key, base = llm_credentials()
    assert key == "test-key"
    assert base == "https://example.invalid/v1"


def test_get_config_is_cached() -> None:
    assert get_config() is get_config()
