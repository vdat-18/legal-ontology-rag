"""Tests for Corrective RAG CLI failure paths."""

from __future__ import annotations

import argparse

import pytest

from legal_rag.corrective_rag.api import cli as crag_cli


def test_index_cli_exits_nonzero_when_config_loading_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Index command must fail the process when configuration is invalid."""
    monkeypatch.setattr(crag_cli, "RAGConfig", _FailingConfig)

    with pytest.raises(SystemExit) as exc_info:
        crag_cli.handle_index(argparse.Namespace(data_dir="data/processed/groundtruth"))

    assert exc_info.value.code == 1


def test_index_cli_exits_nonzero_when_indexing_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Index command must fail the process when Milvus indexing fails."""
    monkeypatch.setattr(crag_cli, "RAGConfig", _WorkingConfig)
    monkeypatch.setattr(crag_cli, "EmbeddingModel", _FakeEmbeddingModel)
    monkeypatch.setattr(crag_cli, "LegalVectorDB", _FailingVectorDb)

    with pytest.raises(SystemExit) as exc_info:
        crag_cli.handle_index(argparse.Namespace(data_dir="data/processed/groundtruth"))

    assert exc_info.value.code == 1


def test_query_cli_exits_nonzero_when_pipeline_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Query command must fail the process when the CRAG pipeline raises."""
    monkeypatch.setattr(crag_cli, "RAGConfig", _WorkingConfig)
    monkeypatch.setattr(crag_cli, "CorrectiveRAGPipeline", _FailingPipeline)

    with pytest.raises(SystemExit) as exc_info:
        crag_cli.handle_query(argparse.Namespace(query="legal question", limit=5))

    assert exc_info.value.code == 1


class _FailingConfig:
    """Config fake that fails like an invalid environment."""

    @classmethod
    def from_env(cls) -> object:
        """Fail during config loading."""
        raise _ConfigError


class _WorkingConfig:
    """Config fake returning fields used by the CLI constructors."""

    @classmethod
    def from_env(cls) -> object:
        """Return a minimal config object."""
        return _FakeConfig()


class _FakeConfig:
    """Minimal CRAG config fake."""

    embedding_profile = "bge_m3"


class _FakeEmbeddingModel:
    """Embedding fake that never loads sentence-transformers."""

    def __init__(self, profile_name: str) -> None:
        self.profile_name = profile_name
        self.closed = False

    def close(self) -> None:
        """Close fake resources."""
        self.closed = True


class _FailingVectorDb:
    """Vector DB fake that fails during indexing."""

    def __init__(self, config: object, embedder: _FakeEmbeddingModel) -> None:
        self.config = config
        self.embedder = embedder
        self.closed = False

    def index_processed_data(self, processed_dir: str) -> None:
        """Fail like a database/schema/indexing error."""
        raise _IndexingError

    def close(self) -> None:
        """Close fake resources."""
        self.closed = True


class _FailingPipeline:
    """Pipeline fake that fails during query execution."""

    def __init__(self, config: object) -> None:
        self.config = config

    def run(self, query: str, limit: int) -> dict[str, object]:
        """Fail like an upstream provider error."""
        raise _PipelineError

    def close(self) -> None:
        """Close fake resources."""


class _ConfigError(Exception):
    """Raised by the config fake."""


class _IndexingError(Exception):
    """Raised by the vector DB fake."""


class _PipelineError(Exception):
    """Raised by the pipeline fake."""
