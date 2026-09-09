"""Tests for the ontology RAG query CLI."""

from __future__ import annotations

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("pymilvus", reason="pymilvus required for CLI tests")

from loguru import logger

from legal_rag import ontology_rag_query_cli
from legal_rag.ontology_rag import pipeline as pipeline_module


def _make_result() -> object:
    """Create a fake result for testing."""

    class FakeRule:
        uri = "rule:001"
        conduct_label_vi = "Không đội mũ bảo hiểm"
        vehicle_categories = ("Xe máy",)
        monetary_penalty_min = Decimal("400000")
        monetary_penalty_max = Decimal("600000")
        currency = "VND"
        point_value = 2
        provision_path = "Điều 5"
        document_number = "NĐ 17/2020"

    class FakeResult:
        query = "Không đội mũ bảo hiểm"
        answer = "Phạt tiền từ 400000 đến 600000 VND, trừ 2 điểm."
        rules = (FakeRule(),)
        sources = ()

    return FakeResult()


def test_query_cli_logs_answer_and_rules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Running the ontology query CLI logs the answer and penalty rules."""
    fake_result = _make_result()
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = fake_result

    monkeypatch.setenv("LLM_BASE_URL", "http://localhost/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("EMBEDDING_PROFILE", "bge_m3")
    monkeypatch.setenv("MILVUS_URI", "http://localhost:19530")
    monkeypatch.setenv("MILVUS_DATABASE", "legal_rag")

    ontology_file = tmp_path / "test.ttl"
    ontology_file.write_text("@prefix test: <http://test.org/> .")

    monkeypatch.setattr(
        sys, "argv", ["legal-ontology-rag-query", "--ontology", str(ontology_file), "Không đội mũ bảo hiểm"]
    )

    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="INFO")

    try:
        with patch.object(pipeline_module, "OntologyRAGPipeline", return_value=mock_pipeline):
            ontology_rag_query_cli.main()
    finally:
        logger.remove(sink_id)

    assert any("Phạt tiền từ 400000 đến 600000 VND, trừ 2 điểm." in message for message in messages)
    assert any("Không đội mũ bảo hiểm" in message for message in messages)


def test_query_cli_with_query_date(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Query CLI accepts query_date parameter."""
    fake_result = _make_result()
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = fake_result

    monkeypatch.setenv("LLM_BASE_URL", "http://localhost/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("EMBEDDING_PROFILE", "bge_m3")
    monkeypatch.setenv("MILVUS_URI", "http://localhost:19530")
    monkeypatch.setenv("MILVUS_DATABASE", "legal_rag")

    ontology_file = tmp_path / "test.ttl"
    ontology_file.write_text("@prefix test: <http://test.org/> .")

    monkeypatch.setattr(
        sys,
        "argv",
        ["legal-ontology-rag-query", "--ontology", str(ontology_file), "--query-date", "2025-06-01", "test query"],
    )

    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="INFO")

    try:
        with patch.object(pipeline_module, "OntologyRAGPipeline", return_value=mock_pipeline):
            ontology_rag_query_cli.main()
    finally:
        logger.remove(sink_id)

    mock_pipeline.run.assert_called_once()
    call_kwargs = mock_pipeline.run.call_args[1]
    assert call_kwargs["query_date"] == date(2025, 6, 1)


def test_query_cli_invalid_query_date(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid query_date logs error and returns."""
    mock_pipeline = MagicMock()

    monkeypatch.setenv("LLM_BASE_URL", "http://localhost/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("EMBEDDING_PROFILE", "bge_m3")
    monkeypatch.setenv("MILVUS_URI", "http://localhost:19530")
    monkeypatch.setenv("MILVUS_DATABASE", "legal_rag")

    ontology_file = tmp_path / "test.ttl"
    ontology_file.write_text("@prefix test: <http://test.org/> .")

    monkeypatch.setattr(
        sys,
        "argv",
        ["legal-ontology-rag-query", "--ontology", str(ontology_file), "--query-date", "invalid", "test query"],
    )

    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="ERROR")

    try:
        with patch.object(pipeline_module, "OntologyRAGPipeline", return_value=mock_pipeline):
            ontology_rag_query_cli.main()
    finally:
        logger.remove(sink_id)

    assert any("Invalid query_date format" in message for message in messages)
    mock_pipeline.run.assert_not_called()
