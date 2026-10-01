"""Tests for building generation requests and returning the generated answer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from legal_rag.baml_client.types import RetrievedChunkContext as BamlRetrievedChunkContext
from legal_rag.naive_rag import generator as generator_module
from legal_rag.naive_rag.generator import BamlGenerationClient, generate_answer
from legal_rag.naive_rag.retriever import RetrievedChunk
from legal_rag.settings import NaiveRagSettings, Settings


class _FakeGenerationClient:
    """Record the call and return a fixed answer."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[BamlRetrievedChunkContext]]] = []

    def generate(self, query_vi: str, chunks: list[BamlRetrievedChunkContext]) -> str:
        """Record the call and return a fixed answer."""
        self.calls.append((query_vi, list(chunks)))
        return "Phạt tiền từ 4 đến 6 triệu đồng."


def _hit(chunk_id: str, text: str) -> RetrievedChunk:
    """Build one retrieved chunk for generator tests."""
    return RetrievedChunk(
        chunk_id=chunk_id, text=text, source_name="168_2024_ND-CP.docx", kind="paragraph_group", score=0.9
    )


def test_generate_answer_builds_chunk_context_from_retrieved_hits() -> None:
    """Each retrieved chunk becomes one RetrievedChunkContext passed to the client."""
    hits = [_hit("chunk-1", "Nội dung 1"), _hit("chunk-2", "Nội dung 2")]
    client = _FakeGenerationClient()

    answer = generate_answer("Mức phạt vượt đèn đỏ?", hits, client)

    assert answer == "Phạt tiền từ 4 đến 6 triệu đồng."
    [(query, chunks)] = client.calls
    assert query == "Mức phạt vượt đèn đỏ?"
    assert [chunk.chunk_id for chunk in chunks] == ["chunk-1", "chunk-2"]
    assert [chunk.text for chunk in chunks] == ["Nội dung 1", "Nội dung 2"]


def test_generate_answer_returns_empty_list_of_hits_unmodified() -> None:
    """No retrieved chunks still produces a valid (empty) chunk context list."""
    client = _FakeGenerationClient()

    generate_answer("Câu hỏi không có ngữ cảnh?", [], client)

    [(_, chunks)] = client.calls
    assert chunks == []


def test_baml_generation_client_uses_shared_caller_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer generation forwards provider-specific settings through the shared bounded caller."""
    captured: dict[str, object] = {}
    fake_caller = SimpleNamespace(options={"client_registry": object()})

    class _FakeCaller:
        @classmethod
        def from_settings(cls, settings: Settings, **kwargs: object) -> object:
            captured["settings"] = settings
            captured.update(kwargs)
            return fake_caller

    settings = Settings(
        "http://provider.test/v1",
        "test-key",
        "test-model",
        30,
        llm_extra_request_body={"thinking": {"type": "disabled"}},
    )
    naive_settings = NaiveRagSettings(
        embedding_profile="bge_m3",
        answer_max_output_tokens=321,
        answer_temperature=0.2,
        max_response_bytes=654,
    )
    monkeypatch.setattr(generator_module, "BamlCaller", _FakeCaller)

    client = BamlGenerationClient(settings, naive_settings)

    assert client._caller is fake_caller
    assert captured == {
        "settings": settings,
        "max_output_tokens": 321,
        "temperature": 0.2,
        "max_response_bytes": 654,
    }
