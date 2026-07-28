"""Tests for building generation requests from retrieved graph facts."""

from __future__ import annotations

from legal_rag.baml_client.types import RetrievedChunkContext as BamlRetrievedChunkContext
from legal_rag.graphiti_rag.generator import generate_answer
from legal_rag.graphiti_rag.retriever import FactSource, RetrievedFact


class _FakeGenerationClient:
    """Record the call and return a fixed answer."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[BamlRetrievedChunkContext]]] = []

    def generate(self, query_vi: str, chunks: list[BamlRetrievedChunkContext]) -> str:
        """Record the call and return a fixed answer."""
        self.calls.append((query_vi, list(chunks)))
        return "Phạt tiền từ 4 đến 6 triệu đồng."


def _fact(chunk_id: str, text: str) -> RetrievedFact:
    """Build one single-source retrieved fact for generator tests."""
    return RetrievedFact(
        fact=text,
        sources=(FactSource(chunk_id=chunk_id, source_name="doc.docx"),),
    )


def test_generate_answer_builds_chunk_context_from_retrieved_facts() -> None:
    """Each retrieved fact becomes one RetrievedChunkContext passed to the client."""
    facts = [_fact("fact-1", "Nội dung 1"), _fact("fact-2", "Nội dung 2")]
    client = _FakeGenerationClient()

    answer = generate_answer("Mức phạt vượt đèn đỏ?", facts, client)

    assert answer == "Phạt tiền từ 4 đến 6 triệu đồng."
    [(query, chunks)] = client.calls
    assert query == "Mức phạt vượt đèn đỏ?"
    assert [chunk.chunk_id for chunk in chunks] == ["fact-1", "fact-2"]
    assert [chunk.text for chunk in chunks] == ["Nội dung 1", "Nội dung 2"]


def test_generate_answer_emits_one_context_per_supporting_source() -> None:
    """A fact backed by several episodes cites each supporting chunk, not just the first."""
    fact = RetrievedFact(
        fact="Nội dung dùng chung",
        sources=(
            FactSource(chunk_id="chunk-1", source_name="doc.docx"),
            FactSource(chunk_id="chunk-9", source_name="doc.docx"),
        ),
    )
    client = _FakeGenerationClient()

    generate_answer("Câu hỏi?", [fact], client)

    [(_, chunks)] = client.calls
    assert [chunk.chunk_id for chunk in chunks] == ["chunk-1", "chunk-9"]
    assert [chunk.text for chunk in chunks] == ["Nội dung dùng chung", "Nội dung dùng chung"]


def test_generate_answer_returns_empty_list_of_facts_unmodified() -> None:
    """No retrieved facts still produces a valid (empty) chunk context list."""
    client = _FakeGenerationClient()

    generate_answer("Câu hỏi không có ngữ cảnh?", [], client)

    [(_, chunks)] = client.calls
    assert chunks == []
