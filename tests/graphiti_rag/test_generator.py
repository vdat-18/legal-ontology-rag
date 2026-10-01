"""Tests for building generation requests from retrieved graph facts."""

from __future__ import annotations

from datetime import date

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


def test_context_carries_the_source_passage_before_the_extracted_fact() -> None:
    """Quoting a provision needs its wording, which the extracted fact has already dropped."""
    client = _FakeGenerationClient()
    fact = RetrievedFact(
        fact="Phạt tiền từ 4.000.000 đến 6.000.000 đồng.",
        sources=(
            FactSource(
                chunk_id="chunk-1",
                source_name="168_2024_ND-CP.docx",
                text="Điều 7. Xử phạt người điều khiển xe mô tô, xe gắn máy...",
            ),
        ),
    )

    generate_answer("Xe máy vượt đèn đỏ phạt bao nhiêu?", [fact], client)

    [(_, chunks)] = client.calls
    assert chunks[0].text.startswith("Điều 7. Xử phạt người điều khiển xe mô tô")
    assert "[Trích xuất] Phạt tiền từ 4.000.000 đến 6.000.000 đồng." in chunks[0].text


def test_context_falls_back_to_the_fact_when_no_passage_was_stored() -> None:
    """An episode indexed without its content must still contribute its fact."""
    client = _FakeGenerationClient()
    fact = RetrievedFact(
        fact="Phạt tiền từ 4.000.000 đến 6.000.000 đồng.",
        sources=(FactSource(chunk_id="chunk-1", source_name="doc.docx"),),
    )

    generate_answer("q", [fact], client)

    [(_, chunks)] = client.calls
    assert chunks[0].text == "Phạt tiền từ 4.000.000 đến 6.000.000 đồng."


def test_context_labels_the_dates_at_which_a_fused_fact_was_retrieved() -> None:
    """Generation can associate a graph fact with each version date that returned it."""
    client = _FakeGenerationClient()
    fact = RetrievedFact(
        fact="Quy định chuyển tiếp",
        sources=(FactSource(chunk_id="chunk-1", source_name="doc.docx"),),
        as_of_dates=(date(2025, 1, 1), date(2026, 3, 1)),
    )

    generate_answer("So sánh quy định", [fact], client)

    [(_, chunks)] = client.calls
    assert chunks[0].text == ("[Có hiệu lực tại mốc truy hồi: 2025-01-01, 2026-03-01]\nQuy định chuyển tiếp")
