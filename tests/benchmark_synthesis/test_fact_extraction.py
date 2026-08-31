"""Tests for chunk windowing and legal-fact extraction validation."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pytest

from legal_rag.baml_client.types import ChunkWindowRole, FactExtractionBatchItem, FactExtractionBatchResult
from legal_rag.baml_client.types import EvidenceEntry as BamlEvidenceEntry
from legal_rag.baml_client.types import LegalFact as BamlLegalFact
from legal_rag.benchmark_synthesis import fact_extraction
from legal_rag.benchmark_synthesis.config import SynthesisSettings
from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.benchmark_synthesis.fact_extraction import (
    BamlFactExtractionClient,
    build_extracted_fact,
    build_reading_window,
    extract_facts_from_document,
)
from legal_rag.benchmark_synthesis.models import LegalFact
from legal_rag.chunker.models import Chunk
from legal_rag.settings import Settings

_SYNTHESIS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "data" / "config" / "groundtruth" / "synthesis_v1.json"
_FACT_EXTRACTION_PROMPT_PATH = Path(__file__).resolve().parents[2] / "baml_src" / "legal_fact_extraction.baml"


def test_fact_extraction_prompt_prioritizes_ground_truth_precision() -> None:
    """The prompt rejects ambiguous facts without skipping clear neighboring rules."""
    prompt = _FACT_EXTRACTION_PROMPT_PATH.read_text(encoding="utf-8")

    assert "Thứ tự ưu tiên bắt buộc" in prompt
    assert "ưu tiên precision\n    hơn recall" in prompt
    assert "Việc bỏ một fact chưa rõ không ảnh hưởng các fact rõ ràng" in prompt
    assert "Lập inventory theo đúng thứ tự Target" in prompt
    assert "vế sau dấu chấm phẩy" in prompt
    assert "Inventory phải bao phủ cả mệnh đề hoàn chỉnh ở đầu hoặc cuối" in prompt
    assert 'Cấu trúc đối lập trạng thái như "chưa/không' in prompt
    assert 'nguồn\n      "A quản lý X; thực hiện Y;" phải bảo toàn cả X và Y' in prompt
    assert "Dẫn chiếu phụ thuộc là thành phần bắt buộc" in prompt
    assert "bỏ fact thay vì suy đoán hoặc xóa locator" in prompt
    assert "Khi chính Target chứa một locator cần thiết" in prompt
    assert "Không cần Target đồng thời nêu mức phạt" in prompt
    assert "Loại nhiệm vụ quản lý, tổ chức thực hiện" in prompt
    assert "Loại giáo dục tại xã, trường giáo dưỡng" in prompt
    assert "Loại danh sách thẩm quyền pha trộn nhiều lĩnh vực" in prompt
    assert "Không trả facts=[] chỉ vì Target không nêu chế tài" in prompt
    assert "lập ánh xạ locator -> chunk_id" in prompt
    assert "không cho\n      phép bỏ các quy tắc hợp lệ khác" in prompt
    assert "toàn bộ trường hợp trong danh sách" in prompt
    assert "Không xuất định nghĩa, bảng thuật ngữ" in prompt
    assert "không viết vòng tròn" in prompt
    assert "giữ nguyên mã đầy đủ gồm tiền tố, số và hậu tố" in prompt
    assert "quy tắc chung của pháp luật xử phạt vi phạm hành chính" in prompt
    assert "hạn chế hoặc\n      từ chối cấp gắn trực tiếp với tước quyền" in prompt
    assert "Mỗi điểm chữ cái hoặc trường hợp" in prompt
    assert "lập bảng kiểm nội bộ theo đúng thứ tự Target" in prompt
    assert "so sánh từng cặp fact" in prompt
    assert "Chọn tiêu đề gần nhất nằm trước" in prompt
    assert "Mọi căn cứ thu hồi đăng ký, giấy phép, phù hiệu" in prompt
    assert "Mỗi mệnh đề hiệu lực hoặc chuyển tiếp" in prompt
    assert 'Cấu trúc giới hạn như "chỉ A, không B"' in prompt
    assert "statement của câu đó bắt buộc nêu mã biển" in prompt
    assert 'Dòng chú thích dạng "Hình ... - Biển số X"' in prompt
    assert "tuyệt đối không xuất câu tác nghiệp không có mã" in prompt
    assert "đối chiếu ngược từng câu tác nghiệp về biển/làn" in prompt
    assert "Khi phân vân giữa giữ và bỏ, giữ quy tắc" not in prompt


def _metadata(**overrides: object) -> DocumentMetadataRecord:
    """Build one resolved document-metadata record for fact-extraction tests."""
    defaults: dict[str, object] = {
        "document_number": "168/2024/NĐ-CP",
        "valid_from": date(2025, 1, 1),
        "valid_to": None,
    }
    defaults.update(overrides)
    return DocumentMetadataRecord(**defaults)


def _chunk(
    chunk_id: str,
    text: str,
    *,
    previous_chunk_id: str | None = None,
    next_chunk_id: str | None = None,
    context_text: str = "",
    chunk_index: int = 0,
) -> Chunk:
    """Build one minimal chunk for fact-extraction tests."""
    return Chunk(
        chunk_id=chunk_id,
        content_sha256="content-hash",
        chunker_config_sha256="config-hash",
        scope_id="scope-id",
        chunk_index=chunk_index,
        kind="paragraph_group",
        source_name="168_2024_ND-CP.docx",
        source_sha256="source-hash",
        record_scope="root",
        parent_attachment_id=None,
        attachment_member_path=None,
        attachment_member_sha256=None,
        unit_ids=(f"unit-{chunk_id}",),
        source_refs=(f"word/document.xml#/body/{chunk_id}",),
        sequence_start=0,
        sequence_end=0,
        text=text,
        context_text=context_text,
        embedding_text=text,
        text_token_count=5,
        embedding_token_count=5,
        previous_chunk_id=previous_chunk_id,
        next_chunk_id=next_chunk_id,
    )


def test_build_reading_window_includes_target_preceding_and_following_chunks() -> None:
    """The local window follows document order and labels both directions explicitly."""
    preceding_two = _chunk("prev2", "Phần trước xa hơn.")
    preceding_one = _chunk("prev1", "Phần ngay trước.", previous_chunk_id="prev2")
    target = _chunk("target", "Nội dung mục tiêu.", previous_chunk_id="prev1", next_chunk_id="next")
    following = _chunk("next", "Phần sau.")
    chunks_by_id = {chunk.chunk_id: chunk for chunk in (preceding_two, preceding_one, target, following)}

    window = build_reading_window(target, chunks_by_id, max_preceding_chunks=2, max_following_chunks=1)

    assert [entry.chunk_id for entry in window] == ["prev2", "prev1", "target", "next"]
    roles_by_id = {entry.chunk_id: entry.role for entry in window}
    assert roles_by_id == {
        "target": ChunkWindowRole.Target,
        "prev2": ChunkWindowRole.Preceding,
        "prev1": ChunkWindowRole.Preceding,
        "next": ChunkWindowRole.Following,
    }


def test_build_reading_window_marks_adjacent_extraction_targets() -> None:
    """The model can distinguish independently processed neighbors from context-only chunks."""
    preceding = _chunk("preceding", "Phần trước.")
    target = _chunk("target", "Nội dung mục tiêu.", previous_chunk_id="preceding", next_chunk_id="following")
    following = _chunk("following", "Phần sau.")
    chunks_by_id = {chunk.chunk_id: chunk for chunk in (preceding, target, following)}

    window = build_reading_window(
        target,
        chunks_by_id,
        max_preceding_chunks=1,
        max_following_chunks=1,
        extraction_target_ids={"preceding", "target", "following"},
    )

    assert {entry.chunk_id: entry.role for entry in window} == {
        "preceding": ChunkWindowRole.PrecedingTarget,
        "target": ChunkWindowRole.Target,
        "following": ChunkWindowRole.FollowingTarget,
    }


def test_build_reading_window_keeps_structural_context_separate_from_source_text() -> None:
    """Parser-derived ancestry reaches the model without becoming evidence text."""
    target = _chunk(
        "target",
        "a) Không chấp hành hiệu lệnh.",
        context_text="[LEGAL_STRUCTURE 1.0.0]\nĐiều 6. Xử phạt người điều khiển xe ô tô\n[/LEGAL_STRUCTURE]",
    )

    entry = build_reading_window(
        target,
        {target.chunk_id: target},
        max_preceding_chunks=0,
        max_following_chunks=0,
    )[0]

    assert entry.context_text == target.context_text
    assert entry.text == target.text


def test_build_reading_window_adds_source_grounded_structural_context() -> None:
    """A distant source heading remains available as evidence without widening local neighbors."""
    context = "[LEGAL_STRUCTURE 1.0.0]\nChương II\nĐiều 6. Xử phạt người điều khiển xe ô tô\n[/LEGAL_STRUCTURE]"
    chapter = _chunk("chapter", "Chương II\n\nQuy định xử phạt", context_text=context, chunk_index=0)
    heading = _chunk(
        "heading",
        "Điều 6. Xử phạt người điều khiển xe ô tô\n\n1. Phạt tiền đối với hành vi sau:",
        context_text=context,
        chunk_index=1,
    )
    distant = _chunk("distant", "Nội dung ở giữa.", context_text=context, chunk_index=2)
    near = _chunk("near", "Phần ngay trước.", previous_chunk_id="distant", context_text=context, chunk_index=3)
    target = _chunk(
        "target",
        "Thực hiện hành vi tại khoản 1 Điều này bị trừ điểm.",
        previous_chunk_id="near",
        context_text=context,
        chunk_index=4,
    )
    chunks_by_id = {chunk.chunk_id: chunk for chunk in (chapter, heading, distant, near, target)}

    window = build_reading_window(target, chunks_by_id, max_preceding_chunks=1, max_following_chunks=0)

    assert [entry.chunk_id for entry in window] == ["heading", "chapter", "near", "target"]
    assert {entry.chunk_id: entry.role for entry in window} == {
        "heading": ChunkWindowRole.ArticleContext,
        "chapter": ChunkWindowRole.ArticleContext,
        "near": ChunkWindowRole.Preceding,
        "target": ChunkWindowRole.Target,
    }


def test_build_reading_window_keeps_prior_context_at_a_structural_boundary() -> None:
    """A target containing the next heading still receives the prior heading from its context variants."""
    context = (
        "[LEGAL_STRUCTURE_VARIANTS 1.0.0]\nĐiều 7. Quy tắc trước\nĐiều 8. Quy tắc sau\n[/LEGAL_STRUCTURE_VARIANTS]"
    )
    prior_heading = _chunk("prior", "Điều 7. Quy tắc trước", context_text=context, chunk_index=1)
    target = _chunk(
        "target",
        "Khoản cuối của Điều này.\n\nĐiều 8. Quy tắc sau",
        context_text=context,
        chunk_index=4,
    )

    window = build_reading_window(
        target,
        {prior_heading.chunk_id: prior_heading, target.chunk_id: target},
        max_preceding_chunks=0,
        max_following_chunks=0,
    )

    assert [entry.chunk_id for entry in window] == ["prior", "target"]
    assert window[0].role == ChunkWindowRole.ArticleContext


def test_build_reading_window_limits_preceding_chunk_depth() -> None:
    """Walking back through previous_chunk_id stops after max_preceding_chunks."""
    far = _chunk("far", "Xa nhất.")
    middle = _chunk("middle", "Giữa.", previous_chunk_id="far")
    near = _chunk("near", "Gần nhất.", previous_chunk_id="middle")
    target = _chunk("target", "Điều 5.", previous_chunk_id="near")
    chunks_by_id = {chunk.chunk_id: chunk for chunk in (far, middle, near, target)}

    window = build_reading_window(target, chunks_by_id, max_preceding_chunks=2, max_following_chunks=1)

    ids = {entry.chunk_id for entry in window}
    assert ids == {"target", "near", "middle"}


def test_build_reading_window_limits_following_chunk_depth() -> None:
    """Walking forward through next_chunk_id stops after max_following_chunks."""
    target = _chunk("target", "Điều 5.", next_chunk_id="near")
    near = _chunk("near", "Gần nhất.", next_chunk_id="middle")
    middle = _chunk("middle", "Giữa.", next_chunk_id="far")
    far = _chunk("far", "Xa nhất.")
    chunks_by_id = {chunk.chunk_id: chunk for chunk in (target, near, middle, far)}

    window = build_reading_window(target, chunks_by_id, max_preceding_chunks=2, max_following_chunks=2)

    ids = {entry.chunk_id for entry in window}
    assert ids == {"target", "near", "middle"}


def _baml_fact(**overrides: object) -> BamlLegalFact:
    """Build one generated BAML legal fact for validation tests."""
    defaults: dict[str, object] = {
        "statement": "Người điều khiển ô tô không chấp hành đèn tín hiệu bị phạt tiền và trừ điểm.",
        "evidence": [BamlEvidenceEntry(chunk_id="chunk-1")],
    }
    defaults.update(overrides)
    return BamlLegalFact(**defaults)


def _build_extracted(raw: BamlLegalFact | None = None, **metadata_overrides: object) -> LegalFact:
    """Build one fact against source text containing the default exact quote."""
    return build_extracted_fact(
        raw or _baml_fact(),
        "chunk-1",
        _metadata(**metadata_overrides),
        {"chunk-1": "Mức phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng."},
    )


def test_build_extracted_fact_keeps_llm_cited_evidence_chunk() -> None:
    """The persisted fact keeps the source chunk selected by the LLM."""
    fact = _build_extracted()

    assert fact.statement == "Người điều khiển ô tô không chấp hành đèn tín hiệu bị phạt tiền và trừ điểm."
    assert fact.evidence[0].chunk_id == "chunk-1"
    assert fact.source_chunk_id == "chunk-1"


def test_build_extracted_fact_stamps_document_number_from_metadata() -> None:
    """document_number comes from the resolved document metadata, never from the reading window."""
    fact = _build_extracted(document_number="336/2025/NĐ-CP")

    assert fact.document_number == "336/2025/NĐ-CP"


def test_build_extracted_fact_falls_back_to_document_effective_dates() -> None:
    """A fact whose window states no effective date inherits the document's."""
    fact = _build_extracted(valid_to=date(2027, 12, 31))

    assert fact.valid_from == date(2025, 1, 1)
    assert fact.valid_to == date(2027, 12, 31)


def test_build_extracted_fact_identity_is_stable_for_identical_content() -> None:
    """Identical statements and provenance produce the same content-addressed identity."""
    first = _build_extracted()
    second = _build_extracted()

    assert first.fact_id == second.fact_id


def test_build_extracted_fact_supports_general_legal_statements() -> None:
    """A general legal rule is represented directly by its self-contained statement."""
    fact = _build_extracted(
        _baml_fact(statement="Quyết định xử phạt phải được gửi cho người vi phạm trong thời hạn luật định.")
    )

    assert fact.statement.startswith("Quyết định xử phạt")


def test_build_extracted_fact_identity_separates_documents_sharing_a_provision_path() -> None:
    """The same provision path in two documents yields two identities, not one merged fact."""
    first = _build_extracted()
    second = _build_extracted(document_number="336/2025/NĐ-CP")

    assert first.fact_id != second.fact_id


def test_build_extracted_fact_identity_changes_with_legal_statement() -> None:
    """Two distinct atomic legal propositions never collapse into one fact identity."""
    first = _build_extracted()
    second = _build_extracted(_baml_fact(statement="Quyết định xử phạt phải được gửi đúng hạn."))

    assert first.fact_id != second.fact_id


def test_build_extracted_fact_rejects_empty_evidence() -> None:
    """A model result without evidence cannot become a persisted fact."""
    with pytest.raises(ValueError, match="at least one evidence chunk"):
        build_extracted_fact(_baml_fact(evidence=[]), "chunk-1", _metadata(), {"chunk-1": "Nội dung."})


def test_build_extracted_fact_rejects_evidence_without_target_chunk() -> None:
    """A context-only citation cannot satisfy target ownership provenance."""
    raw = _baml_fact(evidence=[BamlEvidenceEntry(chunk_id="context")])

    with pytest.raises(ValueError, match="must cite its target chunk"):
        build_extracted_fact(raw, "target", _metadata(), {"context": None, "target": None})


def test_build_extracted_fact_rejects_non_latin_script_text() -> None:
    """Foreign-script model contamination is retried instead of persisted."""
    raw = _baml_fact(statement="Người lái xe phải dừng lại तथा chấp hành hiệu lệnh.")

    with pytest.raises(ValueError, match="Latin-script Vietnamese"):
        _build_extracted(raw)


class _FakeExtractionClient:
    """Return pre-programmed facts for every target in a requested batch."""

    def __init__(self, facts_by_target: dict[str, tuple[BamlLegalFact, ...]]) -> None:
        self._facts_by_target = facts_by_target
        self.batch_sizes: list[int] = []

    def extract_batch(self, items: Sequence[FactExtractionBatchItem]) -> Sequence[FactExtractionBatchResult]:
        """Return one result for each requested target."""
        self.batch_sizes.append(len(items))
        return tuple(
            FactExtractionBatchResult(
                target_chunk_id=item.target_chunk_id,
                facts=list(self._facts_by_target.get(item.target_chunk_id, ())),
            )
            for item in items
        )


def test_extract_facts_from_document_collects_every_fact_per_target() -> None:
    """Every fact the client returns for a target chunk is collected."""
    target = _chunk("target", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng.")
    fact_one = _baml_fact(evidence=[BamlEvidenceEntry(chunk_id="target")])
    fact_two = _baml_fact(
        statement="Người điều khiển ô tô thực hiện hành vi khác bị phạt tiền.",
        evidence=[BamlEvidenceEntry(chunk_id="target")],
    )
    client = _FakeExtractionClient({"target": (fact_one, fact_two)})

    facts = extract_facts_from_document(
        [target],
        [target],
        client,
        _metadata(),
        max_preceding_chunks=2,
        max_following_chunks=1,
        batch_size=5,
        max_retries=2,
        retry_delay_seconds=0,
    )

    assert len(facts) == 2
    assert {fact.statement for fact in facts} == {fact_one.statement, fact_two.statement}


def test_extract_facts_from_document_batches_five_targets_per_request() -> None:
    """Six targets are sent as one batch of five and one final batch of one."""
    targets = tuple(_chunk(f"target-{index}", f"Nội dung {index}.") for index in range(6))
    client = _FakeExtractionClient({})

    extract_facts_from_document(
        targets,
        targets,
        client,
        _metadata(),
        max_preceding_chunks=1,
        max_following_chunks=1,
        batch_size=5,
        max_retries=2,
        retry_delay_seconds=0,
    )

    assert client.batch_sizes == [5, 1]


def test_extract_facts_from_document_runs_batches_concurrently_and_preserves_fact_order() -> None:
    """Configured extraction concurrency allows batches to overlap while final facts stay target ordered."""
    targets = (
        _chunk("target-1", "Phạt tiền từ 4.000.000 đồng."),
        _chunk("target-2", "Phạt tiền từ 6.000.000 đồng."),
    )
    barrier = threading.Barrier(2)
    calls: list[str] = []

    class _ConcurrentClient:
        def extract_batch(self, items: Sequence[FactExtractionBatchItem]) -> Sequence[FactExtractionBatchResult]:
            calls.append(items[0].target_chunk_id)
            barrier.wait(timeout=2)
            return tuple(
                FactExtractionBatchResult(
                    target_chunk_id=item.target_chunk_id,
                    facts=[
                        _baml_fact(
                            statement=f"Người điều khiển thực hiện hành vi {item.target_chunk_id} bị phạt tiền.",
                            evidence=[BamlEvidenceEntry(chunk_id=item.target_chunk_id)],
                        )
                    ],
                )
                for item in items
            )

    facts = extract_facts_from_document(
        targets,
        targets,
        _ConcurrentClient(),
        _metadata(),
        max_preceding_chunks=1,
        max_following_chunks=1,
        batch_size=1,
        max_concurrency=2,
        max_retries=0,
        retry_delay_seconds=0,
    )

    assert set(calls) == {"target-1", "target-2"}
    assert [fact.source_chunk_id for fact in facts] == ["target-1", "target-2"]


def test_extract_facts_from_document_retries_a_failing_batch_then_succeeds() -> None:
    """A batch whose extraction call raises is retried, and a later success produces its facts."""
    target = _chunk("target", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng.")
    fact = _baml_fact(evidence=[BamlEvidenceEntry(chunk_id="target")])
    attempts = 0

    class _FlakyClient:
        def extract_batch(self, items: Sequence[FactExtractionBatchItem]) -> Sequence[FactExtractionBatchResult]:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                msg = "provider timed out"
                raise TimeoutError(msg)
            return (FactExtractionBatchResult(target_chunk_id=items[0].target_chunk_id, facts=[fact]),)

    facts = extract_facts_from_document(
        [target],
        [target],
        _FlakyClient(),
        _metadata(),
        max_preceding_chunks=2,
        max_following_chunks=1,
        batch_size=5,
        max_retries=2,
        retry_delay_seconds=0,
    )

    assert attempts == 2
    assert len(facts) == 1
    assert facts[0].source_chunk_id == "target"


def test_extract_facts_from_document_leaves_failed_batch_incomplete_and_continues() -> None:
    """An exhausted batch stays incomplete while a later batch is still persisted."""
    failing = tuple(_chunk(f"chunk-fail-{index}", "Nội dung gây lỗi mạng.") for index in range(5))
    ok = _chunk("chunk-ok", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng.")
    fact = _baml_fact(evidence=[BamlEvidenceEntry(chunk_id="chunk-ok")])

    class _FailsForFirstBatchClient:
        def extract_batch(self, items: Sequence[FactExtractionBatchItem]) -> Sequence[FactExtractionBatchResult]:
            if len(items) == 5:
                msg = "provider timed out"
                raise TimeoutError(msg)
            return (FactExtractionBatchResult(target_chunk_id="chunk-ok", facts=[fact]),)

    persisted: list[tuple[str, ...]] = []
    with pytest.raises(RuntimeError, match="fact extraction left 1 incomplete"):
        extract_facts_from_document(
            (*failing, ok),
            (*failing, ok),
            _FailsForFirstBatchClient(),
            _metadata(),
            max_preceding_chunks=2,
            max_following_chunks=1,
            batch_size=5,
            max_retries=2,
            retry_delay_seconds=0,
            on_batch_processed=lambda results: persisted.append(tuple(chunk_id for chunk_id, _ in results)),
        )

    assert persisted == [("chunk-ok",)]


def test_extract_facts_from_document_orders_batch_results_by_requested_target() -> None:
    """Out-of-order model results are normalized before persistence."""
    target_one = _chunk("target-1", "Phạt tiền 4.000.000 đồng.")
    target_two = _chunk("target-2", "Trừ 6 điểm giấy phép lái xe.")
    fact_one = _baml_fact(evidence=[BamlEvidenceEntry(chunk_id="target-1")])

    class _ReverseClient:
        def extract_batch(self, items: Sequence[FactExtractionBatchItem]) -> Sequence[FactExtractionBatchResult]:
            return (
                FactExtractionBatchResult(target_chunk_id="target-2", facts=[]),
                FactExtractionBatchResult(target_chunk_id="target-1", facts=[fact_one]),
            )

    calls: list[tuple[str, int]] = []

    extract_facts_from_document(
        [target_one, target_two],
        [target_one, target_two],
        _ReverseClient(),
        _metadata(),
        max_preceding_chunks=2,
        max_following_chunks=1,
        batch_size=5,
        max_retries=2,
        retry_delay_seconds=0,
        on_batch_processed=lambda results: calls.extend((chunk_id, len(facts)) for chunk_id, facts in results),
    )

    assert calls == [("target-1", 1), ("target-2", 0)]


@pytest.mark.parametrize("invalid_results", [(), ("extra",), ("target", "target")])
def test_extract_facts_from_document_retries_invalid_batch_coverage(invalid_results: tuple[str, ...]) -> None:
    """Missing, extra, and duplicate target results are rejected inside the retry boundary."""
    target = _chunk("target", "Phạt tiền 4.000.000 đồng.")
    attempts = 0

    class _InvalidCoverageClient:
        def extract_batch(self, items: Sequence[FactExtractionBatchItem]) -> Sequence[FactExtractionBatchResult]:
            nonlocal attempts
            attempts += 1
            return tuple(FactExtractionBatchResult(target_chunk_id=chunk_id, facts=[]) for chunk_id in invalid_results)

    with pytest.raises(RuntimeError, match="fact extraction left 1 incomplete"):
        extract_facts_from_document(
            [target],
            [target],
            _InvalidCoverageClient(),
            _metadata(),
            max_preceding_chunks=1,
            max_following_chunks=1,
            batch_size=5,
            max_retries=1,
            retry_delay_seconds=0,
        )

    assert attempts == 2


def test_extract_facts_from_document_resolves_context_excluded_from_target_chunks() -> None:
    """A preceding chunk missing from target_chunks still comes from the full document chunk set."""
    preceding = _chunk("preceding", "Trong Nghị định này, xe cơ giới được hiểu là ô tô, mô tô.")
    target = _chunk(
        "target",
        "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng đối với xe cơ giới.",
        previous_chunk_id="preceding",
    )
    seen_roles: dict[str, ChunkWindowRole] = {}

    class _RecordingClient:
        def extract_batch(self, items: Sequence[FactExtractionBatchItem]) -> Sequence[FactExtractionBatchResult]:
            """Record every window entry's role by chunk_id and return no facts."""
            seen_roles.update({entry.chunk_id: entry.role for entry in items[0].window})
            return (FactExtractionBatchResult(target_chunk_id=items[0].target_chunk_id, facts=[]),)

    extract_facts_from_document(
        [target],
        [preceding, target],
        _RecordingClient(),
        _metadata(),
        max_preceding_chunks=2,
        max_following_chunks=1,
        batch_size=5,
        max_retries=2,
        retry_delay_seconds=0,
    )

    assert seen_roles == {"target": ChunkWindowRole.Target, "preceding": ChunkWindowRole.Preceding}


def test_extract_facts_from_document_marks_previously_processed_target_context() -> None:
    """Resume keeps ownership markers for accepted chunks omitted from remaining targets."""
    preceding = _chunk("preceding", "Quy tắc bắt đầu tại đây.")
    target = _chunk("target", "Phần tiếp nối và quy tắc độc lập.", previous_chunk_id="preceding")
    seen_roles: dict[str, ChunkWindowRole] = {}

    class _RecordingClient:
        def extract_batch(self, items: Sequence[FactExtractionBatchItem]) -> Sequence[FactExtractionBatchResult]:
            seen_roles.update({entry.chunk_id: entry.role for entry in items[0].window})
            return (FactExtractionBatchResult(target_chunk_id=items[0].target_chunk_id, facts=[]),)

    extract_facts_from_document(
        [target],
        [preceding, target],
        _RecordingClient(),
        _metadata(),
        max_preceding_chunks=1,
        max_following_chunks=0,
        batch_size=1,
        extraction_target_ids={"preceding", "target"},
        max_retries=0,
        retry_delay_seconds=0,
    )

    assert seen_roles == {"preceding": ChunkWindowRole.PrecedingTarget, "target": ChunkWindowRole.Target}


def _settings() -> Settings:
    """Build runtime settings for an offline BAML request."""
    return Settings(
        llm_base_url="http://localhost:20128/v1",
        llm_api_key="provider-test-key",
        llm_model="provider/test-model",
        llm_timeout_seconds=30,
    )


def _synthesis_settings() -> SynthesisSettings:
    """Build ground-truth synthesis settings for an offline BAML request."""
    return SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH)


def test_baml_fact_extraction_client_parses_typed_batch_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """The generated non-stream client returns typed results keyed by target chunk."""
    client = BamlFactExtractionClient(_settings(), _synthesis_settings())
    monkeypatch.setattr(
        fact_extraction,
        "execute_baml_request",
        lambda *args, **kwargs: (
            '[{"target_chunk_id":"c1","facts":[{'
            '"statement":"Người điều khiển ô tô vượt đèn đỏ bị phạt tiền",'
            '"evidence":[{"chunk_id":"c1"}]}]}]'
        ),
    )
    target = _chunk("c1", "Vượt đèn đỏ phạt 4.000.000 đồng.")
    window = build_reading_window(target, {"c1": target}, max_preceding_chunks=1, max_following_chunks=1)

    results = client.extract_batch([FactExtractionBatchItem(target_chunk_id="c1", window=list(window))])

    assert len(results) == 1
    assert results[0].target_chunk_id == "c1"
    assert results[0].facts[0].statement == "Người điều khiển ô tô vượt đèn đỏ bị phạt tiền"
    assert results[0].facts[0].evidence[0].chunk_id == "c1"
