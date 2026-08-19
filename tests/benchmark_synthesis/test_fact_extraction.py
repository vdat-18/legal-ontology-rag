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


def test_build_reading_window_adds_bounded_same_article_bookends() -> None:
    """Distant first and last chunks from the same article are bounded ArticleContext."""
    article_six = "[LEGAL_STRUCTURE 1.0.0]\nĐiều 6. Xử phạt người điều khiển ô tô\n[/LEGAL_STRUCTURE]"
    article_seven = "[LEGAL_STRUCTURE 1.0.0]\nĐiều 7. Xử phạt người điều khiển mô tô\n[/LEGAL_STRUCTURE]"
    opening = _chunk("opening", "Mức phạt chung.", context_text=article_six, chunk_index=0)
    preceding = _chunk("preceding", "Các hành vi sau đây:", context_text=article_six, chunk_index=1)
    target = _chunk(
        "target",
        "b) Không chấp hành hiệu lệnh.",
        previous_chunk_id="preceding",
        next_chunk_id="following",
        context_text=article_six,
        chunk_index=2,
    )
    following = _chunk("following", "c) Hành vi khác.", context_text=article_six, chunk_index=3)
    middle = _chunk("middle", "Nội dung giữa Điều.", context_text=article_six, chunk_index=4)
    closing = _chunk("closing", "Hình phạt bổ sung.", context_text=article_six, chunk_index=5)
    other = _chunk("other", "Điều khác.", context_text=article_seven, chunk_index=6)
    chunks = (opening, preceding, target, following, middle, closing, other)

    window = build_reading_window(
        target,
        {chunk.chunk_id: chunk for chunk in chunks},
        max_preceding_chunks=1,
        max_following_chunks=1,
        max_article_context_chunks=2,
    )

    assert [entry.chunk_id for entry in window] == ["preceding", "target", "following", "opening", "closing"]
    assert window[-2].role == ChunkWindowRole.ArticleContext
    assert window[-1].role == ChunkWindowRole.ArticleContext


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
        "conduct": "Không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
        "subject": "Người điều khiển ô tô",
        "conditions": [],
        "monetary_penalty": "4.000.000 đồng đến 6.000.000 đồng",
        "license_points_deducted": "4 điểm",
        "additional_penalties": [],
        "remedial_measures": [],
        "valid_from": None,
        "valid_to": None,
        "provision_path": "Điều 6 khoản 9 điểm b",
        "evidence": [BamlEvidenceEntry(chunk_id="chunk-1", quoted_span="phạt tiền từ 4.000.000 đồng")],
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


def test_build_extracted_fact_keeps_exact_llm_cited_evidence() -> None:
    """An already exact cited span remains unchanged."""
    fact = _build_extracted()

    assert fact.evidence[0].chunk_id == "chunk-1"
    assert fact.evidence[0].quoted_span == "phạt tiền từ 4.000.000 đồng"
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


def test_build_extracted_fact_prefers_provision_effective_date_over_document_date() -> None:
    """A provision stating its own effective date keeps it instead of the document-wide one."""
    fact = _build_extracted(_baml_fact(valid_from="2026-01-01"))

    assert fact.valid_from == date(2026, 1, 1)


def test_build_extracted_fact_identity_is_stable_for_identical_identity_fields() -> None:
    """Two facts sharing conduct, subject, document_number, and provision_path share a fact_id."""
    first = _build_extracted()
    second = _build_extracted(_baml_fact(monetary_penalty="5.000.000 đồng"))

    assert first.fact_id == second.fact_id
    assert first.fact_content_sha256 != second.fact_content_sha256


def test_build_extracted_fact_identity_separates_documents_sharing_a_provision_path() -> None:
    """The same provision path in two documents yields two identities, not one merged fact."""
    first = _build_extracted()
    second = _build_extracted(document_number="336/2025/NĐ-CP")

    assert first.fact_id != second.fact_id


def test_build_extracted_fact_splits_non_contiguous_paragraph_quotes() -> None:
    """Exact paragraphs separated by omitted source text become separate evidence references."""
    raw = _baml_fact(evidence=[BamlEvidenceEntry(chunk_id="chunk-1", quoted_span="Quy định chung.\n\nĐiểm b áp dụng.")])

    fact = build_extracted_fact(
        raw,
        "chunk-1",
        _metadata(),
        {"chunk-1": "Quy định chung.\n\nĐiểm a áp dụng.\n\nĐiểm b áp dụng."},
    )

    assert tuple(reference.quoted_span for reference in fact.evidence) == ("Quy định chung.", "Điểm b áp dụng.")


def test_build_extracted_fact_rejects_non_exact_evidence() -> None:
    """A quote that cannot be bound exactly raises inside the retry boundary."""
    with pytest.raises(ValueError, match="not exact source text"):
        build_extracted_fact(
            _baml_fact(),
            "chunk-1",
            _metadata(),
            {"chunk-1": "Nội dung hoàn toàn khác."},
        )


def test_build_extracted_fact_rejects_empty_evidence() -> None:
    """A model result without evidence cannot become a persisted fact."""
    with pytest.raises(ValueError, match="at least one evidence span"):
        build_extracted_fact(_baml_fact(evidence=[]), "chunk-1", _metadata(), {"chunk-1": "Nội dung."})


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
    fact_one = _baml_fact(evidence=[BamlEvidenceEntry(chunk_id="target", quoted_span="Phạt tiền")])
    fact_two = _baml_fact(
        conduct="Hành vi khác", evidence=[BamlEvidenceEntry(chunk_id="target", quoted_span="Phạt tiền")]
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
    assert {fact.conduct for fact in facts} == {fact_one.conduct, fact_two.conduct}


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
                            conduct=f"Hành vi {item.target_chunk_id}",
                            evidence=[BamlEvidenceEntry(chunk_id=item.target_chunk_id, quoted_span="Phạt tiền")],
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
    fact = _baml_fact(evidence=[BamlEvidenceEntry(chunk_id="target", quoted_span="Phạt tiền")])
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
    fact = _baml_fact(evidence=[BamlEvidenceEntry(chunk_id="chunk-ok", quoted_span="Phạt tiền")])

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
    fact_one = _baml_fact(evidence=[BamlEvidenceEntry(chunk_id="target-1", quoted_span="Phạt")])

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
            '[{"target_chunk_id":"c1","facts":[{"conduct":"Vượt đèn đỏ",'
            '"subject":"Người điều khiển ô tô","conditions":[],'
            '"monetary_penalty":"4.000.000 đồng","license_points_deducted":null,'
            '"additional_penalties":[],"remedial_measures":[],"valid_from":null,"valid_to":null,'
            '"provision_path":"Điều 6",'
            '"evidence":[{"chunk_id":"c1","quoted_span":"4.000.000 đồng"}]}]}]'
        ),
    )
    target = _chunk("c1", "Vượt đèn đỏ phạt 4.000.000 đồng.")
    window = build_reading_window(target, {"c1": target}, max_preceding_chunks=1, max_following_chunks=1)

    results = client.extract_batch([FactExtractionBatchItem(target_chunk_id="c1", window=list(window))])

    assert len(results) == 1
    assert results[0].target_chunk_id == "c1"
    assert results[0].facts[0].conduct == "Vượt đèn đỏ"
    assert results[0].facts[0].evidence[0].chunk_id == "c1"
