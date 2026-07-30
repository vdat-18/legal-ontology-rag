"""Tests for chunk windowing and legal-fact extraction validation."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pytest

from legal_rag.baml_client.types import ChunkWindowRole
from legal_rag.baml_client.types import EvidenceEntry as BamlEvidenceEntry
from legal_rag.baml_client.types import LegalFact as BamlLegalFact
from legal_rag.benchmark_synthesis import fact_extraction
from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.benchmark_synthesis.fact_extraction import (
    BamlFactExtractionClient,
    build_extracted_fact,
    build_provision_index,
    build_reading_window,
    cited_articles,
    extract_facts_from_document,
)
from legal_rag.chunker.models import Chunk
from legal_rag.settings import EvaluationSettings, Settings


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
) -> Chunk:
    """Build one minimal chunk for fact-extraction tests."""
    return Chunk(
        chunk_id=chunk_id,
        content_sha256="content-hash",
        chunker_config_sha256="config-hash",
        scope_id="scope-id",
        chunk_index=0,
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
        context_text="",
        embedding_text=text,
        text_token_count=5,
        embedding_token_count=5,
        previous_chunk_id=previous_chunk_id,
        next_chunk_id=next_chunk_id,
    )


def test_build_provision_index_only_indexes_chunks_opening_an_article() -> None:
    """Only chunks whose text starts with an article heading are indexed."""
    heading = _chunk("c1", "Điều 6. Xử phạt người điều khiển xe ô tô.")
    continuation = _chunk("c2", "9. Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng.")
    unrelated = _chunk("c3", "Chương II. Quy định chung.")

    index = build_provision_index([heading, continuation, unrelated])

    assert index.chunk_ids_by_article == {"6": ("c1",)}


def test_cited_articles_excludes_own_heading() -> None:
    """A chunk citing its own article number is not a cross-reference."""
    chunk = _chunk("c1", "Điều 6. Áp dụng theo Điều 6 và tham chiếu Điều 5.")

    assert cited_articles(chunk) == frozenset({"5"})


def test_cited_articles_returns_empty_when_no_citation_present() -> None:
    """A chunk without any article mention has no cited articles."""
    chunk = _chunk("c1", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng.")

    assert cited_articles(chunk) == frozenset()


def test_build_reading_window_includes_target_preceding_following_and_cited_chunks() -> None:
    """The window combines target, preceding chunks, one following chunk, and citation-resolved chunks by role."""
    heading = _chunk("art5", "Điều 5. Thẩm quyền xử phạt.")
    preceding_two = _chunk("prev2", "Phần trước xa hơn.")
    preceding_one = _chunk("prev1", "Phần ngay trước.", previous_chunk_id="prev2")
    target = _chunk("target", "Áp dụng theo Điều 5.", previous_chunk_id="prev1", next_chunk_id="next")
    following = _chunk("next", "Phần sau.")
    chunks_by_id = {chunk.chunk_id: chunk for chunk in (heading, preceding_two, preceding_one, target, following)}
    provision_index = build_provision_index([heading, preceding_two, preceding_one, target, following])

    window = build_reading_window(target, chunks_by_id, provision_index)

    roles_by_id = {entry.chunk_id: entry.role for entry in window}
    assert roles_by_id == {
        "target": ChunkWindowRole.Target,
        "prev1": ChunkWindowRole.Adjacent,
        "prev2": ChunkWindowRole.Adjacent,
        "next": ChunkWindowRole.Adjacent,
        "art5": ChunkWindowRole.CitedProvision,
    }
    art5_entry = next(entry for entry in window if entry.chunk_id == "art5")
    assert art5_entry.provision_path == "Điều 5"


def test_build_reading_window_limits_preceding_chunk_depth() -> None:
    """Walking back through previous_chunk_id stops after max_preceding_chunks."""
    far = _chunk("far", "Xa nhất.")
    middle = _chunk("middle", "Giữa.", previous_chunk_id="far")
    near = _chunk("near", "Gần nhất.", previous_chunk_id="middle")
    target = _chunk("target", "Điều 5.", previous_chunk_id="near")
    chunks_by_id = {chunk.chunk_id: chunk for chunk in (far, middle, near, target)}
    provision_index = build_provision_index([far, middle, near, target])

    window = build_reading_window(target, chunks_by_id, provision_index, max_preceding_chunks=2)

    ids = {entry.chunk_id for entry in window}
    assert ids == {"target", "near", "middle"}


def test_build_reading_window_limits_following_chunk_depth() -> None:
    """Walking forward through next_chunk_id stops after max_following_chunks."""
    target = _chunk("target", "Điều 5.", next_chunk_id="near")
    near = _chunk("near", "Gần nhất.", next_chunk_id="middle")
    middle = _chunk("middle", "Giữa.", next_chunk_id="far")
    far = _chunk("far", "Xa nhất.")
    chunks_by_id = {chunk.chunk_id: chunk for chunk in (target, near, middle, far)}
    provision_index = build_provision_index([target, near, middle, far])

    window = build_reading_window(target, chunks_by_id, provision_index, max_following_chunks=2)

    ids = {entry.chunk_id for entry in window}
    assert ids == {"target", "near", "middle"}


def test_build_reading_window_deduplicates_chunk_appearing_in_multiple_roles() -> None:
    """A chunk that is both adjacent and cited only appears once, with its first role."""
    heading = _chunk("prev", "Điều 5. Thẩm quyền xử phạt.")
    target = _chunk("target", "Áp dụng theo Điều 5.", previous_chunk_id="prev")
    chunks_by_id = {chunk.chunk_id: chunk for chunk in (heading, target)}
    provision_index = build_provision_index([heading, target])

    window = build_reading_window(target, chunks_by_id, provision_index)

    assert len(window) == 2
    prev_entry = next(entry for entry in window if entry.chunk_id == "prev")
    assert prev_entry.role == ChunkWindowRole.Adjacent


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


def test_build_extracted_fact_keeps_llm_cited_evidence_as_is() -> None:
    """chunk_id and quoted_span in evidence come straight from the LLM, untouched."""
    fact = build_extracted_fact(_baml_fact(), "chunk-1", _metadata())

    assert fact.evidence[0].chunk_id == "chunk-1"
    assert fact.evidence[0].quoted_span == "phạt tiền từ 4.000.000 đồng"
    assert fact.source_chunk_id == "chunk-1"


def test_build_extracted_fact_stamps_document_number_from_metadata() -> None:
    """document_number comes from the resolved document metadata, never from the reading window."""
    fact = build_extracted_fact(_baml_fact(), "chunk-1", _metadata(document_number="336/2025/NĐ-CP"))

    assert fact.document_number == "336/2025/NĐ-CP"


def test_build_extracted_fact_falls_back_to_document_effective_dates() -> None:
    """A fact whose window states no effective date inherits the document's."""
    fact = build_extracted_fact(_baml_fact(), "chunk-1", _metadata(valid_to=date(2027, 12, 31)))

    assert fact.valid_from == date(2025, 1, 1)
    assert fact.valid_to == date(2027, 12, 31)


def test_build_extracted_fact_prefers_provision_effective_date_over_document_date() -> None:
    """A provision stating its own effective date keeps it instead of the document-wide one."""
    fact = build_extracted_fact(_baml_fact(valid_from="2026-01-01"), "chunk-1", _metadata())

    assert fact.valid_from == date(2026, 1, 1)


def test_build_extracted_fact_identity_is_stable_for_identical_identity_fields() -> None:
    """Two facts sharing conduct, subject, document_number, and provision_path share a fact_id."""
    first = build_extracted_fact(_baml_fact(), "chunk-1", _metadata())
    second = build_extracted_fact(_baml_fact(monetary_penalty="5.000.000 đồng"), "chunk-1", _metadata())

    assert first.fact_id == second.fact_id
    assert first.fact_content_sha256 != second.fact_content_sha256


def test_build_extracted_fact_identity_separates_documents_sharing_a_provision_path() -> None:
    """The same provision path in two documents yields two identities, not one merged fact."""
    first = build_extracted_fact(_baml_fact(), "chunk-1", _metadata())
    second = build_extracted_fact(_baml_fact(), "chunk-1", _metadata(document_number="336/2025/NĐ-CP"))

    assert first.fact_id != second.fact_id


def test_build_extracted_fact_trusts_llm_evidence_without_validation() -> None:
    """No defensive check runs; an empty evidence list simply produces empty evidence."""
    fact = build_extracted_fact(_baml_fact(evidence=[]), "chunk-1", _metadata())

    assert fact.evidence == ()


class _FakeExtractionClient:
    """Return pre-programmed facts keyed by the window's target chunk_id."""

    def __init__(self, facts_by_target: dict[str, tuple[BamlLegalFact, ...]]) -> None:
        self._facts_by_target = facts_by_target

    def extract(self, window: Sequence[object]) -> tuple[BamlLegalFact, ...]:
        """Return the facts configured for this window's target chunk."""
        target_chunk_id = next(entry.chunk_id for entry in window if entry.role == ChunkWindowRole.Target)
        return self._facts_by_target.get(target_chunk_id, ())


def test_extract_facts_from_document_collects_every_fact_per_target() -> None:
    """Every fact the client returns for a target chunk is collected."""
    target = _chunk("target", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng.")
    fact_one = _baml_fact(evidence=[BamlEvidenceEntry(chunk_id="target", quoted_span="phạt tiền")])
    fact_two = _baml_fact(
        conduct="Hành vi khác", evidence=[BamlEvidenceEntry(chunk_id="target", quoted_span="phạt tiền")]
    )
    client = _FakeExtractionClient({"target": (fact_one, fact_two)})

    facts = extract_facts_from_document([target], [target], client, _metadata())

    assert len(facts) == 2
    assert {fact.conduct for fact in facts} == {fact_one.conduct, fact_two.conduct}


def test_extract_facts_from_document_retries_a_failing_target_then_succeeds() -> None:
    """A target whose extraction call raises is retried, and a later success still produces its facts."""
    target = _chunk("target", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng.")
    fact = _baml_fact(evidence=[BamlEvidenceEntry(chunk_id="target", quoted_span="phạt tiền")])
    attempts = 0

    class _FlakyClient:
        def extract(self, window: Sequence[object]) -> tuple[BamlLegalFact, ...]:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                msg = "provider timed out"
                raise TimeoutError(msg)
            return (fact,)

    facts = extract_facts_from_document([target], [target], _FlakyClient(), _metadata())

    assert attempts == 2
    assert len(facts) == 1
    assert facts[0].source_chunk_id == "target"


def test_extract_facts_from_document_skips_a_target_that_fails_every_attempt() -> None:
    """A target that keeps raising is skipped, but other targets are still extracted."""
    failing = _chunk("chunk-fail", "Nội dung gây lỗi mạng.")
    ok = _chunk("chunk-ok", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng.")
    fact = _baml_fact(evidence=[BamlEvidenceEntry(chunk_id="chunk-ok", quoted_span="phạt tiền")])

    class _FailsForOneChunkClient:
        def extract(self, window: Sequence[object]) -> tuple[BamlLegalFact, ...]:
            target_chunk_id = next(entry.chunk_id for entry in window if entry.role == ChunkWindowRole.Target)
            if target_chunk_id == "chunk-fail":
                msg = "provider timed out"
                raise TimeoutError(msg)
            return (fact,)

    persisted: list[str] = []
    with pytest.raises(RuntimeError, match="fact extraction left 1 incomplete"):
        extract_facts_from_document(
            [failing, ok],
            [failing, ok],
            _FailsForOneChunkClient(),
            _metadata(),
            on_target_processed=lambda chunk_id, _facts: persisted.append(chunk_id),
        )

    assert persisted == ["chunk-ok"]


def test_extract_facts_from_document_calls_on_target_processed_after_each_target() -> None:
    """on_target_processed receives the chunk_id and that target's own facts, once per target."""
    target_one = _chunk("target-1", "Phạt tiền 4.000.000 đồng.")
    target_two = _chunk("target-2", "Trừ 6 điểm giấy phép lái xe.")
    fact_one = _baml_fact(evidence=[BamlEvidenceEntry(chunk_id="target-1", quoted_span="phạt")])
    client = _FakeExtractionClient({"target-1": (fact_one,), "target-2": ()})
    calls: list[tuple[str, int]] = []

    extract_facts_from_document(
        [target_one, target_two],
        [target_one, target_two],
        client,
        _metadata(),
        on_target_processed=lambda chunk_id, facts: calls.append((chunk_id, len(facts))),
    )

    assert calls == [("target-1", 1), ("target-2", 0)]


def test_extract_facts_from_document_resolves_context_excluded_from_target_chunks() -> None:
    """A preceding chunk missing from target_chunks is still resolved as Adjacent context from document_chunks."""
    preceding = _chunk("preceding", "Trong Nghị định này, xe cơ giới được hiểu là ô tô, mô tô.")
    target = _chunk(
        "target",
        "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng đối với xe cơ giới.",
        previous_chunk_id="preceding",
    )
    seen_roles: dict[str, ChunkWindowRole] = {}

    class _RecordingClient:
        def extract(self, window: Sequence[object]) -> tuple[BamlLegalFact, ...]:
            """Record every window entry's role by chunk_id and return no facts."""
            seen_roles.update({entry.chunk_id: entry.role for entry in window})
            return ()

    extract_facts_from_document([target], [preceding, target], _RecordingClient(), _metadata())

    assert seen_roles == {"target": ChunkWindowRole.Target, "preceding": ChunkWindowRole.Adjacent}


def _settings() -> Settings:
    """Build runtime settings for an offline BAML request."""
    return Settings(
        llm_base_url="http://localhost:20128/v1",
        llm_api_key="provider-test-key",
        llm_model="provider/test-model",
        llm_timeout_seconds=30,
    )


def _evaluation_settings() -> EvaluationSettings:
    """Build ground-truth synthesis settings for an offline BAML request."""
    return EvaluationSettings(
        embedding_profile="qwen3_0_6b",
        span_match_threshold=0.85,
        bundle_overlap_threshold=0.5,
        max_quality_retries=2,
    )


def test_baml_fact_extraction_client_parses_typed_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """The generated non-stream client returns typed candidate facts."""
    client = BamlFactExtractionClient(_settings(), _evaluation_settings())
    monkeypatch.setattr(
        fact_extraction,
        "execute_baml_request",
        lambda *args, **kwargs: (
            '[{"conduct":"Vượt đèn đỏ","subject":"Người điều khiển ô tô","conditions":[],'
            '"monetary_penalty":"4.000.000 đồng","license_points_deducted":null,'
            '"additional_penalties":[],"remedial_measures":[],"valid_from":null,"valid_to":null,'
            '"provision_path":"Điều 6",'
            '"evidence":[{"chunk_id":"c1","quoted_span":"4.000.000 đồng"}]}]'
        ),
    )
    target = _chunk("c1", "Vượt đèn đỏ phạt 4.000.000 đồng.")
    provision_index = build_provision_index([target])
    window = build_reading_window(target, {"c1": target}, provision_index)

    facts = client.extract(window)

    assert len(facts) == 1
    assert facts[0].conduct == "Vượt đèn đỏ"
    assert facts[0].evidence[0].chunk_id == "c1"
