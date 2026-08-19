"""Tests for typed LLM ontology extraction and evidence binding."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from legal_rag.baml_client.types import (
    ChunkWindowEntry,
    ChunkWindowRole,
    FactExtractionBatchItem,
    OntologyBoundRuleEntry,
    OntologyConsequenceEntry,
    OntologyConsequenceKind,
    OntologyCurrency,
    OntologyEvidenceEntry,
    OntologyPopulationBatchResult,
    OntologyPopulationBindingBatchResult,
    OntologyRuleEntry,
    OntologySubjectKind,
    OntologyTargetConductBatchResult,
    OntologyTargetConductEntry,
)
from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.benchmark_synthesis.llm_execution import IncompleteLlmTasksError, LlmTaskExhaustedError
from legal_rag.chunker.models import Chunk
from legal_rag.ontology_population.config import PopulationSettings
from legal_rag.ontology_population.extraction import (
    build_population_binding_items,
    build_population_reading_window,
    build_population_record,
    extract_population_from_document,
    finalize_population_binding_results,
)
from legal_rag.ontology_population.models import OntologyPopulationRecord

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _PopulationClient:
    """Return one typed rule for each requested target."""

    def __init__(self, quoted_span: str) -> None:
        self._quoted_span = quoted_span

    def extract_batch(self, items: Sequence[FactExtractionBatchItem]) -> Sequence[OntologyPopulationBatchResult]:
        """Build source-grounded typed BAML responses."""
        return tuple(
            OntologyPopulationBatchResult(
                target_chunk_id=item.target_chunk_id,
                rules=[
                    OntologyRuleEntry(
                        conduct_label_vi="Điều khiển xe không gắn biển số",
                        subject_kind=OntologySubjectKind.NaturalPerson,
                        role_labels_vi=["Người điều khiển"],
                        vehicle_category_labels_vi=["Xe mô tô"],
                        condition_root_id=None,
                        conditions=[],
                        consequences=[
                            OntologyConsequenceEntry(
                                consequence_id="money",
                                kind=OntologyConsequenceKind.MonetaryPenalty,
                                minimum_amount="4000000",
                                maximum_amount="6000000",
                                currency=OntologyCurrency.VND,
                                point_value=None,
                                consequence_label_vi=None,
                                minimum_duration_months=None,
                                maximum_duration_months=None,
                                target_label_vi=None,
                                responsible_role_label_vi=None,
                                evidence=[
                                    OntologyEvidenceEntry(chunk_id=item.target_chunk_id, quoted_span=self._quoted_span)
                                ],
                            )
                        ],
                        valid_from=None,
                        valid_to=None,
                        provision_path="Điều 13 khoản 9 điểm b",
                        evidence=[OntologyEvidenceEntry(chunk_id=item.target_chunk_id, quoted_span=self._quoted_span)],
                    )
                ],
            )
            for item in items
        )


def test_extraction_binds_evidence_and_uses_reviewed_document_validity(
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """Typed model output becomes deterministic source-grounded population records."""
    quoted_span = legal_chunk.text.split(". ", maxsplit=1)[0] + "."
    settings = replace(
        PopulationSettings.from_path(PROJECT_ROOT / "ontology" / "population_contract.yaml"),
        max_retries=0,
        retry_delay_seconds=0,
    )

    first = extract_population_from_document(
        (legal_chunk,), (legal_chunk,), _PopulationClient(quoted_span), document_metadata, settings
    )
    second = extract_population_from_document(
        (legal_chunk,), (legal_chunk,), _PopulationClient(quoted_span), document_metadata, settings
    )

    assert first == second
    assert first[0].valid_from == document_metadata.valid_from
    assert first[0].evidence[0].exact_text == quoted_span
    assert str(first[0].consequences[0].minimum_amount) == "4000000"


def test_extraction_rejects_non_exact_model_evidence(
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """A model paraphrase cannot enter the population record."""
    settings = replace(
        PopulationSettings.from_path(PROJECT_ROOT / "ontology" / "population_contract.yaml"),
        max_retries=0,
        retry_delay_seconds=0,
    )

    with pytest.raises(LlmTaskExhaustedError) as error:
        extract_population_from_document(
            (legal_chunk,),
            (legal_chunk,),
            _PopulationClient("Nội dung do mô hình tự diễn giải."),
            document_metadata,
            settings,
        )
    assert "not exact source text" in str(error.value.__cause__)


def test_extraction_runs_batches_concurrently_and_preserves_target_order(
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """Concurrent completion cannot change final population-record ordering."""
    targets = (
        replace(legal_chunk, chunk_id="target-1", chunk_index=0, next_chunk_id="target-2"),
        replace(legal_chunk, chunk_id="target-2", chunk_index=1, previous_chunk_id="target-1"),
    )
    barrier = threading.Barrier(2)
    second_checkpointed = threading.Event()
    calls: list[str] = []
    completion_order: list[str] = []
    quoted_span = legal_chunk.text.split(". ", maxsplit=1)[0] + "."

    class _ConcurrentPopulationClient(_PopulationClient):
        def extract_batch(self, items: Sequence[FactExtractionBatchItem]) -> Sequence[OntologyPopulationBatchResult]:
            target_id = items[0].target_chunk_id
            calls.append(target_id)
            barrier.wait(timeout=2)
            if target_id == "target-1":
                assert second_checkpointed.wait(timeout=2)
            return super().extract_batch(items)

    def checkpoint(batch_results: Sequence[tuple[str, tuple[OntologyPopulationRecord, ...]]]) -> None:
        """Record main-thread checkpoint order and release the first target."""
        target_id = batch_results[0][0]
        completion_order.append(target_id)
        if target_id == "target-2":
            second_checkpointed.set()

    settings = replace(
        PopulationSettings.from_path(PROJECT_ROOT / "ontology" / "population_contract.yaml"),
        max_retries=0,
        retry_delay_seconds=0,
    )

    records = extract_population_from_document(
        targets,
        targets,
        _ConcurrentPopulationClient(quoted_span),
        document_metadata,
        settings,
        max_concurrency=2,
        on_batch_processed=checkpoint,
    )

    assert set(calls) == {"target-1", "target-2"}
    assert completion_order == ["target-2", "target-1"]
    assert [record.source_chunk_id for record in records] == ["target-1", "target-2"]


def test_extraction_reports_all_concurrent_exhausted_batches(
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """Independent failures are reported together after successful checkpoints finish."""
    targets = (
        replace(legal_chunk, chunk_id="target-1", chunk_index=0),
        replace(legal_chunk, chunk_id="target-2", chunk_index=1),
    )
    settings = replace(
        PopulationSettings.from_path(PROJECT_ROOT / "ontology" / "population_contract.yaml"),
        max_retries=0,
        retry_delay_seconds=0,
    )

    with pytest.raises(IncompleteLlmTasksError) as error:
        extract_population_from_document(
            targets,
            targets,
            _PopulationClient("Nội dung không có trong source."),
            document_metadata,
            settings,
            max_concurrency=2,
        )

    assert len(error.value.failures) == 2


def test_population_window_includes_bounded_same_article_context(legal_chunk: Chunk) -> None:
    """Distant consequence catalogs in the same article reach the population model."""
    article_six = "[LEGAL_STRUCTURE 1.0.0]\nĐiều 6. Quy tắc A\n[/LEGAL_STRUCTURE]"
    article_seven = "[LEGAL_STRUCTURE 1.0.0]\nĐiều 7. Quy tắc B\n[/LEGAL_STRUCTURE]"
    target = replace(
        legal_chunk,
        chunk_id="target",
        chunk_index=2,
        context_text=article_six,
        previous_chunk_id="near",
    )
    near = replace(legal_chunk, chunk_id="near", chunk_index=1, context_text=article_six)
    distant = replace(legal_chunk, chunk_id="distant", chunk_index=8, context_text=article_six)
    other_article = replace(legal_chunk, chunk_id="other", chunk_index=9, context_text=article_seven)

    window = build_population_reading_window(
        target,
        {chunk.chunk_id: chunk for chunk in (near, target, distant, other_article)},
        max_preceding_chunks=1,
        max_following_chunks=0,
        max_article_context_chunks=3,
    )

    assert [entry.chunk_id for entry in window] == ["near", "target", "distant"]
    assert {entry.chunk_id: entry.role for entry in window} == {
        "near": ChunkWindowRole.Preceding,
        "target": ChunkWindowRole.Target,
        "distant": ChunkWindowRole.ArticleContext,
    }


def test_population_window_rejects_unbounded_article_context(legal_chunk: Chunk) -> None:
    """A large article fails explicitly instead of silently dropping distant consequences."""
    context = "[LEGAL_STRUCTURE 1.0.0]\nĐiều 6. Quy tắc A\n[/LEGAL_STRUCTURE]"
    target = replace(legal_chunk, chunk_id="target", chunk_index=0, context_text=context)
    distant = replace(legal_chunk, chunk_id="distant", chunk_index=1, context_text=context)

    with pytest.raises(ValueError, match="exceeding configured maximum"):
        build_population_reading_window(
            target,
            {target.chunk_id: target, distant.chunk_id: distant},
            max_preceding_chunks=0,
            max_following_chunks=0,
            max_article_context_chunks=1,
        )


def test_population_window_uses_local_context_for_structural_variants(legal_chunk: Chunk) -> None:
    """A boundary target does not mix same-article context from multiple variants."""
    variant_context = "\n".join(
        (
            "[LEGAL_STRUCTURE_VARIANTS 1.0.0]",
            "Điều A. Quy tắc thứ nhất",
            "Điều B. Quy tắc thứ hai",
            "[/LEGAL_STRUCTURE_VARIANTS]",
        )
    )
    single_context = "[LEGAL_STRUCTURE 1.0.0]\nĐiều A. Quy tắc thứ nhất\n[/LEGAL_STRUCTURE]"
    target = replace(
        legal_chunk,
        chunk_id="target",
        chunk_index=1,
        context_text=variant_context,
        previous_chunk_id="near",
    )
    near = replace(legal_chunk, chunk_id="near", chunk_index=0, context_text=single_context)
    distant = replace(legal_chunk, chunk_id="distant", chunk_index=5, context_text=single_context)

    window = build_population_reading_window(
        target,
        {chunk.chunk_id: chunk for chunk in (near, target, distant)},
        max_preceding_chunks=1,
        max_following_chunks=0,
        max_article_context_chunks=3,
    )

    assert {entry.chunk_id for entry in window} == {"near", "target"}
    assert all(entry.role != ChunkWindowRole.ArticleContext for entry in window)


def test_population_binding_inventory_requires_target_only_evidence(legal_chunk: Chunk) -> None:
    """A conduct inventory cannot cite context that the inventory stage never receives."""
    item, inventory, _ = _target_inventory_fixture(legal_chunk)
    inventory.conducts[0].evidence = [OntologyEvidenceEntry(chunk_id="context", quoted_span=legal_chunk.text)]

    with pytest.raises(KeyError, match="context"):
        build_population_binding_items((item,), (inventory,))


def test_population_binding_rejects_conduct_outside_locked_inventory(legal_chunk: Chunk) -> None:
    """The consequence stage cannot create a context-only conduct."""
    item, inventory, evidence = _target_inventory_fixture(legal_chunk)
    binding_item = build_population_binding_items((item,), (inventory,))[0]
    result = OntologyPopulationBindingBatchResult(
        target_chunk_id=legal_chunk.chunk_id,
        rules=[
            OntologyBoundRuleEntry(
                target_conduct_id="conduct_from_context",
                rule=_rule_with_consequence(_money_consequence(evidence), evidence),
            )
        ],
    )

    with pytest.raises(ValueError, match="locked Target conduct"):
        finalize_population_binding_results((binding_item,), (result,))


@pytest.mark.parametrize(
    ("conduct_label", "provision_path", "message"),
    [
        ("Hành vi bị đổi tên", "Điều kiểm tra", "cannot rename"),
        ("Hành vi được kiểm tra", "Điều khác", "cannot change"),
    ],
)
def test_population_binding_preserves_locked_conduct_identity(
    conduct_label: str,
    provision_path: str,
    message: str,
    legal_chunk: Chunk,
) -> None:
    """The consequence stage cannot reinterpret the Target label or locator."""
    item, inventory, evidence = _target_inventory_fixture(legal_chunk)
    binding_item = build_population_binding_items((item,), (inventory,))[0]
    rule = _rule_with_consequence(_money_consequence(evidence), evidence)
    rule.conduct_label_vi = conduct_label
    rule.provision_path = provision_path
    result = OntologyPopulationBindingBatchResult(
        target_chunk_id=legal_chunk.chunk_id,
        rules=[OntologyBoundRuleEntry(target_conduct_id="conduct_1", rule=rule)],
    )

    with pytest.raises(ValueError, match=message):
        finalize_population_binding_results((binding_item,), (result,))


def test_population_binding_returns_existing_population_result(legal_chunk: Chunk) -> None:
    """Two-stage extraction keeps the downstream population result API unchanged."""
    item, inventory, evidence = _target_inventory_fixture(legal_chunk)
    binding_item = build_population_binding_items((item,), (inventory,))[0]
    rule = _rule_with_consequence(_money_consequence(evidence), evidence)
    result = OntologyPopulationBindingBatchResult(
        target_chunk_id=legal_chunk.chunk_id,
        rules=[OntologyBoundRuleEntry(target_conduct_id="conduct_1", rule=rule)],
    )

    finalized = finalize_population_binding_results((binding_item,), (result,))

    assert finalized == (OntologyPopulationBatchResult(target_chunk_id=legal_chunk.chunk_id, rules=[rule]),)


def test_population_record_requires_target_conduct_evidence(
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """Context-only conduct evidence cannot create a rule for the target chunk."""
    context_chunk_id = "context"
    evidence = OntologyEvidenceEntry(chunk_id=context_chunk_id, quoted_span=legal_chunk.text)
    raw = _rule_with_consequence(
        OntologyConsequenceEntry(
            consequence_id="money",
            kind=OntologyConsequenceKind.MonetaryPenalty,
            minimum_amount="4000000",
            maximum_amount="6000000",
            currency=OntologyCurrency.VND,
            point_value=None,
            consequence_label_vi=None,
            minimum_duration_months=None,
            maximum_duration_months=None,
            target_label_vi=None,
            responsible_role_label_vi=None,
            evidence=[evidence],
        ),
        evidence,
    )

    with pytest.raises(ValueError, match="target chunk"):
        build_population_record(
            raw,
            legal_chunk.chunk_id,
            document_metadata,
            {context_chunk_id: legal_chunk.text, legal_chunk.chunk_id: legal_chunk.text},
        )


def test_population_record_rejects_remedial_measure_without_responsible_role(
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """An incomplete remedial measure fails before persisted RDF validation."""
    evidence = OntologyEvidenceEntry(chunk_id=legal_chunk.chunk_id, quoted_span=legal_chunk.text)
    raw = OntologyRuleEntry(
        conduct_label_vi="Vi phạm hành chính nói chung",
        subject_kind=None,
        role_labels_vi=[],
        vehicle_category_labels_vi=[],
        condition_root_id=None,
        conditions=[],
        consequences=[
            OntologyConsequenceEntry(
                consequence_id="remedy",
                kind=OntologyConsequenceKind.RemedialMeasure,
                minimum_amount=None,
                maximum_amount=None,
                currency=None,
                point_value=None,
                consequence_label_vi="Buộc khôi phục tình trạng ban đầu",
                minimum_duration_months=None,
                maximum_duration_months=None,
                target_label_vi="tình trạng ban đầu",
                responsible_role_label_vi=None,
                evidence=[evidence],
            )
        ],
        valid_from=None,
        valid_to=None,
        provision_path="Điều 3 khoản 3 điểm a",
        evidence=[evidence],
    )

    with pytest.raises(ValueError, match="requires a source-grounded responsible role"):
        build_population_record(
            raw,
            legal_chunk.chunk_id,
            document_metadata,
            {legal_chunk.chunk_id: legal_chunk.text},
        )


@pytest.mark.parametrize("kind", [OntologyConsequenceKind.AdditionalSanction, OntologyConsequenceKind.RemedialMeasure])
def test_population_record_requires_named_non_numeric_consequence(
    kind: OntologyConsequenceKind,
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """Non-numeric consequences retain the source-grounded legal effect name."""
    evidence = OntologyEvidenceEntry(chunk_id=legal_chunk.chunk_id, quoted_span=legal_chunk.text)
    raw = _rule_with_consequence(
        OntologyConsequenceEntry(
            consequence_id="effect",
            kind=kind,
            minimum_amount=None,
            maximum_amount=None,
            currency=None,
            point_value=None,
            consequence_label_vi=None,
            minimum_duration_months=None,
            maximum_duration_months=None,
            target_label_vi="Đối tượng chịu tác động",
            responsible_role_label_vi="Chủ thể thực hiện" if kind == OntologyConsequenceKind.RemedialMeasure else None,
            evidence=[evidence],
        ),
        evidence,
    )

    with pytest.raises(ValueError, match="consequence label"):
        build_population_record(raw, legal_chunk.chunk_id, document_metadata, {legal_chunk.chunk_id: legal_chunk.text})


@pytest.mark.parametrize(
    ("kind", "minimum", "maximum", "message"),
    [
        (OntologyConsequenceKind.AdditionalSanction, 10, None, "both lower and upper"),
        (OntologyConsequenceKind.AdditionalSanction, 0, 12, "positive and ordered"),
        (OntologyConsequenceKind.AdditionalSanction, 12, 10, "positive and ordered"),
        (OntologyConsequenceKind.MonetaryPenalty, 10, 12, "Only AdditionalSanction"),
    ],
)
def test_population_record_rejects_invalid_duration_bounds(
    kind: OntologyConsequenceKind,
    minimum: int | None,
    maximum: int | None,
    message: str,
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """Duration normalization is complete, positive, ordered, and sanction-specific."""
    evidence = OntologyEvidenceEntry(chunk_id=legal_chunk.chunk_id, quoted_span=legal_chunk.text)
    raw = _rule_with_consequence(
        OntologyConsequenceEntry(
            consequence_id="effect",
            kind=kind,
            minimum_amount="1" if kind == OntologyConsequenceKind.MonetaryPenalty else None,
            maximum_amount="2" if kind == OntologyConsequenceKind.MonetaryPenalty else None,
            currency=OntologyCurrency.VND if kind == OntologyConsequenceKind.MonetaryPenalty else None,
            point_value=None,
            consequence_label_vi="Tác động pháp lý" if kind == OntologyConsequenceKind.AdditionalSanction else None,
            minimum_duration_months=minimum,
            maximum_duration_months=maximum,
            target_label_vi="Đối tượng chịu tác động" if kind == OntologyConsequenceKind.AdditionalSanction else None,
            responsible_role_label_vi=None,
            evidence=[evidence],
        ),
        evidence,
    )

    with pytest.raises(ValueError, match=message):
        build_population_record(raw, legal_chunk.chunk_id, document_metadata, {legal_chunk.chunk_id: legal_chunk.text})


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        (OntologyConsequenceKind.MonetaryPenalty, "at most one MonetaryPenalty"),
        (OntologyConsequenceKind.PointDeduction, "at most one PointDeduction"),
    ],
)
def test_population_record_rejects_multiple_exclusive_consequences(
    kind: OntologyConsequenceKind,
    message: str,
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """Alternative numeric consequences must be emitted as separate rule variants."""
    evidence = OntologyEvidenceEntry(chunk_id=legal_chunk.chunk_id, quoted_span=legal_chunk.text)
    consequences = [
        OntologyConsequenceEntry(
            consequence_id=f"effect_{index}",
            kind=kind,
            minimum_amount=str(index) if kind == OntologyConsequenceKind.MonetaryPenalty else None,
            maximum_amount=str(index + 1) if kind == OntologyConsequenceKind.MonetaryPenalty else None,
            currency=OntologyCurrency.VND if kind == OntologyConsequenceKind.MonetaryPenalty else None,
            point_value=index if kind == OntologyConsequenceKind.PointDeduction else None,
            consequence_label_vi=None,
            minimum_duration_months=None,
            maximum_duration_months=None,
            target_label_vi="Giấy phép lái xe" if kind == OntologyConsequenceKind.PointDeduction else None,
            responsible_role_label_vi=None,
            evidence=[evidence],
        )
        for index in (2, 4)
    ]
    raw = OntologyRuleEntry(
        conduct_label_vi="Hành vi được kiểm tra",
        subject_kind=None,
        role_labels_vi=[],
        vehicle_category_labels_vi=[],
        condition_root_id=None,
        conditions=[],
        consequences=consequences,
        valid_from=None,
        valid_to=None,
        provision_path="Điều kiểm tra",
        evidence=[evidence],
    )

    with pytest.raises(ValueError, match=message):
        build_population_record(raw, legal_chunk.chunk_id, document_metadata, {legal_chunk.chunk_id: legal_chunk.text})


def _rule_with_consequence(
    consequence: OntologyConsequenceEntry,
    evidence: OntologyEvidenceEntry,
) -> OntologyRuleEntry:
    """Build a minimal typed rule for consequence-boundary tests."""
    return OntologyRuleEntry(
        conduct_label_vi="Hành vi được kiểm tra",
        subject_kind=None,
        role_labels_vi=[],
        vehicle_category_labels_vi=[],
        condition_root_id=None,
        conditions=[],
        consequences=[consequence],
        valid_from=None,
        valid_to=None,
        provision_path="Điều kiểm tra",
        evidence=[evidence],
    )


def _target_inventory_fixture(
    legal_chunk: Chunk,
) -> tuple[FactExtractionBatchItem, OntologyTargetConductBatchResult, OntologyEvidenceEntry]:
    """Build one locked Target conduct and its isolated extraction input."""
    evidence = OntologyEvidenceEntry(chunk_id=legal_chunk.chunk_id, quoted_span=legal_chunk.text)
    item = FactExtractionBatchItem(
        target_chunk_id=legal_chunk.chunk_id,
        window=[
            ChunkWindowEntry(
                chunk_id=legal_chunk.chunk_id,
                role=ChunkWindowRole.Target,
                context_text=legal_chunk.context_text,
                text=legal_chunk.text,
            )
        ],
    )
    inventory = OntologyTargetConductBatchResult(
        target_chunk_id=legal_chunk.chunk_id,
        conducts=[
            OntologyTargetConductEntry(
                conduct_id="conduct_1",
                conduct_label_vi="Hành vi được kiểm tra",
                provision_path="Điều kiểm tra",
                evidence=[evidence],
            )
        ],
    )
    return item, inventory, evidence


def _money_consequence(evidence: OntologyEvidenceEntry) -> OntologyConsequenceEntry:
    """Build one complete monetary consequence for binding tests."""
    return OntologyConsequenceEntry(
        consequence_id="money",
        kind=OntologyConsequenceKind.MonetaryPenalty,
        minimum_amount="1",
        maximum_amount="2",
        currency=OntologyCurrency.VND,
        point_value=None,
        consequence_label_vi=None,
        minimum_duration_months=None,
        maximum_duration_months=None,
        target_label_vi=None,
        responsible_role_label_vi=None,
        evidence=[evidence],
    )
