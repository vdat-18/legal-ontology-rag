"""Tests for typed LLM ontology extraction and evidence binding."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import replace
from datetime import date
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
    OntologyEvidenceReference,
    OntologyPopulationBatchResult,
    OntologyPopulationBindingBatchResult,
    OntologyRuleEntry,
    OntologySubjectKind,
    OntologyTargetConductBatchResult,
    OntologyTargetConductEntry,
    OntologyTargetConductInput,
    OntologyValidityMode,
)
from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord, TemporalValidityOverride
from legal_rag.benchmark_synthesis.llm_execution import IncompleteLlmTasksError, LlmTaskExhaustedError
from legal_rag.chunker.models import Chunk
from legal_rag.ontology_population.config import PopulationSettings
from legal_rag.ontology_population.evidence import evidence_source, fragment_evidence_sources
from legal_rag.ontology_population.extraction import (
    build_population_binding_items,
    build_population_reading_window,
    build_population_record,
    build_population_run_id,
    extract_population_from_document,
    extract_target_conducts,
    finalize_population_binding_results,
)
from legal_rag.ontology_population.models import OntologyPopulationRecord
from legal_rag.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ONTOLOGY_SEMANTIC_SHA256 = "ontology-semantic-sha256"
POPULATION_RUN_ID = "population-run-id"
EXTRACTION_RUN = {
    "ontology_semantic_sha256": ONTOLOGY_SEMANTIC_SHA256,
    "population_run_id": POPULATION_RUN_ID,
}
RECORD_RUN = {"ontology_version": "0.2.0", **EXTRACTION_RUN}


def test_target_conduct_extraction_uses_configured_batch_size(legal_chunk: Chunk) -> None:
    """Step 1 groups target chunks using its dedicated contract limit."""
    targets = tuple(replace(legal_chunk, chunk_id=f"target-{index}", chunk_index=index) for index in range(11))
    batch_sizes = []

    class _ConductClient:
        def extract_conduct_batch(
            self, items: Sequence[OntologyTargetConductInput]
        ) -> Sequence[OntologyTargetConductBatchResult]:
            batch_sizes.append(len(items))
            return tuple(
                OntologyTargetConductBatchResult(target_chunk_id=item.target_chunk_id, conducts=[]) for item in items
            )

    settings = replace(
        PopulationSettings.from_path(PROJECT_ROOT / "ontology" / "population_contract.yaml"),
        max_retries=0,
        retry_delay_seconds=0,
    )

    records = extract_target_conducts(targets, targets, _ConductClient(), settings, run_id="conduct-run")

    assert batch_sizes == [1] * len(targets)
    assert [record.target_chunk_id for record in records] == [chunk.chunk_id for chunk in targets]


def test_target_conduct_extraction_uses_dedicated_preceding_locator_window(legal_chunk: Chunk) -> None:
    """Step 1 exposes its bounded locator window while target evidence remains isolated."""
    chunks = (
        replace(legal_chunk, chunk_id="preceding-3", chunk_index=0, next_chunk_id="preceding-2"),
        replace(
            legal_chunk,
            chunk_id="preceding-2",
            chunk_index=1,
            previous_chunk_id="preceding-3",
            next_chunk_id="preceding-1",
        ),
        replace(
            legal_chunk,
            chunk_id="preceding-1",
            chunk_index=2,
            previous_chunk_id="preceding-2",
            next_chunk_id="target",
        ),
        replace(
            legal_chunk,
            chunk_id="target",
            chunk_index=3,
            previous_chunk_id="preceding-1",
            next_chunk_id="following",
        ),
        replace(legal_chunk, chunk_id="following", chunk_index=4, previous_chunk_id="target"),
    )
    received: list[OntologyTargetConductInput] = []

    class _ConductClient:
        def extract_conduct_batch(
            self, items: Sequence[OntologyTargetConductInput]
        ) -> Sequence[OntologyTargetConductBatchResult]:
            received.extend(items)
            return tuple(
                OntologyTargetConductBatchResult(target_chunk_id=item.target_chunk_id, conducts=[]) for item in items
            )

    settings = replace(
        PopulationSettings.from_path(PROJECT_ROOT / "ontology" / "population_contract.yaml"),
        max_locator_context_chunks=3,
        max_preceding_chunks=1,
        max_retries=0,
        retry_delay_seconds=0,
    )

    extract_target_conducts((chunks[3],), chunks, _ConductClient(), settings, run_id="conduct-run")

    assert "chunk_id=preceding-3" in received[0].locator_context
    assert "chunk_id=preceding-2" in received[0].locator_context
    assert "chunk_id=preceding-1" in received[0].locator_context
    assert "chunk_id=following" not in received[0].locator_context
    assert {source.chunk_id for source in received[0].evidence_sources} == {"target"}


def test_population_run_identity_tracks_semantic_provider_configuration() -> None:
    """Resume identity changes with model behavior but not credential rotation."""
    settings = Settings(
        llm_base_url="https://provider.test/v1",
        llm_api_key="key-one",
        llm_model="model-one",
        llm_timeout_seconds=30,
    )
    run_id = build_population_run_id("pilot-run", settings)

    assert build_population_run_id("pilot-run", replace(settings, llm_api_key="key-two")) == run_id
    assert build_population_run_id("pilot-run", replace(settings, llm_model="model-two")) != run_id
    assert (
        build_population_run_id(
            "pilot-run",
            replace(settings, llm_extra_request_body={"thinking": {"type": "disabled"}}),
        )
        != run_id
    )


class _PopulationClient:
    """Return one typed rule for each requested target."""

    def extract_batch(
        self,
        items: Sequence[FactExtractionBatchItem],
        metadata: DocumentMetadataRecord,
    ) -> Sequence[OntologyPopulationBatchResult]:
        """Build source-grounded typed BAML responses."""
        results = []
        for item in items:
            target = next(entry for entry in item.window if entry.role == ChunkWindowRole.Target)
            reference = OntologyEvidenceReference(
                evidence_id=fragment_evidence_sources(target.chunk_id, target.text)[0].evidence_id
            )
            results.append(
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
                                    evidence=[reference],
                                )
                            ],
                            validity_mode=OntologyValidityMode.DocumentDefault,
                            valid_from=None,
                            valid_to=None,
                            provision_path="Điều 13 khoản 9 điểm b",
                            evidence=[reference],
                        )
                    ],
                )
            )
        return tuple(results)


def test_extraction_binds_evidence_and_uses_reviewed_document_validity(
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """Typed model output becomes deterministic source-grounded population records."""
    evidence_text = fragment_evidence_sources(legal_chunk.chunk_id, legal_chunk.text)[0].text
    settings = replace(
        PopulationSettings.from_path(PROJECT_ROOT / "ontology" / "population_contract.yaml"),
        max_retries=0,
        retry_delay_seconds=0,
    )

    first = extract_population_from_document(
        (legal_chunk,), (legal_chunk,), _PopulationClient(), document_metadata, settings, **EXTRACTION_RUN
    )
    second = extract_population_from_document(
        (legal_chunk,), (legal_chunk,), _PopulationClient(), document_metadata, settings, **EXTRACTION_RUN
    )

    assert first == second
    assert first[0].valid_from == document_metadata.valid_from
    assert first[0].evidence[0].exact_text == evidence_text
    assert str(first[0].consequences[0].minimum_amount) == "4000000"


@pytest.mark.parametrize(
    ("validity_mode", "raw_valid_from", "expected_valid_from"),
    [
        (OntologyValidityMode.ExplicitInterval, "2026-01-01", date(2026, 1, 1)),
        (OntologyValidityMode.Unresolved, None, None),
    ],
)
def test_population_record_respects_typed_provision_validity(
    validity_mode: OntologyValidityMode,
    raw_valid_from: str | None,
    expected_valid_from: date | None,
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """A provision-specific decision overrides or suppresses document-default validity."""
    source, evidence = _source_reference(legal_chunk.chunk_id, legal_chunk.text)
    raw = _rule_with_consequence(_money_consequence(evidence), evidence)
    raw.validity_mode = validity_mode
    raw.valid_from = raw_valid_from

    record = build_population_record(raw, legal_chunk.chunk_id, document_metadata, [source], **RECORD_RUN)

    assert record.valid_from == expected_valid_from


def test_population_binding_receives_document_temporal_context(
    legal_chunk: Chunk,
) -> None:
    """Step 2 receives default validity and typed provision-specific overrides for semantic matching."""
    metadata = DocumentMetadataRecord(
        document_number="168/2024/NĐ-CP",
        valid_from=date(2025, 1, 1),
        valid_to=None,
        temporal_overrides=(
            TemporalValidityOverride(
                affected_provision_paths=("Điểm m khoản 3 Điều 6",),
                valid_from=date(2026, 1, 1),
                valid_to=None,
                unresolved_basis_vi=None,
            ),
        ),
    )
    item, inventory, _ = _target_inventory_fixture(legal_chunk)

    binding = build_population_binding_items((item,), (inventory,), metadata)[0]

    assert binding.temporal_overrides[0].affected_provision_paths == ["Điểm m khoản 3 Điều 6"]
    assert binding.temporal_overrides[0].valid_from == "2026-01-01"


def test_extraction_rejects_unknown_evidence_reference(
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """Ontology population accepts only source IDs supplied to the model."""
    settings = replace(
        PopulationSettings.from_path(PROJECT_ROOT / "ontology" / "population_contract.yaml"),
        max_retries=0,
        retry_delay_seconds=0,
    )

    class _UnknownEvidenceClient(_PopulationClient):
        def extract_batch(
            self,
            items: Sequence[FactExtractionBatchItem],
            metadata: DocumentMetadataRecord,
        ) -> Sequence[OntologyPopulationBatchResult]:
            results = list(super().extract_batch(items, metadata))
            results[0].rules[0].evidence = [OntologyEvidenceReference(evidence_id="ev_unknown")]
            return tuple(results)

    with pytest.raises(LlmTaskExhaustedError) as error:
        extract_population_from_document(
            (legal_chunk,),
            (legal_chunk,),
            _UnknownEvidenceClient(),
            document_metadata,
            settings,
            **EXTRACTION_RUN,
        )
    assert "Unknown evidence reference" in str(error.value.__cause__)


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

    class _ConcurrentPopulationClient(_PopulationClient):
        def extract_batch(
            self,
            items: Sequence[FactExtractionBatchItem],
            metadata: DocumentMetadataRecord,
        ) -> Sequence[OntologyPopulationBatchResult]:
            target_id = items[0].target_chunk_id
            calls.append(target_id)
            barrier.wait(timeout=2)
            if target_id == "target-1":
                assert second_checkpointed.wait(timeout=2)
            return super().extract_batch(items, metadata)

    def checkpoint(batch_results: Sequence[tuple[str, tuple[OntologyPopulationRecord, ...]]]) -> None:
        """Record main-thread checkpoint order and release the first target."""
        target_id = batch_results[0][0]
        completion_order.append(target_id)
        if target_id == "target-2":
            second_checkpointed.set()

    settings = replace(
        PopulationSettings.from_path(PROJECT_ROOT / "ontology" / "population_contract.yaml"),
        batch_size=1,
        max_retries=0,
        retry_delay_seconds=0,
    )

    records = extract_population_from_document(
        targets,
        targets,
        _ConcurrentPopulationClient(),
        document_metadata,
        settings,
        **EXTRACTION_RUN,
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

    class _FailingPopulationClient:
        def extract_batch(
            self,
            items: Sequence[FactExtractionBatchItem],
            metadata: DocumentMetadataRecord,
        ) -> Sequence[OntologyPopulationBatchResult]:
            """Simulate an invalid typed LLM response for each independent batch."""
            raise ValueError(items[0].target_chunk_id)

    targets = (
        replace(legal_chunk, chunk_id="target-1", chunk_index=0),
        replace(legal_chunk, chunk_id="target-2", chunk_index=1),
    )
    settings = replace(
        PopulationSettings.from_path(PROJECT_ROOT / "ontology" / "population_contract.yaml"),
        batch_size=1,
        max_retries=0,
        retry_delay_seconds=0,
    )

    with pytest.raises(IncompleteLlmTasksError) as error:
        extract_population_from_document(
            targets,
            targets,
            _FailingPopulationClient(),
            document_metadata,
            settings,
            **EXTRACTION_RUN,
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


def test_population_binding_inventory_is_forwarded_without_semantic_revalidation(
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """Typed inventory output is forwarded without a second evidence gate."""
    item, inventory, _ = _target_inventory_fixture(legal_chunk)
    inventory.conducts[0].evidence = [OntologyEvidenceReference(evidence_id="ev_context")]

    binding = build_population_binding_items((item,), (inventory,), document_metadata)[0]

    assert binding.target_conducts == inventory.conducts


def test_population_binding_rejects_conduct_outside_locked_inventory(
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """The consequence stage cannot create a context-only conduct."""
    item, inventory, evidence = _target_inventory_fixture(legal_chunk)
    binding_item = build_population_binding_items((item,), (inventory,), document_metadata)[0]
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
    ("conduct_label", "provision_path"),
    [
        ("Hành vi bị đổi tên", "Điều kiểm tra"),
        ("Hành vi được kiểm tra", "Điều khác"),
    ],
)
def test_population_binding_restores_locked_conduct_identity(
    conduct_label: str,
    provision_path: str,
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """The binding stage cannot rename or relocate a locked conduct."""
    item, inventory, evidence = _target_inventory_fixture(legal_chunk)
    binding_item = build_population_binding_items((item,), (inventory,), document_metadata)[0]
    rule = _rule_with_consequence(_money_consequence(evidence), evidence)
    rule.conduct_label_vi = conduct_label
    rule.provision_path = provision_path
    result = OntologyPopulationBindingBatchResult(
        target_chunk_id=legal_chunk.chunk_id,
        rules=[OntologyBoundRuleEntry(target_conduct_id="conduct_1", rule=rule)],
    )

    finalized = finalize_population_binding_results((binding_item,), (result,))

    assert finalized[0].rules == [rule]
    assert rule.conduct_label_vi == inventory.conducts[0].conduct_label_vi
    assert rule.provision_path == inventory.conducts[0].provision_path


def test_population_binding_rejects_omitted_locked_conduct(
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """Every locked conduct must produce at least one population rule."""
    item, inventory, evidence = _target_inventory_fixture(legal_chunk)
    inventory.conducts.append(
        OntologyTargetConductEntry(
            conduct_id="conduct_2",
            conduct_label_vi="Hành vi thứ hai",
            provision_path="Điều kiểm tra > Điểm b",
            evidence=[evidence],
        )
    )
    binding_item = build_population_binding_items((item,), (inventory,), document_metadata)[0]
    result = OntologyPopulationBindingBatchResult(
        target_chunk_id=legal_chunk.chunk_id,
        rules=[
            OntologyBoundRuleEntry(
                target_conduct_id="conduct_1",
                rule=_rule_with_consequence(_money_consequence(evidence), evidence),
            )
        ],
    )

    with pytest.raises(ValueError, match="omitted locked Target conduct: conduct_2"):
        finalize_population_binding_results((binding_item,), (result,))


def test_population_binding_rejects_rule_without_consequence(
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """A bound conduct cannot be persisted as a rule with no legal consequence."""
    item, inventory, evidence = _target_inventory_fixture(legal_chunk)
    binding_item = build_population_binding_items((item,), (inventory,), document_metadata)[0]
    rule = _rule_with_consequence(_money_consequence(evidence), evidence)
    rule.consequences = []
    result = OntologyPopulationBindingBatchResult(
        target_chunk_id=legal_chunk.chunk_id,
        rules=[OntologyBoundRuleEntry(target_conduct_id="conduct_1", rule=rule)],
    )

    with pytest.raises(ValueError, match="no consequence for locked Target conduct: conduct_1"):
        finalize_population_binding_results((binding_item,), (result,))


def test_population_binding_returns_existing_population_result(
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """Two-stage extraction keeps the downstream population result API unchanged."""
    item, inventory, evidence = _target_inventory_fixture(legal_chunk)
    binding_item = build_population_binding_items((item,), (inventory,), document_metadata)[0]
    rule = _rule_with_consequence(_money_consequence(evidence), evidence)
    result = OntologyPopulationBindingBatchResult(
        target_chunk_id=legal_chunk.chunk_id,
        rules=[OntologyBoundRuleEntry(target_conduct_id="conduct_1", rule=rule)],
    )

    finalized = finalize_population_binding_results((binding_item,), (result,))

    assert finalized == (OntologyPopulationBatchResult(target_chunk_id=legal_chunk.chunk_id, rules=[rule]),)


def test_population_record_accepts_context_evidence(
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """Typed context evidence is persisted without a target-only semantic gate."""
    context_chunk_id = "context"
    source, evidence = _source_reference(context_chunk_id, legal_chunk.text)
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

    record = build_population_record(
        raw,
        legal_chunk.chunk_id,
        document_metadata,
        [source, evidence_source(legal_chunk.chunk_id, legal_chunk.text)],
        **RECORD_RUN,
    )

    assert {item.chunk_id for item in record.evidence} == {context_chunk_id}


def test_population_record_preserves_incomplete_remedial_measure(
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """Optional remedial fields remain null instead of triggering semantic validation."""
    source, evidence = _source_reference(legal_chunk.chunk_id, legal_chunk.text)
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
        validity_mode=OntologyValidityMode.DocumentDefault,
        valid_from=None,
        valid_to=None,
        provision_path="Điều 3 khoản 3 điểm a",
        evidence=[evidence],
    )

    record = build_population_record(
        raw,
        legal_chunk.chunk_id,
        document_metadata,
        [source],
        **RECORD_RUN,
    )

    assert record.consequences[0].responsible_role_label_vi is None


@pytest.mark.parametrize("kind", [OntologyConsequenceKind.AdditionalSanction, OntologyConsequenceKind.RemedialMeasure])
def test_population_record_preserves_unnamed_non_numeric_consequence(
    kind: OntologyConsequenceKind,
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """Missing optional consequence labels do not block typed persistence."""
    source, evidence = _source_reference(legal_chunk.chunk_id, legal_chunk.text)
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

    record = build_population_record(raw, legal_chunk.chunk_id, document_metadata, [source], **RECORD_RUN)

    assert record.consequences[0].consequence_label_vi is None


@pytest.mark.parametrize(
    ("kind", "minimum", "maximum"),
    [
        (OntologyConsequenceKind.AdditionalSanction, 10, None),
        (OntologyConsequenceKind.AdditionalSanction, 0, 12),
        (OntologyConsequenceKind.AdditionalSanction, 12, 10),
        (OntologyConsequenceKind.MonetaryPenalty, 10, 12),
    ],
)
def test_population_record_preserves_typed_duration_bounds(
    kind: OntologyConsequenceKind,
    minimum: int | None,
    maximum: int | None,
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """Duration values are persisted without duplicating semantic rules in code."""
    source, evidence = _source_reference(legal_chunk.chunk_id, legal_chunk.text)
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

    record = build_population_record(raw, legal_chunk.chunk_id, document_metadata, [source], **RECORD_RUN)

    assert record.consequences[0].minimum_duration_months == minimum
    assert record.consequences[0].maximum_duration_months == maximum


@pytest.mark.parametrize(
    "kind",
    [
        OntologyConsequenceKind.MonetaryPenalty,
        OntologyConsequenceKind.PointDeduction,
    ],
)
def test_population_record_preserves_multiple_typed_consequences(
    kind: OntologyConsequenceKind,
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """Typed LLM consequences are not split or rejected by hardcoded subtype rules."""
    source, evidence = _source_reference(legal_chunk.chunk_id, legal_chunk.text)
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
        validity_mode=OntologyValidityMode.DocumentDefault,
        valid_from=None,
        valid_to=None,
        provision_path="Điều kiểm tra",
        evidence=[evidence],
    )

    record = build_population_record(raw, legal_chunk.chunk_id, document_metadata, [source], **RECORD_RUN)

    assert len(record.consequences) == 2


def _rule_with_consequence(
    consequence: OntologyConsequenceEntry,
    evidence: OntologyEvidenceReference,
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
        validity_mode=OntologyValidityMode.DocumentDefault,
        valid_from=None,
        valid_to=None,
        provision_path="Điều kiểm tra",
        evidence=[evidence],
    )


def _target_inventory_fixture(
    legal_chunk: Chunk,
) -> tuple[FactExtractionBatchItem, OntologyTargetConductBatchResult, OntologyEvidenceReference]:
    """Build one locked Target conduct and its isolated extraction input."""
    source = fragment_evidence_sources(legal_chunk.chunk_id, legal_chunk.text)[0]
    evidence = OntologyEvidenceReference(evidence_id=source.evidence_id)
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


def _money_consequence(evidence: OntologyEvidenceReference) -> OntologyConsequenceEntry:
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


def _source_reference(chunk_id: str, text: str):
    """Build matching source and model reference for one test fragment."""
    source = evidence_source(chunk_id, text)
    return source, OntologyEvidenceReference(evidence_id=source.evidence_id)
