"""Extract typed ontology population records from chunk reading windows."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from decimal import Decimal
from functools import partial
from typing import Protocol

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import (
    ChunkWindowEntry,
    ChunkWindowRole,
    FactExtractionBatchItem,
    OntologyChunkWindowEntry,
    OntologyEvidenceReference,
    OntologyEvidenceSource,
    OntologyPopulationBatchResult,
    OntologyPopulationBindingBatchResult,
    OntologyPopulationBindingItem,
    OntologyTargetConductBatchResult,
    OntologyTargetConductInput,
)
from legal_rag.baml_client.types import OntologyRuleEntry as BamlRule
from legal_rag.baml_transport import BamlCaller
from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.benchmark_synthesis.fact_extraction import build_reading_window
from legal_rag.benchmark_synthesis.llm_execution import (
    IncompleteLlmTasksError,
    LlmTaskExhaustedError,
    LlmTaskFailure,
    execute_llm_task,
)
from legal_rag.chunker.models import Chunk
from legal_rag.ontology_population.config import PopulationSettings
from legal_rag.ontology_population.evidence import fragment_evidence_sources, resolve_evidence_references
from legal_rag.ontology_population.models import (
    OntologyPopulationRecord,
    PopulationCondition,
    PopulationConsequence,
    PopulationEvidence,
    population_identity,
    validate_population_condition_tree,
)
from legal_rag.settings import Settings

ONTOLOGY_POPULATION_PROMPT_VERSION = "0.16.0"


def build_population_run_id(pilot_run_id: str, settings: Settings) -> str:
    """Fingerprint every external input that can change population output."""
    return population_identity(
        {
            "pilot_run_id": pilot_run_id,
            "prompt_version": ONTOLOGY_POPULATION_PROMPT_VERSION,
            "llm_base_url": settings.llm_base_url,
            "llm_model": settings.llm_model,
            "llm_extra_request_body": dict(settings.llm_extra_request_body),
        }
    )


class OntologyPopulationClient(Protocol):
    """Boundary implemented by an ontology-population language-model client."""

    def extract_batch(self, items: Sequence[FactExtractionBatchItem]) -> Sequence[OntologyPopulationBatchResult]:
        """Return typed population candidates for reading windows."""


class BamlOntologyPopulationClient:
    """Typed BAML population client backed by an OpenAI-compatible provider."""

    def __init__(self, settings: Settings, population_settings: PopulationSettings) -> None:
        """Create a bounded runtime BAML client."""
        self._caller = BamlCaller.from_settings(
            settings,
            max_output_tokens=population_settings.max_output_tokens,
            temperature=population_settings.temperature,
            max_response_bytes=population_settings.max_response_bytes,
        )

    def extract_batch(self, items: Sequence[FactExtractionBatchItem]) -> Sequence[OntologyPopulationBatchResult]:
        """Inventory target conducts before binding consequences from context."""
        target_inputs = tuple(_target_conduct_input(item) for item in items)
        inventory_request = b.request.ExtractOntologyTargetConductBatch(
            list(target_inputs), baml_options=self._caller.options
        )
        inventory_output = self._caller.execute(inventory_request)
        inventories = b.parse.ExtractOntologyTargetConductBatch(inventory_output, baml_options=self._caller.options)
        binding_items = build_population_binding_items(items, inventories)
        if not any(item.target_conducts for item in binding_items):
            return tuple(
                OntologyPopulationBatchResult(target_chunk_id=item.target_chunk_id, rules=[]) for item in binding_items
            )
        binding_request = b.request.BindOntologyPopulationBatch(list(binding_items), baml_options=self._caller.options)
        binding_output = self._caller.execute(binding_request)
        binding_results = b.parse.BindOntologyPopulationBatch(binding_output, baml_options=self._caller.options)
        return finalize_population_binding_results(binding_items, binding_results)


def _target_conduct_input(item: FactExtractionBatchItem) -> OntologyTargetConductInput:
    """Build the isolated Target-only input for conduct inventory."""
    target = _target_entry(item)
    return OntologyTargetConductInput(
        target_chunk_id=item.target_chunk_id,
        structural_context=target.context_text,
        evidence_sources=list(fragment_evidence_sources(target.chunk_id, target.text)),
    )


def build_population_binding_items(
    items: Sequence[FactExtractionBatchItem],
    inventories: Sequence[OntologyTargetConductBatchResult],
) -> tuple[OntologyPopulationBindingItem, ...]:
    """Validate Target-only inventories and attach them to full reading windows."""
    requested_ids = [item.target_chunk_id for item in items]
    inventory_ids = [result.target_chunk_id for result in inventories]
    if len(inventory_ids) != len(set(inventory_ids)) or set(inventory_ids) != set(requested_ids):
        msg = "Target conduct inventory must contain exactly one result per target chunk"
        raise ValueError(msg)
    inventories_by_target = {result.target_chunk_id: result for result in inventories}
    binding_items = []
    for item in items:
        inventory = inventories_by_target[item.target_chunk_id]
        conduct_ids = [conduct.conduct_id for conduct in inventory.conducts]
        if len(conduct_ids) != len(set(conduct_ids)):
            msg = "Target conduct IDs must be unique within each target chunk"
            raise ValueError(msg)
        for conduct in inventory.conducts:
            if not conduct.evidence:
                msg = "Every target conduct must cite Target evidence"
                raise ValueError(msg)
            target_sources = fragment_evidence_sources(item.target_chunk_id, _target_entry(item).text)
            bound_evidence = resolve_evidence_references(conduct.evidence, target_sources)
            if any(entry.chunk_id != item.target_chunk_id for entry in bound_evidence):
                msg = "Target conduct evidence must reference its target chunk"
                raise ValueError(msg)
        binding_items.append(
            OntologyPopulationBindingItem(
                target_chunk_id=item.target_chunk_id,
                window=[
                    OntologyChunkWindowEntry(
                        role=entry.role,
                        chunk_id=entry.chunk_id,
                        structural_context=entry.context_text,
                        evidence_sources=list(fragment_evidence_sources(entry.chunk_id, entry.text)),
                    )
                    for entry in item.window
                ],
                target_conducts=list(inventory.conducts),
            )
        )
    return tuple(binding_items)


def finalize_population_binding_results(
    items: Sequence[OntologyPopulationBindingItem],
    results: Sequence[OntologyPopulationBindingBatchResult],
) -> tuple[OntologyPopulationBatchResult, ...]:
    """Reject rules that diverge from the locked Target conduct inventory."""
    requested_ids = [item.target_chunk_id for item in items]
    result_ids = [result.target_chunk_id for result in results]
    if len(result_ids) != len(set(result_ids)) or set(result_ids) != set(requested_ids):
        msg = "Population binding response must contain exactly one result per target chunk"
        raise ValueError(msg)
    items_by_target = {item.target_chunk_id: item for item in items}
    results_by_target = {result.target_chunk_id: result for result in results}
    finalized = []
    for target_id in requested_ids:
        item = items_by_target[target_id]
        conducts_by_id = {conduct.conduct_id: conduct for conduct in item.target_conducts}
        rules = []
        for bound in results_by_target[target_id].rules:
            conduct = conducts_by_id.get(bound.target_conduct_id)
            if conduct is None:
                msg = "Population rule must reference a locked Target conduct"
                raise ValueError(msg)
            if bound.rule.conduct_label_vi != conduct.conduct_label_vi:
                msg = "Population rule cannot rename its locked Target conduct"
                raise ValueError(msg)
            if bound.rule.provision_path != conduct.provision_path:
                msg = "Population rule cannot change its locked Target provision path"
                raise ValueError(msg)
            inventory_evidence = {entry.evidence_id for entry in conduct.evidence}
            rule_evidence = {entry.evidence_id for entry in bound.rule.evidence}
            if not inventory_evidence.issubset(rule_evidence):
                msg = "Population rule must retain its locked Target conduct evidence"
                raise ValueError(msg)
            rules.append(bound.rule)
        finalized.append(OntologyPopulationBatchResult(target_chunk_id=target_id, rules=rules))
    return tuple(finalized)


def extract_population_from_document(
    target_chunks: Sequence[Chunk],
    document_chunks: Sequence[Chunk],
    client: OntologyPopulationClient,
    metadata: DocumentMetadataRecord,
    population_settings: PopulationSettings,
    *,
    ontology_semantic_sha256: str,
    population_run_id: str,
    max_concurrency: int = 1,
    on_batch_processed: Callable[[Sequence[tuple[str, tuple[OntologyPopulationRecord, ...]]]], None] | None = None,
) -> tuple[OntologyPopulationRecord, ...]:
    """Extract population records concurrently and checkpoint each successful batch."""
    chunks_by_id = {chunk.chunk_id: chunk for chunk in document_chunks}
    batches = [
        target_chunks[batch_start : batch_start + population_settings.batch_size]
        for batch_start in range(0, len(target_chunks), population_settings.batch_size)
    ]
    batch_results_by_index: dict[int, tuple[tuple[str, tuple[OntologyPopulationRecord, ...]], ...]] = {}
    failures: list[LlmTaskFailure] = []
    exhausted_errors: list[LlmTaskExhaustedError] = []
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {}
        for batch_index, batch in enumerate(batches):
            items = tuple(
                FactExtractionBatchItem(
                    target_chunk_id=target.chunk_id,
                    window=list(
                        build_population_reading_window(
                            target,
                            chunks_by_id,
                            max_preceding_chunks=population_settings.max_preceding_chunks,
                            max_following_chunks=population_settings.max_following_chunks,
                            max_article_context_chunks=population_settings.max_article_context_chunks,
                        )
                    ),
                )
                for target in batch
            )
            task_key = _batch_task_key(items)
            future = executor.submit(
                execute_llm_task,
                task_key,
                partial(
                    _extract_batch_records,
                    client,
                    items,
                    metadata,
                    ontology_version=population_settings.ontology_version,
                    ontology_semantic_sha256=ontology_semantic_sha256,
                    population_run_id=population_run_id,
                ),
                max_retries=population_settings.max_retries,
                retry_delay_seconds=population_settings.retry_delay_seconds,
            )
            futures[future] = (batch_index, task_key)

        for future in as_completed(futures):
            batch_index, task_key = futures[future]
            try:
                batch_results = future.result()
            except LlmTaskExhaustedError as error:
                failures.append(LlmTaskFailure(task_key=task_key, error=str(error)))
                exhausted_errors.append(error)
                continue
            batch_results_by_index[batch_index] = batch_results
            if on_batch_processed is not None:
                on_batch_processed(batch_results)
    if failures:
        if len(exhausted_errors) == 1:
            raise exhausted_errors[0]
        incomplete_error = IncompleteLlmTasksError.for_stage("ontology population", failures)
        raise incomplete_error
    return tuple(
        record
        for batch_index in range(len(batches))
        for _, target_records in batch_results_by_index[batch_index]
        for record in target_records
    )


def _extract_batch_records(
    client: OntologyPopulationClient,
    items: Sequence[FactExtractionBatchItem],
    metadata: DocumentMetadataRecord,
    *,
    ontology_version: str,
    ontology_semantic_sha256: str,
    population_run_id: str,
) -> tuple[tuple[str, tuple[OntologyPopulationRecord, ...]], ...]:
    """Extract one batch and preserve the model-selected source evidence."""
    results = client.extract_batch(items)
    requested_ids = [item.target_chunk_id for item in items]
    result_ids = [result.target_chunk_id for result in results]
    if len(result_ids) != len(set(result_ids)) or set(result_ids) != set(requested_ids):
        msg = "Ontology population response must contain exactly one result per target chunk"
        raise ValueError(msg)
    results_by_target = {result.target_chunk_id: result for result in results}
    evidence_sources_by_target = {
        item.target_chunk_id: tuple(
            source for entry in item.window for source in fragment_evidence_sources(entry.chunk_id, entry.text)
        )
        for item in items
    }
    return tuple(
        (
            target_id,
            tuple(
                build_population_record(
                    rule,
                    target_id,
                    metadata,
                    evidence_sources_by_target[target_id],
                    ontology_version=ontology_version,
                    ontology_semantic_sha256=ontology_semantic_sha256,
                    population_run_id=population_run_id,
                )
                for rule in results_by_target[target_id].rules
            ),
        )
        for target_id in requested_ids
    )


def build_population_record(
    raw: BamlRule,
    source_chunk_id: str,
    metadata: DocumentMetadataRecord,
    evidence_sources: Sequence[OntologyEvidenceSource],
    *,
    ontology_version: str,
    ontology_semantic_sha256: str,
    population_run_id: str,
) -> OntologyPopulationRecord:
    """Convert one typed model result into a stable reviewable record."""
    bound_rule_evidence = bind_population_evidence(raw.evidence, evidence_sources)
    if not any(item.chunk_id == source_chunk_id for item in bound_rule_evidence):
        msg = "Population rule conduct must retain evidence from its target chunk"
        raise ValueError(msg)
    for kind in ("MonetaryPenalty", "PointDeduction"):
        if sum(consequence.kind.value == kind for consequence in raw.consequences) > 1:
            msg = f"A population rule variant may contain at most one {kind}"
            raise ValueError(msg)
    if any(
        consequence.kind.value == "RemedialMeasure" and consequence.responsible_role_label_vi is None
        for consequence in raw.consequences
    ):
        msg = "RemedialMeasure requires a source-grounded responsible role"
        raise ValueError(msg)
    if any(
        consequence.kind.value in {"AdditionalSanction", "RemedialMeasure"} and consequence.consequence_label_vi is None
        for consequence in raw.consequences
    ):
        msg = "AdditionalSanction and RemedialMeasure require a source-grounded consequence label"
        raise ValueError(msg)
    for consequence in raw.consequences:
        durations = (consequence.minimum_duration_months, consequence.maximum_duration_months)
        if consequence.kind.value != "AdditionalSanction" and any(value is not None for value in durations):
            msg = "Only AdditionalSanction may define duration month bounds"
            raise ValueError(msg)
        if (durations[0] is None) != (durations[1] is None):
            msg = "Consequence duration requires both lower and upper month bounds"
            raise ValueError(msg)
        if durations[0] is not None and (durations[0] < 1 or durations[1] < durations[0]):
            msg = "Consequence duration month bounds must be positive and ordered"
            raise ValueError(msg)
    evidence = bound_rule_evidence
    conditions = tuple(
        PopulationCondition(
            condition_id=condition.condition_id,
            kind=condition.kind.value,
            text_vi=condition.text_vi,
            operand_ids=tuple(condition.operand_ids),
            evidence=bind_population_evidence(condition.evidence, evidence_sources),
        )
        for condition in raw.conditions
    )
    validate_population_condition_tree(raw.condition_root_id, conditions)
    consequences = tuple(
        PopulationConsequence(
            consequence_id=consequence.consequence_id,
            kind=consequence.kind.value,
            minimum_amount=Decimal(consequence.minimum_amount) if consequence.minimum_amount is not None else None,
            maximum_amount=Decimal(consequence.maximum_amount) if consequence.maximum_amount is not None else None,
            currency_code=consequence.currency.value if consequence.currency else None,
            point_value=consequence.point_value,
            target_label_vi=consequence.target_label_vi,
            responsible_role_label_vi=consequence.responsible_role_label_vi,
            evidence=bind_population_evidence(consequence.evidence, evidence_sources),
            consequence_label_vi=consequence.consequence_label_vi,
            minimum_duration_months=consequence.minimum_duration_months,
            maximum_duration_months=consequence.maximum_duration_months,
        )
        for consequence in raw.consequences
    )
    content = {
        "source_chunk_id": source_chunk_id,
        "conduct_label_vi": raw.conduct_label_vi,
        "subject_kind": raw.subject_kind.value if raw.subject_kind else None,
        "role_labels_vi": raw.role_labels_vi,
        "vehicle_category_labels_vi": raw.vehicle_category_labels_vi,
        "condition_root_id": raw.condition_root_id,
        "conditions": [condition.to_dict() for condition in conditions],
        "consequences": [consequence.to_dict() for consequence in consequences],
        "valid_from": raw.valid_from or (metadata.valid_from.isoformat() if metadata.valid_from else None),
        "valid_to": raw.valid_to or (metadata.valid_to.isoformat() if metadata.valid_to else None),
        "document_number": metadata.document_number,
        "provision_path": raw.provision_path,
        "evidence": [item.to_dict() for item in evidence],
        "ontology_version": ontology_version,
        "ontology_semantic_sha256": ontology_semantic_sha256,
        "population_run_id": population_run_id,
    }
    return OntologyPopulationRecord(
        population_id=population_identity(content),
        source_chunk_id=source_chunk_id,
        conduct_label_vi=raw.conduct_label_vi,
        subject_kind=raw.subject_kind.value if raw.subject_kind else None,
        role_labels_vi=tuple(raw.role_labels_vi),
        vehicle_category_labels_vi=tuple(raw.vehicle_category_labels_vi),
        condition_root_id=raw.condition_root_id,
        conditions=conditions,
        consequences=consequences,
        valid_from=date.fromisoformat(raw.valid_from) if raw.valid_from else metadata.valid_from,
        valid_to=date.fromisoformat(raw.valid_to) if raw.valid_to else metadata.valid_to,
        document_number=metadata.document_number,
        provision_path=raw.provision_path,
        evidence=evidence,
        ontology_version=ontology_version,
        ontology_semantic_sha256=ontology_semantic_sha256,
        population_run_id=population_run_id,
    )


def bind_population_evidence(
    entries: Sequence[OntologyEvidenceReference], evidence_sources: Sequence[OntologyEvidenceSource]
) -> tuple[PopulationEvidence, ...]:
    """Resolve typed model references to authoritative source fragments."""
    return resolve_evidence_references(entries, evidence_sources)


def _target_entry(item: FactExtractionBatchItem) -> ChunkWindowEntry:
    """Return the sole Target entry from one population reading window."""
    targets = [entry for entry in item.window if entry.role == ChunkWindowRole.Target]
    if len(targets) != 1:
        msg = "Population reading window must contain exactly one Target entry"
        raise ValueError(msg)
    return targets[0]


def build_population_reading_window(
    target: Chunk,
    chunks_by_id: Mapping[str, Chunk],
    *,
    max_preceding_chunks: int,
    max_following_chunks: int,
    max_article_context_chunks: int,
) -> tuple[ChunkWindowEntry, ...]:
    """Add bounded same-article context so distant consequence references remain visible."""
    local_window = build_reading_window(
        target,
        chunks_by_id,
        max_preceding_chunks=max_preceding_chunks,
        max_following_chunks=max_following_chunks,
    )
    target_articles = _structural_articles(target.context_text)
    if len(target_articles) != 1:
        return local_window

    article = target_articles[0]
    article_chunks = tuple(
        chunk
        for chunk in sorted(chunks_by_id.values(), key=lambda item: item.chunk_index)
        if chunk.scope_id == target.scope_id and article in _structural_articles(chunk.context_text)
    )
    if len(article_chunks) > max_article_context_chunks:
        msg = (
            f"Article context for target {target.chunk_id} has {len(article_chunks)} chunks, "
            f"exceeding configured maximum {max_article_context_chunks}"
        )
        raise ValueError(msg)

    entries = {entry.chunk_id: entry for entry in local_window}
    for chunk in article_chunks:
        entries.setdefault(
            chunk.chunk_id,
            ChunkWindowEntry(
                chunk_id=chunk.chunk_id,
                role=ChunkWindowRole.ArticleContext,
                context_text=chunk.context_text,
                text=chunk.text,
            ),
        )
    article_chunk_ids = {chunk.chunk_id for chunk in article_chunks}
    return tuple(entries[chunk.chunk_id] for chunk in article_chunks) + tuple(
        entry for entry in local_window if entry.chunk_id not in article_chunk_ids
    )


def _structural_articles(context_text: str) -> tuple[str, ...]:
    """Read article labels from the parser-owned structural-context serialization."""
    return tuple(dict.fromkeys(line.strip() for line in context_text.splitlines() if line.strip().startswith("Điều ")))


def _batch_task_key(items: Sequence[FactExtractionBatchItem]) -> str:
    """Return a deterministic retry key for one population batch."""
    identity = "\0".join(
        (
            ONTOLOGY_POPULATION_PROMPT_VERSION,
            *(f"{item.target_chunk_id}:{entry.role.value}:{entry.chunk_id}" for item in items for entry in item.window),
        )
    )
    return f"ontology-population:{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"
