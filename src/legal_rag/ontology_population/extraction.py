"""Extract typed ontology population records from chunk reading windows."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from decimal import Decimal
from functools import partial
from typing import Protocol

from baml_py.logging import set_log_level

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import FactExtractionBatchItem, OntologyPopulationBatchResult
from legal_rag.baml_client.types import OntologyEvidenceEntry as BamlEvidence
from legal_rag.baml_client.types import OntologyRuleEntry as BamlRule
from legal_rag.baml_transport import build_client_registry, execute_baml_request
from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.benchmark_synthesis.evidence_binding import bind_evidence_references
from legal_rag.benchmark_synthesis.fact_extraction import build_reading_window
from legal_rag.benchmark_synthesis.llm_execution import execute_llm_task
from legal_rag.benchmark_synthesis.models import EvidenceReference
from legal_rag.chunker.models import Chunk
from legal_rag.ontology_population.config import PopulationSettings
from legal_rag.ontology_population.models import (
    OntologyPopulationRecord,
    PopulationCondition,
    PopulationConsequence,
    PopulationEvidence,
    population_identity,
)
from legal_rag.settings import Settings

ONTOLOGY_POPULATION_PROMPT_VERSION = "0.6.0"


class OntologyPopulationClient(Protocol):
    """Boundary implemented by an ontology-population language-model client."""

    def extract_batch(self, items: Sequence[FactExtractionBatchItem]) -> Sequence[OntologyPopulationBatchResult]:
        """Return typed population candidates for reading windows."""


class BamlOntologyPopulationClient:
    """Typed BAML population client backed by an OpenAI-compatible provider."""

    def __init__(self, settings: Settings, population_settings: PopulationSettings) -> None:
        """Create a bounded runtime BAML client."""
        set_log_level("OFF")
        self._registry = build_client_registry(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_output_tokens=population_settings.max_output_tokens,
            temperature=population_settings.temperature,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        self._timeout_seconds = settings.llm_timeout_seconds
        self._max_response_bytes = population_settings.max_response_bytes

    def extract_batch(self, items: Sequence[FactExtractionBatchItem]) -> Sequence[OntologyPopulationBatchResult]:
        """Run one bounded BAML call over population windows."""
        request = b.request.ExtractOntologyPopulationBatch(
            list(items), baml_options={"client_registry": self._registry}
        )
        raw_output = execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)
        return b.parse.ExtractOntologyPopulationBatch(raw_output, baml_options={"client_registry": self._registry})


def extract_population_from_document(
    target_chunks: Sequence[Chunk],
    document_chunks: Sequence[Chunk],
    client: OntologyPopulationClient,
    metadata: DocumentMetadataRecord,
    population_settings: PopulationSettings,
    *,
    on_batch_processed: Callable[[Sequence[tuple[str, tuple[OntologyPopulationRecord, ...]]]], None] | None = None,
) -> tuple[OntologyPopulationRecord, ...]:
    """Extract source-grounded population records in bounded batches."""
    chunks_by_id = {chunk.chunk_id: chunk for chunk in document_chunks}
    records: list[OntologyPopulationRecord] = []
    for batch_start in range(0, len(target_chunks), population_settings.batch_size):
        batch = target_chunks[batch_start : batch_start + population_settings.batch_size]
        items = tuple(
            FactExtractionBatchItem(
                target_chunk_id=target.chunk_id,
                window=list(
                    build_reading_window(
                        target,
                        chunks_by_id,
                        max_preceding_chunks=population_settings.max_preceding_chunks,
                        max_following_chunks=population_settings.max_following_chunks,
                    )
                ),
            )
            for target in batch
        )
        batch_results = execute_llm_task(
            _batch_task_key(items),
            partial(_extract_batch_records, client, items, metadata),
            max_retries=population_settings.max_retries,
            retry_delay_seconds=population_settings.retry_delay_seconds,
        )
        for _, target_records in batch_results:
            records.extend(target_records)
        if on_batch_processed is not None:
            on_batch_processed(batch_results)
    return tuple(records)


def _extract_batch_records(
    client: OntologyPopulationClient,
    items: Sequence[FactExtractionBatchItem],
    metadata: DocumentMetadataRecord,
) -> tuple[tuple[str, tuple[OntologyPopulationRecord, ...]], ...]:
    """Extract one batch and bind every assertion to exact source evidence."""
    results = client.extract_batch(items)
    requested_ids = [item.target_chunk_id for item in items]
    result_ids = [result.target_chunk_id for result in results]
    if len(result_ids) != len(set(result_ids)) or set(result_ids) != set(requested_ids):
        msg = "Ontology population response must contain exactly one result per target chunk"
        raise ValueError(msg)
    results_by_target = {result.target_chunk_id: result for result in results}
    text_by_target = {item.target_chunk_id: {entry.chunk_id: entry.text for entry in item.window} for item in items}
    return tuple(
        (
            target_id,
            tuple(
                build_population_record(rule, target_id, metadata, text_by_target[target_id])
                for rule in results_by_target[target_id].rules
            ),
        )
        for target_id in requested_ids
    )


def build_population_record(
    raw: BamlRule,
    source_chunk_id: str,
    metadata: DocumentMetadataRecord,
    chunk_text_by_id: Mapping[str, str],
) -> OntologyPopulationRecord:
    """Convert one typed model result into a stable reviewable record."""
    if any(
        consequence.kind.value == "RemedialMeasure" and consequence.responsible_role_label_vi is None
        for consequence in raw.consequences
    ):
        msg = "RemedialMeasure requires a source-grounded responsible role"
        raise ValueError(msg)
    evidence = bind_population_evidence(raw.evidence, chunk_text_by_id)
    conditions = tuple(
        PopulationCondition(
            condition_id=condition.condition_id,
            kind=condition.kind.value,
            text_vi=condition.text_vi,
            operand_ids=tuple(condition.operand_ids),
            evidence=bind_population_evidence(condition.evidence, chunk_text_by_id),
        )
        for condition in raw.conditions
    )
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
            evidence=bind_population_evidence(consequence.evidence, chunk_text_by_id),
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
    )


def bind_population_evidence(
    entries: Sequence[BamlEvidence], chunk_text_by_id: Mapping[str, str]
) -> tuple[PopulationEvidence, ...]:
    """Bind BAML evidence to exact contiguous source spans."""
    bound = bind_evidence_references(
        tuple(EvidenceReference(chunk_id=entry.chunk_id, quoted_span=entry.quoted_span) for entry in entries),
        chunk_text_by_id,
    )
    return tuple(PopulationEvidence(chunk_id=item.chunk_id, exact_text=item.quoted_span) for item in bound)


def _batch_task_key(items: Sequence[FactExtractionBatchItem]) -> str:
    """Return a deterministic retry key for one population batch."""
    target_ids = "\0".join(item.target_chunk_id for item in items)
    return f"ontology-population:{hashlib.sha256(target_ids.encode('utf-8')).hexdigest()}"
