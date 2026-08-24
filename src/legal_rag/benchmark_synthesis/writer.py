"""Atomic persistence for ground-truth synthesis JSONL outputs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from legal_rag.benchmark_synthesis.candidate_generation import (
    CANDIDATE_SCHEMA_VERSION,
    CandidateDraft,
    ClaimEvidence,
    RequiredClaim,
)
from legal_rag.benchmark_synthesis.candidate_record import (
    CANDIDATE_PROGRESS_SCHEMA_VERSION,
    CandidateProgressRecord,
)
from legal_rag.benchmark_synthesis.chunk_screening import CHUNK_SCREENING_SCHEMA_VERSION, ChunkScreeningRecord
from legal_rag.benchmark_synthesis.document_metadata import (
    DOCUMENT_METADATA_SCHEMA_VERSION,
    DocumentMetadataRecord,
)
from legal_rag.benchmark_synthesis.evidence_graph import (
    EVIDENCE_CANDIDATE_SET_SCHEMA_VERSION,
    EVIDENCE_RELATION_PROGRESS_SCHEMA_VERSION,
    EVIDENCE_RELATION_SCHEMA_VERSION,
    EvidenceCandidateSet,
    EvidenceRelation,
    EvidenceRelationProgressRecord,
)
from legal_rag.benchmark_synthesis.fact_extraction import (
    EXTRACTION_PROGRESS_SCHEMA_VERSION,
    ExtractionProgressRecord,
)
from legal_rag.benchmark_synthesis.models import LEGAL_FACT_SCHEMA_VERSION, EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.scenario_sampling import SCENARIO_SCHEMA_VERSION, Scenario
from legal_rag.io_utils import write_json_atomic, write_jsonl_atomic


def write_chunk_screening(output_path: Path, records: Sequence[ChunkScreeningRecord]) -> None:
    """Atomically replace one UTF-8 JSONL output with chunk-screening records."""
    write_jsonl_atomic(output_path, (record.to_dict() for record in records))


def read_chunk_screening(input_path: Path) -> tuple[ChunkScreeningRecord, ...]:
    """Read a chunk-screening JSONL file back into typed records."""
    records: list[ChunkScreeningRecord] = []
    with input_path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != CHUNK_SCREENING_SCHEMA_VERSION:
                msg = f"Unsupported chunk-screening schema: {payload['schema_version']}"
                raise ValueError(msg)
            records.append(
                ChunkScreeningRecord(
                    chunk_id=payload["chunk_id"],
                    has_penalty_content=payload["has_penalty_content"],
                    reason=payload["reason"],
                )
            )
    return tuple(records)


def write_document_metadata(output_path: Path, record: DocumentMetadataRecord) -> None:
    """Atomically replace one UTF-8 JSONL output with one document's metadata record."""
    write_jsonl_atomic(output_path, (record.to_dict(),))


def read_document_metadata(input_path: Path) -> DocumentMetadataRecord:
    """Read one document's metadata JSONL file back into a typed record."""
    with input_path.open(encoding="utf-8") as source_file:
        payload = json.loads(source_file.readline())
    if payload["schema_version"] != DOCUMENT_METADATA_SCHEMA_VERSION:
        msg = f"Unsupported document-metadata schema: {payload['schema_version']}"
        raise ValueError(msg)
    return DocumentMetadataRecord(
        document_number=payload["document_number"],
        valid_from=_parse_date(payload["valid_from"]),
        valid_to=_parse_date(payload["valid_to"]),
    )


def write_legal_facts(output_path: Path, facts: Sequence[LegalFact]) -> None:
    """Atomically replace one UTF-8 JSONL output with legal-fact records."""
    write_jsonl_atomic(output_path, (fact.to_dict() for fact in facts))


def read_legal_facts(input_path: Path) -> tuple[LegalFact, ...]:
    """Read a legal-facts JSONL file back into typed records."""
    facts: list[LegalFact] = []
    with input_path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != LEGAL_FACT_SCHEMA_VERSION:
                msg = f"Unsupported legal-fact schema: {payload['schema_version']}"
                raise ValueError(msg)
            facts.append(
                LegalFact(
                    fact_id=payload["fact_id"],
                    statement=payload["statement"],
                    valid_from=_parse_date(payload["valid_from"]),
                    valid_to=_parse_date(payload["valid_to"]),
                    document_number=payload["document_number"],
                    evidence=tuple(EvidenceReference(chunk_id=entry["chunk_id"]) for entry in payload["evidence"]),
                    source_chunk_id=payload["source_chunk_id"],
                )
            )
    return tuple(facts)


def write_extraction_progress(output_path: Path, records: Sequence[ExtractionProgressRecord]) -> None:
    """Atomically replace one UTF-8 JSONL output with extraction-progress records."""
    write_jsonl_atomic(output_path, (record.to_dict() for record in records))


def read_extraction_progress(input_path: Path) -> tuple[ExtractionProgressRecord, ...]:
    """Read an extraction-progress JSONL file back into typed records."""
    records: list[ExtractionProgressRecord] = []
    with input_path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != EXTRACTION_PROGRESS_SCHEMA_VERSION:
                msg = f"Unsupported extraction-progress schema: {payload['schema_version']}"
                raise ValueError(msg)
            records.append(ExtractionProgressRecord(chunk_id=payload["chunk_id"]))
    return tuple(records)


def write_evidence_relations(output_path: Path, relations: Sequence[EvidenceRelation]) -> None:
    """Atomically replace one UTF-8 JSONL output with evidence-relation records."""
    write_jsonl_atomic(output_path, (relation.to_dict() for relation in relations))


def write_evidence_candidate_set(output_path: Path, candidate_set: EvidenceCandidateSet) -> None:
    """Atomically replace one evidence-candidate-set JSON output."""
    write_json_atomic(output_path, candidate_set.to_dict())


def read_evidence_candidate_set(input_path: Path) -> EvidenceCandidateSet:
    """Read one fingerprinted evidence-candidate-set JSON output."""
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if payload["schema_version"] != EVIDENCE_CANDIDATE_SET_SCHEMA_VERSION:
        msg = f"Unsupported evidence-candidate-set schema: {payload['schema_version']}"
        raise ValueError(msg)
    pairs = tuple(tuple(pair) for pair in payload["pairs"])
    if pairs != tuple(sorted(set(pairs))) or any(len(pair) != 2 or pair[0] >= pair[1] for pair in pairs):
        msg = "Evidence candidate pairs must be unique, sorted fact-id pairs"
        raise ValueError(msg)
    return EvidenceCandidateSet(run_id=payload["run_id"], pairs=pairs)


def read_evidence_relations(input_path: Path) -> tuple[EvidenceRelation, ...]:
    """Read an evidence-relations JSONL file back into typed records."""
    relations: list[EvidenceRelation] = []
    with input_path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != EVIDENCE_RELATION_SCHEMA_VERSION:
                msg = f"Unsupported evidence-relation schema: {payload['schema_version']}"
                raise ValueError(msg)
            relations.append(
                EvidenceRelation(
                    relation_id=payload["relation_id"],
                    fact_ids=tuple(payload["fact_ids"]),
                    bundle_kind=payload["bundle_kind"],
                    bridge=payload["bridge"],
                    confirmed_by_model=payload["confirmed_by_model"],
                    run_id=payload["run_id"],
                )
            )
    return tuple(relations)


def write_evidence_relation_progress(output_path: Path, records: Sequence[EvidenceRelationProgressRecord]) -> None:
    """Atomically replace one UTF-8 JSONL output with evidence-relation-progress records."""
    write_jsonl_atomic(output_path, (record.to_dict() for record in records))


def read_evidence_relation_progress(input_path: Path) -> tuple[EvidenceRelationProgressRecord, ...]:
    """Read an evidence-relation-progress JSONL file back into typed records."""
    records: list[EvidenceRelationProgressRecord] = []
    with input_path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != EVIDENCE_RELATION_PROGRESS_SCHEMA_VERSION:
                msg = f"Unsupported evidence-relation-progress schema: {payload['schema_version']}"
                raise ValueError(msg)
            records.append(
                EvidenceRelationProgressRecord(
                    fact_id_a=payload["fact_id_a"],
                    fact_id_b=payload["fact_id_b"],
                    run_id=payload["run_id"],
                )
            )
    return tuple(records)


def write_scenarios(output_path: Path, scenarios: Sequence[Scenario]) -> None:
    """Atomically replace one UTF-8 JSONL output with scenario records."""
    write_jsonl_atomic(output_path, (scenario.to_dict() for scenario in scenarios))


def read_scenarios(input_path: Path) -> tuple[Scenario, ...]:
    """Read a scenarios JSONL file back into typed records."""
    scenarios: list[Scenario] = []
    with input_path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != SCENARIO_SCHEMA_VERSION:
                msg = f"Unsupported scenario schema: {payload['schema_version']}"
                raise ValueError(msg)
            scenarios.append(
                Scenario(
                    scenario_id=payload["scenario_id"],
                    bundle_kind=payload["bundle_kind"],
                    query_form=payload["query_form"],
                    answerability=payload["answerability"],
                    persona=payload["persona"],
                    query_style=payload["query_style"],
                    query_length=payload["query_length"],
                    reasoning_hops=payload["reasoning_hops"],
                    fact_ids=tuple(payload["fact_ids"]),
                    bridge=payload["bridge"],
                    gold_chunk_ids=tuple(payload["gold_chunk_ids"]),
                )
            )
    return tuple(scenarios)


def write_candidates(output_path: Path, drafts: Sequence[CandidateDraft]) -> None:
    """Atomically replace one UTF-8 JSONL output with candidate drafts."""
    write_jsonl_atomic(output_path, (draft.to_dict() for draft in drafts))


def write_candidate_progress(output_path: Path, records: Sequence[CandidateProgressRecord]) -> None:
    """Atomically replace one UTF-8 JSONL output with candidate checkpoints."""
    write_jsonl_atomic(output_path, (record.to_dict() for record in records))


def read_candidate_progress(input_path: Path) -> tuple[CandidateProgressRecord, ...]:
    """Read candidate checkpoints back into typed records."""
    records: list[CandidateProgressRecord] = []
    with input_path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != CANDIDATE_PROGRESS_SCHEMA_VERSION:
                msg = f"Unsupported candidate-progress schema: {payload['schema_version']}"
                raise ValueError(msg)
            records.append(
                CandidateProgressRecord(
                    task_id=payload["task_id"],
                    scenario_id=payload["scenario_id"],
                    attempt_index=payload["attempt_index"],
                    stage=payload["stage"],
                    draft=_parse_candidate_draft(payload["draft"]),
                )
            )
    return tuple(records)


def read_candidates(input_path: Path) -> tuple[CandidateDraft, ...]:
    """Read a candidates JSONL file back into typed records."""
    drafts: list[CandidateDraft] = []
    with input_path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != CANDIDATE_SCHEMA_VERSION:
                msg = f"Unsupported candidate schema: {payload['schema_version']}"
                raise ValueError(msg)
            drafts.append(_parse_candidate_draft(payload))
    return tuple(drafts)


def _parse_candidate_draft(payload: dict[str, object]) -> CandidateDraft:
    """Parse the CandidateDraft portion of one candidate JSONL record."""
    return CandidateDraft(
        sample_id=payload["sample_id"],
        scenario_id=payload["scenario_id"],
        bundle_kind=payload["bundle_kind"],
        bridge=payload["bridge"],
        query_vi=payload["query_vi"],
        reference_answer_vi=payload["reference_answer_vi"],
        reasoning_vi=payload["reasoning_vi"],
        required_claims=tuple(
            RequiredClaim(
                claim=claim["claim"],
                fact_id=claim["fact_id"],
                requested_chunk_ids=tuple(claim["requested_chunk_ids"]),
                evidence=tuple(ClaimEvidence(chunk_id=entry["chunk_id"]) for entry in claim["evidence"]),
            )
            for claim in payload["required_claims"]
        ),
        evolutions=tuple(payload["evolutions"]),
        gold_chunk_ids=tuple(payload["gold_chunk_ids"]),
        query_form=payload["query_form"],
        persona=payload["persona"],
        query_style=payload["query_style"],
        query_length=payload["query_length"],
        answerability=payload["answerability"],
        reasoning_hops=payload["reasoning_hops"],
        source_fact_ids=tuple(payload["source_fact_ids"]),
        generator_model=payload["generator_model"],
    )


def _parse_date(value: str | None) -> date | None:
    """Parse an ISO date string, returning None for a null field."""
    return date.fromisoformat(value) if value else None
