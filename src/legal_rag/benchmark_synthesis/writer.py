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
from legal_rag.benchmark_synthesis.chunk_screening import (
    CHUNK_SCREENING_SCHEMA_VERSION,
    LEGACY_CHUNK_SCREENING_SCHEMA_VERSION,
    ChunkScreeningRecord,
)
from legal_rag.benchmark_synthesis.document_metadata import (
    DOCUMENT_METADATA_SCHEMA_VERSION,
    DocumentMetadataRecord,
)
from legal_rag.benchmark_synthesis.evidence_graph import (
    EVIDENCE_CANDIDATE_SET_SCHEMA_VERSION,
    EVIDENCE_RELATION_PROGRESS_SCHEMA_VERSION,
    EVIDENCE_RELATION_SCHEMA_VERSION,
    LEGACY_EVIDENCE_RELATION_PROGRESS_SCHEMA_VERSION,
    LEGACY_EVIDENCE_RELATION_PROMPT_VERSION,
    LEGACY_EVIDENCE_RELATION_SCHEMA_VERSION,
    EvidenceCandidateSet,
    EvidenceRelation,
    EvidenceRelationProgressRecord,
)
from legal_rag.benchmark_synthesis.fact_extraction import (
    EXTRACTION_PROGRESS_SCHEMA_VERSION,
    LEGACY_EXTRACTION_PROGRESS_SCHEMA_VERSION,
    ExtractionProgressRecord,
)
from legal_rag.benchmark_synthesis.models import LEGAL_FACT_SCHEMA_VERSION, EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.multi_hop_3 import (
    MULTI_HOP_3_BUNDLE_SCHEMA_VERSION,
    MULTI_HOP_3_CANDIDATE_SET_SCHEMA_VERSION,
    MULTI_HOP_3_PROGRESS_SCHEMA_VERSION,
    ConfirmedMultiHop3Bundle,
    MultiHop3CandidateSet,
    MultiHop3ProgressRecord,
)
from legal_rag.benchmark_synthesis.multi_hop_4 import (
    MULTI_HOP_4_BUNDLE_SCHEMA_VERSION,
    MULTI_HOP_4_CANDIDATE_SET_SCHEMA_VERSION,
    MULTI_HOP_4_PROGRESS_SCHEMA_VERSION,
    ConfirmedMultiHop4Bundle,
    MultiHop4Candidate,
    MultiHop4CandidateSet,
    MultiHop4ProgressRecord,
)
from legal_rag.benchmark_synthesis.multi_hop_5 import (
    MULTI_HOP_5_BUNDLE_SCHEMA_VERSION,
    MULTI_HOP_5_CANDIDATE_SET_SCHEMA_VERSION,
    MULTI_HOP_5_PROGRESS_SCHEMA_VERSION,
    ConfirmedMultiHop5Bundle,
    MultiHop5Candidate,
    MultiHop5CandidateSet,
    MultiHop5ProgressRecord,
)
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
            if payload["schema_version"] not in {
                LEGACY_CHUNK_SCREENING_SCHEMA_VERSION,
                CHUNK_SCREENING_SCHEMA_VERSION,
            }:
                msg = f"Unsupported chunk-screening schema: {payload['schema_version']}"
                raise ValueError(msg)
            records.append(
                ChunkScreeningRecord(
                    chunk_id=payload["chunk_id"],
                    has_penalty_content=payload["has_penalty_content"],
                    has_document_metadata=payload["has_document_metadata"],
                    reason=payload["reason"],
                    prompt_version=payload.get("prompt_version"),
                    model=payload.get("model"),
                    screening_run_id=payload.get("screening_run_id"),
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
            if payload["schema_version"] not in {
                LEGACY_EXTRACTION_PROGRESS_SCHEMA_VERSION,
                EXTRACTION_PROGRESS_SCHEMA_VERSION,
            }:
                msg = f"Unsupported extraction-progress schema: {payload['schema_version']}"
                raise ValueError(msg)
            records.append(
                ExtractionProgressRecord(
                    chunk_id=payload["chunk_id"],
                    prompt_version=payload.get("prompt_version"),
                    model=payload.get("model"),
                    extraction_run_id=payload.get("extraction_run_id"),
                )
            )
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
            if payload["schema_version"] not in {
                LEGACY_EVIDENCE_RELATION_SCHEMA_VERSION,
                EVIDENCE_RELATION_SCHEMA_VERSION,
            }:
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
                    prompt_version=payload.get(
                        "prompt_version",
                        LEGACY_EVIDENCE_RELATION_PROMPT_VERSION,
                    ),
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
            if payload["schema_version"] not in {
                LEGACY_EVIDENCE_RELATION_PROGRESS_SCHEMA_VERSION,
                EVIDENCE_RELATION_PROGRESS_SCHEMA_VERSION,
            }:
                msg = f"Unsupported evidence-relation-progress schema: {payload['schema_version']}"
                raise ValueError(msg)
            records.append(
                EvidenceRelationProgressRecord(
                    fact_id_a=payload["fact_id_a"],
                    fact_id_b=payload["fact_id_b"],
                    run_id=payload["run_id"],
                    prompt_version=payload.get(
                        "prompt_version",
                        LEGACY_EVIDENCE_RELATION_PROMPT_VERSION,
                    ),
                    confirmed_by_model=payload.get("confirmed_by_model"),
                )
            )
    return tuple(records)


def write_scenarios(output_path: Path, scenarios: Sequence[Scenario]) -> None:
    """Atomically replace one UTF-8 JSONL output with scenario records."""
    write_jsonl_atomic(output_path, (scenario.to_dict() for scenario in scenarios))


def write_multi_hop_3_candidate_set(output_path: Path, candidate_set: MultiHop3CandidateSet) -> None:
    """Atomically replace one multi-hop-3 candidate cache."""
    write_json_atomic(output_path, candidate_set.to_dict())


def read_multi_hop_3_candidate_set(input_path: Path) -> MultiHop3CandidateSet:
    """Read and validate one multi-hop-3 candidate cache."""
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if payload["schema_version"] != MULTI_HOP_3_CANDIDATE_SET_SCHEMA_VERSION:
        msg = f"Unsupported multi-hop-3 candidate schema: {payload['schema_version']}"
        raise ValueError(msg)
    triples = tuple(tuple(triple) for triple in payload["triples"])
    if any(len(triple) != 3 or tuple(sorted(set(triple))) != triple for triple in triples):
        msg = "Multi-hop-3 candidates must contain canonical distinct fact-id triples"
        raise ValueError(msg)
    if len(triples) != len(set(triples)):
        msg = "Multi-hop-3 candidates must be unique"
        raise ValueError(msg)
    return MultiHop3CandidateSet(run_id=payload["run_id"], triples=triples)


def write_multi_hop_3_bundles(output_path: Path, bundles: Sequence[ConfirmedMultiHop3Bundle]) -> None:
    """Atomically replace one confirmed multi-hop-3 bundle output."""
    write_jsonl_atomic(output_path, (bundle.to_dict() for bundle in bundles))


def read_multi_hop_3_bundles(input_path: Path) -> tuple[ConfirmedMultiHop3Bundle, ...]:
    """Read confirmed multi-hop-3 bundles."""
    bundles: list[ConfirmedMultiHop3Bundle] = []
    with input_path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != MULTI_HOP_3_BUNDLE_SCHEMA_VERSION:
                msg = f"Unsupported multi-hop-3 bundle schema: {payload['schema_version']}"
                raise ValueError(msg)
            bundles.append(
                ConfirmedMultiHop3Bundle(
                    bundle_id=payload["bundle_id"],
                    fact_ids=tuple(payload["fact_ids"]),
                    bridge=payload["bridge"],
                    confirmed_by_model=payload["confirmed_by_model"],
                    run_id=payload["run_id"],
                    prompt_version=payload["prompt_version"],
                )
            )
    return tuple(bundles)


def write_multi_hop_3_progress(output_path: Path, records: Sequence[MultiHop3ProgressRecord]) -> None:
    """Atomically replace one multi-hop-3 progress output."""
    write_jsonl_atomic(output_path, (record.to_dict() for record in records))


def read_multi_hop_3_progress(input_path: Path) -> tuple[MultiHop3ProgressRecord, ...]:
    """Read multi-hop-3 confirmation progress."""
    records: list[MultiHop3ProgressRecord] = []
    with input_path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != MULTI_HOP_3_PROGRESS_SCHEMA_VERSION:
                msg = f"Unsupported multi-hop-3 progress schema: {payload['schema_version']}"
                raise ValueError(msg)
            records.append(
                MultiHop3ProgressRecord(
                    candidate_id=payload["candidate_id"],
                    run_id=payload["run_id"],
                    prompt_version=payload["prompt_version"],
                    confirmed_by_model=payload.get("confirmed_by_model"),
                )
            )
    return tuple(records)


def write_multi_hop_4_candidate_set(output_path: Path, candidate_set: MultiHop4CandidateSet) -> None:
    """Atomically replace one multi-hop-4 candidate cache."""
    write_json_atomic(output_path, candidate_set.to_dict())


def read_multi_hop_4_candidate_set(input_path: Path) -> MultiHop4CandidateSet:
    """Read and validate one multi-hop-4 candidate cache."""
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if payload["schema_version"] != MULTI_HOP_4_CANDIDATE_SET_SCHEMA_VERSION:
        msg = f"Unsupported multi-hop-4 candidate schema: {payload['schema_version']}"
        raise ValueError(msg)
    candidates = tuple(
        MultiHop4Candidate(
            fact_ids=tuple(candidate["fact_ids"]),
            source_bundle_id=candidate["source_bundle_id"],
        )
        for candidate in payload["candidates"]
    )
    if any(
        len(candidate.fact_ids) != 4 or tuple(sorted(set(candidate.fact_ids))) != candidate.fact_ids
        for candidate in candidates
    ):
        msg = "Multi-hop-4 candidates must contain canonical distinct four-fact IDs"
        raise ValueError(msg)
    if len({candidate.fact_ids for candidate in candidates}) != len(candidates):
        msg = "Multi-hop-4 candidates must contain unique fact-id sets"
        raise ValueError(msg)
    return MultiHop4CandidateSet(run_id=payload["run_id"], candidates=candidates)


def write_multi_hop_4_bundles(output_path: Path, bundles: Sequence[ConfirmedMultiHop4Bundle]) -> None:
    """Atomically replace one confirmed multi-hop-4 bundle output."""
    write_jsonl_atomic(output_path, (bundle.to_dict() for bundle in bundles))


def read_multi_hop_4_bundles(input_path: Path) -> tuple[ConfirmedMultiHop4Bundle, ...]:
    """Read confirmed multi-hop-4 bundles."""
    bundles: list[ConfirmedMultiHop4Bundle] = []
    with input_path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != MULTI_HOP_4_BUNDLE_SCHEMA_VERSION:
                msg = f"Unsupported multi-hop-4 bundle schema: {payload['schema_version']}"
                raise ValueError(msg)
            fact_ids = tuple(payload["fact_ids"])
            if len(fact_ids) != 4 or tuple(sorted(set(fact_ids))) != fact_ids:
                msg = "Multi-hop-4 bundles must contain canonical distinct four-fact IDs"
                raise ValueError(msg)
            bundles.append(
                ConfirmedMultiHop4Bundle(
                    bundle_id=payload["bundle_id"],
                    fact_ids=fact_ids,
                    bridge=payload["bridge"],
                    confirmed_by_model=payload["confirmed_by_model"],
                    run_id=payload["run_id"],
                    prompt_version=payload["prompt_version"],
                )
            )
    return tuple(bundles)


def write_multi_hop_4_progress(output_path: Path, records: Sequence[MultiHop4ProgressRecord]) -> None:
    """Atomically replace one multi-hop-4 progress output."""
    write_jsonl_atomic(output_path, (record.to_dict() for record in records))


def read_multi_hop_4_progress(input_path: Path) -> tuple[MultiHop4ProgressRecord, ...]:
    """Read multi-hop-4 confirmation progress."""
    records: list[MultiHop4ProgressRecord] = []
    with input_path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != MULTI_HOP_4_PROGRESS_SCHEMA_VERSION:
                msg = f"Unsupported multi-hop-4 progress schema: {payload['schema_version']}"
                raise ValueError(msg)
            records.append(
                MultiHop4ProgressRecord(
                    candidate_id=payload["candidate_id"],
                    run_id=payload["run_id"],
                    prompt_version=payload["prompt_version"],
                    confirmed_by_model=payload.get("confirmed_by_model"),
                )
            )
    return tuple(records)


def write_multi_hop_5_candidate_set(output_path: Path, candidate_set: MultiHop5CandidateSet) -> None:
    """Atomically replace one multi-hop-5 candidate cache."""
    write_json_atomic(output_path, candidate_set.to_dict())


def read_multi_hop_5_candidate_set(input_path: Path) -> MultiHop5CandidateSet:
    """Read and validate one multi-hop-5 candidate cache."""
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if payload["schema_version"] != MULTI_HOP_5_CANDIDATE_SET_SCHEMA_VERSION:
        msg = f"Unsupported multi-hop-5 candidate schema: {payload['schema_version']}"
        raise ValueError(msg)
    candidates = tuple(
        MultiHop5Candidate(
            fact_ids=tuple(candidate["fact_ids"]),
            source_bundle_id=candidate["source_bundle_id"],
        )
        for candidate in payload["candidates"]
    )
    if any(
        len(candidate.fact_ids) != 5 or tuple(sorted(set(candidate.fact_ids))) != candidate.fact_ids
        for candidate in candidates
    ):
        msg = "Multi-hop-5 candidates must contain canonical distinct five-fact IDs"
        raise ValueError(msg)
    if len({candidate.fact_ids for candidate in candidates}) != len(candidates):
        msg = "Multi-hop-5 candidates must contain unique fact-id sets"
        raise ValueError(msg)
    return MultiHop5CandidateSet(run_id=payload["run_id"], candidates=candidates)


def write_multi_hop_5_bundles(output_path: Path, bundles: Sequence[ConfirmedMultiHop5Bundle]) -> None:
    """Atomically replace one confirmed multi-hop-5 bundle output."""
    write_jsonl_atomic(output_path, (bundle.to_dict() for bundle in bundles))


def read_multi_hop_5_bundles(input_path: Path) -> tuple[ConfirmedMultiHop5Bundle, ...]:
    """Read confirmed multi-hop-5 bundles."""
    bundles: list[ConfirmedMultiHop5Bundle] = []
    with input_path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != MULTI_HOP_5_BUNDLE_SCHEMA_VERSION:
                msg = f"Unsupported multi-hop-5 bundle schema: {payload['schema_version']}"
                raise ValueError(msg)
            fact_ids = tuple(payload["fact_ids"])
            if len(fact_ids) != 5 or tuple(sorted(set(fact_ids))) != fact_ids:
                msg = "Multi-hop-5 bundles must contain canonical distinct five-fact IDs"
                raise ValueError(msg)
            bundles.append(
                ConfirmedMultiHop5Bundle(
                    bundle_id=payload["bundle_id"],
                    fact_ids=fact_ids,
                    bridge=payload["bridge"],
                    confirmed_by_model=payload["confirmed_by_model"],
                    run_id=payload["run_id"],
                    prompt_version=payload["prompt_version"],
                )
            )
    return tuple(bundles)


def write_multi_hop_5_progress(output_path: Path, records: Sequence[MultiHop5ProgressRecord]) -> None:
    """Atomically replace one multi-hop-5 progress output."""
    write_jsonl_atomic(output_path, (record.to_dict() for record in records))


def read_multi_hop_5_progress(input_path: Path) -> tuple[MultiHop5ProgressRecord, ...]:
    """Read multi-hop-5 confirmation progress."""
    records: list[MultiHop5ProgressRecord] = []
    with input_path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != MULTI_HOP_5_PROGRESS_SCHEMA_VERSION:
                msg = f"Unsupported multi-hop-5 progress schema: {payload['schema_version']}"
                raise ValueError(msg)
            records.append(
                MultiHop5ProgressRecord(
                    candidate_id=payload["candidate_id"],
                    run_id=payload["run_id"],
                    prompt_version=payload["prompt_version"],
                    confirmed_by_model=payload.get("confirmed_by_model"),
                )
            )
    return tuple(records)


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
        query=payload["query"],
        reference_answer=payload["reference_answer"],
        reasoning=payload["reasoning"],
        required_claims=tuple(
            RequiredClaim(
                claim=claim["claim"],
                fact_id=claim["fact_id"],
                evidence=tuple(ClaimEvidence(chunk_id=entry["chunk_id"]) for entry in claim["evidence"]),
            )
            for claim in payload["required_claims"]
        ),
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
