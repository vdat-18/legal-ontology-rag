"""Atomic persistence for ground-truth synthesis JSONL outputs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from legal_rag.benchmark_synthesis.binding_gate import BindingCheckResult, BindingViolation
from legal_rag.benchmark_synthesis.candidate_generation import (
    CANDIDATE_SCHEMA_VERSION,
    CandidateDraft,
    ClaimEvidence,
    FactAnswerFields,
    RequiredClaim,
)
from legal_rag.benchmark_synthesis.candidate_record import CandidateRecord
from legal_rag.benchmark_synthesis.evidence_graph import EVIDENCE_RELATION_SCHEMA_VERSION, EvidenceRelation
from legal_rag.benchmark_synthesis.models import LEGAL_FACT_SCHEMA_VERSION, EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.scenario_sampling import SCENARIO_SCHEMA_VERSION, Scenario
from legal_rag.benchmark_synthesis.verification import VerificationResult
from legal_rag.io_utils import write_jsonl_atomic


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
                    fact_content_sha256=payload["fact_content_sha256"],
                    conduct=payload["conduct"],
                    subject=payload["subject"],
                    conditions=tuple(payload["conditions"]),
                    monetary_penalty=payload["monetary_penalty"],
                    license_points_deducted=payload["license_points_deducted"],
                    additional_penalties=tuple(payload["additional_penalties"]),
                    remedial_measures=tuple(payload["remedial_measures"]),
                    valid_from=_parse_date(payload["valid_from"]),
                    valid_to=_parse_date(payload["valid_to"]),
                    document_number=payload["document_number"],
                    provision_path=payload["provision_path"],
                    evidence=tuple(
                        EvidenceReference(chunk_id=entry["chunk_id"], quoted_span=entry["quoted_span"])
                        for entry in payload["evidence"]
                    ),
                )
            )
    return tuple(facts)


def write_evidence_relations(output_path: Path, relations: Sequence[EvidenceRelation]) -> None:
    """Atomically replace one UTF-8 JSONL output with evidence-relation records."""
    write_jsonl_atomic(output_path, (relation.to_dict() for relation in relations))


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
                    relation_type=payload["relation_type"],
                    from_fact_id=payload["from_fact_id"],
                    to_fact_id=payload["to_fact_id"],
                    bridge=tuple(payload["bridge"]),
                    confirmed_by_model=payload["confirmed_by_model"],
                )
            )
    return tuple(relations)


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
                    query_type=payload["query_type"],
                    query_form=payload["query_form"],
                    answerability=payload["answerability"],
                    persona=payload["persona"],
                    query_style=payload["query_style"],
                    query_length=payload["query_length"],
                    reasoning_hops=payload["reasoning_hops"],
                    as_of_date=_parse_date(payload["as_of_date"]),
                    fact_ids=tuple(payload["fact_ids"]),
                    bridge=tuple(payload["bridge"]),
                    gold_chunk_ids=tuple(payload["gold_chunk_ids"]),
                )
            )
    return tuple(scenarios)


def write_candidates(output_path: Path, records: Sequence[CandidateRecord]) -> None:
    """Atomically replace one UTF-8 JSONL output with candidate records."""
    write_jsonl_atomic(output_path, (record.to_dict() for record in records))


def read_candidates(input_path: Path) -> tuple[CandidateRecord, ...]:
    """Read a candidates JSONL file back into typed records."""
    records: list[CandidateRecord] = []
    with input_path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != CANDIDATE_SCHEMA_VERSION:
                msg = f"Unsupported candidate schema: {payload['schema_version']}"
                raise ValueError(msg)
            records.append(
                CandidateRecord(
                    draft=_parse_candidate_draft(payload),
                    binding=_parse_binding_check_result(payload["binding_check"]),
                    verification=(
                        _parse_verification_result(payload["verification"])
                        if payload["verification"] is not None
                        else None
                    ),
                )
            )
    return tuple(records)


def _parse_candidate_draft(payload: dict[str, object]) -> CandidateDraft:
    """Parse the CandidateDraft portion of one candidate JSONL record."""
    return CandidateDraft(
        sample_id=payload["sample_id"],
        scenario_id=payload["scenario_id"],
        query_vi=payload["query_vi"],
        reference_answer_vi=payload["reference_answer_vi"],
        reasoning_vi=payload["reasoning_vi"],
        required_claims=tuple(
            RequiredClaim(
                claim=claim["claim"],
                fact_id=claim["fact_id"],
                evidence=tuple(
                    ClaimEvidence(chunk_id=entry["chunk_id"], quoted_span=entry["quoted_span"])
                    for entry in claim["evidence"]
                ),
            )
            for claim in payload["required_claims"]
        ),
        answer_fields=tuple(
            FactAnswerFields(
                fact_id=fields["fact_id"],
                monetary_penalty=fields["monetary_penalty"],
                license_points_deducted=fields["license_points_deducted"],
                additional_penalties=tuple(fields["additional_penalties"]),
                remedial_measures=tuple(fields["remedial_measures"]),
            )
            for fields in payload["answer_fields"]
        ),
        evolutions=tuple(payload["evolutions"]),
        gold_chunk_ids=tuple(payload["gold_chunk_ids"]),
        query_type=payload["query_type"],
        query_form=payload["query_form"],
        persona=payload["persona"],
        query_style=payload["query_style"],
        query_length=payload["query_length"],
        answerability=payload["answerability"],
        reasoning_hops=payload["reasoning_hops"],
        as_of_date=_parse_date(payload["as_of_date"]),
        source_fact_ids=tuple(payload["source_fact_ids"]),
        source_fact_content_sha256=tuple(payload["source_fact_content_sha256"]),
        generator_model=payload["generator_model"],
    )


def _parse_binding_check_result(payload: dict[str, object]) -> BindingCheckResult:
    """Parse the gate-1 outcome portion of one candidate JSONL record."""
    return BindingCheckResult(
        passed=payload["passed"],
        grounding_score=payload["grounding_score"],
        violations=tuple(
            BindingViolation(claim=violation["claim"], reason=violation["reason"])
            for violation in payload["violations"]
        ),
    )


def _parse_verification_result(payload: dict[str, object]) -> VerificationResult:
    """Parse the gate-2 outcome portion of one candidate JSONL record."""
    return VerificationResult(
        passed=payload["passed"],
        is_standalone=payload["is_standalone"],
        is_specific=payload["is_specific"],
        is_question_grounded=payload["is_question_grounded"],
        is_answer_grounded=payload["is_answer_grounded"],
        is_domain_relevant=payload["is_domain_relevant"],
        is_temporal_correct=payload["is_temporal_correct"],
        necessary_reasoning_hops=payload["necessary_reasoning_hops"],
        observed_query_type=payload["observed_query_type"],
        observed_answerability=payload["observed_answerability"],
        reasoning=payload["reasoning"],
        verifier_model=payload["verifier_model"],
    )


def _parse_date(value: str | None) -> date | None:
    """Parse an ISO date string, returning None for a null field."""
    return date.fromisoformat(value) if value else None
