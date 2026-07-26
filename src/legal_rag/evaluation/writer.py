"""Atomic persistence for the fact-ledger pipeline's JSONL outputs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

from legal_rag.evaluation.evidence_graph import EVIDENCE_RELATION_SCHEMA_VERSION, EvidenceRelation
from legal_rag.evaluation.fact_ledger import FACT_LEDGER_SCHEMA_VERSION, FactLedgerEntry
from legal_rag.evaluation.models import LEGAL_FACT_SCHEMA_VERSION, EvidenceReference, LegalFact
from legal_rag.evaluation.scenario_sampling import SCENARIO_SCHEMA_VERSION, Scenario
from legal_rag.io_utils import write_jsonl_atomic


def write_legal_facts(output_path: Path, facts: Sequence[LegalFact]) -> None:
    """Atomically replace one UTF-8 JSONL output with legal-fact records."""
    write_jsonl_atomic(output_path, (fact.to_dict() for fact in facts))


def write_fact_ledger(output_path: Path, entries: Sequence[FactLedgerEntry]) -> None:
    """Atomically replace one UTF-8 JSONL output with fact-ledger records."""
    write_jsonl_atomic(output_path, (entry.to_dict() for entry in entries))


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
                        EvidenceReference(
                            chunk_id=entry["chunk_id"],
                            quoted_span=entry["quoted_span"],
                            unit_ids=tuple(entry["unit_ids"]),
                            source_refs=tuple(entry["source_refs"]),
                        )
                        for entry in payload["evidence"]
                    ),
                )
            )
    return tuple(facts)


def read_fact_ledger(input_path: Path) -> tuple[FactLedgerEntry, ...]:
    """Read a fact-ledger JSONL file back into typed entries.

    A missing file means no ledger exists yet, which is the expected state
    before the first extraction run; it is not an error condition.
    """
    if not input_path.exists():
        return ()
    entries: list[FactLedgerEntry] = []
    with input_path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != FACT_LEDGER_SCHEMA_VERSION:
                msg = f"Unsupported fact-ledger schema: {payload['schema_version']}"
                raise ValueError(msg)
            entries.append(
                FactLedgerEntry(
                    fact_id=payload["fact_id"],
                    fact_content_sha256=payload["fact_content_sha256"],
                    status=payload["status"],
                    reviewer=payload["reviewer"],
                    reviewed_at=datetime.fromisoformat(payload["reviewed_at"]) if payload["reviewed_at"] else None,
                )
            )
    return tuple(entries)


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


def _parse_date(value: str | None) -> date | None:
    """Parse an ISO date string, returning None for a null field."""
    return date.fromisoformat(value) if value else None
