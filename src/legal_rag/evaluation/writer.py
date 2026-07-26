"""Atomic persistence for the fact-ledger pipeline's JSONL outputs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

from legal_rag.evaluation.fact_ledger import FACT_LEDGER_SCHEMA_VERSION, FactLedgerEntry
from legal_rag.evaluation.models import LEGAL_FACT_SCHEMA_VERSION, EvidenceReference, LegalFact
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


def _parse_date(value: str | None) -> date | None:
    """Parse an ISO date string, returning None for a null field."""
    return date.fromisoformat(value) if value else None
