"""Interactive terminal review for legal facts awaiting expert approval."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from legal_rag.benchmark_synthesis.models import LegalFact


def _parser() -> argparse.ArgumentParser:
    """Build the fact-review command-line interface."""
    parser = argparse.ArgumentParser(description="Review legal facts awaiting expert approval.")
    parser.add_argument("facts_path", type=Path, help="Legal-facts JSONL path")
    parser.add_argument("ledger_path", type=Path, help="Fact-ledger JSONL path, read and rewritten")
    parser.add_argument("--reviewer", required=True, help="Reviewer identity recorded on each decision")
    return parser


def _print_fact(fact: LegalFact) -> None:
    """Print one legal fact for terminal review."""
    print(f"\nfact_id: {fact.fact_id}")
    print(f"document: {fact.document_number} — {fact.provision_path}")
    print(f"conduct: {fact.conduct}")
    print(f"subject: {fact.subject}")
    if fact.conditions:
        print(f"conditions: {', '.join(fact.conditions)}")
    print(f"monetary_penalty: {fact.monetary_penalty}")
    print(f"license_points_deducted: {fact.license_points_deducted}")
    if fact.additional_penalties:
        print(f"additional_penalties: {', '.join(fact.additional_penalties)}")
    if fact.remedial_measures:
        print(f"remedial_measures: {', '.join(fact.remedial_measures)}")
    print(f"valid_from: {fact.valid_from}  valid_to: {fact.valid_to}")
    for evidence in fact.evidence:
        print(f"  [{evidence.chunk_id}] {evidence.quoted_span}")


def main() -> None:
    """Walk every fact awaiting expert review and record an approve/reject decision."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.benchmark_synthesis.fact_ledger import approve, reject
    from legal_rag.benchmark_synthesis.writer import read_fact_ledger, read_legal_facts, write_fact_ledger

    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)

    facts_by_id = {fact.fact_id: fact for fact in read_legal_facts(arguments.facts_path)}
    entries = list(read_fact_ledger(arguments.ledger_path))
    approved_count = 0
    rejected_count = 0

    for index, entry in enumerate(entries):
        if entry.status != "needs_expert_review":
            continue
        _print_fact(facts_by_id[entry.fact_id])
        decision = input("Approve? [y]es / [n]o / [s]kip: ").strip().lower()
        reviewed_at = datetime.now(UTC)
        if decision == "y":
            entries[index] = approve(entry, reviewer=arguments.reviewer, reviewed_at=reviewed_at)
            approved_count += 1
        elif decision == "n":
            entries[index] = reject(entry, reviewer=arguments.reviewer, reviewed_at=reviewed_at)
            rejected_count += 1

    write_fact_ledger(arguments.ledger_path, entries)
    logger.info(
        "Reviewed facts: {} approved, {} rejected, ledger written to {}",
        approved_count,
        rejected_count,
        arguments.ledger_path,
    )


if __name__ == "__main__":
    main()
