"""Command-line interface for exporting candidates as evaluation ground truth."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the ground-truth export command-line interface."""
    parser = argparse.ArgumentParser(description="Export synthesized candidates for Legal RAG evaluation.")
    parser.add_argument("candidates_path", type=Path, help="Synthesized candidates JSONL path")
    parser.add_argument("output_path", type=Path, help="Destination evaluation ground-truth JSONL path")
    parser.add_argument(
        "--multi-turn",
        action="store_true",
        help="Treat candidates input as multi-turn conversations instead of single-turn samples",
    )
    parser.add_argument(
        "--legal-facts",
        type=Path,
        default=None,
        help="Legal-facts JSONL used to persist authoritative version dates for temporal candidates",
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=None,
        help="Reviewed scenarios JSONL; required with --multi-turn to recover each target's bundle kind",
    )
    return parser


def main() -> None:
    """Export synthesized candidates using the schema consumed by legal-evaluate."""
    from legal_rag.benchmark_synthesis.groundtruth_export import (
        export_ground_truth,
        export_multi_turn_ground_truth,
    )
    from legal_rag.benchmark_synthesis.writer import (
        read_candidates,
        read_legal_facts,
        read_multi_turn_candidates_for_export,
        read_scenarios,
    )
    from legal_rag.io_utils import write_jsonl_atomic

    arguments = _parser().parse_args()
    facts_by_id = None
    if arguments.legal_facts is not None:
        facts = read_legal_facts(arguments.legal_facts)
        facts_by_id = {fact.fact_id: fact for fact in facts}
        if len(facts_by_id) != len(facts):
            msg = f"Duplicate fact_id in legal facts: {arguments.legal_facts}"
            raise ValueError(msg)
    if arguments.multi_turn:
        if arguments.scenarios is None:
            msg = "--scenarios is required with --multi-turn"
            raise SystemExit(msg)
        conversations = read_multi_turn_candidates_for_export(arguments.candidates_path)
        scenarios = read_scenarios(arguments.scenarios)
        bundle_kinds = {scenario.scenario_id: scenario.bundle_kind for scenario in scenarios}
        if len(bundle_kinds) != len(scenarios):
            msg = f"Duplicate scenario_id in scenarios: {arguments.scenarios}"
            raise ValueError(msg)
        records = export_multi_turn_ground_truth(conversations, facts_by_id, bundle_kinds)
    else:
        candidates = read_candidates(arguments.candidates_path)
        records = export_ground_truth(candidates, facts_by_id)
    write_jsonl_atomic(arguments.output_path, records)
    logger.info("Exported {} evaluation samples to {}", len(records), arguments.output_path)


if __name__ == "__main__":
    main()
