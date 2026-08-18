"""Command-line interface for ontology population input readiness."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the population-preflight command-line parser."""
    parser = argparse.ArgumentParser(description="Validate ontology population inputs without calling an LLM.")
    parser.add_argument("chunks_path", type=Path, help="Final chunk JSONL path")
    parser.add_argument("screening_path", type=Path, help="Chunk-screening JSONL path")
    parser.add_argument("metadata_path", type=Path, help="Document-metadata JSONL path")
    parser.add_argument("contract_path", type=Path, help="Versioned ontology population contract YAML path")
    parser.add_argument("report_output_path", type=Path, help="Machine-readable population-readiness JSON report")
    parser.add_argument("--target-limit", type=int, help="Maximum accepted chunks selected for a pilot run")
    return parser


def main() -> None:
    """Persist structural readiness and fail after recording any blocking issue."""
    from legal_rag.io_utils import write_json_atomic
    from legal_rag.ontology_population.config import PopulationSettings
    from legal_rag.ontology_population.preflight import load_population_inputs, run_population_preflight

    arguments = _parser().parse_args()
    settings = PopulationSettings.from_path(arguments.contract_path)
    bundle = load_population_inputs(arguments.chunks_path, arguments.screening_path, arguments.metadata_path)
    report = run_population_preflight(
        bundle,
        settings,
        (
            ("chunks", arguments.chunks_path),
            ("screening", arguments.screening_path),
            ("metadata", arguments.metadata_path),
            ("population_contract", arguments.contract_path),
        ),
        datetime.now(UTC),
        target_limit=arguments.target_limit,
    )
    write_json_atomic(arguments.report_output_path, report.to_dict())
    logger.info(
        "Ontology population preflight completed with status {}; {} targets in {} estimated LLM batches",
        report.overall_status,
        len(report.selected_target_ids),
        report.estimated_llm_batches,
    )
    if report.overall_status != "ready":
        msg = f"Ontology population inputs are blocked; inspect {arguments.report_output_path}"
        raise ValueError(msg)


if __name__ == "__main__":
    main()
