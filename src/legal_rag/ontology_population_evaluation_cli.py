"""Command-line interface for ontology population gold evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the population-evaluation command parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate ontology population JSONL against reviewed deterministic gold assertions."
    )
    parser.add_argument("records_path", type=Path, help="Ontology population JSONL path")
    parser.add_argument("progress_path", type=Path, help="Ontology population progress JSONL path")
    parser.add_argument("gold_path", type=Path, help="Reviewed population gold YAML path")
    parser.add_argument("report_path", type=Path, help="Machine-readable evaluation report destination")
    return parser


def main() -> None:
    """Evaluate one completed population pilot and persist its report."""
    from legal_rag.io_utils import write_json_atomic
    from legal_rag.ontology_population.evaluation import PopulationGoldStandard, evaluate_population_records
    from legal_rag.ontology_population.writer import read_population_progress, read_population_records

    arguments = _parser().parse_args()
    report = evaluate_population_records(
        read_population_records(arguments.records_path),
        read_population_progress(arguments.progress_path),
        PopulationGoldStandard.from_path(arguments.gold_path),
    )
    write_json_atomic(arguments.report_path, report.to_dict())
    logger.info(
        "Evaluated {} population records across {} target chunks: {} issue(s)",
        report.population_record_count,
        report.target_chunk_count,
        len(report.issues),
    )
    if report.overall_status != "passed":
        msg = f"Ontology population evaluation failed; inspect {arguments.report_path}"
        raise ValueError(msg)
