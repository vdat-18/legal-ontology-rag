"""Command-line interface for structural cross-chunk assembly candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the assembly-candidate command-line parser."""
    parser = argparse.ArgumentParser(
        description="Build bounded structural candidate windows for downstream LLM assembly confirmation."
    )
    parser.add_argument("chunks_path", type=Path, help="Final chunk JSONL path")
    parser.add_argument("records_path", type=Path, help="Ontology population JSONL path")
    parser.add_argument("assembly_contract_path", type=Path, help="Versioned assembly contract YAML path")
    parser.add_argument("output_path", type=Path, help="Assembly candidate-set JSON destination")
    return parser


def main() -> None:
    """Build and persist structural candidate windows without semantic decisions."""
    from legal_rag.chunker.writer import read_chunks
    from legal_rag.ontology_population.assembly_candidates import build_assembly_candidates
    from legal_rag.ontology_population.assembly_config import AssemblySettings
    from legal_rag.ontology_population.assembly_writer import write_assembly_candidates
    from legal_rag.ontology_population.writer import read_population_records

    arguments = _parser().parse_args()
    chunks = read_chunks(arguments.chunks_path)
    records = read_population_records(arguments.records_path)
    settings = AssemblySettings.from_path(arguments.assembly_contract_path)
    candidates = build_assembly_candidates(records, chunks, settings)
    write_assembly_candidates(arguments.output_path, candidates)
    logger.info(
        "Wrote {} structural assembly candidate windows with run id {} to {}",
        len(candidates.windows),
        candidates.run_id,
        arguments.output_path,
    )


if __name__ == "__main__":
    main()
