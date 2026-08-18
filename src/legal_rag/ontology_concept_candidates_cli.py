"""Command-line interface for exhaustive ontology concept candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the concept-candidate command-line parser."""
    parser = argparse.ArgumentParser(
        description="Build exhaustive same-kind concept pairs without semantic matching or scoring."
    )
    parser.add_argument("canonical_records_path", type=Path, help="Canonical multi-document population JSONL path")
    parser.add_argument("concept_contract_path", type=Path, help="Versioned concept-resolution contract YAML")
    parser.add_argument("output_path", type=Path, help="Concept candidate-set JSON destination")
    return parser


def main() -> None:
    """Build and persist structural concept candidate pairs offline."""
    from legal_rag.ontology_concept.candidates import build_concept_candidates
    from legal_rag.ontology_concept.config import ConceptResolutionSettings
    from legal_rag.ontology_concept.writer import write_concept_candidates
    from legal_rag.ontology_population.writer import read_population_records

    arguments = _parser().parse_args()
    records = read_population_records(arguments.canonical_records_path)
    settings = ConceptResolutionSettings.from_path(arguments.concept_contract_path)
    candidates = build_concept_candidates(records, settings)
    write_concept_candidates(arguments.output_path, candidates)
    logger.info(
        "Wrote {} concept occurrences and {} exhaustive candidate pairs with run id {} to {}",
        len(candidates.occurrences),
        len(candidates.pairs),
        candidates.run_id,
        arguments.output_path,
    )


if __name__ == "__main__":
    main()
