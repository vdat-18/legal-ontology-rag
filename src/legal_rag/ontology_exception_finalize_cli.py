"""Command-line interface for legal exceptions."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the exception-finalization command-line parser."""
    parser = argparse.ArgumentParser(description="Apply typed exceptions and write a canonical ontology graph.")
    parser.add_argument("corpus_manifest_path", type=Path, help="Ordered multi-document corpus manifest JSON path")
    parser.add_argument("canonical_records_path", type=Path, help="Canonical multi-document population JSONL path")
    parser.add_argument("change_events_path", type=Path, help="Legal-change event JSONL path")
    parser.add_argument("proposals_path", type=Path, help="LLM exception proposal JSONL path")
    parser.add_argument("population_contract_path", type=Path, help="Versioned population contract YAML path")
    parser.add_argument("exception_records_output_path", type=Path, help="Canonical legal-exception JSONL destination")
    parser.add_argument("graph_output_path", type=Path, help="Canonical Turtle destination")
    return parser


def main() -> None:
    """Apply exceptions and materialize the complete ontology graph."""
    from legal_rag.ontology_change.corpus import read_change_corpus_manifest, replace_change_corpus_records
    from legal_rag.ontology_change.writer import read_legal_change_events
    from legal_rag.ontology_exception.finalization import finalize_legal_exceptions
    from legal_rag.ontology_exception.materializer import materialize_legal_exception_graph
    from legal_rag.ontology_exception.writer import (
        read_exception_proposals,
        write_legal_exception_records,
    )
    from legal_rag.ontology_population.config import PopulationSettings
    from legal_rag.ontology_population.materializer import write_population_graph
    from legal_rag.ontology_population.writer import read_population_records

    arguments = _parser().parse_args()
    manifest_corpus = read_change_corpus_manifest(arguments.corpus_manifest_path)
    population_records = read_population_records(arguments.canonical_records_path)
    corpus = replace_change_corpus_records(manifest_corpus, population_records)
    change_events = read_legal_change_events(arguments.change_events_path)
    proposals = read_exception_proposals(arguments.proposals_path)
    population_settings = PopulationSettings.from_path(arguments.population_contract_path)
    exceptions = finalize_legal_exceptions(population_records, proposals)
    graph = materialize_legal_exception_graph(
        corpus,
        population_records,
        change_events,
        exceptions,
        population_settings.instance_namespace,
    )
    write_legal_exception_records(arguments.exception_records_output_path, exceptions)
    write_population_graph(arguments.graph_output_path, graph)
    logger.info(
        "Finalized {} legal exceptions into {} RDF triples",
        len(exceptions),
        len(graph),
    )


if __name__ == "__main__":
    main()
