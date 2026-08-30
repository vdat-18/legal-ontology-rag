"""Command-line interface for cross-document legal changes."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the legal-change finalization command-line parser."""
    parser = argparse.ArgumentParser(description="Apply typed legal changes and write a multi-document ontology graph.")
    parser.add_argument("corpus_manifest_path", type=Path, help="Ordered multi-document corpus manifest JSON path")
    parser.add_argument("proposals_path", type=Path, help="LLM legal-change proposal JSONL path")
    parser.add_argument("population_contract_path", type=Path, help="Versioned population contract YAML path")
    parser.add_argument("canonical_records_output_path", type=Path, help="Canonical multi-document population JSONL")
    parser.add_argument("change_events_output_path", type=Path, help="Legal-change event JSONL")
    parser.add_argument("graph_output_path", type=Path, help="Canonical Turtle destination")
    return parser


def main() -> None:
    """Apply legal changes and materialize the complete corpus graph."""
    from legal_rag.ontology_change.corpus import read_change_corpus_manifest
    from legal_rag.ontology_change.finalization import finalize_legal_changes
    from legal_rag.ontology_change.materializer import materialize_legal_change_graph
    from legal_rag.ontology_change.writer import (
        read_change_proposals,
        write_legal_change_events,
    )
    from legal_rag.ontology_population.config import PopulationSettings
    from legal_rag.ontology_population.materializer import write_population_graph
    from legal_rag.ontology_population.writer import write_population_records

    arguments = _parser().parse_args()
    corpus = read_change_corpus_manifest(arguments.corpus_manifest_path)
    records = tuple(record for document in corpus.documents for record in document.records)
    proposals = read_change_proposals(arguments.proposals_path)
    population_settings = PopulationSettings.from_path(arguments.population_contract_path)
    canonical, events = finalize_legal_changes(records, proposals)
    graph = materialize_legal_change_graph(corpus, canonical, events, population_settings.instance_namespace)
    write_population_records(arguments.canonical_records_output_path, canonical)
    write_legal_change_events(arguments.change_events_output_path, events)
    write_population_graph(arguments.graph_output_path, graph)
    logger.info(
        "Finalized {} rule versions with {} legal changes into {} RDF triples",
        len(canonical),
        len(events),
        len(graph),
    )


if __name__ == "__main__":
    main()
