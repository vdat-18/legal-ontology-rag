"""Command-line interface for human-reviewed ontology rule assembly."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the assembly-finalization command-line parser."""
    parser = argparse.ArgumentParser(
        description="Apply accepted human reviews and write a SHACL-validated canonical instance graph."
    )
    parser.add_argument("chunks_path", type=Path, help="Final chunk JSONL path")
    parser.add_argument("metadata_path", type=Path, help="Document-metadata JSONL path")
    parser.add_argument("records_path", type=Path, help="Original population JSONL path")
    parser.add_argument("proposals_path", type=Path, help="LLM assembly-proposal JSONL path")
    parser.add_argument("reviews_path", type=Path, help="Completed human-review JSONL path")
    parser.add_argument("population_contract_path", type=Path, help="Versioned population contract YAML path")
    parser.add_argument("assembly_contract_path", type=Path, help="Versioned assembly contract YAML path")
    parser.add_argument("canonical_records_output_path", type=Path, help="Canonical population JSONL destination")
    parser.add_argument("graph_output_path", type=Path, help="Canonical validated Turtle destination")
    return parser


def main() -> None:
    """Finalize accepted assemblies and validate the canonical graph."""
    from legal_rag.benchmark_synthesis.writer import read_document_metadata
    from legal_rag.chunker.writer import read_chunks
    from legal_rag.ontology_population.assembly_config import AssemblySettings
    from legal_rag.ontology_population.assembly_finalization import finalize_assemblies
    from legal_rag.ontology_population.assembly_writer import read_assembly_proposals, read_assembly_reviews
    from legal_rag.ontology_population.config import PopulationSettings
    from legal_rag.ontology_population.materializer import (
        materialize_population_graph,
        validate_population_graph,
        write_population_graph,
    )
    from legal_rag.ontology_population.writer import read_population_records, write_population_records
    from legal_rag.ontology_validation.paths import OntologyPaths

    arguments = _parser().parse_args()
    chunks = read_chunks(arguments.chunks_path)
    metadata = read_document_metadata(arguments.metadata_path)
    records = read_population_records(arguments.records_path)
    proposals = read_assembly_proposals(arguments.proposals_path)
    reviews = read_assembly_reviews(arguments.reviews_path)
    assembly_settings = AssemblySettings.from_path(arguments.assembly_contract_path)
    population_settings = PopulationSettings.from_path(arguments.population_contract_path)
    canonical = finalize_assemblies(records, proposals, reviews, assembly_settings)
    graph = materialize_population_graph(chunks, metadata, canonical, population_settings.instance_namespace)
    project_root = Path(__file__).resolve().parents[2]
    validate_population_graph(graph, OntologyPaths(project_root))
    write_population_records(arguments.canonical_records_output_path, canonical)
    write_population_graph(arguments.graph_output_path, graph)
    logger.info(
        "Finalized {} source records into {} canonical rules and wrote {} RDF triples",
        len(records),
        len(canonical),
        len(graph),
    )


if __name__ == "__main__":
    main()
