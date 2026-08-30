"""Command-line interface for ontology rule assembly."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the assembly-finalization command-line parser."""
    parser = argparse.ArgumentParser(description="Apply typed assembly proposals and write a canonical instance graph.")
    parser.add_argument("chunks_path", type=Path, help="Final chunk JSONL path")
    parser.add_argument("metadata_path", type=Path, help="Document-metadata JSONL path")
    parser.add_argument("records_path", type=Path, help="Original population JSONL path")
    parser.add_argument("proposals_path", type=Path, help="LLM assembly-proposal JSONL path")
    parser.add_argument("population_contract_path", type=Path, help="Versioned population contract YAML path")
    parser.add_argument("canonical_records_output_path", type=Path, help="Canonical population JSONL destination")
    parser.add_argument("graph_output_path", type=Path, help="Canonical Turtle destination")
    return parser


def main() -> None:
    """Apply assembly proposals and materialize the canonical graph."""
    from legal_rag.benchmark_synthesis.writer import read_document_metadata
    from legal_rag.chunker.writer import read_chunks
    from legal_rag.ontology_population.assembly_finalization import finalize_assemblies
    from legal_rag.ontology_population.assembly_writer import read_assembly_proposals
    from legal_rag.ontology_population.config import PopulationSettings
    from legal_rag.ontology_population.materializer import (
        materialize_population_graph,
        write_population_graph,
    )
    from legal_rag.ontology_population.writer import read_population_records, write_population_records

    arguments = _parser().parse_args()
    chunks = read_chunks(arguments.chunks_path)
    metadata = read_document_metadata(arguments.metadata_path)
    records = read_population_records(arguments.records_path)
    proposals = read_assembly_proposals(arguments.proposals_path)
    population_settings = PopulationSettings.from_path(arguments.population_contract_path)
    canonical = finalize_assemblies(records, proposals)
    graph = materialize_population_graph(chunks, metadata, canonical, population_settings.instance_namespace)
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
