"""Command-line interface for canonical ontology concepts."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the concept-finalization command-line parser."""
    parser = argparse.ArgumentParser(description="Apply typed concept relations and write a canonical ontology graph.")
    parser.add_argument("corpus_manifest_path", type=Path, help="Ordered multi-document corpus manifest JSON path")
    parser.add_argument("canonical_records_path", type=Path, help="Canonical multi-document population JSONL path")
    parser.add_argument("change_events_path", type=Path, help="Legal-change event JSONL path")
    parser.add_argument("exception_records_path", type=Path, help="Canonical legal-exception JSONL path")
    parser.add_argument("candidates_path", type=Path, help="Concept candidate-set JSON path")
    parser.add_argument("proposals_path", type=Path, help="LLM concept proposal JSONL path")
    parser.add_argument("population_contract_path", type=Path, help="Versioned population contract YAML path")
    parser.add_argument("concept_contract_path", type=Path, help="Versioned concept-resolution contract YAML")
    parser.add_argument("concepts_output_path", type=Path, help="Canonical concept JSONL destination")
    parser.add_argument("relations_output_path", type=Path, help="Concept relation JSONL destination")
    parser.add_argument("graph_output_path", type=Path, help="Canonical Turtle destination")
    return parser


def main() -> None:
    """Apply concept decisions and materialize the complete ontology graph."""
    from legal_rag.ontology_change.corpus import read_change_corpus_manifest, replace_change_corpus_records
    from legal_rag.ontology_change.writer import read_legal_change_events
    from legal_rag.ontology_concept.candidates import extract_concept_occurrences
    from legal_rag.ontology_concept.config import ConceptResolutionSettings
    from legal_rag.ontology_concept.finalization import finalize_concept_resolution
    from legal_rag.ontology_concept.materializer import materialize_concept_resolution_graph
    from legal_rag.ontology_concept.writer import (
        read_concept_candidates,
        read_concept_proposals,
        write_canonical_concepts,
        write_canonical_relations,
    )
    from legal_rag.ontology_exception.writer import read_legal_exception_records
    from legal_rag.ontology_population.config import PopulationSettings
    from legal_rag.ontology_population.materializer import write_population_graph
    from legal_rag.ontology_population.writer import read_population_records

    arguments = _parser().parse_args()
    population_records = read_population_records(arguments.canonical_records_path)
    concept_settings = ConceptResolutionSettings.from_path(arguments.concept_contract_path)
    candidates = read_concept_candidates(arguments.candidates_path)
    current_occurrences = extract_concept_occurrences(population_records, concept_settings.included_kinds)
    if candidates.occurrences != current_occurrences:
        msg = "Concept candidates do not match the current canonical population"
        raise ValueError(msg)
    manifest_corpus = read_change_corpus_manifest(arguments.corpus_manifest_path)
    corpus = replace_change_corpus_records(manifest_corpus, population_records)
    change_events = read_legal_change_events(arguments.change_events_path)
    exceptions = read_legal_exception_records(arguments.exception_records_path)
    proposals = read_concept_proposals(arguments.proposals_path)
    concepts, relations = finalize_concept_resolution(
        candidates.occurrences,
        proposals,
    )
    population_settings = PopulationSettings.from_path(arguments.population_contract_path)
    graph = materialize_concept_resolution_graph(
        corpus,
        population_records,
        change_events,
        exceptions,
        candidates.occurrences,
        concepts,
        relations,
        population_settings.instance_namespace,
    )
    write_canonical_concepts(arguments.concepts_output_path, concepts)
    write_canonical_relations(arguments.relations_output_path, relations)
    write_population_graph(arguments.graph_output_path, graph)
    logger.info(
        "Finalized {} canonical concepts and {} relations into {} RDF triples",
        len(concepts),
        len(relations),
        len(graph),
    )


if __name__ == "__main__":
    main()
