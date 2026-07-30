"""Command-line interface for evidence-relation confirmation and scenario sampling."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the evidence-graph command-line interface."""
    parser = argparse.ArgumentParser(
        description="Confirm evidence relations between approved facts and sample scenarios."
    )
    parser.add_argument("facts_path", type=Path, help="Legal-facts JSONL path")
    parser.add_argument("relations_output_path", type=Path, help="Destination evidence-relations JSONL path")
    parser.add_argument(
        "relations_progress_output_path", type=Path, help="Destination evidence-relation-progress JSONL path"
    )
    parser.add_argument("scenarios_output_path", type=Path, help="Destination scenarios JSONL path")
    parser.add_argument("--query-taxonomy", type=Path, required=True, help="Query-taxonomy config JSON path")
    parser.add_argument("--personas", type=Path, required=True, help="Persona config JSON path")
    parser.add_argument(
        "--scenario-distribution", type=Path, required=True, help="Scenario-distribution config JSON path"
    )
    parser.add_argument("--total-samples", type=int, required=True, help="Target scenario count across all bundles")
    parser.add_argument("--synthesis-config", type=Path, required=True, help="Versioned synthesis config JSON path")
    return parser


def main() -> None:
    """Confirm evidence relations not yet checked among facts, then sample scenarios from the result."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.benchmark_synthesis.config import SynthesisSettings
    from legal_rag.benchmark_synthesis.evidence_graph import (
        BamlEvidenceRelationClient,
        EvidenceRelation,
        EvidenceRelationProgressRecord,
        confirm_evidence_relations,
        find_candidate_fact_pairs,
        sample_bundles,
    )
    from legal_rag.benchmark_synthesis.scenario_sampling import (
        ScenarioSamplingConfig,
        load_personas,
        load_query_taxonomy,
        load_scenario_distribution,
        sample_scenarios,
    )
    from legal_rag.benchmark_synthesis.writer import (
        read_evidence_relation_progress,
        read_evidence_relations,
        read_legal_facts,
        write_evidence_relation_progress,
        write_evidence_relations,
        write_scenarios,
    )
    from legal_rag.embedding.encoder import EmbeddingEncoder
    from legal_rag.embedding.profiles import load_configured_embedding_profile
    from legal_rag.settings import Settings

    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    settings = Settings.from_environment()
    synthesis_settings = SynthesisSettings.from_path(arguments.synthesis_config)

    facts = read_legal_facts(arguments.facts_path)
    logger.info("Loaded {} facts from {}", len(facts), arguments.facts_path)

    profile = load_configured_embedding_profile(synthesis_settings.embedding_profile)
    encoder = EmbeddingEncoder.from_profile(profile)
    candidate_pairs = find_candidate_fact_pairs(
        facts,
        encoder,
        max_candidates_per_fact=synthesis_settings.relation_max_candidates_per_fact,
        min_similarity=synthesis_settings.relation_min_similarity,
        pool_multiplier=synthesis_settings.relation_pool_multiplier,
        rrf_k=synthesis_settings.relation_rrf_k,
    )

    existing_relations = (
        read_evidence_relations(arguments.relations_output_path) if arguments.relations_output_path.exists() else ()
    )
    relations_by_id = {relation.relation_id: relation for relation in existing_relations}
    existing_progress = (
        read_evidence_relation_progress(arguments.relations_progress_output_path)
        if arguments.relations_progress_output_path.exists()
        else ()
    )
    checked_pairs = {(record.fact_id_a, record.fact_id_b) for record in existing_progress}
    remaining_pairs = tuple(pair for pair in candidate_pairs if pair not in checked_pairs)
    logger.info(
        "Loaded {} candidate pairs; {} already checked, {} remaining",
        len(candidate_pairs),
        len(checked_pairs),
        len(remaining_pairs),
    )

    def _persist_pair(pair: tuple[str, str], relation: EvidenceRelation | None) -> None:
        """Rewrite the relations file, then record the pair as checked, so a crash between the two only costs a retry."""
        if relation is not None:
            relations_by_id[relation.relation_id] = relation
        write_evidence_relations(arguments.relations_output_path, tuple(relations_by_id.values()))
        checked_pairs.add(pair)
        write_evidence_relation_progress(
            arguments.relations_progress_output_path,
            tuple(
                EvidenceRelationProgressRecord(fact_id_a=fact_id_a, fact_id_b=fact_id_b)
                for fact_id_a, fact_id_b in candidate_pairs
                if (fact_id_a, fact_id_b) in checked_pairs
            ),
        )
        logger.debug(
            "Checked pair ({}, {}): {}", pair[0], pair[1], "related" if relation is not None else "not related"
        )

    if remaining_pairs:
        relation_client = BamlEvidenceRelationClient(settings, synthesis_settings)
        confirm_evidence_relations(
            facts,
            remaining_pairs,
            relation_client,
            max_retries=synthesis_settings.llm_max_retries,
            on_pair_confirmed=_persist_pair,
        )

    relations = tuple(relations_by_id.values())
    write_evidence_relations(arguments.relations_output_path, relations)
    checked_candidate_count = len(checked_pairs & set(candidate_pairs))
    logger.info(
        "Confirmed {} of {} candidate pairs ({} still unchecked), wrote relations to {}",
        len(relations),
        len(candidate_pairs),
        len(candidate_pairs) - checked_candidate_count,
        arguments.relations_output_path,
    )

    bundles = sample_bundles(facts, relations, overlap_threshold=synthesis_settings.bundle_overlap_threshold)
    scenario_config = ScenarioSamplingConfig(
        query_taxonomy=load_query_taxonomy(arguments.query_taxonomy),
        personas=load_personas(arguments.personas),
        distribution=load_scenario_distribution(arguments.scenario_distribution),
    )
    facts_by_id = {fact.fact_id: fact for fact in facts}
    scenarios = sample_scenarios(bundles, facts_by_id, scenario_config, total_samples=arguments.total_samples)
    write_scenarios(arguments.scenarios_output_path, scenarios)
    logger.info(
        "Sampled {} bundles into {} scenarios, wrote scenarios to {}",
        len(bundles),
        len(scenarios),
        arguments.scenarios_output_path,
    )


if __name__ == "__main__":
    main()
