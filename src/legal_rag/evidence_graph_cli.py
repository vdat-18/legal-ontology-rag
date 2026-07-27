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
    parser.add_argument("ledger_path", type=Path, help="Fact-ledger JSONL path; only approved facts are used")
    parser.add_argument("relations_output_path", type=Path, help="Destination evidence-relations JSONL path")
    parser.add_argument("scenarios_output_path", type=Path, help="Destination scenarios JSONL path")
    parser.add_argument("--query-taxonomy", type=Path, required=True, help="Query-taxonomy config JSON path")
    parser.add_argument("--personas", type=Path, required=True, help="Persona config JSON path")
    parser.add_argument(
        "--scenario-distribution", type=Path, required=True, help="Scenario-distribution config JSON path"
    )
    parser.add_argument("--total-samples", type=int, required=True, help="Target scenario count across all bundles")
    return parser


def main() -> None:
    """Confirm evidence relations among approved facts and sample scenarios from the result."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.benchmark_synthesis.evidence_graph import (
        BamlEvidenceRelationClient,
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
        read_fact_ledger,
        read_legal_facts,
        write_evidence_relations,
        write_scenarios,
    )
    from legal_rag.embedding.encoder import EmbeddingEncoder
    from legal_rag.embedding.profiles import load_configured_embedding_profile
    from legal_rag.settings import EvaluationSettings, Settings

    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    settings = Settings.from_environment()
    evaluation_settings = EvaluationSettings.from_environment()

    facts = read_legal_facts(arguments.facts_path)
    ledger_entries = read_fact_ledger(arguments.ledger_path)
    approved_fact_ids = {entry.fact_id for entry in ledger_entries if entry.status == "approved"}
    approved_facts = [fact for fact in facts if fact.fact_id in approved_fact_ids]
    logger.info("Using {} of {} facts with an approved ledger entry", len(approved_facts), len(facts))

    profile = load_configured_embedding_profile(evaluation_settings.embedding_profile)
    encoder = EmbeddingEncoder.from_profile(profile)
    candidate_pairs = find_candidate_fact_pairs(approved_facts, encoder)
    relation_client = BamlEvidenceRelationClient(settings, evaluation_settings)
    relations = confirm_evidence_relations(approved_facts, candidate_pairs, relation_client)
    write_evidence_relations(arguments.relations_output_path, relations)
    logger.info(
        "Confirmed {} of {} candidate pairs, wrote relations to {}",
        len(relations),
        len(candidate_pairs),
        arguments.relations_output_path,
    )

    bundles = sample_bundles(approved_facts, relations, overlap_threshold=evaluation_settings.bundle_overlap_threshold)
    scenario_config = ScenarioSamplingConfig(
        query_taxonomy=load_query_taxonomy(arguments.query_taxonomy),
        personas=load_personas(arguments.personas),
        distribution=load_scenario_distribution(arguments.scenario_distribution),
    )
    facts_by_id = {fact.fact_id: fact for fact in approved_facts}
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
