"""Command-line interface for single-turn candidate generation and both gates."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the candidate-generation command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate single-turn candidates from scenarios and run gate 1 and gate 2."
    )
    parser.add_argument("facts_path", type=Path, help="Legal-facts JSONL path")
    parser.add_argument("ledger_path", type=Path, help="Fact-ledger JSONL path; only approved facts are trusted")
    parser.add_argument("scenarios_path", type=Path, help="Scenarios JSONL path")
    parser.add_argument("chunks_dir", type=Path, help="Directory containing one or more *.chunks.jsonl files")
    parser.add_argument("candidates_output_path", type=Path, help="Destination single-turn candidates JSONL path")
    parser.add_argument("--query-taxonomy", type=Path, required=True, help="Query-taxonomy config JSON path")
    parser.add_argument("--personas", type=Path, required=True, help="Persona config JSON path")
    parser.add_argument("--evolution", type=Path, required=True, help="Evolution-operation config JSON path")
    parser.add_argument(
        "--candidates-per-scenario", type=int, required=True, help="Number of generation attempts per scenario"
    )
    return parser


def main() -> None:
    """Generate candidates for every scenario, then run gate 1 and, when it passes, gate 2."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.benchmark_synthesis.binding_gate import check_binding
    from legal_rag.benchmark_synthesis.candidate_generation import (
        BamlCandidateGenerationClient,
        evolve_candidate,
        generate_candidate,
        refine_surface,
    )
    from legal_rag.benchmark_synthesis.candidate_record import CandidateRecord
    from legal_rag.benchmark_synthesis.scenario_sampling import (
        load_evolution_config,
        load_persona_descriptions,
        load_query_taxonomy,
        load_query_type_descriptions,
    )
    from legal_rag.benchmark_synthesis.verification import BamlVerificationClient, verify_candidate
    from legal_rag.benchmark_synthesis.writer import (
        read_fact_ledger,
        read_legal_facts,
        read_scenarios,
        write_candidates,
    )
    from legal_rag.chunker.writer import read_chunks
    from legal_rag.settings import EvaluationSettings, Settings

    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    settings = Settings.from_environment()
    evaluation_settings = EvaluationSettings.from_environment()

    facts_by_id = {fact.fact_id: fact for fact in read_legal_facts(arguments.facts_path)}
    ledger_by_fact_id = {entry.fact_id: entry for entry in read_fact_ledger(arguments.ledger_path)}
    scenarios = read_scenarios(arguments.scenarios_path)

    chunks_by_id = {}
    for chunks_path in sorted(arguments.chunks_dir.glob("*.chunks.jsonl")):
        for chunk in read_chunks(chunks_path):
            chunks_by_id[chunk.chunk_id] = chunk
    logger.info("Loaded {} chunks from {}", len(chunks_by_id), arguments.chunks_dir)

    query_taxonomy = load_query_taxonomy(arguments.query_taxonomy)
    query_type_descriptions = load_query_type_descriptions(arguments.query_taxonomy)
    persona_descriptions = load_persona_descriptions(arguments.personas)
    evolution_config = load_evolution_config(arguments.evolution)
    evolution_operations = evolution_config.operation_ids[: evolution_config.max_rounds]

    generation_client = BamlCandidateGenerationClient(settings, evaluation_settings)
    verification_client = BamlVerificationClient(settings, evaluation_settings)

    records: list[CandidateRecord] = []
    for scenario in scenarios:
        for _attempt in range(arguments.candidates_per_scenario):
            draft = generate_candidate(
                scenario,
                facts_by_id,
                persona_descriptions[scenario.persona],
                query_type_descriptions[scenario.query_type],
                generation_client,
                generator_model=settings.llm_model,
            )
            draft = refine_surface(draft, generation_client, max_retries=evaluation_settings.max_quality_retries)
            if evolution_operations:
                draft = evolve_candidate(draft, evolution_operations, generation_client)

            binding = check_binding(
                draft,
                facts_by_id,
                chunks_by_id,
                ledger_by_fact_id,
                span_match_threshold=evaluation_settings.span_match_threshold,
            )

            verification = None
            if binding.passed:
                verification = verify_candidate(
                    draft,
                    facts_by_id,
                    query_taxonomy.query_type_ids,
                    verification_client,
                    verifier_model=settings.llm_model,
                )

            records.append(CandidateRecord(draft=draft, binding=binding, verification=verification))

    write_candidates(arguments.candidates_output_path, records)

    rejected_binding = sum(1 for record in records if record.review_status == "rejected_by_binding_gate")
    rejected_verification = sum(1 for record in records if record.review_status == "rejected_by_verification")
    verified = sum(1 for record in records if record.review_status == "automatically_verified")
    logger.info(
        "Generated {} candidates from {} scenarios: {} rejected by gate 1, {} rejected by gate 2, "
        "{} automatically verified. Wrote {}",
        len(records),
        len(scenarios),
        rejected_binding,
        rejected_verification,
        verified,
        arguments.candidates_output_path,
    )


if __name__ == "__main__":
    main()
