"""Command-line interface for evidence-relation confirmation and scenario sampling."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from legal_rag.benchmark_synthesis.evidence_graph import EvidenceRelation, EvidenceRelationProgressRecord
    from legal_rag.benchmark_synthesis.llm_execution import LlmTaskFailure


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
    parser.add_argument(
        "--candidate-pairs-cache",
        type=Path,
        help="Fingerprint-validated candidate-pair JSON cache; defaults beside the relations output",
    )
    parser.add_argument(
        "--prepare-candidates-only",
        action="store_true",
        help="Build or validate the candidate-pair cache, then exit before any relation LLM calls",
    )
    parser.add_argument(
        "--allow-incomplete-relations",
        action="store_true",
        help="Sample scenarios from confirmed relations even if some candidate pairs failed after retries",
    )
    parser.add_argument("--query-taxonomy", type=Path, required=True, help="Query-taxonomy config JSON path")
    parser.add_argument("--personas", type=Path, required=True, help="Persona config JSON path")
    parser.add_argument(
        "--scenario-distribution", type=Path, required=True, help="Scenario-distribution config JSON path"
    )
    parser.add_argument("--total-samples", type=int, required=True, help="Target scenario count across all bundles")
    parser.add_argument("--synthesis-config", type=Path, required=True, help="Versioned synthesis config JSON path")
    return parser


def _validate_resume_state(
    existing_relations: Sequence[EvidenceRelation],
    existing_progress: Sequence[EvidenceRelationProgressRecord],
    candidate_pairs: Sequence[tuple[str, str]],
    relation_run_id: str,
) -> tuple[dict[str, EvidenceRelation], set[tuple[str, str]]]:
    """Validate persisted relation state and return mutable resume indexes."""
    if any(relation.run_id != relation_run_id for relation in existing_relations):
        msg = "Existing evidence relations belong to a different synthesis run"
        raise ValueError(msg)
    relations_by_id = {relation.relation_id: relation for relation in existing_relations}
    if len(relations_by_id) != len(existing_relations):
        msg = "Existing evidence relations contain duplicate relation IDs"
        raise ValueError(msg)
    if any(record.run_id != relation_run_id for record in existing_progress):
        msg = "Existing evidence-relation progress belongs to a different synthesis run"
        raise ValueError(msg)
    checked_pairs = {(record.fact_id_a, record.fact_id_b) for record in existing_progress}
    if len(checked_pairs) != len(existing_progress):
        msg = "Evidence-relation progress contains duplicate candidate pairs"
        raise ValueError(msg)
    candidate_pair_set = set(candidate_pairs)
    if not checked_pairs <= candidate_pair_set:
        msg = "Evidence-relation progress contains pairs outside the current candidate set"
        raise ValueError(msg)
    existing_relation_pairs = {
        tuple(sorted((relation.from_fact_id, relation.to_fact_id))) for relation in existing_relations
    }
    if len(existing_relation_pairs) != len(existing_relations):
        msg = "Existing evidence relations contain multiple results for one candidate pair"
        raise ValueError(msg)
    if not existing_relation_pairs <= candidate_pair_set:
        msg = "Evidence relations contain pairs outside the current candidate set"
        raise ValueError(msg)
    return relations_by_id, checked_pairs


def _log_relation_confirmation_failures(failures: Sequence[LlmTaskFailure]) -> bool:
    """Log any batches that failed after retries and report whether the stage was incomplete."""
    if not failures:
        return False
    logger.warning(
        "Evidence relation confirmation left {} batch(es) incomplete: {}",
        len(failures),
        ", ".join(failure.task_key for failure in failures),
    )
    return True


def main() -> None:
    """Confirm evidence relations not yet checked among facts, then sample scenarios from the result."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.benchmark_synthesis.config import SynthesisSettings
    from legal_rag.benchmark_synthesis.embedding import SynthesisEmbeddingEncoder
    from legal_rag.benchmark_synthesis.evidence_graph import (
        BamlEvidenceRelationClient,
        EvidenceCandidateSet,
        EvidenceRelationProgressRecord,
        build_evidence_candidate_run_id,
        build_evidence_relation_run_id,
        confirm_evidence_relations,
        evidence_relation_sort_key,
        find_candidate_fact_pairs,
        sample_bundles,
    )
    from legal_rag.benchmark_synthesis.llm_execution import IncompleteLlmTasksError
    from legal_rag.benchmark_synthesis.scenario_sampling import (
        ScenarioSamplingConfig,
        load_personas,
        load_query_taxonomy,
        load_scenario_distribution,
        sample_scenarios,
    )
    from legal_rag.benchmark_synthesis.writer import (
        read_evidence_candidate_set,
        read_evidence_relation_progress,
        read_evidence_relations,
        read_legal_facts,
        write_evidence_candidate_set,
        write_evidence_relation_progress,
        write_evidence_relations,
        write_scenarios,
    )
    from legal_rag.embedding.profiles import load_configured_embedding_profile
    from legal_rag.settings import Settings, SynthesisEmbeddingSettings

    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    settings = Settings.from_environment()
    embedding_settings = SynthesisEmbeddingSettings.from_environment()
    synthesis_settings = SynthesisSettings.from_path(arguments.synthesis_config)

    facts = read_legal_facts(arguments.facts_path)
    logger.info("Loaded {} facts from {}", len(facts), arguments.facts_path)

    profile = load_configured_embedding_profile(synthesis_settings.embedding_profile)
    candidate_run_id = build_evidence_candidate_run_id(
        facts,
        profile,
        synthesis_settings,
        embedding_base_url=embedding_settings.base_url,
        embedding_model=embedding_settings.model,
    )
    relation_run_id = build_evidence_relation_run_id(
        facts,
        profile,
        synthesis_settings,
        embedding_base_url=embedding_settings.base_url,
        embedding_model=embedding_settings.model,
        llm_base_url=settings.llm_base_url,
        llm_model=settings.llm_model,
    )
    candidate_cache_path = arguments.candidate_pairs_cache or arguments.relations_output_path.with_name(
        "evidence_relation_candidates.json"
    )
    if candidate_cache_path.exists():
        candidate_set = read_evidence_candidate_set(candidate_cache_path)
        if candidate_set.run_id != candidate_run_id:
            msg = "Existing evidence candidates belong to a different candidate-selection run"
            raise ValueError(msg)
        candidate_pairs = candidate_set.pairs
        logger.info("Loaded {} cached candidate pairs from {}", len(candidate_pairs), candidate_cache_path)
    else:
        from legal_rag.benchmark_synthesis.vietnamese_lexical import VnCoreNlpWordSegmenter
        from legal_rag.settings import SynthesisVnCoreNlpSettings

        logger.info(
            "Candidate cache missing; encoding {} facts with profile {} before relation confirmation",
            len(facts),
            profile.profile_id,
        )
        encoder = SynthesisEmbeddingEncoder(
            embedding_settings,
            profile,
            batch_size=synthesis_settings.embedding_batch_size,
            max_response_bytes=synthesis_settings.embedding_max_response_bytes,
        )
        vncorenlp_settings = SynthesisVnCoreNlpSettings.from_environment()
        word_segmenter = VnCoreNlpWordSegmenter(
            java_path=vncorenlp_settings.java_path,
            model_dir=vncorenlp_settings.model_dir,
            max_heap_megabytes=synthesis_settings.vncorenlp_max_heap_megabytes,
            timeout_seconds=synthesis_settings.vncorenlp_timeout_seconds,
        )
        candidate_pairs = find_candidate_fact_pairs(
            facts,
            encoder,
            word_segmenter,
            synthesis_settings.evidence_retrieval,
        )
        write_evidence_candidate_set(
            candidate_cache_path,
            EvidenceCandidateSet(run_id=candidate_run_id, pairs=candidate_pairs),
        )
        logger.info("Selected {} candidate pairs, wrote {}", len(candidate_pairs), candidate_cache_path)
    if arguments.prepare_candidates_only:
        logger.info("Candidate preparation complete: {} pairs; no relation LLM calls were made", len(candidate_pairs))
        return

    existing_relations = (
        read_evidence_relations(arguments.relations_output_path) if arguments.relations_output_path.exists() else ()
    )
    existing_progress = (
        read_evidence_relation_progress(arguments.relations_progress_output_path)
        if arguments.relations_progress_output_path.exists()
        else ()
    )
    relations_by_id, checked_pairs = _validate_resume_state(
        existing_relations,
        existing_progress,
        candidate_pairs,
        relation_run_id,
    )
    remaining_pairs = tuple(pair for pair in candidate_pairs if pair not in checked_pairs)
    logger.info(
        "Loaded {} candidate pairs; {} already checked, {} remaining",
        len(candidate_pairs),
        len(checked_pairs),
        len(remaining_pairs),
    )

    def _persist_pair(pair: tuple[str, str], relation: EvidenceRelation | None) -> None:
        """Rewrite the relations file, then record the pair as checked, so a crash between the two only costs a retry."""
        stale_relation_ids = [
            relation_id
            for relation_id, existing in relations_by_id.items()
            if tuple(sorted((existing.from_fact_id, existing.to_fact_id))) == pair
        ]
        for relation_id in stale_relation_ids:
            del relations_by_id[relation_id]
        if relation is not None:
            relations_by_id[relation.relation_id] = relation
        write_evidence_relations(
            arguments.relations_output_path,
            tuple(sorted(relations_by_id.values(), key=evidence_relation_sort_key)),
        )
        checked_pairs.add(pair)
        write_evidence_relation_progress(
            arguments.relations_progress_output_path,
            tuple(
                EvidenceRelationProgressRecord(fact_id_a=fact_id_a, fact_id_b=fact_id_b, run_id=relation_run_id)
                for fact_id_a, fact_id_b in candidate_pairs
                if (fact_id_a, fact_id_b) in checked_pairs
            ),
        )
        logger.debug(
            "Checked pair ({}, {}): {}", pair[0], pair[1], "related" if relation is not None else "not related"
        )

    if remaining_pairs:
        relation_client = BamlEvidenceRelationClient(settings, synthesis_settings, run_id=relation_run_id)
        _, failures = confirm_evidence_relations(
            facts,
            remaining_pairs,
            relation_client,
            batch_size=synthesis_settings.evidence_relation_batch_size,
            max_concurrency=synthesis_settings.evidence_relation_max_concurrency,
            max_retries=synthesis_settings.llm_max_retries,
            retry_delay_seconds=synthesis_settings.llm_retry_delay_seconds,
            on_pair_confirmed=_persist_pair,
        )
        if _log_relation_confirmation_failures(failures) and not arguments.allow_incomplete_relations:
            incomplete_error = IncompleteLlmTasksError.for_stage("evidence relation confirmation", failures)
            raise incomplete_error
    else:
        write_evidence_relations(
            arguments.relations_output_path,
            tuple(sorted(relations_by_id.values(), key=evidence_relation_sort_key)),
        )
        write_evidence_relation_progress(
            arguments.relations_progress_output_path,
            tuple(
                EvidenceRelationProgressRecord(fact_id_a=fact_id_a, fact_id_b=fact_id_b, run_id=relation_run_id)
                for fact_id_a, fact_id_b in candidate_pairs
                if (fact_id_a, fact_id_b) in checked_pairs
            ),
        )
    relations = tuple(sorted(relations_by_id.values(), key=evidence_relation_sort_key))
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
