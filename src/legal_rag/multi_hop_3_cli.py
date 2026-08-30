"""Command-line interface for confirming three-fact multi-hop bundles."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from legal_rag.benchmark_synthesis.config import SynthesisSettings
    from legal_rag.benchmark_synthesis.evidence_graph import EvidenceRelation
    from legal_rag.benchmark_synthesis.llm_execution import LlmTaskFailure
    from legal_rag.benchmark_synthesis.multi_hop_3 import (
        ConfirmedMultiHop3Bundle,
        MultiHop3ProgressRecord,
    )


def _parser() -> argparse.ArgumentParser:
    """Build the multi-hop-3 command-line interface."""
    parser = argparse.ArgumentParser(description="Confirm three-fact multi-hop bundles and resample scenarios.")
    parser.add_argument("facts_path", type=Path, help="Legal-facts JSONL path")
    parser.add_argument("relations_path", type=Path, help="Confirmed evidence-relations JSONL path")
    parser.add_argument("bundles_output_path", type=Path, help="Confirmed multi-hop-3 bundles JSONL destination")
    parser.add_argument("progress_output_path", type=Path, help="Multi-hop-3 progress JSONL destination")
    parser.add_argument("scenarios_output_path", type=Path, help="Scenarios JSONL destination")
    parser.add_argument("--candidate-cache", type=Path, required=True, help="Fingerprint-validated triple cache")
    parser.add_argument("--prepare-candidates-only", action="store_true", help="Prepare candidates without LLM calls")
    parser.add_argument(
        "--allow-incomplete-bundles",
        action="store_true",
        help="Sample scenarios even when some candidate batches failed after retries",
    )
    parser.add_argument("--reconfirm-candidate-id", action="append", default=[], help="Candidate ID to confirm again")
    parser.add_argument(
        "--reconfirm-candidate-ids", type=Path, help="JSON string array of candidate IDs to confirm again"
    )
    parser.add_argument(
        "--reconfirm-accepted",
        action="store_true",
        help="Confirm every currently accepted multi-hop-3 bundle again",
    )
    parser.add_argument(
        "--migrate-relation-pair-ids",
        action="append",
        default=[],
        type=Path,
        help="JSON fact-pair selection used to migrate state after the relation graph changes",
    )
    parser.add_argument("--personas", type=Path, required=True, help="Persona config JSON path")
    parser.add_argument("--scenario-distribution", type=Path, required=True, help="Scenario distribution JSON path")
    parser.add_argument("--total-samples", type=int, required=True, help="Target scenario count")
    parser.add_argument("--synthesis-config", type=Path, required=True, help="Versioned synthesis config JSON path")
    return parser


def _validate_resume_state(
    existing_bundles: Sequence[ConfirmedMultiHop3Bundle],
    existing_progress: Sequence[MultiHop3ProgressRecord],
    candidate_ids: frozenset[str],
    run_id: str,
    reconfirm_ids: frozenset[str],
    migration_invalidated_ids: frozenset[str] | None = None,
) -> tuple[dict[str, ConfirmedMultiHop3Bundle], dict[str, MultiHop3ProgressRecord]]:
    """Validate persisted state and invalidate only explicitly selected candidates."""
    from legal_rag.benchmark_synthesis.multi_hop_3 import (
        MULTI_HOP_3_PROMPT_VERSION,
        multi_hop_3_candidate_id,
    )
    from legal_rag.benchmark_synthesis.multi_hop_3_selection import validate_selective_confirmation

    bundles_by_id = {bundle.bundle_id: bundle for bundle in existing_bundles}
    if len(bundles_by_id) != len(existing_bundles):
        msg = "Existing multi-hop-3 bundles contain duplicate bundle IDs"
        raise ValueError(msg)
    progress_by_id = {record.candidate_id: record for record in existing_progress}
    if len(progress_by_id) != len(existing_progress):
        msg = "Existing multi-hop-3 progress contains duplicate candidate IDs"
        raise ValueError(msg)
    if migration_invalidated_ids is not None:
        progress_by_id = {
            candidate_id: record
            for candidate_id, record in progress_by_id.items()
            if candidate_id in candidate_ids and candidate_id not in migration_invalidated_ids
        }
    if not progress_by_id.keys() <= candidate_ids:
        msg = "Multi-hop-3 progress contains candidates outside the current candidate set"
        raise ValueError(msg)
    if not reconfirm_ids and any(
        record.run_id != run_id
        and record.prompt_version == MULTI_HOP_3_PROMPT_VERSION
        and record.confirmed_by_model is None
        for record in existing_progress
    ):
        msg = "Existing multi-hop-3 progress belongs to a different synthesis run"
        raise ValueError(msg)

    bundle_candidate_ids = [multi_hop_3_candidate_id(bundle.fact_ids) for bundle in existing_bundles]
    if len(bundle_candidate_ids) != len(set(bundle_candidate_ids)):
        msg = "Existing multi-hop-3 bundles contain multiple results for one candidate"
        raise ValueError(msg)
    if migration_invalidated_ids is not None:
        bundles_by_id = {
            bundle_id: bundle
            for bundle_id, bundle in bundles_by_id.items()
            if (candidate_id := multi_hop_3_candidate_id(bundle.fact_ids)) in candidate_ids
            and candidate_id not in migration_invalidated_ids
        }
        bundle_candidate_ids = [multi_hop_3_candidate_id(bundle.fact_ids) for bundle in bundles_by_id.values()]
    if not set(bundle_candidate_ids) <= candidate_ids:
        msg = "Multi-hop-3 bundles contain candidates outside the current candidate set"
        raise ValueError(msg)

    _validate_bundle_provenance(tuple(bundles_by_id.values()), bundle_candidate_ids, progress_by_id, run_id)

    if reconfirm_ids:
        validate_selective_confirmation(candidate_ids, frozenset(progress_by_id), reconfirm_ids)
        progress_by_id = {
            candidate_id: record for candidate_id, record in progress_by_id.items() if candidate_id not in reconfirm_ids
        }
        bundles_by_id = {
            bundle_id: bundle
            for bundle_id, bundle in bundles_by_id.items()
            if multi_hop_3_candidate_id(bundle.fact_ids) not in reconfirm_ids
        }
    return bundles_by_id, progress_by_id


def _validate_bundle_provenance(
    bundles: Sequence[ConfirmedMultiHop3Bundle],
    candidate_ids: Sequence[str],
    progress_by_id: dict[str, MultiHop3ProgressRecord],
    run_id: str,
) -> None:
    """Validate completed bundle provenance while allowing current-run crash replay."""
    for bundle, candidate_id in zip(bundles, candidate_ids, strict=True):
        if candidate_id not in progress_by_id:
            if bundle.run_id == run_id:
                continue
            msg = "Existing multi-hop-3 bundles belong to a different synthesis run"
            raise ValueError(msg)
        progress = progress_by_id[candidate_id]
        if bundle.run_id != progress.run_id or bundle.prompt_version != progress.prompt_version:
            msg = "Multi-hop-3 bundle provenance must match its progress record"
            raise ValueError(msg)
        if progress.confirmed_by_model is not None and bundle.confirmed_by_model != progress.confirmed_by_model:
            msg = "Multi-hop-3 bundle model must match its progress record"
            raise ValueError(msg)


def _load_or_create_candidates(
    candidate_cache: Path,
    relations: Sequence[EvidenceRelation],
    synthesis_settings: SynthesisSettings,
    changed_relation_pairs: frozenset[tuple[str, str]],
) -> tuple[str, tuple[tuple[str, str, str], ...], frozenset[str] | None]:
    """Load a compatible candidate cache or build it from the pair graph."""
    from legal_rag.benchmark_synthesis.multi_hop_3 import (
        MultiHop3CandidateSet,
        build_multi_hop_3_candidate_run_id,
        find_multi_hop_3_candidates,
        multi_hop_3_candidate_id,
    )
    from legal_rag.benchmark_synthesis.writer import (
        read_multi_hop_3_candidate_set,
        write_multi_hop_3_candidate_set,
    )

    run_id = build_multi_hop_3_candidate_run_id(relations, synthesis_settings.multi_hop_3_candidate_limit)
    if candidate_cache.exists():
        candidate_set = read_multi_hop_3_candidate_set(candidate_cache)
        if candidate_set.run_id == run_id:
            logger.info("Loaded {} cached multi-hop-3 candidates from {}", len(candidate_set.triples), candidate_cache)
            return run_id, candidate_set.triples, None
        if not changed_relation_pairs:
            msg = "Existing multi-hop-3 candidates belong to a different pair graph or selection run"
            raise ValueError(msg)
        candidates = find_multi_hop_3_candidates(relations, limit=synthesis_settings.multi_hop_3_candidate_limit)
        previous_candidates = frozenset(candidate_set.triples)
        invalidated_ids = frozenset(
            multi_hop_3_candidate_id(candidate)
            for candidate in candidates
            if candidate not in previous_candidates
            or any(tuple(sorted(pair)) in changed_relation_pairs for pair in combinations(candidate, 2))
        )
        write_multi_hop_3_candidate_set(
            candidate_cache,
            MultiHop3CandidateSet(run_id=run_id, triples=candidates),
        )
        logger.info(
            "Migrated multi-hop-3 candidates: retained={}, invalidated={}, removed={}",
            len(candidates) - len(invalidated_ids),
            len(invalidated_ids),
            len(previous_candidates - frozenset(candidates)),
        )
        return run_id, candidates, invalidated_ids

    candidates = find_multi_hop_3_candidates(relations, limit=synthesis_settings.multi_hop_3_candidate_limit)
    write_multi_hop_3_candidate_set(
        candidate_cache,
        MultiHop3CandidateSet(run_id=run_id, triples=candidates),
    )
    logger.info("Selected {} multi-hop-3 candidates, wrote {}", len(candidates), candidate_cache)
    return run_id, candidates, None


def _load_changed_relation_pairs(paths: Sequence[Path]) -> frozenset[tuple[str, str]]:
    """Load the union of explicit relation-pair selections used for migration."""
    from legal_rag.benchmark_synthesis.evidence_relation_selection import load_reconfirm_pairs

    pairs: set[tuple[str, str]] = set()
    for path in paths:
        pairs.update(load_reconfirm_pairs((), path))
    return frozenset(pairs)


def _validate_requested_operations(
    reconfirm_ids: frozenset[str],
    changed_relation_pairs: frozenset[tuple[str, str]],
) -> None:
    """Keep selective reconfirmation and graph migration as separate operations."""
    if reconfirm_ids and changed_relation_pairs:
        msg = "Candidate reconfirmation and relation-graph migration must run separately"
        raise ValueError(msg)


def _require_completed_confirmation(failures: Sequence[LlmTaskFailure], *, allow_incomplete: bool) -> None:
    """Reject incomplete confirmation unless the caller explicitly accepts partial output."""
    if not failures:
        return
    logger.warning("Multi-hop-3 confirmation left {} batch(es) incomplete", len(failures))
    if allow_incomplete:
        return
    from legal_rag.benchmark_synthesis.llm_execution import IncompleteLlmTasksError

    incomplete_error = IncompleteLlmTasksError.for_stage("multi-hop-3 confirmation", failures)
    raise incomplete_error


def main() -> None:
    """Confirm missing three-fact bundles, checkpoint progress, and resample scenarios."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.benchmark_synthesis.config import SynthesisSettings
    from legal_rag.benchmark_synthesis.evidence_graph import sample_bundles
    from legal_rag.benchmark_synthesis.multi_hop_3 import (
        MULTI_HOP_3_PROMPT_VERSION,
        BamlMultiHop3Client,
        MultiHop3ProgressRecord,
        build_multi_hop_3_run_id,
        confirm_multi_hop_3_bundles,
        multi_hop_3_candidate_id,
    )
    from legal_rag.benchmark_synthesis.multi_hop_3_selection import (
        include_accepted_candidate_ids,
        load_reconfirm_candidate_ids,
    )
    from legal_rag.benchmark_synthesis.scenario_sampling import (
        ScenarioSamplingConfig,
        load_personas,
        load_scenario_distribution,
        sample_scenarios,
    )
    from legal_rag.benchmark_synthesis.writer import (
        read_evidence_relations,
        read_legal_facts,
        read_multi_hop_3_bundles,
        read_multi_hop_3_progress,
        write_multi_hop_3_bundles,
        write_multi_hop_3_progress,
        write_scenarios,
    )
    from legal_rag.settings import Settings

    started_at = time.perf_counter()
    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    settings = Settings.from_environment()
    synthesis_settings = SynthesisSettings.from_path(arguments.synthesis_config)
    reconfirm_ids = load_reconfirm_candidate_ids(
        arguments.reconfirm_candidate_id,
        arguments.reconfirm_candidate_ids,
    )
    changed_relation_pairs = _load_changed_relation_pairs(arguments.migrate_relation_pair_ids)
    _validate_requested_operations(reconfirm_ids, changed_relation_pairs)

    facts = read_legal_facts(arguments.facts_path)
    relations = read_evidence_relations(arguments.relations_path)
    candidate_run_id, candidates, migration_invalidated_ids = _load_or_create_candidates(
        arguments.candidate_cache,
        relations,
        synthesis_settings,
        changed_relation_pairs,
    )
    if arguments.prepare_candidates_only:
        return

    run_id = build_multi_hop_3_run_id(
        candidate_run_id,
        synthesis_settings,
        llm_base_url=settings.llm_base_url,
        llm_model=settings.llm_model,
    )
    existing_bundles = (
        read_multi_hop_3_bundles(arguments.bundles_output_path) if arguments.bundles_output_path.exists() else ()
    )
    accepted_fact_ids = (bundle.fact_ids for bundle in existing_bundles) if arguments.reconfirm_accepted else ()
    reconfirm_ids = include_accepted_candidate_ids(
        reconfirm_ids,
        accepted_fact_ids,
        multi_hop_3_candidate_id,
    )
    _validate_requested_operations(reconfirm_ids, changed_relation_pairs)
    existing_progress = (
        read_multi_hop_3_progress(arguments.progress_output_path) if arguments.progress_output_path.exists() else ()
    )
    candidate_ids = frozenset(multi_hop_3_candidate_id(triple) for triple in candidates)
    bundles_by_id, progress_by_id = _validate_resume_state(
        existing_bundles,
        existing_progress,
        candidate_ids,
        run_id,
        reconfirm_ids,
        migration_invalidated_ids,
    )
    if reconfirm_ids:
        logger.info("Selectively invalidated {} multi-hop-3 candidate(s)", len(reconfirm_ids))
    if migration_invalidated_ids is not None:
        logger.info(
            "Migrated multi-hop-3 state: retained_progress={}, remaining={}",
            len(progress_by_id),
            len(candidate_ids - progress_by_id.keys()),
        )
    candidates_by_id = {multi_hop_3_candidate_id(triple): triple for triple in candidates}
    remaining = tuple(triple for triple in candidates if multi_hop_3_candidate_id(triple) not in progress_by_id)
    logger.info(
        "Loaded {} multi-hop-3 candidates; {} already checked, {} remaining",
        len(candidates),
        len(candidates) - len(remaining),
        len(remaining),
    )

    def _persist_batch(batch_results: Sequence[tuple[tuple[str, str, str], ConfirmedMultiHop3Bundle | None]]) -> None:
        """Replace completed candidate outcomes and atomically checkpoint the batch."""
        completed_ids = {multi_hop_3_candidate_id(triple) for triple, _bundle in batch_results}
        stale_bundle_ids = [
            bundle_id
            for bundle_id, bundle in bundles_by_id.items()
            if multi_hop_3_candidate_id(bundle.fact_ids) in completed_ids
        ]
        for bundle_id in stale_bundle_ids:
            del bundles_by_id[bundle_id]
        bundles_by_id.update({bundle.bundle_id: bundle for _triple, bundle in batch_results if bundle is not None})
        for candidate_id in completed_ids:
            progress_by_id[candidate_id] = MultiHop3ProgressRecord(
                candidate_id=candidate_id,
                run_id=run_id,
                prompt_version=MULTI_HOP_3_PROMPT_VERSION,
                confirmed_by_model=settings.llm_model,
            )
        write_multi_hop_3_bundles(
            arguments.bundles_output_path,
            tuple(sorted(bundles_by_id.values(), key=lambda bundle: (bundle.fact_ids, bundle.bridge))),
        )
        write_multi_hop_3_progress(
            arguments.progress_output_path,
            tuple(
                progress_by_id[multi_hop_3_candidate_id(triple)]
                for triple in candidates
                if multi_hop_3_candidate_id(triple) in progress_by_id
            ),
        )
        logger.debug("Checkpointed {} multi-hop-3 candidate(s)", len(batch_results))

    if reconfirm_ids or migration_invalidated_ids is not None:
        write_multi_hop_3_bundles(
            arguments.bundles_output_path,
            tuple(sorted(bundles_by_id.values(), key=lambda bundle: (bundle.fact_ids, bundle.bridge))),
        )
        write_multi_hop_3_progress(
            arguments.progress_output_path,
            tuple(progress_by_id[candidate_id] for candidate_id in candidates_by_id if candidate_id in progress_by_id),
        )

    if remaining:
        client = BamlMultiHop3Client(settings, synthesis_settings, run_id=run_id)
        _, failures = confirm_multi_hop_3_bundles(
            facts,
            relations,
            remaining,
            client,
            batch_size=synthesis_settings.multi_hop_3_batch_size,
            max_concurrency=synthesis_settings.multi_hop_3_max_concurrency,
            max_retries=synthesis_settings.llm_max_retries,
            retry_delay_seconds=synthesis_settings.llm_retry_delay_seconds,
            on_batch_confirmed=_persist_batch,
        )
        _require_completed_confirmation(failures, allow_incomplete=arguments.allow_incomplete_bundles)

    confirmed_bundles = tuple(sorted(bundles_by_id.values(), key=lambda bundle: (bundle.fact_ids, bundle.bridge)))
    base_bundles = sample_bundles(facts, relations, overlap_threshold=synthesis_settings.bundle_overlap_threshold)
    all_bundles = (*base_bundles, *(bundle.to_evidence_bundle() for bundle in confirmed_bundles))
    scenario_config = ScenarioSamplingConfig(
        personas=load_personas(arguments.personas),
        distribution=load_scenario_distribution(arguments.scenario_distribution),
    )
    scenarios = sample_scenarios(
        all_bundles,
        {fact.fact_id: fact for fact in facts},
        scenario_config,
        total_samples=arguments.total_samples,
    )
    write_scenarios(arguments.scenarios_output_path, scenarios)
    logger.info(
        "Confirmed {} multi-hop-3 bundles and wrote {} scenarios in {:.2f} seconds",
        len(confirmed_bundles),
        len(scenarios),
        time.perf_counter() - started_at,
    )


if __name__ == "__main__":
    main()
