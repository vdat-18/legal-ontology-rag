"""Command-line interface for confirming four-fact multi-hop bundles."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from legal_rag.benchmark_synthesis.config import SynthesisSettings
    from legal_rag.benchmark_synthesis.evidence_graph import EvidenceRelation
    from legal_rag.benchmark_synthesis.llm_execution import LlmTaskFailure
    from legal_rag.benchmark_synthesis.multi_hop_3 import ConfirmedMultiHop3Bundle
    from legal_rag.benchmark_synthesis.multi_hop_4 import (
        ConfirmedMultiHop4Bundle,
        MultiHop4Candidate,
        MultiHop4ProgressRecord,
    )


def _parser() -> argparse.ArgumentParser:
    """Build the multi-hop-4 command-line interface."""
    parser = argparse.ArgumentParser(description="Confirm four-fact multi-hop bundles and resample scenarios.")
    parser.add_argument("facts_path", type=Path, help="Legal-facts JSONL path")
    parser.add_argument("relations_path", type=Path, help="Confirmed evidence-relations JSONL path")
    parser.add_argument("source_bundles_path", type=Path, help="Confirmed multi-hop-3 bundles JSONL path")
    parser.add_argument("bundles_output_path", type=Path, help="Confirmed multi-hop-4 bundles JSONL destination")
    parser.add_argument("progress_output_path", type=Path, help="Multi-hop-4 progress JSONL destination")
    parser.add_argument("scenarios_output_path", type=Path, help="Scenarios JSONL destination")
    parser.add_argument("--candidate-cache", type=Path, required=True, help="Fingerprint-validated candidate cache")
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
        help="Confirm every currently accepted multi-hop-4 bundle again",
    )
    parser.add_argument("--personas", type=Path, required=True, help="Persona config JSON path")
    parser.add_argument("--scenario-distribution", type=Path, required=True, help="Scenario distribution JSON path")
    parser.add_argument("--total-samples", type=int, required=True, help="Target scenario count")
    parser.add_argument("--synthesis-config", type=Path, required=True, help="Versioned synthesis config JSON path")
    return parser


def _validate_resume_state(
    existing_bundles: Sequence[ConfirmedMultiHop4Bundle],
    existing_progress: Sequence[MultiHop4ProgressRecord],
    candidate_ids: frozenset[str],
    run_id: str,
    reconfirm_ids: frozenset[str],
) -> tuple[dict[str, ConfirmedMultiHop4Bundle], dict[str, MultiHop4ProgressRecord]]:
    """Validate persisted state and invalidate only explicitly selected candidates."""
    from legal_rag.benchmark_synthesis.multi_hop_4 import MULTI_HOP_4_PROMPT_VERSION, multi_hop_4_candidate_id

    bundles_by_id = {bundle.bundle_id: bundle for bundle in existing_bundles}
    if len(bundles_by_id) != len(existing_bundles):
        msg = "Existing multi-hop-4 bundles contain duplicate bundle IDs"
        raise ValueError(msg)
    progress_by_id = {record.candidate_id: record for record in existing_progress}
    if len(progress_by_id) != len(existing_progress):
        msg = "Existing multi-hop-4 progress contains duplicate candidate IDs"
        raise ValueError(msg)
    if not progress_by_id.keys() <= candidate_ids:
        msg = "Multi-hop-4 progress contains candidates outside the current candidate set"
        raise ValueError(msg)
    if not reconfirm_ids and any(
        record.run_id != run_id
        and record.prompt_version == MULTI_HOP_4_PROMPT_VERSION
        and record.confirmed_by_model is None
        for record in existing_progress
    ):
        msg = "Existing multi-hop-4 progress belongs to a different synthesis run"
        raise ValueError(msg)

    bundle_candidate_ids = [multi_hop_4_candidate_id(bundle.fact_ids) for bundle in existing_bundles]
    if len(bundle_candidate_ids) != len(set(bundle_candidate_ids)):
        msg = "Existing multi-hop-4 bundles contain multiple results for one candidate"
        raise ValueError(msg)
    if not set(bundle_candidate_ids) <= candidate_ids:
        msg = "Multi-hop-4 bundles contain candidates outside the current candidate set"
        raise ValueError(msg)
    _validate_bundle_provenance(existing_bundles, bundle_candidate_ids, progress_by_id, run_id)

    if reconfirm_ids:
        unknown_ids = reconfirm_ids - candidate_ids
        if unknown_ids:
            msg = f"Unknown multi-hop-4 candidates requested for reconfirmation: {sorted(unknown_ids)}"
            raise ValueError(msg)
        if candidate_ids - reconfirm_ids - progress_by_id.keys():
            msg = "Selective multi-hop-4 confirmation requires prior progress for every unselected candidate"
            raise ValueError(msg)
        progress_by_id = {
            candidate_id: record for candidate_id, record in progress_by_id.items() if candidate_id not in reconfirm_ids
        }
        bundles_by_id = {
            bundle_id: bundle
            for bundle_id, bundle in bundles_by_id.items()
            if multi_hop_4_candidate_id(bundle.fact_ids) not in reconfirm_ids
        }
    return bundles_by_id, progress_by_id


def _validate_bundle_provenance(
    bundles: Sequence[ConfirmedMultiHop4Bundle],
    candidate_ids: Sequence[str],
    progress_by_id: dict[str, MultiHop4ProgressRecord],
    run_id: str,
) -> None:
    """Validate completed bundle provenance while allowing current-run crash replay."""
    for bundle, candidate_id in zip(bundles, candidate_ids, strict=True):
        if candidate_id not in progress_by_id:
            if bundle.run_id == run_id:
                continue
            msg = "Existing multi-hop-4 bundles belong to a different synthesis run"
            raise ValueError(msg)
        progress = progress_by_id[candidate_id]
        if bundle.run_id != progress.run_id or bundle.prompt_version != progress.prompt_version:
            msg = "Multi-hop-4 bundle provenance must match its progress record"
            raise ValueError(msg)
        if progress.confirmed_by_model is not None and bundle.confirmed_by_model != progress.confirmed_by_model:
            msg = "Multi-hop-4 bundle model must match its progress record"
            raise ValueError(msg)


def _require_completed_confirmation(failures: Sequence[LlmTaskFailure], *, allow_incomplete: bool) -> None:
    """Reject incomplete confirmation unless the caller explicitly accepts partial output."""
    if not failures:
        return
    logger.warning("Multi-hop-4 confirmation left {} batch(es) incomplete", len(failures))
    if allow_incomplete:
        return
    from legal_rag.benchmark_synthesis.llm_execution import IncompleteLlmTasksError

    incomplete_error = IncompleteLlmTasksError.for_stage("multi-hop-4 confirmation", failures)
    raise incomplete_error


def _load_or_create_candidates(
    candidate_cache: Path,
    source_bundles: Sequence[ConfirmedMultiHop3Bundle],
    relations: Sequence[EvidenceRelation],
    synthesis_settings: SynthesisSettings,
) -> tuple[str, tuple[MultiHop4Candidate, ...]]:
    """Load a compatible cache or create four-fact candidates."""
    from legal_rag.benchmark_synthesis.multi_hop_4 import (
        MultiHop4CandidateSet,
        build_multi_hop_4_candidate_run_id,
        find_multi_hop_4_candidates,
    )
    from legal_rag.benchmark_synthesis.writer import (
        read_multi_hop_4_candidate_set,
        write_multi_hop_4_candidate_set,
    )

    candidate_limit = synthesis_settings.multi_hop_4_candidate_limit
    run_id = build_multi_hop_4_candidate_run_id(source_bundles, relations, candidate_limit)
    if candidate_cache.exists():
        candidate_set = read_multi_hop_4_candidate_set(candidate_cache)
        if candidate_set.run_id != run_id:
            msg = "Existing multi-hop-4 candidates belong to a different source bundle or relation run"
            raise ValueError(msg)
        logger.info("Loaded {} cached multi-hop-4 candidates from {}", len(candidate_set.candidates), candidate_cache)
        return run_id, candidate_set.candidates
    candidates = find_multi_hop_4_candidates(source_bundles, relations, limit=candidate_limit)
    write_multi_hop_4_candidate_set(
        candidate_cache,
        MultiHop4CandidateSet(run_id=run_id, candidates=candidates),
    )
    logger.info("Selected {} multi-hop-4 candidates, wrote {}", len(candidates), candidate_cache)
    return run_id, candidates


def main() -> None:
    """Confirm missing four-fact bundles, checkpoint progress, and resample scenarios."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.benchmark_synthesis.config import SynthesisSettings
    from legal_rag.benchmark_synthesis.evidence_graph import sample_bundles
    from legal_rag.benchmark_synthesis.multi_hop_3_selection import (
        include_accepted_candidate_ids,
        load_reconfirm_candidate_ids,
    )
    from legal_rag.benchmark_synthesis.multi_hop_4 import (
        MULTI_HOP_4_PROMPT_VERSION,
        BamlMultiHop4Client,
        MultiHop4ProgressRecord,
        build_multi_hop_4_run_id,
        confirm_multi_hop_4_bundles,
        multi_hop_4_candidate_id,
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
        read_multi_hop_4_bundles,
        read_multi_hop_4_progress,
        write_scenarios,
    )
    from legal_rag.settings import Settings

    started_at = time.perf_counter()
    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    synthesis_settings = SynthesisSettings.from_path(arguments.synthesis_config)
    facts = read_legal_facts(arguments.facts_path)
    relations = read_evidence_relations(arguments.relations_path)
    source_bundles = read_multi_hop_3_bundles(arguments.source_bundles_path)
    candidate_run_id, candidates = _load_or_create_candidates(
        arguments.candidate_cache,
        source_bundles,
        relations,
        synthesis_settings,
    )
    if arguments.prepare_candidates_only:
        return

    settings = Settings.from_environment()
    run_id = build_multi_hop_4_run_id(
        candidate_run_id,
        synthesis_settings,
        llm_base_url=settings.llm_base_url,
        llm_model=settings.llm_model,
    )
    reconfirm_ids = load_reconfirm_candidate_ids(
        arguments.reconfirm_candidate_id,
        arguments.reconfirm_candidate_ids,
    )
    existing_bundles = (
        read_multi_hop_4_bundles(arguments.bundles_output_path) if arguments.bundles_output_path.exists() else ()
    )
    accepted_fact_ids = (bundle.fact_ids for bundle in existing_bundles) if arguments.reconfirm_accepted else ()
    reconfirm_ids = include_accepted_candidate_ids(
        reconfirm_ids,
        accepted_fact_ids,
        multi_hop_4_candidate_id,
    )
    existing_progress = (
        read_multi_hop_4_progress(arguments.progress_output_path) if arguments.progress_output_path.exists() else ()
    )
    candidate_ids = frozenset(multi_hop_4_candidate_id(candidate.fact_ids) for candidate in candidates)
    bundles_by_id, progress_by_id = _validate_resume_state(
        existing_bundles,
        existing_progress,
        candidate_ids,
        run_id,
        reconfirm_ids,
    )
    if reconfirm_ids:
        logger.info("Selectively invalidated {} multi-hop-4 candidate(s)", len(reconfirm_ids))
    remaining = tuple(
        candidate for candidate in candidates if multi_hop_4_candidate_id(candidate.fact_ids) not in progress_by_id
    )
    logger.info(
        "Loaded {} multi-hop-4 candidates; {} already checked, {} remaining",
        len(candidates),
        len(candidates) - len(remaining),
        len(remaining),
    )

    def _persist_batch(
        batch_results: Sequence[tuple[MultiHop4Candidate, ConfirmedMultiHop4Bundle | None]],
    ) -> None:
        """Replace completed outcomes and atomically checkpoint one batch."""
        completed_ids = {multi_hop_4_candidate_id(candidate.fact_ids) for candidate, _bundle in batch_results}
        stale_bundle_ids = [
            bundle_id
            for bundle_id, bundle in bundles_by_id.items()
            if multi_hop_4_candidate_id(bundle.fact_ids) in completed_ids
        ]
        for bundle_id in stale_bundle_ids:
            del bundles_by_id[bundle_id]
        bundles_by_id.update({bundle.bundle_id: bundle for _candidate, bundle in batch_results if bundle is not None})
        for candidate_id in completed_ids:
            progress_by_id[candidate_id] = MultiHop4ProgressRecord(
                candidate_id=candidate_id,
                run_id=run_id,
                prompt_version=MULTI_HOP_4_PROMPT_VERSION,
                confirmed_by_model=settings.llm_model,
            )
        _write_state(
            arguments.bundles_output_path,
            arguments.progress_output_path,
            candidates,
            bundles_by_id,
            progress_by_id,
        )
        logger.debug("Checkpointed {} multi-hop-4 candidate(s)", len(batch_results))

    if reconfirm_ids:
        _write_state(
            arguments.bundles_output_path,
            arguments.progress_output_path,
            candidates,
            bundles_by_id,
            progress_by_id,
        )
    if remaining:
        client = BamlMultiHop4Client(settings, synthesis_settings, run_id=run_id)
        _, failures = confirm_multi_hop_4_bundles(
            facts,
            relations,
            source_bundles,
            remaining,
            client,
            batch_size=synthesis_settings.multi_hop_4_batch_size,
            max_concurrency=synthesis_settings.multi_hop_4_max_concurrency,
            max_retries=synthesis_settings.llm_max_retries,
            retry_delay_seconds=synthesis_settings.llm_retry_delay_seconds,
            on_batch_confirmed=_persist_batch,
        )
        _require_completed_confirmation(failures, allow_incomplete=arguments.allow_incomplete_bundles)

    confirmed_bundles = tuple(sorted(bundles_by_id.values(), key=lambda bundle: (bundle.fact_ids, bundle.bridge)))
    base_bundles = sample_bundles(facts, relations, overlap_threshold=synthesis_settings.bundle_overlap_threshold)
    all_bundles = (
        *base_bundles,
        *(bundle.to_evidence_bundle() for bundle in source_bundles),
        *(bundle.to_evidence_bundle() for bundle in confirmed_bundles),
    )
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
        "Confirmed {} multi-hop-4 bundles and wrote {} scenarios in {:.2f} seconds",
        len(confirmed_bundles),
        len(scenarios),
        time.perf_counter() - started_at,
    )


def _write_state(
    bundles_output_path: Path,
    progress_output_path: Path,
    candidates: Sequence[MultiHop4Candidate],
    bundles_by_id: dict[str, ConfirmedMultiHop4Bundle],
    progress_by_id: dict[str, MultiHop4ProgressRecord],
) -> None:
    """Write deterministic bundle and progress checkpoints."""
    from legal_rag.benchmark_synthesis.multi_hop_4 import multi_hop_4_candidate_id
    from legal_rag.benchmark_synthesis.writer import write_multi_hop_4_bundles, write_multi_hop_4_progress

    write_multi_hop_4_bundles(
        bundles_output_path,
        tuple(sorted(bundles_by_id.values(), key=lambda bundle: (bundle.fact_ids, bundle.bridge))),
    )
    write_multi_hop_4_progress(
        progress_output_path,
        tuple(
            progress_by_id[multi_hop_4_candidate_id(candidate.fact_ids)]
            for candidate in candidates
            if multi_hop_4_candidate_id(candidate.fact_ids) in progress_by_id
        ),
    )


if __name__ == "__main__":
    main()
