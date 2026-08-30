"""Command-line interface for scope-filtered ground-truth scenario sampling."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from legal_rag.benchmark_synthesis.bundle_scope import BundleScopeRecord
    from legal_rag.benchmark_synthesis.config import SynthesisSettings
    from legal_rag.benchmark_synthesis.evidence_graph import EvidenceBundle, EvidenceRelation
    from legal_rag.benchmark_synthesis.models import LegalFact
    from legal_rag.benchmark_synthesis.multi_hop_3 import ConfirmedMultiHop3Bundle
    from legal_rag.benchmark_synthesis.multi_hop_4 import ConfirmedMultiHop4Bundle
    from legal_rag.benchmark_synthesis.multi_hop_5 import ConfirmedMultiHop5Bundle


def _parser() -> argparse.ArgumentParser:
    """Build the scope-filtered scenario-sampling command-line interface."""
    parser = argparse.ArgumentParser(description="Filter evidence bundles by domain and sample scenarios.")
    parser.add_argument("facts_path", type=Path, help="Legal-facts JSONL path")
    parser.add_argument("relations_path", type=Path, help="Evidence-relations JSONL path")
    parser.add_argument("multi_hop_3_bundles_path", type=Path, help="Confirmed three-fact bundles JSONL path")
    parser.add_argument("multi_hop_4_bundles_path", type=Path, help="Confirmed four-fact bundles JSONL path")
    parser.add_argument("multi_hop_5_bundles_path", type=Path, help="Confirmed five-fact bundles JSONL path")
    parser.add_argument("scope_output_path", type=Path, help="Checkpointed bundle-scope decisions JSONL path")
    parser.add_argument("scenarios_output_path", type=Path, help="Destination scope-filtered scenarios JSONL path")
    parser.add_argument("--personas", type=Path, required=True, help="Persona config JSON path")
    parser.add_argument(
        "--scenario-distribution", type=Path, required=True, help="Scenario-distribution config JSON path"
    )
    parser.add_argument("--total-samples", type=int, required=True, help="Target number of scenarios")
    parser.add_argument(
        "--reconfirm-bundle-id",
        action="append",
        default=[],
        help="Existing bundle ID to reconfirm; repeat as needed",
    )
    parser.add_argument(
        "--reconfirm-bundle-ids",
        type=Path,
        help="JSON string array of existing bundle IDs to reconfirm",
    )
    parser.add_argument("--synthesis-config", type=Path, required=True, help="Versioned synthesis config JSON path")
    return parser


def main() -> None:
    """Confirm bundle scope with checkpoints, then sample only accepted bundles."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.benchmark_synthesis.bundle_scope import (
        BamlBundleScopeClient,
        build_bundle_scope_run_id,
        confirm_bundle_scopes,
    )
    from legal_rag.benchmark_synthesis.bundle_scope_selection import (
        load_reconfirm_bundle_ids,
        validate_selective_scope_confirmation,
    )
    from legal_rag.benchmark_synthesis.config import SynthesisSettings
    from legal_rag.benchmark_synthesis.llm_execution import IncompleteLlmTasksError
    from legal_rag.benchmark_synthesis.scenario_sampling import (
        ScenarioSamplingConfig,
        load_personas,
        load_scenario_distribution,
        sample_scenarios,
    )
    from legal_rag.benchmark_synthesis.writer import (
        read_bundle_scope_records,
        read_evidence_relations,
        read_legal_facts,
        read_multi_hop_3_bundles,
        read_multi_hop_4_bundles,
        read_multi_hop_5_bundles,
        write_bundle_scope_records,
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

    facts = read_legal_facts(arguments.facts_path)
    relations = read_evidence_relations(arguments.relations_path)
    multi_hop_3 = read_multi_hop_3_bundles(arguments.multi_hop_3_bundles_path)
    multi_hop_4 = read_multi_hop_4_bundles(arguments.multi_hop_4_bundles_path)
    multi_hop_5 = read_multi_hop_5_bundles(arguments.multi_hop_5_bundles_path)
    bundles = _all_bundles(facts, relations, multi_hop_3, multi_hop_4, multi_hop_5, synthesis_settings)
    bundle_ids = frozenset(bundle.bundle_id for bundle in bundles)

    run_id = build_bundle_scope_run_id(
        bundles,
        facts,
        synthesis_settings,
        llm_base_url=settings.llm_base_url,
        llm_model=settings.llm_model,
    )
    reconfirm_ids = load_reconfirm_bundle_ids(
        arguments.reconfirm_bundle_id,
        arguments.reconfirm_bundle_ids,
    )
    existing_records = (
        read_bundle_scope_records(arguments.scope_output_path) if arguments.scope_output_path.exists() else ()
    )
    records_by_id = _prepare_resume_records(
        existing_records,
        bundle_ids,
        run_id,
        reconfirm_ids,
        validate_selective_scope_confirmation,
    )

    pending = tuple(bundle for bundle in bundles if bundle.bundle_id not in records_by_id)
    logger.info(
        "Loaded {} bundles; {} already scope-checked, {} remaining",
        len(bundles),
        len(records_by_id),
        len(pending),
    )

    def _persist(batch_records: Sequence[BundleScopeRecord]) -> None:
        records_by_id.update({record.bundle_id: record for record in batch_records})
        write_bundle_scope_records(
            arguments.scope_output_path,
            tuple(records_by_id[bundle.bundle_id] for bundle in bundles if bundle.bundle_id in records_by_id),
        )

    if pending:
        client = BamlBundleScopeClient(settings, synthesis_settings)
        _, failures = confirm_bundle_scopes(
            pending,
            facts,
            client,
            run_id=run_id,
            model=settings.llm_model,
            batch_size=synthesis_settings.bundle_scope_batch_size,
            max_concurrency=synthesis_settings.bundle_scope_max_concurrency,
            max_retries=synthesis_settings.llm_max_retries,
            retry_delay_seconds=synthesis_settings.llm_retry_delay_seconds,
            on_batch_confirmed=_persist,
        )
        if failures:
            incomplete_error = IncompleteLlmTasksError.for_stage("bundle scope confirmation", failures)
            raise incomplete_error

    ordered_records = tuple(records_by_id[bundle.bundle_id] for bundle in bundles)
    write_bundle_scope_records(arguments.scope_output_path, ordered_records)
    accepted_bundles = tuple(bundle for bundle in bundles if records_by_id[bundle.bundle_id].is_in_scope)
    if not accepted_bundles:
        msg = "Bundle scope confirmation accepted no bundles"
        raise ValueError(msg)

    scenario_config = ScenarioSamplingConfig(
        personas=load_personas(arguments.personas),
        distribution=load_scenario_distribution(arguments.scenario_distribution),
    )
    scenarios = sample_scenarios(
        accepted_bundles,
        {fact.fact_id: fact for fact in facts},
        scenario_config,
        total_samples=arguments.total_samples,
    )
    write_scenarios(arguments.scenarios_output_path, scenarios)
    logger.info(
        "Accepted {}/{} bundles and wrote {} scenarios in {:.2f} seconds",
        len(accepted_bundles),
        len(bundles),
        len(scenarios),
        time.perf_counter() - started_at,
    )


def _all_bundles(
    facts: Sequence[LegalFact],
    relations: Sequence[EvidenceRelation],
    multi_hop_3: Sequence[ConfirmedMultiHop3Bundle],
    multi_hop_4: Sequence[ConfirmedMultiHop4Bundle],
    multi_hop_5: Sequence[ConfirmedMultiHop5Bundle],
    synthesis_settings: SynthesisSettings,
) -> tuple[EvidenceBundle, ...]:
    """Build the deterministic bundle pool from every completed Step 4 output."""
    from legal_rag.benchmark_synthesis.evidence_graph import sample_bundles

    bundles = (
        *sample_bundles(facts, relations, overlap_threshold=synthesis_settings.bundle_overlap_threshold),
        *(bundle.to_evidence_bundle() for bundle in multi_hop_3),
        *(bundle.to_evidence_bundle() for bundle in multi_hop_4),
        *(bundle.to_evidence_bundle() for bundle in multi_hop_5),
    )
    bundles_by_id = {bundle.bundle_id: bundle for bundle in bundles}
    if len(bundles_by_id) != len(bundles):
        msg = "Step 4 outputs contain duplicate evidence bundle IDs"
        raise ValueError(msg)
    return tuple(sorted(bundles_by_id.values(), key=lambda bundle: bundle.bundle_id))


def _prepare_resume_records(
    existing_records: Sequence[BundleScopeRecord],
    bundle_ids: frozenset[str],
    run_id: str,
    reconfirm_ids: frozenset[str],
    validate_selective: Callable[[frozenset[str], frozenset[str], frozenset[str]], None],
) -> dict[str, BundleScopeRecord]:
    """Validate existing decisions and remove explicitly selected bundle IDs."""
    records_by_id = {record.bundle_id: record for record in existing_records}
    if len(records_by_id) != len(existing_records):
        msg = "Existing bundle-scope decisions contain duplicate bundle IDs"
        raise ValueError(msg)
    if not records_by_id.keys() <= bundle_ids:
        msg = "Existing bundle-scope decisions contain bundles outside the current bundle pool"
        raise ValueError(msg)
    if reconfirm_ids:
        validate_selective(bundle_ids, frozenset(records_by_id), reconfirm_ids)
        for bundle_id in reconfirm_ids:
            records_by_id.pop(bundle_id, None)
        logger.info("Selectively invalidated {} bundle-scope decision(s)", len(reconfirm_ids))
    if any(record.run_id != run_id for record in records_by_id.values()):
        msg = "Existing bundle-scope decisions belong to a different synthesis run"
        raise ValueError(msg)
    return records_by_id


if __name__ == "__main__":
    main()
