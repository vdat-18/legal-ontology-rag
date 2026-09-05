"""Command-line interface for global ontology assembly conflict resolution."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from legal_rag.ontology_proposal_execution import DEFAULT_ONTOLOGY_MAX_CONCURRENCY


def _parser() -> argparse.ArgumentParser:
    """Build the assembly conflict-resolution parser."""
    parser = argparse.ArgumentParser(description="Review rule-assembly proposal components with one global view.")
    parser.add_argument("chunks_path", type=Path)
    parser.add_argument("metadata_path", type=Path)
    parser.add_argument("records_path", type=Path)
    parser.add_argument("proposals_path", type=Path)
    parser.add_argument("assembly_contract_path", type=Path)
    parser.add_argument("resolved_proposals_output_path", type=Path)
    parser.add_argument("progress_output_path", type=Path)
    parser.add_argument(
        "--rerun-component-ids-path",
        type=Path,
        help="Reviewed JSON selection containing only proposal components that must be resolved again",
    )
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_ONTOLOGY_MAX_CONCURRENCY)
    return parser


def main() -> None:
    """Review proposal components while checkpointing every model decision."""
    from legal_rag.benchmark_synthesis.writer import read_document_metadata
    from legal_rag.chunker.writer import read_chunks
    from legal_rag.ontology_population.assembly_config import AssemblySettings
    from legal_rag.ontology_population.assembly_models import AssemblyProposal, AssemblyResolutionProgress
    from legal_rag.ontology_population.assembly_resolution import (
        BamlAssemblyResolutionClient,
        build_assembly_proposal_components,
        build_assembly_resolution_rerun_id,
        build_assembly_resolution_run_id,
        resolve_assembly_conflicts,
    )
    from legal_rag.ontology_population.assembly_resolution_selection import (
        load_assembly_resolution_selection,
        select_assembly_resolution_rerun_components,
    )
    from legal_rag.ontology_population.assembly_writer import (
        read_assembly_proposals,
        read_assembly_resolution_progress,
        write_assembly_proposals,
        write_assembly_resolution_progress,
    )
    from legal_rag.ontology_population.writer import read_population_records
    from legal_rag.ontology_proposal_resume import replace_replayed_proposals
    from legal_rag.settings import Settings

    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=True)
    provider_settings = Settings.from_environment()
    assembly_settings = AssemblySettings.from_path(arguments.assembly_contract_path)
    proposals = read_assembly_proposals(arguments.proposals_path)
    components = build_assembly_proposal_components(proposals)
    component_ids = {component.component_id for component in components}
    base_run_id = build_assembly_resolution_run_id(proposals, assembly_settings, provider_settings)

    existing = (
        read_assembly_proposals(arguments.resolved_proposals_output_path)
        if arguments.resolved_proposals_output_path.exists()
        else ()
    )
    existing_progress = (
        read_assembly_resolution_progress(arguments.progress_output_path)
        if arguments.progress_output_path.exists()
        else ()
    )
    progress_by_id = {item.component_id: item for item in existing_progress}
    if len(progress_by_id) != len(existing_progress) or not set(progress_by_id) <= component_ids:
        msg = "Existing assembly resolution progress references invalid proposal components"
        raise ValueError(msg)

    resolved_by_id: dict[str, AssemblyProposal] = {}
    for proposal in existing:
        if proposal.candidate_id not in component_ids:
            msg = "Existing resolved assembly proposals reference an invalid proposal component"
            raise ValueError(msg)
        resolved_by_id[proposal.proposal_id] = proposal
    if len(resolved_by_id) != len(existing):
        msg = "Existing resolved assembly proposal identifiers must be unique"
        raise ValueError(msg)

    if arguments.rerun_component_ids_path is None:
        run_id = base_run_id
        if any(item.run_id != run_id for item in (*existing, *existing_progress)):
            msg = "Existing assembly resolution outputs belong to a different proposal, prompt, or model run"
            raise ValueError(msg)
        selected_components = components
    else:
        selection = load_assembly_resolution_selection(arguments.rerun_component_ids_path)
        selected_components = select_assembly_resolution_rerun_components(components, existing_progress, selection)
        run_id = build_assembly_resolution_rerun_id(
            base_run_id,
            tuple(component.component_id for component in selected_components),
        )

    all_proposals_by_id = resolved_by_id
    remaining = tuple(
        component
        for component in selected_components
        if component.component_id not in progress_by_id or progress_by_id[component.component_id].run_id != run_id
    )

    def persist_batch(batch_results: Sequence[tuple[str, tuple[AssemblyProposal, ...]]]) -> None:
        """Persist resolved proposals before marking their components complete."""
        replace_replayed_proposals(all_proposals_by_id, batch_results)
        write_assembly_proposals(
            arguments.resolved_proposals_output_path,
            tuple(sorted(all_proposals_by_id.values(), key=lambda proposal: proposal.proposal_id)),
        )
        progress_by_id.update(
            {
                component_id: AssemblyResolutionProgress(component_id=component_id, run_id=run_id)
                for component_id, _ in batch_results
            }
        )
        write_assembly_resolution_progress(
            arguments.progress_output_path,
            tuple(
                progress_by_id[component.component_id]
                for component in components
                if component.component_id in progress_by_id
            ),
        )

    if remaining:
        records = read_population_records(arguments.records_path)
        chunks = read_chunks(arguments.chunks_path)
        metadata = read_document_metadata(arguments.metadata_path)
        client = BamlAssemblyResolutionClient(provider_settings, assembly_settings)
        resolve_assembly_conflicts(
            remaining,
            proposals,
            records,
            chunks,
            metadata,
            client,
            assembly_settings,
            run_id=run_id,
            max_concurrency=arguments.max_concurrency,
            on_batch_processed=persist_batch,
        )

    write_assembly_proposals(
        arguments.resolved_proposals_output_path,
        tuple(sorted(all_proposals_by_id.values(), key=lambda proposal: proposal.proposal_id)),
    )
    write_assembly_resolution_progress(
        arguments.progress_output_path,
        tuple(
            progress_by_id[component.component_id]
            for component in components
            if component.component_id in progress_by_id
        ),
    )
    logger.info(
        "Reviewed {}/{} assembly proposal components and wrote {} final proposals",
        len(progress_by_id),
        len(components),
        len(all_proposals_by_id),
    )


if __name__ == "__main__":
    main()
