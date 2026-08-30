"""Command-line interface for LLM-assisted ontology rule-assembly proposals."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the assembly-proposal command-line parser."""
    parser = argparse.ArgumentParser(description="Generate and persist typed rule-assembly proposals.")
    parser.add_argument("chunks_path", type=Path, help="Final chunk JSONL path")
    parser.add_argument("metadata_path", type=Path, help="Document-metadata JSONL path")
    parser.add_argument("records_path", type=Path, help="Ontology population JSONL path")
    parser.add_argument("candidates_path", type=Path, help="Assembly candidate-set JSON path")
    parser.add_argument("assembly_contract_path", type=Path, help="Versioned assembly contract YAML path")
    parser.add_argument("proposals_output_path", type=Path, help="Assembly-proposal JSONL destination")
    parser.add_argument("progress_output_path", type=Path, help="Assembly-proposal progress JSONL destination")
    return parser


def main() -> None:
    """Generate typed proposals while checkpointing every candidate window."""
    from legal_rag.benchmark_synthesis.writer import read_document_metadata
    from legal_rag.chunker.writer import read_chunks
    from legal_rag.ontology_population.assembly_config import AssemblySettings
    from legal_rag.ontology_population.assembly_models import AssemblyProposal, AssemblyProposalProgress
    from legal_rag.ontology_population.assembly_proposals import (
        BamlAssemblyProposalClient,
        build_assembly_proposal_run_id,
        propose_assemblies,
    )
    from legal_rag.ontology_population.assembly_writer import (
        read_assembly_candidates,
        read_assembly_progress,
        read_assembly_proposals,
        write_assembly_progress,
        write_assembly_proposals,
    )
    from legal_rag.ontology_population.writer import read_population_records
    from legal_rag.ontology_proposal_resume import (
        load_compatible_proposal_resume_state,
        replace_replayed_proposals,
    )
    from legal_rag.settings import Settings

    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=True)
    provider_settings = Settings.from_environment()
    assembly_settings = AssemblySettings.from_path(arguments.assembly_contract_path)
    candidates = read_assembly_candidates(arguments.candidates_path)
    run_id = build_assembly_proposal_run_id(candidates.run_id, assembly_settings, provider_settings)
    chunks = read_chunks(arguments.chunks_path)
    records = read_population_records(arguments.records_path)
    metadata = read_document_metadata(arguments.metadata_path)
    existing_proposals = (
        read_assembly_proposals(arguments.proposals_output_path) if arguments.proposals_output_path.exists() else ()
    )
    existing_progress = (
        read_assembly_progress(arguments.progress_output_path) if arguments.progress_output_path.exists() else ()
    )
    candidate_ids = {window.candidate_id for window in candidates.windows}
    proposals_by_id, processed_ids = load_compatible_proposal_resume_state(
        existing_proposals,
        existing_progress,
        candidate_ids,
        run_id,
        workflow_label="assembly",
        candidate_scope="windows",
    )
    remaining = tuple(window for window in candidates.windows if window.candidate_id not in processed_ids)

    def persist_batch(batch_results: Sequence[tuple[str, tuple[AssemblyProposal, ...]]]) -> None:
        """Persist proposals before marking their candidate windows complete."""
        replace_replayed_proposals(proposals_by_id, batch_results)
        write_assembly_proposals(arguments.proposals_output_path, tuple(proposals_by_id.values()))
        processed_ids.update(candidate_id for candidate_id, _ in batch_results)
        write_assembly_progress(
            arguments.progress_output_path,
            tuple(
                AssemblyProposalProgress(candidate_id=window.candidate_id, run_id=run_id)
                for window in candidates.windows
                if window.candidate_id in processed_ids
            ),
        )

    if remaining:
        client = BamlAssemblyProposalClient(provider_settings, assembly_settings)
        propose_assemblies(
            remaining,
            records,
            chunks,
            metadata,
            client,
            assembly_settings,
            run_id=run_id,
            on_batch_processed=persist_batch,
        )

    proposals = tuple(proposals_by_id.values())
    write_assembly_proposals(arguments.proposals_output_path, proposals)
    write_assembly_progress(
        arguments.progress_output_path,
        tuple(
            AssemblyProposalProgress(candidate_id=window.candidate_id, run_id=run_id)
            for window in candidates.windows
            if window.candidate_id in processed_ids
        ),
    )
    logger.info(
        "Completed {}/{} candidate windows and wrote {} typed proposals",
        len(processed_ids),
        len(candidates.windows),
        len(proposals),
    )


if __name__ == "__main__":
    main()
