"""Command-line interface for LLM-assisted cross-document change proposals."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from legal_rag.ontology_proposal_execution import DEFAULT_ONTOLOGY_MAX_CONCURRENCY


def _parser() -> argparse.ArgumentParser:
    """Build the legal-change proposal command-line parser."""
    parser = argparse.ArgumentParser(description="Generate and persist typed source-grounded legal-change proposals.")
    parser.add_argument("corpus_manifest_path", type=Path, help="Ordered multi-document corpus manifest JSON path")
    parser.add_argument("candidates_path", type=Path, help="Legal-change candidate-set JSON path")
    parser.add_argument("change_contract_path", type=Path, help="Versioned change-resolution contract YAML path")
    parser.add_argument("proposals_output_path", type=Path, help="Legal-change proposal JSONL destination")
    parser.add_argument("progress_output_path", type=Path, help="Legal-change proposal progress JSONL destination")
    parser.add_argument(
        "--rerun-candidate-ids-path",
        type=Path,
        help="Reviewed JSON selection containing only legal-change candidates that must be proposed again",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_ONTOLOGY_MAX_CONCURRENCY,
        help="Maximum concurrent legal-change batches",
    )
    return parser


def main() -> None:
    """Generate typed legal-change proposals with resumable checkpoints."""
    from legal_rag.ontology_change.config import ChangeResolutionSettings
    from legal_rag.ontology_change.corpus import read_change_corpus_manifest
    from legal_rag.ontology_change.models import ChangeProposalProgress, LegalChangeProposal
    from legal_rag.ontology_change.proposals import (
        BamlChangeProposalClient,
        build_change_proposal_rerun_id,
        build_change_proposal_run_id,
        propose_legal_changes,
    )
    from legal_rag.ontology_change.selection import load_change_rerun_selection, select_change_rerun_candidates
    from legal_rag.ontology_change.writer import (
        read_change_candidates,
        read_change_progress,
        read_change_proposals,
        write_change_progress,
        write_change_proposals,
    )
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
    change_settings = ChangeResolutionSettings.from_path(arguments.change_contract_path)
    candidates = read_change_candidates(arguments.candidates_path)
    corpus = read_change_corpus_manifest(arguments.corpus_manifest_path)
    base_run_id = build_change_proposal_run_id(candidates.run_id, change_settings, provider_settings)
    existing_proposals = (
        read_change_proposals(arguments.proposals_output_path) if arguments.proposals_output_path.exists() else ()
    )
    existing_progress = (
        read_change_progress(arguments.progress_output_path) if arguments.progress_output_path.exists() else ()
    )
    candidate_ids = {window.candidate_id for window in candidates.windows}
    progress_by_id = {item.candidate_id: item for item in existing_progress}
    if arguments.rerun_candidate_ids_path is None:
        run_id = base_run_id
        proposals_by_id, processed_ids = load_compatible_proposal_resume_state(
            existing_proposals,
            existing_progress,
            candidate_ids,
            run_id,
            workflow_label="legal-change",
            candidate_scope="windows",
        )
        selected_windows = candidates.windows
    else:
        selection = load_change_rerun_selection(arguments.rerun_candidate_ids_path)
        selected_windows = select_change_rerun_candidates(candidates.windows, existing_progress, selection)
        run_id = build_change_proposal_rerun_id(
            base_run_id,
            tuple(window.candidate_id for window in selected_windows),
        )
        proposals_by_id = {proposal.proposal_id: proposal for proposal in existing_proposals}
        if len(proposals_by_id) != len(existing_proposals):
            msg = "Existing legal-change proposal identifiers must be unique"
            raise ValueError(msg)
        if any(proposal.candidate_id not in candidate_ids for proposal in existing_proposals):
            msg = "Existing legal-change proposals reference unknown candidate windows"
            raise ValueError(msg)
        processed_ids = set(progress_by_id)
    remaining = tuple(
        window
        for window in selected_windows
        if window.candidate_id not in processed_ids or progress_by_id[window.candidate_id].run_id != run_id
    )

    def persist_batch(batch_results: Sequence[tuple[str, tuple[LegalChangeProposal, ...]]]) -> None:
        """Persist proposals before marking candidate windows complete."""
        replace_replayed_proposals(proposals_by_id, batch_results)
        write_change_proposals(arguments.proposals_output_path, tuple(proposals_by_id.values()))
        processed_ids.update(candidate_id for candidate_id, _ in batch_results)
        progress_by_id.update(
            {
                candidate_id: ChangeProposalProgress(candidate_id=candidate_id, run_id=run_id)
                for candidate_id, _ in batch_results
            }
        )
        write_change_progress(
            arguments.progress_output_path,
            tuple(
                progress_by_id[window.candidate_id]
                for window in candidates.windows
                if window.candidate_id in processed_ids
            ),
        )

    if remaining:
        client = BamlChangeProposalClient(provider_settings, change_settings)
        propose_legal_changes(
            remaining,
            corpus,
            client,
            change_settings,
            run_id=run_id,
            max_concurrency=arguments.max_concurrency,
            on_batch_processed=persist_batch,
        )

    proposals = tuple(proposals_by_id.values())
    write_change_proposals(arguments.proposals_output_path, proposals)
    write_change_progress(
        arguments.progress_output_path,
        tuple(
            progress_by_id[window.candidate_id] for window in candidates.windows if window.candidate_id in processed_ids
        ),
    )
    logger.info(
        "Completed {}/{} candidate windows and wrote {} legal-change proposals",
        len(processed_ids),
        len(candidates.windows),
        len(proposals),
    )


if __name__ == "__main__":
    main()
