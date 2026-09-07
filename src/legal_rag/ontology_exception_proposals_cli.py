"""Command-line interface for LLM-assisted legal-exception proposals."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from legal_rag.ontology_proposal_execution import DEFAULT_ONTOLOGY_MAX_CONCURRENCY


def _parser() -> argparse.ArgumentParser:
    """Build the exception-proposal command-line parser."""
    parser = argparse.ArgumentParser(
        description="Generate and persist typed source-grounded exception and citation proposals."
    )
    parser.add_argument("corpus_manifest_path", type=Path, help="Ordered multi-document corpus manifest JSON path")
    parser.add_argument("canonical_records_path", type=Path, help="Canonical multi-document population JSONL path")
    parser.add_argument("candidates_path", type=Path, help="Exception candidate-set JSON path")
    parser.add_argument("exception_contract_path", type=Path, help="Versioned exception-resolution contract YAML")
    parser.add_argument("proposals_output_path", type=Path, help="Exception proposal JSONL destination")
    parser.add_argument("progress_output_path", type=Path, help="Exception proposal progress JSONL destination")
    parser.add_argument(
        "--rerun-candidate-ids-path",
        type=Path,
        help="Reviewed JSON selection containing only exception candidates that must be proposed again",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_ONTOLOGY_MAX_CONCURRENCY,
        help="Maximum concurrent exception batches",
    )
    return parser


def main() -> None:
    """Generate typed exception proposals with resumable checkpoints."""
    from legal_rag.ontology_change.corpus import read_change_corpus_manifest, replace_change_corpus_records
    from legal_rag.ontology_exception.config import ExceptionResolutionSettings
    from legal_rag.ontology_exception.models import ExceptionProposalProgress, LegalExceptionProposal
    from legal_rag.ontology_exception.proposals import (
        BamlExceptionProposalClient,
        build_exception_proposal_rerun_id,
        build_exception_proposal_run_id,
        propose_legal_exceptions,
    )
    from legal_rag.ontology_exception.selection import (
        load_exception_rerun_selection,
        select_exception_rerun_candidates,
    )
    from legal_rag.ontology_exception.writer import (
        read_exception_candidates,
        read_exception_progress,
        read_exception_proposals,
        write_exception_progress,
        write_exception_proposals,
    )
    from legal_rag.ontology_population.writer import read_population_records
    from legal_rag.ontology_proposal_resume import replace_replayed_proposals
    from legal_rag.settings import Settings

    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=True)
    provider_settings = Settings.from_environment()
    exception_settings = ExceptionResolutionSettings.from_path(arguments.exception_contract_path)
    candidates = read_exception_candidates(arguments.candidates_path)
    manifest_corpus = read_change_corpus_manifest(arguments.corpus_manifest_path)
    records = read_population_records(arguments.canonical_records_path)
    corpus = replace_change_corpus_records(manifest_corpus, records)
    base_run_id = build_exception_proposal_run_id(candidates.run_id, exception_settings, provider_settings)
    existing_proposals = (
        read_exception_proposals(arguments.proposals_output_path) if arguments.proposals_output_path.exists() else ()
    )
    existing_progress = (
        read_exception_progress(arguments.progress_output_path) if arguments.progress_output_path.exists() else ()
    )
    candidate_ids = {window.candidate_id for window in candidates.windows}
    proposals_by_id = {proposal.proposal_id: proposal for proposal in existing_proposals}
    if len(proposals_by_id) != len(existing_proposals):
        msg = "Existing exception proposal identifiers must be unique"
        raise ValueError(msg)
    if any(proposal.candidate_id not in candidate_ids for proposal in existing_proposals):
        msg = "Existing exception proposals reference unknown candidate windows"
        raise ValueError(msg)
    progress_by_id = {item.candidate_id: item for item in existing_progress}
    if len(progress_by_id) != len(existing_progress) or not set(progress_by_id) <= candidate_ids:
        msg = "Existing exception progress references invalid candidate windows"
        raise ValueError(msg)

    if arguments.rerun_candidate_ids_path is None:
        run_id = base_run_id
        if any(item.run_id != run_id for item in (*existing_proposals, *existing_progress)):
            msg = "Existing exception outputs belong to a different candidate, prompt, or model run"
            raise ValueError(msg)
        selected_windows = candidates.windows
    else:
        selection = load_exception_rerun_selection(arguments.rerun_candidate_ids_path)
        selected_windows = select_exception_rerun_candidates(candidates.windows, existing_progress, selection)
        run_id = build_exception_proposal_rerun_id(
            base_run_id,
            tuple(window.candidate_id for window in selected_windows),
        )
    remaining = tuple(
        window
        for window in selected_windows
        if window.candidate_id not in progress_by_id or progress_by_id[window.candidate_id].run_id != run_id
    )

    def persist_batch(batch_results: Sequence[tuple[str, tuple[LegalExceptionProposal, ...]]]) -> None:
        """Persist proposals before marking candidate windows complete."""
        replace_replayed_proposals(proposals_by_id, batch_results)
        write_exception_proposals(arguments.proposals_output_path, tuple(proposals_by_id.values()))
        progress_by_id.update(
            {
                candidate_id: ExceptionProposalProgress(candidate_id=candidate_id, run_id=run_id)
                for candidate_id, _ in batch_results
            }
        )
        write_exception_progress(
            arguments.progress_output_path,
            tuple(
                progress_by_id[window.candidate_id]
                for window in candidates.windows
                if window.candidate_id in progress_by_id
            ),
        )

    if remaining:
        client = BamlExceptionProposalClient(provider_settings, exception_settings)
        propose_legal_exceptions(
            remaining,
            corpus,
            client,
            exception_settings,
            run_id=run_id,
            max_concurrency=arguments.max_concurrency,
            on_batch_processed=persist_batch,
        )

    proposals = tuple(sorted(proposals_by_id.values(), key=lambda proposal: proposal.proposal_id))
    write_exception_proposals(arguments.proposals_output_path, proposals)
    write_exception_progress(
        arguments.progress_output_path,
        tuple(
            progress_by_id[window.candidate_id]
            for window in candidates.windows
            if window.candidate_id in progress_by_id
        ),
    )
    logger.info(
        "Completed {}/{} candidate windows and wrote {} exception proposals",
        len(progress_by_id),
        len(candidates.windows),
        len(proposals),
    )


if __name__ == "__main__":
    main()
