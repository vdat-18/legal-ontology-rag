"""Command-line interface for LLM-assisted legal-exception proposals."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from loguru import logger


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
    parser.add_argument("--max-concurrency", type=int, default=1, help="Maximum concurrent exception batches")
    return parser


def main() -> None:
    """Generate typed exception proposals with resumable checkpoints."""
    from legal_rag.ontology_change.corpus import read_change_corpus_manifest, replace_change_corpus_records
    from legal_rag.ontology_exception.config import ExceptionResolutionSettings
    from legal_rag.ontology_exception.models import ExceptionProposalProgress, LegalExceptionProposal
    from legal_rag.ontology_exception.proposals import (
        BamlExceptionProposalClient,
        build_exception_proposal_run_id,
        propose_legal_exceptions,
    )
    from legal_rag.ontology_exception.writer import (
        read_exception_candidates,
        read_exception_progress,
        read_exception_proposals,
        write_exception_progress,
        write_exception_proposals,
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
    exception_settings = ExceptionResolutionSettings.from_path(arguments.exception_contract_path)
    candidates = read_exception_candidates(arguments.candidates_path)
    manifest_corpus = read_change_corpus_manifest(arguments.corpus_manifest_path)
    records = read_population_records(arguments.canonical_records_path)
    corpus = replace_change_corpus_records(manifest_corpus, records)
    run_id = build_exception_proposal_run_id(candidates.run_id, exception_settings, provider_settings)
    existing_proposals = (
        read_exception_proposals(arguments.proposals_output_path) if arguments.proposals_output_path.exists() else ()
    )
    existing_progress = (
        read_exception_progress(arguments.progress_output_path) if arguments.progress_output_path.exists() else ()
    )
    candidate_ids = {window.candidate_id for window in candidates.windows}
    proposals_by_id, processed_ids = load_compatible_proposal_resume_state(
        existing_proposals,
        existing_progress,
        candidate_ids,
        run_id,
        workflow_label="exception",
        candidate_scope="windows",
    )
    remaining = tuple(window for window in candidates.windows if window.candidate_id not in processed_ids)

    def persist_batch(batch_results: Sequence[tuple[str, tuple[LegalExceptionProposal, ...]]]) -> None:
        """Persist proposals before marking candidate windows complete."""
        replace_replayed_proposals(proposals_by_id, batch_results)
        write_exception_proposals(arguments.proposals_output_path, tuple(proposals_by_id.values()))
        processed_ids.update(candidate_id for candidate_id, _ in batch_results)
        write_exception_progress(
            arguments.progress_output_path,
            tuple(
                ExceptionProposalProgress(candidate_id=window.candidate_id, run_id=run_id)
                for window in candidates.windows
                if window.candidate_id in processed_ids
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

    proposals = tuple(proposals_by_id.values())
    write_exception_proposals(arguments.proposals_output_path, proposals)
    write_exception_progress(
        arguments.progress_output_path,
        tuple(
            ExceptionProposalProgress(candidate_id=window.candidate_id, run_id=run_id)
            for window in candidates.windows
            if window.candidate_id in processed_ids
        ),
    )
    logger.info(
        "Completed {}/{} candidate windows and wrote {} exception proposals",
        len(processed_ids),
        len(candidates.windows),
        len(proposals),
    )


if __name__ == "__main__":
    main()
