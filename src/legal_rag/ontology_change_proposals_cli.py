"""Command-line interface for LLM-assisted cross-document change proposals."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the legal-change proposal command-line parser."""
    parser = argparse.ArgumentParser(
        description="Generate exact-evidenced legal-change proposals for mandatory human review."
    )
    parser.add_argument("corpus_manifest_path", type=Path, help="Ordered multi-document corpus manifest JSON path")
    parser.add_argument("candidates_path", type=Path, help="Legal-change candidate-set JSON path")
    parser.add_argument("change_contract_path", type=Path, help="Versioned change-resolution contract YAML path")
    parser.add_argument("proposals_output_path", type=Path, help="Legal-change proposal JSONL destination")
    parser.add_argument("progress_output_path", type=Path, help="Legal-change proposal progress JSONL destination")
    parser.add_argument("reviews_output_path", type=Path, help="Human-review JSONL destination/template")
    return parser


def main() -> None:
    """Generate reviewable legal-change proposals with resumable checkpoints."""
    from legal_rag.ontology_change.config import ChangeResolutionSettings
    from legal_rag.ontology_change.corpus import read_change_corpus_manifest
    from legal_rag.ontology_change.models import ChangeProposalProgress, LegalChangeProposal
    from legal_rag.ontology_change.proposals import (
        BamlChangeProposalClient,
        build_change_proposal_run_id,
        propose_legal_changes,
    )
    from legal_rag.ontology_change.writer import (
        pending_change_review_template,
        read_change_candidates,
        read_change_progress,
        read_change_proposals,
        read_change_reviews,
        write_change_progress,
        write_change_proposals,
        write_change_reviews,
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
    run_id = build_change_proposal_run_id(candidates.run_id, change_settings, provider_settings)
    existing_proposals = (
        read_change_proposals(arguments.proposals_output_path) if arguments.proposals_output_path.exists() else ()
    )
    existing_progress = (
        read_change_progress(arguments.progress_output_path) if arguments.progress_output_path.exists() else ()
    )
    candidate_ids = {window.candidate_id for window in candidates.windows}
    proposals_by_id, processed_ids = load_compatible_proposal_resume_state(
        existing_proposals,
        existing_progress,
        candidate_ids,
        run_id,
        workflow_label="legal-change",
        candidate_scope="windows",
    )
    remaining = tuple(window for window in candidates.windows if window.candidate_id not in processed_ids)

    def persist_batch(batch_results: Sequence[tuple[str, tuple[LegalChangeProposal, ...]]]) -> None:
        """Persist proposals before marking candidate windows complete."""
        replace_replayed_proposals(proposals_by_id, batch_results)
        write_change_proposals(arguments.proposals_output_path, tuple(proposals_by_id.values()))
        processed_ids.update(candidate_id for candidate_id, _ in batch_results)
        write_change_progress(
            arguments.progress_output_path,
            tuple(
                ChangeProposalProgress(candidate_id=window.candidate_id, run_id=run_id)
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
            on_batch_processed=persist_batch,
        )

    proposals = tuple(proposals_by_id.values())
    write_change_proposals(arguments.proposals_output_path, proposals)
    write_change_progress(
        arguments.progress_output_path,
        tuple(
            ChangeProposalProgress(candidate_id=window.candidate_id, run_id=run_id)
            for window in candidates.windows
            if window.candidate_id in processed_ids
        ),
    )
    existing_reviews = (
        read_change_reviews(arguments.reviews_output_path) if arguments.reviews_output_path.exists() else ()
    )
    reviews_by_id = {review.proposal_id: review for review in existing_reviews}
    if len(reviews_by_id) != len(existing_reviews):
        msg = "Existing legal-change review identifiers must be unique"
        raise ValueError(msg)
    for review in pending_change_review_template(proposals):
        reviews_by_id.setdefault(review.proposal_id, review)
    write_change_reviews(
        arguments.reviews_output_path,
        tuple(reviews_by_id[proposal.proposal_id] for proposal in proposals),
    )
    logger.info(
        "Completed {}/{} candidate windows and wrote {} legal-change proposals for human review",
        len(processed_ids),
        len(candidates.windows),
        len(proposals),
    )


if __name__ == "__main__":
    main()
