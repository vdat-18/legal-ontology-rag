"""Command-line interface for LLM-assisted ontology concept proposals."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from loguru import logger


def _parser() -> argparse.ArgumentParser:
    """Build the concept-proposal command-line parser."""
    parser = argparse.ArgumentParser(
        description="Generate exact-evidenced concept relation proposals for mandatory human review."
    )
    parser.add_argument("corpus_manifest_path", type=Path, help="Ordered multi-document corpus manifest JSON path")
    parser.add_argument("candidates_path", type=Path, help="Concept candidate-set JSON path")
    parser.add_argument("concept_contract_path", type=Path, help="Versioned concept-resolution contract YAML")
    parser.add_argument("proposals_output_path", type=Path, help="Concept proposal JSONL destination")
    parser.add_argument("progress_output_path", type=Path, help="Concept proposal progress JSONL destination")
    parser.add_argument("reviews_output_path", type=Path, help="Human-review JSONL destination/template")
    return parser


def main() -> None:
    """Generate reviewable concept proposals with resumable checkpoints."""
    from legal_rag.ontology_change.corpus import read_change_corpus_manifest
    from legal_rag.ontology_concept.config import ConceptResolutionSettings
    from legal_rag.ontology_concept.models import ConceptProposalProgress, ConceptRelationProposal
    from legal_rag.ontology_concept.proposals import (
        BamlConceptProposalClient,
        build_concept_proposal_run_id,
        propose_concept_relations,
    )
    from legal_rag.ontology_concept.writer import (
        pending_concept_review_template,
        read_concept_candidates,
        read_concept_progress,
        read_concept_proposals,
        read_concept_reviews,
        write_concept_progress,
        write_concept_proposals,
        write_concept_reviews,
    )
    from legal_rag.settings import Settings

    arguments = _parser().parse_args()
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=True)
    provider_settings = Settings.from_environment()
    concept_settings = ConceptResolutionSettings.from_path(arguments.concept_contract_path)
    candidates = read_concept_candidates(arguments.candidates_path)
    corpus = read_change_corpus_manifest(arguments.corpus_manifest_path)
    chunk_text_by_id = {chunk.chunk_id: chunk.text for document in corpus.documents for chunk in document.chunks}
    run_id = build_concept_proposal_run_id(candidates.run_id, concept_settings, provider_settings)
    existing_proposals = (
        read_concept_proposals(arguments.proposals_output_path) if arguments.proposals_output_path.exists() else ()
    )
    existing_progress = (
        read_concept_progress(arguments.progress_output_path) if arguments.progress_output_path.exists() else ()
    )
    if any(item.run_id != run_id for item in existing_proposals) or any(
        item.run_id != run_id for item in existing_progress
    ):
        msg = "Existing concept outputs belong to a different candidate, prompt, or model run"
        raise ValueError(msg)
    candidate_ids = {item.candidate_id for item in candidates.pairs}
    processed_ids = {item.candidate_id for item in existing_progress}
    if not processed_ids <= candidate_ids:
        msg = "Existing concept progress references unknown candidate pairs"
        raise ValueError(msg)
    proposals_by_id = {item.proposal_id: item for item in existing_proposals}
    if len(proposals_by_id) != len(existing_proposals):
        msg = "Existing concept proposal identifiers must be unique"
        raise ValueError(msg)
    remaining = tuple(item for item in candidates.pairs if item.candidate_id not in processed_ids)

    def persist_batch(batch_results: Sequence[tuple[str, tuple[ConceptRelationProposal, ...]]]) -> None:
        """Persist proposals before marking candidate pairs complete."""
        for _, proposals in batch_results:
            proposals_by_id.update({item.proposal_id: item for item in proposals})
        write_concept_proposals(arguments.proposals_output_path, tuple(proposals_by_id.values()))
        processed_ids.update(candidate_id for candidate_id, _ in batch_results)
        write_concept_progress(
            arguments.progress_output_path,
            tuple(
                ConceptProposalProgress(candidate_id=item.candidate_id, run_id=run_id)
                for item in candidates.pairs
                if item.candidate_id in processed_ids
            ),
        )

    if remaining:
        client = BamlConceptProposalClient(provider_settings, concept_settings)
        propose_concept_relations(
            remaining,
            candidates.occurrences,
            chunk_text_by_id,
            client,
            concept_settings,
            run_id=run_id,
            on_batch_processed=persist_batch,
        )
    proposals = tuple(proposals_by_id.values())
    write_concept_proposals(arguments.proposals_output_path, proposals)
    write_concept_progress(
        arguments.progress_output_path,
        tuple(
            ConceptProposalProgress(candidate_id=item.candidate_id, run_id=run_id)
            for item in candidates.pairs
            if item.candidate_id in processed_ids
        ),
    )
    existing_reviews = (
        read_concept_reviews(arguments.reviews_output_path) if arguments.reviews_output_path.exists() else ()
    )
    reviews_by_id = {item.proposal_id: item for item in existing_reviews}
    if len(reviews_by_id) != len(existing_reviews):
        msg = "Existing concept review identifiers must be unique"
        raise ValueError(msg)
    for review in pending_concept_review_template(proposals):
        reviews_by_id.setdefault(review.proposal_id, review)
    write_concept_reviews(
        arguments.reviews_output_path,
        tuple(reviews_by_id[item.proposal_id] for item in proposals),
    )
    logger.info(
        "Completed {}/{} candidate pairs and wrote {} concept proposals for human review",
        len(processed_ids),
        len(candidates.pairs),
        len(proposals),
    )


if __name__ == "__main__":
    main()
