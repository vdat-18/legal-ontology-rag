"""Command-line interface for LLM-assisted ontology concept proposals."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from legal_rag.ontology_proposal_execution import DEFAULT_ONTOLOGY_MAX_CONCURRENCY


def _parser() -> argparse.ArgumentParser:
    """Build the concept-proposal command-line parser."""
    parser = argparse.ArgumentParser(
        description="Generate and persist typed source-grounded concept relation proposals."
    )
    parser.add_argument("corpus_manifest_path", type=Path, help="Ordered multi-document corpus manifest JSON path")
    parser.add_argument("candidates_path", type=Path, help="Concept candidate-set JSON path")
    parser.add_argument("concept_contract_path", type=Path, help="Versioned concept-resolution contract YAML")
    parser.add_argument("proposals_output_path", type=Path, help="Concept proposal JSONL destination")
    parser.add_argument("progress_output_path", type=Path, help="Concept proposal progress JSONL destination")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_ONTOLOGY_MAX_CONCURRENCY,
        help="Maximum concurrent concept proposal batches",
    )
    return parser


def main() -> None:
    """Generate typed concept proposals with resumable checkpoints."""
    from legal_rag.ontology_change.corpus import read_change_corpus_manifest
    from legal_rag.ontology_concept.config import ConceptResolutionSettings
    from legal_rag.ontology_concept.models import ConceptProposalProgress, ConceptRelationProposal
    from legal_rag.ontology_concept.proposals import (
        BamlConceptProposalClient,
        build_concept_proposal_run_id,
        propose_concept_relations,
    )
    from legal_rag.ontology_concept.writer import (
        read_concept_candidates,
        read_concept_progress,
        read_concept_proposals,
        write_concept_progress,
        write_concept_proposals,
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
    concept_settings = ConceptResolutionSettings.from_path(arguments.concept_contract_path)
    candidates = read_concept_candidates(arguments.candidates_path)
    read_change_corpus_manifest(arguments.corpus_manifest_path)
    run_id = build_concept_proposal_run_id(candidates.run_id, concept_settings, provider_settings)
    existing_proposals = (
        read_concept_proposals(arguments.proposals_output_path) if arguments.proposals_output_path.exists() else ()
    )
    existing_progress = (
        read_concept_progress(arguments.progress_output_path) if arguments.progress_output_path.exists() else ()
    )
    candidate_ids = {item.candidate_id for item in candidates.pairs}
    proposals_by_id, processed_ids = load_compatible_proposal_resume_state(
        existing_proposals,
        existing_progress,
        candidate_ids,
        run_id,
        workflow_label="concept",
        candidate_scope="pairs",
    )
    remaining = tuple(item for item in candidates.pairs if item.candidate_id not in processed_ids)

    def persist_batch(batch_results: Sequence[tuple[str, tuple[ConceptRelationProposal, ...]]]) -> None:
        """Persist proposals before marking candidate pairs complete."""
        replace_replayed_proposals(proposals_by_id, batch_results)
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
            client,
            concept_settings,
            run_id=run_id,
            max_concurrency=arguments.max_concurrency,
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
    logger.info(
        "Completed {}/{} candidate pairs and wrote {} concept proposals",
        len(processed_ids),
        len(candidates.pairs),
        len(proposals),
    )


if __name__ == "__main__":
    main()
