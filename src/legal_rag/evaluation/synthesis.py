"""Adapt legal episodes to Ragas and preserve their source provenance."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path
from typing import cast
from uuid import NAMESPACE_URL, UUID, uuid5

from langchain_core.documents import Document
from ragas.dataset_schema import SingleTurnSample
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import QueryDistribution
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from ragas.testset.synthesizers.testset_schema import Testset
from ragas.testset.transforms import Transforms

from legal_rag.errors import (
    DuplicateReferenceContextError,
    EmptyReferenceContextBatchError,
    InvalidSyntheticSampleError,
    InvalidTestsetSizeError,
    UnknownReferenceContextError,
)
from legal_rag.evaluation.schemas import GroundTruthCandidate, ReferenceContext
from legal_rag.ingestion import LegalEpisode


def prepare_ragas_documents(
    episodes: Iterable[LegalEpisode],
) -> tuple[list[Document], dict[str, ReferenceContext]]:
    """Convert legal episodes to Ragas documents with exact provenance metadata."""
    documents: list[Document] = []
    contexts_by_text: dict[str, ReferenceContext] = {}
    context_ids: set[str] = set()

    for episode in episodes:
        context_id = episode.name
        if context_id in context_ids:
            raise DuplicateReferenceContextError(context_id)

        context = ReferenceContext(
            context_id=context_id,
            text=episode.body,
            source_description=episode.source_description,
            effective_date=episode.reference_time.date(),
        )
        duplicate = contexts_by_text.get(context.text)
        if duplicate is not None:
            raise DuplicateReferenceContextError(duplicate.context_id, context.context_id)

        context_ids.add(context_id)
        contexts_by_text[context.text] = context
        documents.append(
            Document(
                page_content=context.text,
                metadata={
                    "context_id": context.context_id,
                    "source_description": context.source_description,
                    "effective_date": context.effective_date.isoformat(),
                },
            )
        )

    if not documents:
        raise EmptyReferenceContextBatchError

    return documents, contexts_by_text


def candidates_from_testset(
    testset: Testset,
    contexts_by_text: Mapping[str, ReferenceContext],
    *,
    as_of_date: date | None = None,
) -> list[GroundTruthCandidate]:
    """Convert validated Ragas samples into project-owned candidate records."""
    candidates: list[GroundTruthCandidate] = []

    for testset_sample in testset.samples:
        sample = testset_sample.eval_sample
        if not isinstance(sample, SingleTurnSample):
            raise InvalidSyntheticSampleError("eval_sample")

        question = _required_text(sample.user_input, "user_input")
        reference_answer = _required_text(sample.reference, "reference")
        reference_contexts = sample.reference_contexts
        if not reference_contexts:
            raise InvalidSyntheticSampleError("reference_contexts")

        source_contexts = tuple(_source_context(text, contexts_by_text) for text in reference_contexts)
        candidates.append(
            GroundTruthCandidate(
                candidate_id=_candidate_id(question, source_contexts, as_of_date),
                question=question,
                reference_answer=reference_answer,
                reference_contexts=source_contexts,
                synthesizer_name=testset_sample.synthesizer_name,
                persona_name=sample.persona_name,
                query_style=sample.query_style,
                query_length=sample.query_length,
                as_of_date=as_of_date,
            )
        )

    return candidates


def synthesize_single_hop_candidates(
    generator: TestsetGenerator,
    query_synthesizer: SingleHopSpecificQuerySynthesizer,
    episodes: Iterable[LegalEpisode],
    *,
    testset_size: int,
    transforms: Transforms | None = None,
    as_of_date: date | None = None,
) -> list[GroundTruthCandidate]:
    """Generate single-hop candidates using caller-supplied Ragas components."""
    if testset_size < 1:
        raise InvalidTestsetSizeError

    documents, contexts_by_text = prepare_ragas_documents(episodes)
    query_distribution: QueryDistribution = [(query_synthesizer, 1.0)]
    generated = generator.generate_with_chunks(
        documents,
        testset_size=testset_size,
        transforms=transforms,
        query_distribution=query_distribution,
    )
    testset = cast(Testset, generated)
    return candidates_from_testset(testset, contexts_by_text, as_of_date=as_of_date)


def write_candidates_jsonl(candidates: Iterable[GroundTruthCandidate], path: Path) -> None:
    """Write candidates as UTF-8 JSON Lines for review and versioning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for candidate in candidates:
            stream.write(candidate.model_dump_json())
            stream.write("\n")


def _required_text(value: str | None, field_name: str) -> str:
    if value is None or not value.strip():
        raise InvalidSyntheticSampleError(field_name)
    return value


def _source_context(text: str, contexts_by_text: Mapping[str, ReferenceContext]) -> ReferenceContext:
    context = contexts_by_text.get(text)
    if context is None:
        raise UnknownReferenceContextError
    return context


def _candidate_id(question: str, contexts: tuple[ReferenceContext, ...], as_of_date: date | None) -> UUID:
    identity = json.dumps(
        {
            "as_of_date": as_of_date.isoformat() if as_of_date is not None else None,
            "context_ids": [context.context_id for context in contexts],
            "question": question,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return uuid5(NAMESPACE_URL, identity)
