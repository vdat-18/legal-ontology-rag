"""Offline tests for synthetic evaluation candidate generation."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import cast

import pytest
from ragas.dataset_schema import SingleTurnSample
from ragas.testset import TestsetGenerator as RagasTestsetGenerator
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from ragas.testset.synthesizers.testset_schema import Testset as RagasTestset
from ragas.testset.synthesizers.testset_schema import TestsetSample as RagasTestsetSample

from legal_rag.errors import EmptyReferenceContextBatchError, UnknownReferenceContextError
from legal_rag.evaluation import (
    ApprovalStatus,
    candidates_from_testset,
    prepare_ragas_documents,
    synthesize_single_hop_candidates,
    write_candidates_jsonl,
)
from legal_rag.ingestion import LegalEpisode


def _episode() -> LegalEpisode:
    """Build one deterministic source episode."""
    return LegalEpisode(
        name="168/2024/ND-CP - Article 10",
        body="Article 10 source text",
        source_description="Decree 168/2024/ND-CP",
        reference_time=datetime(2025, 1, 1, tzinfo=UTC),
    )


def _testset(context: str = "Article 10 source text") -> RagasTestset:
    """Build one deterministic Ragas testset."""
    return RagasTestset(
        samples=[
            RagasTestsetSample(
                eval_sample=SingleTurnSample(
                    user_input="What penalty applies?",
                    reference="The source defines the applicable penalty.",
                    reference_contexts=[context],
                    persona_name="Road user",
                    query_style="PERFECT_GRAMMAR",
                    query_length="SHORT",
                ),
                synthesizer_name="single_hop_specific_query_synthesizer",
            )
        ]
    )


def test_prepare_ragas_documents_preserves_provenance() -> None:
    """Ragas documents retain the source identifier and effective date."""
    documents, contexts_by_text = prepare_ragas_documents([_episode()])

    assert len(documents) == 1
    assert documents[0].page_content == "Article 10 source text"
    assert documents[0].metadata == {
        "context_id": "168/2024/ND-CP - Article 10",
        "source_description": "Decree 168/2024/ND-CP",
        "effective_date": "2025-01-01",
    }
    assert contexts_by_text[documents[0].page_content].context_id == "168/2024/ND-CP - Article 10"


def test_prepare_ragas_documents_rejects_empty_source() -> None:
    """Synthesis fails before Ragas when no legal source is available."""
    with pytest.raises(EmptyReferenceContextBatchError):
        prepare_ragas_documents([])


def test_candidates_from_testset_builds_stable_pending_candidate() -> None:
    """The same Ragas sample produces the same review candidate identifier."""
    _, contexts_by_text = prepare_ragas_documents([_episode()])

    first = candidates_from_testset(_testset(), contexts_by_text, as_of_date=date(2025, 2, 1))[0]
    second = candidates_from_testset(_testset(), contexts_by_text, as_of_date=date(2025, 2, 1))[0]

    assert first.candidate_id == second.candidate_id
    assert first.question == "What penalty applies?"
    assert first.reference_contexts[0].effective_date.isoformat() == "2025-01-01"
    assert first.as_of_date == date(2025, 2, 1)
    assert first.approval_status is ApprovalStatus.PENDING


def test_candidate_id_distinguishes_reference_dates() -> None:
    """Temporal variants of one question remain distinct candidates."""
    _, contexts_by_text = prepare_ragas_documents([_episode()])

    earlier = candidates_from_testset(_testset(), contexts_by_text, as_of_date=date(2025, 2, 1))[0]
    later = candidates_from_testset(_testset(), contexts_by_text, as_of_date=date(2026, 2, 1))[0]

    assert earlier.candidate_id != later.candidate_id


def test_candidates_from_testset_rejects_unknown_context() -> None:
    """Generated content must map exactly to a source episode."""
    _, contexts_by_text = prepare_ragas_documents([_episode()])

    with pytest.raises(UnknownReferenceContextError):
        candidates_from_testset(_testset("Unknown source text"), contexts_by_text)


def test_synthesize_single_hop_candidates_passes_explicit_policy() -> None:
    """The adapter passes the caller's synthesizer and size to Ragas."""
    expected = _testset()

    class FakeGenerator:
        """Capture Ragas arguments without calling a model."""

        def __init__(self) -> None:
            """Initialize captured arguments."""
            self.documents = []
            self.size = 0
            self.distribution = []

        def generate_with_chunks(self, documents, *, testset_size, transforms, query_distribution):
            """Return a fixed testset and retain explicit inputs for assertions."""
            self.documents = documents
            self.size = testset_size
            self.distribution = query_distribution
            return expected

    fake_generator = FakeGenerator()
    query_synthesizer = cast(SingleHopSpecificQuerySynthesizer, object())

    candidates = synthesize_single_hop_candidates(
        cast(RagasTestsetGenerator, fake_generator),
        query_synthesizer,
        [_episode()],
        testset_size=1,
    )

    assert len(candidates) == 1
    assert fake_generator.documents[0].page_content == "Article 10 source text"
    assert fake_generator.size == 1
    assert fake_generator.distribution == [(query_synthesizer, 1.0)]


def test_write_candidates_jsonl(tmp_path) -> None:
    """Candidate output is valid UTF-8 JSON Lines."""
    _, contexts_by_text = prepare_ragas_documents([_episode()])
    candidate = candidates_from_testset(_testset(), contexts_by_text)[0]
    output_path = tmp_path / "evaluation" / "candidates.jsonl"

    write_candidates_jsonl([candidate], output_path)

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["candidate_id"] == str(candidate.candidate_id)
    assert records[0]["approval_status"] == "pending"
