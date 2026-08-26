"""Tests for typed exact-evidenced concept proposals."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from legal_rag.baml_client.types import (
    ConceptRelationProposalEntry,
    ConceptResolutionPairInput,
    ConceptResolutionPairResult,
    OntologyConceptRelation,
    OntologyEvidenceReference,
)
from legal_rag.benchmark_synthesis.llm_execution import LlmTaskExhaustedError
from legal_rag.ontology_change.corpus import ChangeCorpus
from legal_rag.ontology_concept.candidates import build_concept_candidates
from legal_rag.ontology_concept.config import ConceptResolutionSettings
from legal_rag.ontology_concept.proposals import propose_concept_relations

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _ConceptClient:
    """Return one configurable typed relation without an external model."""

    model = "test-model"

    def __init__(self, *, paraphrase: bool = False, invalid_label: bool = False, empty: bool = False) -> None:
        self._paraphrase = paraphrase
        self._invalid_label = invalid_label
        self._empty = empty
        self.inputs: tuple[ConceptResolutionPairInput, ...] = ()

    def propose_batch(self, pairs: Sequence[ConceptResolutionPairInput]) -> Sequence[ConceptResolutionPairResult]:
        """Capture complete typed pairs and propose equivalence only for roles."""
        self.inputs = tuple(pairs)
        results = []
        for pair in pairs:
            proposal = None
            if not self._empty and pair.left.kind.value == "LegalSubjectRole":
                left_evidence = [
                    OntologyEvidenceReference(evidence_id=source.evidence_id) for source in pair.left.evidence
                ]
                if self._paraphrase:
                    left_evidence = [OntologyEvidenceReference(evidence_id="ev_unknown")]
                proposal = ConceptRelationProposalEntry(
                    relation=OntologyConceptRelation.SameConcept,
                    preferred_label_vi="Nhãn không thuộc pair" if self._invalid_label else pair.left.label_vi,
                    left_evidence=left_evidence,
                    right_evidence=[
                        OntologyEvidenceReference(evidence_id=source.evidence_id) for source in pair.right.evidence
                    ],
                    rationale_vi="Hai occurrence biểu đạt cùng vai trò pháp lý trong source.",
                )
            results.append(ConceptResolutionPairResult(candidate_id=pair.candidate_id, proposal=proposal))
        return tuple(results)


def _propose(corpus: ChangeCorpus, client: _ConceptClient):
    """Run one proposal pass with retries disabled."""
    settings = replace(
        ConceptResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "concept_resolution_contract.yaml"),
        max_retries=0,
        retry_delay_seconds=0,
    )
    candidates = build_concept_candidates(corpus.documents[0].records, settings)
    return propose_concept_relations(
        candidates.pairs,
        candidates.occurrences,
        client,
        settings,
        run_id="concept-run",
    )


def test_proposal_preserves_context_and_binds_both_sides_exactly(exception_corpus: ChangeCorpus) -> None:
    """Typed inputs retain provenance and accepted evidence belongs to both source occurrences."""
    client = _ConceptClient()
    proposals = _propose(exception_corpus, client)

    assert len(proposals) == 1
    assert proposals[0].relation == "SameConcept"
    assert proposals[0].preferred_label_vi == "Người điều khiển"
    assert proposals[0].left_evidence
    assert proposals[0].right_evidence
    assert all(pair.left.document_number == "168/2024/NĐ-CP" for pair in client.inputs)
    assert {pair.left.kind.value for pair in client.inputs} == {
        "RegulatedConduct",
        "LegalSubjectRole",
        "VehicleCategory",
    }


@pytest.mark.parametrize("client", [_ConceptClient(paraphrase=True), _ConceptClient(invalid_label=True)])
def test_proposal_rejects_paraphrase_or_unowned_preferred_label(
    client: _ConceptClient,
    exception_corpus: ChangeCorpus,
) -> None:
    """The model cannot invent source evidence or a canonical label."""
    with pytest.raises(LlmTaskExhaustedError):
        _propose(exception_corpus, client)


def test_zero_proposal_results_are_valid_and_checkpointable(exception_corpus: ChangeCorpus) -> None:
    """A pair without a supported relation completes without fabricating a decision."""
    processed = []
    settings = replace(
        ConceptResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "concept_resolution_contract.yaml"),
        max_retries=0,
        retry_delay_seconds=0,
    )
    candidates = build_concept_candidates(exception_corpus.documents[0].records, settings)
    proposals = propose_concept_relations(
        candidates.pairs,
        candidates.occurrences,
        _ConceptClient(empty=True),
        settings,
        run_id="concept-run",
        on_batch_processed=processed.extend,
    )

    assert proposals == ()
    assert {candidate_id for candidate_id, _ in processed} == {item.candidate_id for item in candidates.pairs}
