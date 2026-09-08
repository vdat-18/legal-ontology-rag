"""Tests for typed exact-evidenced concept proposals."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from legal_rag.baml_client.types import (
    ConceptRelationProposalEntry,
    ConceptResolutionInventoryInput,
    ConceptResolutionInventoryResult,
    OntologyConceptRelation,
    OntologyEvidenceReference,
)
from legal_rag.benchmark_synthesis.llm_execution import LlmTaskExhaustedError
from legal_rag.ontology_change.corpus import ChangeCorpus
from legal_rag.ontology_concept.candidates import build_concept_candidates
from legal_rag.ontology_concept.config import ConceptResolutionSettings
from legal_rag.ontology_concept.finalization import finalize_concept_resolution
from legal_rag.ontology_concept.proposals import propose_concept_relations

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _ConceptClient:
    """Return one configurable typed relation without an external model."""

    model = "test-model"

    def __init__(self, *, paraphrase: bool = False, invalid_label: bool = False, empty: bool = False) -> None:
        self._paraphrase = paraphrase
        self._invalid_label = invalid_label
        self._empty = empty
        self.inputs: tuple[ConceptResolutionInventoryInput, ...] = ()

    def propose_batch(
        self,
        inventories: Sequence[ConceptResolutionInventoryInput],
    ) -> Sequence[ConceptResolutionInventoryResult]:
        """Capture complete inventories and propose one role equivalence."""
        self.inputs += tuple(inventories)
        results = []
        for inventory in inventories:
            proposals = []
            if not self._empty and inventory.kind.value == "LegalSubjectRole":
                left, right = inventory.occurrences
                left_evidence = [OntologyEvidenceReference(evidence_id=source.evidence_id) for source in left.evidence]
                if self._paraphrase:
                    left_evidence = [OntologyEvidenceReference(evidence_id="ev_unknown")]
                proposals.append(
                    ConceptRelationProposalEntry(
                        left_occurrence_id=left.occurrence_id,
                        right_occurrence_id=right.occurrence_id,
                        relation=OntologyConceptRelation.SameConcept,
                        preferred_occurrence_id="unknown" if self._invalid_label else left.occurrence_id,
                        left_evidence=left_evidence,
                        right_evidence=[
                            OntologyEvidenceReference(evidence_id=source.evidence_id) for source in right.evidence
                        ],
                        rationale_vi="Hai occurrence biểu đạt cùng vai trò pháp lý trong source.",
                    )
                )
            results.append(ConceptResolutionInventoryResult(candidate_id=inventory.candidate_id, proposals=proposals))
        return tuple(results)


class _TransitivePreferredClient:
    """Return a connected identity group whose preferred member is outside one edge."""

    def __init__(self, *, outside_component: bool = False) -> None:
        self._outside_component = outside_component

    def propose_batch(
        self,
        inventories: Sequence[ConceptResolutionInventoryInput],
    ) -> Sequence[ConceptResolutionInventoryResult]:
        """Use one component-wide preferred occurrence across a three-node chain."""
        results = []
        for inventory in inventories:
            proposals = []
            if inventory.kind.value == "LegalSubjectRole":
                first, second, third = inventory.occurrences
                pairs = ((first, second),) if self._outside_component else ((first, second), (second, third))
                preferred = third if self._outside_component else first
                for left, right in pairs:
                    proposals.append(
                        ConceptRelationProposalEntry(
                            left_occurrence_id=left.occurrence_id,
                            right_occurrence_id=right.occurrence_id,
                            relation=OntologyConceptRelation.SameConcept,
                            preferred_occurrence_id=preferred.occurrence_id,
                            left_evidence=[
                                OntologyEvidenceReference(evidence_id=source.evidence_id) for source in left.evidence
                            ],
                            right_evidence=[
                                OntologyEvidenceReference(evidence_id=source.evidence_id) for source in right.evidence
                            ],
                            rationale_vi="Ba occurrence cùng chỉ một vai trò pháp lý.",
                        )
                    )
            results.append(ConceptResolutionInventoryResult(candidate_id=inventory.candidate_id, proposals=proposals))
        return tuple(results)


def test_concept_prompt_requires_complete_inventory_audit() -> None:
    """Step 6 must audit every supplied label before returning sparse relations."""
    prompt = (PROJECT_ROOT / "baml_src/ontology_concept_resolution.baml").read_text(encoding="utf-8")
    compact_prompt = " ".join(prompt.split())

    for phrase in (
        "tập kiểm kê đóng",
        "rà toàn bộ occurrence",
        "không chỉ trả một số ví dụ nổi bật",
        "Làm theo hai pha",
        "phân hoạch toàn bộ bảng nhãn",
        "Pha taxonomy",
        "cùng chỉ một loại chủ thể hoặc phương tiện",
        "Không yêu cầu chúng xuất hiện trong cùng hành vi",
        "khác biệt giữa các quy định chứa occurrence không tự làm thay đổi identity",
        "chỉ diễn đạt lại chức năng vốn có",
        "không tạo thêm điều kiện phân loại thực sự",
        "một tập con pháp lý có thể phân biệt",
        "không SameConcept với khái niệm bao quát",
        "không phải Related",
        "Related chỉ dùng cho liên hệ có giá trị nhưng không có identity hoặc bao hàm",
        "bỏ qua khác biệt chữ hoa/thường và cách viết",
        "phải nằm cùng nhóm SameConcept",
        "Mỗi nhóm SameConcept có N thành viên phải xuất đúng N-1 proposal",
        "một graph liên thông",
        "không bắt buộc là endpoint của từng proposal",
        "cùng preferred occurrence",
        "Không xuất thêm các cặp bắc cầu",
        "xét từng nhóm identity như một concept",
        "Ưu tiên cha trực tiếp bao quát gần nhất",
        "Mỗi concept chuyên biệt phải có quan hệ với cha trực tiếp",
        "concept phương tiện rất rộng vẫn được dùng làm gốc taxonomy",
        "dùng bảng nhãn làm checklist",
        "đủ N-1 proposal",
        "quan hệ bao hàm bị ghi nhầm thành Related",
        "Bảng nhãn bắt buộc kiểm kê",
        "Occurrences và evidence chi tiết",
        "Existing proposals là baseline từ lượt trước",
        "giữ lại mọi proposal vẫn đúng",
        "tập thay thế hoàn chỉnh",
        "Existing proposals cần audit và hợp nhất",
    ):
        assert phrase in compact_prompt


def _propose(corpus: ChangeCorpus, client: _ConceptClient):
    """Run one proposal pass with retries disabled."""
    settings = replace(
        ConceptResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "concept_resolution_contract.yaml"),
        max_retries=0,
        retry_delay_seconds=0,
    )
    target = replace(corpus.documents[0].records[1], role_labels_vi=("Người lái xe",))
    records = (corpus.documents[0].records[0], target)
    candidates = build_concept_candidates(records, settings)
    return propose_concept_relations(
        candidates.inventories,
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
    assert proposals[0].preferred_label_vi in {"Người điều khiển", "Người lái xe"}
    assert proposals[0].left_evidence
    assert proposals[0].right_evidence
    assert all(
        occurrence.document_number == "168/2024/NĐ-CP"
        for inventory in client.inputs
        for occurrence in inventory.occurrences
    )
    assert {inventory.kind.value for inventory in client.inputs} == {"LegalSubjectRole", "VehicleCategory"}
    assert all(inventory.existing_proposals == [] for inventory in client.inputs)


def test_proposal_passes_existing_relations_as_revision_context(exception_corpus: ChangeCorpus) -> None:
    """Selective reruns receive prior proposals so valid decisions can be retained."""
    baseline = _propose(exception_corpus, _ConceptClient())
    client = _ConceptClient()
    settings = replace(
        ConceptResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "concept_resolution_contract.yaml"),
        max_retries=0,
        retry_delay_seconds=0,
    )
    target = replace(exception_corpus.documents[0].records[1], role_labels_vi=("Người lái xe",))
    records = (exception_corpus.documents[0].records[0], target)
    candidates = build_concept_candidates(records, settings)

    propose_concept_relations(
        candidates.inventories,
        candidates.occurrences,
        client,
        settings,
        run_id="concept-rerun",
        existing_proposals=baseline,
    )
    role_input = next(inventory for inventory in client.inputs if inventory.kind.value == "LegalSubjectRole")

    assert len(role_input.existing_proposals) == 1
    assert role_input.existing_proposals[0].relation == OntologyConceptRelation.SameConcept


def test_proposal_rejects_unknown_evidence_reference(exception_corpus: ChangeCorpus) -> None:
    """Typed evidence IDs must still resolve to the supplied input boundary."""
    with pytest.raises(LlmTaskExhaustedError):
        _propose(exception_corpus, _ConceptClient(paraphrase=True))


def test_proposal_rejects_preferred_occurrence_outside_inventory(exception_corpus: ChangeCorpus) -> None:
    """Canonical wording must be selected through one supplied inventory occurrence."""
    with pytest.raises(LlmTaskExhaustedError):
        _propose(exception_corpus, _ConceptClient(invalid_label=True))


def test_proposal_accepts_component_preferred_occurrence_outside_one_pair(
    exception_corpus: ChangeCorpus,
) -> None:
    """A transitive group may reuse its canonical member on an edge that omits it."""
    settings = replace(
        ConceptResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "concept_resolution_contract.yaml"),
        max_retries=0,
        retry_delay_seconds=0,
    )
    first = replace(exception_corpus.documents[0].records[0], role_labels_vi=("Chủ thể A",))
    second = replace(exception_corpus.documents[0].records[1], role_labels_vi=("Chủ thể B",))
    third = replace(second, population_id="population-third", role_labels_vi=("Chủ thể C",))
    candidates = build_concept_candidates((first, second, third), settings)

    proposals = propose_concept_relations(
        candidates.inventories,
        candidates.occurrences,
        _TransitivePreferredClient(),
        settings,
        run_id="concept-run",
    )
    concepts, _ = finalize_concept_resolution(candidates.occurrences, proposals)
    role = next(concept for concept in concepts if len(concept.occurrence_ids) == 3)

    assert len(proposals) == 2
    assert role.preferred_label_vi == "Chủ thể A"


def test_proposal_rejects_preferred_occurrence_outside_component(exception_corpus: ChangeCorpus) -> None:
    """An inventory occurrence outside the proposed identity component cannot be canonical."""
    settings = replace(
        ConceptResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "concept_resolution_contract.yaml"),
        max_retries=0,
        retry_delay_seconds=0,
    )
    first = replace(exception_corpus.documents[0].records[0], role_labels_vi=("Chủ thể A",))
    second = replace(exception_corpus.documents[0].records[1], role_labels_vi=("Chủ thể B",))
    third = replace(second, population_id="population-third", role_labels_vi=("Chủ thể C",))
    candidates = build_concept_candidates((first, second, third), settings)

    with pytest.raises(LlmTaskExhaustedError):
        propose_concept_relations(
            candidates.inventories,
            candidates.occurrences,
            _TransitivePreferredClient(outside_component=True),
            settings,
            run_id="concept-run",
        )


def test_zero_proposal_results_are_valid_and_checkpointable(exception_corpus: ChangeCorpus) -> None:
    """A pair without a supported relation completes without fabricating a decision."""
    processed = []
    settings = replace(
        ConceptResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "concept_resolution_contract.yaml"),
        max_retries=0,
        retry_delay_seconds=0,
    )
    target = replace(exception_corpus.documents[0].records[1], role_labels_vi=("Người lái xe",))
    records = (exception_corpus.documents[0].records[0], target)
    candidates = build_concept_candidates(records, settings)
    proposals = propose_concept_relations(
        candidates.inventories,
        candidates.occurrences,
        _ConceptClient(empty=True),
        settings,
        run_id="concept-run",
        on_batch_processed=processed.extend,
    )

    assert proposals == ()
    assert {candidate_id for candidate_id, _ in processed} == {item.candidate_id for item in candidates.inventories}
