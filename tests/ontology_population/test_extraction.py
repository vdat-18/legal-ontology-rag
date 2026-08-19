"""Tests for typed LLM ontology extraction and evidence binding."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from legal_rag.baml_client.types import (
    ChunkWindowRole,
    FactExtractionBatchItem,
    OntologyConsequenceEntry,
    OntologyConsequenceKind,
    OntologyCurrency,
    OntologyEvidenceEntry,
    OntologyPopulationBatchResult,
    OntologyRuleEntry,
    OntologySubjectKind,
)
from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.benchmark_synthesis.llm_execution import LlmTaskExhaustedError
from legal_rag.chunker.models import Chunk
from legal_rag.ontology_population.config import PopulationSettings
from legal_rag.ontology_population.extraction import (
    build_population_reading_window,
    build_population_record,
    extract_population_from_document,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _PopulationClient:
    """Return one typed rule for each requested target."""

    def __init__(self, quoted_span: str) -> None:
        self._quoted_span = quoted_span

    def extract_batch(self, items: Sequence[FactExtractionBatchItem]) -> Sequence[OntologyPopulationBatchResult]:
        """Build source-grounded typed BAML responses."""
        return tuple(
            OntologyPopulationBatchResult(
                target_chunk_id=item.target_chunk_id,
                rules=[
                    OntologyRuleEntry(
                        conduct_label_vi="Điều khiển xe không gắn biển số",
                        subject_kind=OntologySubjectKind.NaturalPerson,
                        role_labels_vi=["Người điều khiển"],
                        vehicle_category_labels_vi=["Xe mô tô"],
                        condition_root_id=None,
                        conditions=[],
                        consequences=[
                            OntologyConsequenceEntry(
                                consequence_id="money",
                                kind=OntologyConsequenceKind.MonetaryPenalty,
                                minimum_amount="4000000",
                                maximum_amount="6000000",
                                currency=OntologyCurrency.VND,
                                point_value=None,
                                consequence_label_vi=None,
                                minimum_duration_months=None,
                                maximum_duration_months=None,
                                target_label_vi=None,
                                responsible_role_label_vi=None,
                                evidence=[
                                    OntologyEvidenceEntry(chunk_id=item.target_chunk_id, quoted_span=self._quoted_span)
                                ],
                            )
                        ],
                        valid_from=None,
                        valid_to=None,
                        provision_path="Điều 13 khoản 9 điểm b",
                        evidence=[OntologyEvidenceEntry(chunk_id=item.target_chunk_id, quoted_span=self._quoted_span)],
                    )
                ],
            )
            for item in items
        )


def test_extraction_binds_evidence_and_uses_reviewed_document_validity(
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """Typed model output becomes deterministic source-grounded population records."""
    quoted_span = legal_chunk.text.split(". ", maxsplit=1)[0] + "."
    settings = replace(
        PopulationSettings.from_path(PROJECT_ROOT / "ontology" / "population_contract.yaml"),
        max_retries=0,
        retry_delay_seconds=0,
    )

    first = extract_population_from_document(
        (legal_chunk,), (legal_chunk,), _PopulationClient(quoted_span), document_metadata, settings
    )
    second = extract_population_from_document(
        (legal_chunk,), (legal_chunk,), _PopulationClient(quoted_span), document_metadata, settings
    )

    assert first == second
    assert first[0].valid_from == document_metadata.valid_from
    assert first[0].evidence[0].exact_text == quoted_span
    assert str(first[0].consequences[0].minimum_amount) == "4000000"


def test_extraction_rejects_non_exact_model_evidence(
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """A model paraphrase cannot enter the population record."""
    settings = replace(
        PopulationSettings.from_path(PROJECT_ROOT / "ontology" / "population_contract.yaml"),
        max_retries=0,
        retry_delay_seconds=0,
    )

    with pytest.raises(LlmTaskExhaustedError) as error:
        extract_population_from_document(
            (legal_chunk,),
            (legal_chunk,),
            _PopulationClient("Nội dung do mô hình tự diễn giải."),
            document_metadata,
            settings,
        )
    assert "not exact source text" in str(error.value.__cause__)


def test_population_window_includes_bounded_same_article_context(legal_chunk: Chunk) -> None:
    """Distant consequence catalogs in the same article reach the population model."""
    article_six = "[LEGAL_STRUCTURE 1.0.0]\nĐiều 6. Quy tắc A\n[/LEGAL_STRUCTURE]"
    article_seven = "[LEGAL_STRUCTURE 1.0.0]\nĐiều 7. Quy tắc B\n[/LEGAL_STRUCTURE]"
    target = replace(
        legal_chunk,
        chunk_id="target",
        chunk_index=2,
        context_text=article_six,
        previous_chunk_id="near",
    )
    near = replace(legal_chunk, chunk_id="near", chunk_index=1, context_text=article_six)
    distant = replace(legal_chunk, chunk_id="distant", chunk_index=8, context_text=article_six)
    other_article = replace(legal_chunk, chunk_id="other", chunk_index=9, context_text=article_seven)

    window = build_population_reading_window(
        target,
        {chunk.chunk_id: chunk for chunk in (near, target, distant, other_article)},
        max_preceding_chunks=1,
        max_following_chunks=0,
        max_article_context_chunks=3,
    )

    assert [entry.chunk_id for entry in window] == ["near", "target", "distant"]
    assert {entry.chunk_id: entry.role for entry in window} == {
        "near": ChunkWindowRole.Adjacent,
        "target": ChunkWindowRole.Target,
        "distant": ChunkWindowRole.ArticleContext,
    }


def test_population_window_rejects_unbounded_article_context(legal_chunk: Chunk) -> None:
    """A large article fails explicitly instead of silently dropping distant consequences."""
    context = "[LEGAL_STRUCTURE 1.0.0]\nĐiều 6. Quy tắc A\n[/LEGAL_STRUCTURE]"
    target = replace(legal_chunk, chunk_id="target", chunk_index=0, context_text=context)
    distant = replace(legal_chunk, chunk_id="distant", chunk_index=1, context_text=context)

    with pytest.raises(ValueError, match="exceeding configured maximum"):
        build_population_reading_window(
            target,
            {target.chunk_id: target, distant.chunk_id: distant},
            max_preceding_chunks=0,
            max_following_chunks=0,
            max_article_context_chunks=1,
        )


def test_population_window_uses_local_context_for_structural_variants(legal_chunk: Chunk) -> None:
    """A boundary target does not mix same-article context from multiple variants."""
    variant_context = "\n".join(
        (
            "[LEGAL_STRUCTURE_VARIANTS 1.0.0]",
            "Điều A. Quy tắc thứ nhất",
            "Điều B. Quy tắc thứ hai",
            "[/LEGAL_STRUCTURE_VARIANTS]",
        )
    )
    single_context = "[LEGAL_STRUCTURE 1.0.0]\nĐiều A. Quy tắc thứ nhất\n[/LEGAL_STRUCTURE]"
    target = replace(
        legal_chunk,
        chunk_id="target",
        chunk_index=1,
        context_text=variant_context,
        previous_chunk_id="near",
    )
    near = replace(legal_chunk, chunk_id="near", chunk_index=0, context_text=single_context)
    distant = replace(legal_chunk, chunk_id="distant", chunk_index=5, context_text=single_context)

    window = build_population_reading_window(
        target,
        {chunk.chunk_id: chunk for chunk in (near, target, distant)},
        max_preceding_chunks=1,
        max_following_chunks=0,
        max_article_context_chunks=3,
    )

    assert {entry.chunk_id for entry in window} == {"near", "target"}
    assert all(entry.role != ChunkWindowRole.ArticleContext for entry in window)


def test_population_record_requires_target_conduct_evidence(
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """Context-only conduct evidence cannot create a rule for the target chunk."""
    context_chunk_id = "context"
    evidence = OntologyEvidenceEntry(chunk_id=context_chunk_id, quoted_span=legal_chunk.text)
    raw = _rule_with_consequence(
        OntologyConsequenceEntry(
            consequence_id="money",
            kind=OntologyConsequenceKind.MonetaryPenalty,
            minimum_amount="4000000",
            maximum_amount="6000000",
            currency=OntologyCurrency.VND,
            point_value=None,
            consequence_label_vi=None,
            minimum_duration_months=None,
            maximum_duration_months=None,
            target_label_vi=None,
            responsible_role_label_vi=None,
            evidence=[evidence],
        ),
        evidence,
    )

    with pytest.raises(ValueError, match="target chunk"):
        build_population_record(
            raw,
            legal_chunk.chunk_id,
            document_metadata,
            {context_chunk_id: legal_chunk.text, legal_chunk.chunk_id: legal_chunk.text},
        )


def test_population_record_rejects_remedial_measure_without_responsible_role(
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """An incomplete remedial measure fails before persisted RDF validation."""
    evidence = OntologyEvidenceEntry(chunk_id=legal_chunk.chunk_id, quoted_span=legal_chunk.text)
    raw = OntologyRuleEntry(
        conduct_label_vi="Vi phạm hành chính nói chung",
        subject_kind=None,
        role_labels_vi=[],
        vehicle_category_labels_vi=[],
        condition_root_id=None,
        conditions=[],
        consequences=[
            OntologyConsequenceEntry(
                consequence_id="remedy",
                kind=OntologyConsequenceKind.RemedialMeasure,
                minimum_amount=None,
                maximum_amount=None,
                currency=None,
                point_value=None,
                consequence_label_vi="Buộc khôi phục tình trạng ban đầu",
                minimum_duration_months=None,
                maximum_duration_months=None,
                target_label_vi="tình trạng ban đầu",
                responsible_role_label_vi=None,
                evidence=[evidence],
            )
        ],
        valid_from=None,
        valid_to=None,
        provision_path="Điều 3 khoản 3 điểm a",
        evidence=[evidence],
    )

    with pytest.raises(ValueError, match="requires a source-grounded responsible role"):
        build_population_record(
            raw,
            legal_chunk.chunk_id,
            document_metadata,
            {legal_chunk.chunk_id: legal_chunk.text},
        )


@pytest.mark.parametrize("kind", [OntologyConsequenceKind.AdditionalSanction, OntologyConsequenceKind.RemedialMeasure])
def test_population_record_requires_named_non_numeric_consequence(
    kind: OntologyConsequenceKind,
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """Non-numeric consequences retain the source-grounded legal effect name."""
    evidence = OntologyEvidenceEntry(chunk_id=legal_chunk.chunk_id, quoted_span=legal_chunk.text)
    raw = _rule_with_consequence(
        OntologyConsequenceEntry(
            consequence_id="effect",
            kind=kind,
            minimum_amount=None,
            maximum_amount=None,
            currency=None,
            point_value=None,
            consequence_label_vi=None,
            minimum_duration_months=None,
            maximum_duration_months=None,
            target_label_vi="Đối tượng chịu tác động",
            responsible_role_label_vi="Chủ thể thực hiện" if kind == OntologyConsequenceKind.RemedialMeasure else None,
            evidence=[evidence],
        ),
        evidence,
    )

    with pytest.raises(ValueError, match="consequence label"):
        build_population_record(raw, legal_chunk.chunk_id, document_metadata, {legal_chunk.chunk_id: legal_chunk.text})


@pytest.mark.parametrize(
    ("kind", "minimum", "maximum", "message"),
    [
        (OntologyConsequenceKind.AdditionalSanction, 10, None, "both lower and upper"),
        (OntologyConsequenceKind.AdditionalSanction, 0, 12, "positive and ordered"),
        (OntologyConsequenceKind.AdditionalSanction, 12, 10, "positive and ordered"),
        (OntologyConsequenceKind.MonetaryPenalty, 10, 12, "Only AdditionalSanction"),
    ],
)
def test_population_record_rejects_invalid_duration_bounds(
    kind: OntologyConsequenceKind,
    minimum: int | None,
    maximum: int | None,
    message: str,
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """Duration normalization is complete, positive, ordered, and sanction-specific."""
    evidence = OntologyEvidenceEntry(chunk_id=legal_chunk.chunk_id, quoted_span=legal_chunk.text)
    raw = _rule_with_consequence(
        OntologyConsequenceEntry(
            consequence_id="effect",
            kind=kind,
            minimum_amount="1" if kind == OntologyConsequenceKind.MonetaryPenalty else None,
            maximum_amount="2" if kind == OntologyConsequenceKind.MonetaryPenalty else None,
            currency=OntologyCurrency.VND if kind == OntologyConsequenceKind.MonetaryPenalty else None,
            point_value=None,
            consequence_label_vi="Tác động pháp lý" if kind == OntologyConsequenceKind.AdditionalSanction else None,
            minimum_duration_months=minimum,
            maximum_duration_months=maximum,
            target_label_vi="Đối tượng chịu tác động" if kind == OntologyConsequenceKind.AdditionalSanction else None,
            responsible_role_label_vi=None,
            evidence=[evidence],
        ),
        evidence,
    )

    with pytest.raises(ValueError, match=message):
        build_population_record(raw, legal_chunk.chunk_id, document_metadata, {legal_chunk.chunk_id: legal_chunk.text})


def _rule_with_consequence(
    consequence: OntologyConsequenceEntry,
    evidence: OntologyEvidenceEntry,
) -> OntologyRuleEntry:
    """Build a minimal typed rule for consequence-boundary tests."""
    return OntologyRuleEntry(
        conduct_label_vi="Hành vi được kiểm tra",
        subject_kind=None,
        role_labels_vi=[],
        vehicle_category_labels_vi=[],
        condition_root_id=None,
        conditions=[],
        consequences=[consequence],
        valid_from=None,
        valid_to=None,
        provision_path="Điều kiểm tra",
        evidence=[evidence],
    )
