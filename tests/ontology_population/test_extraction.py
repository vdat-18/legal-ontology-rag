"""Tests for typed LLM ontology extraction and evidence binding."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from legal_rag.baml_client.types import (
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
from legal_rag.ontology_population.extraction import extract_population_from_document

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
