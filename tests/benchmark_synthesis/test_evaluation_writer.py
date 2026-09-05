"""Tests for atomic ground-truth synthesis persistence."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from legal_rag.benchmark_synthesis.candidate_generation import (
    CandidateDraft,
    ClaimEvidence,
    RequiredClaim,
)
from legal_rag.benchmark_synthesis.chunk_screening import ChunkScreeningRecord
from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord, TemporalValidityOverride
from legal_rag.benchmark_synthesis.evidence_graph import (
    EvidenceCandidateSet,
    EvidenceRelation,
    EvidenceRelationProgressRecord,
)
from legal_rag.benchmark_synthesis.fact_extraction import ExtractionProgressRecord
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.scenario_sampling import Scenario
from legal_rag.benchmark_synthesis.writer import (
    read_candidates,
    read_chunk_screening,
    read_document_metadata,
    read_evidence_candidate_set,
    read_evidence_relation_progress,
    read_evidence_relations,
    read_extraction_progress,
    read_legal_facts,
    read_scenarios,
    write_candidates,
    write_chunk_screening,
    write_document_metadata,
    write_evidence_candidate_set,
    write_evidence_relation_progress,
    write_evidence_relations,
    write_extraction_progress,
    write_legal_facts,
    write_scenarios,
)


def _fact() -> LegalFact:
    """Build one legal fact covering every nullable and tuple field."""
    return LegalFact(
        fact_id="fact-1",
        statement="Người điều khiển ô tô vượt đèn đỏ bị phạt tiền.",
        valid_from=None,
        valid_to=None,
        document_number="168/2024/NĐ-CP",
        evidence=(EvidenceReference(chunk_id="chunk-1"),),
    )


def test_legal_facts_round_trip_through_jsonl(tmp_path: Path) -> None:
    """A written legal fact reads back with identical field values."""
    output_path = tmp_path / "legal_facts.jsonl"
    fact = _fact()

    write_legal_facts(output_path, [fact])
    facts = read_legal_facts(output_path)

    assert facts == (fact,)


def test_chunk_screening_round_trips_through_jsonl(tmp_path: Path) -> None:
    """A written chunk-screening record reads back with identical field values."""
    output_path = tmp_path / "chunk_screening.jsonl"
    record = ChunkScreeningRecord(
        chunk_id="chunk-1",
        has_penalty_content=True,
        has_document_metadata=True,
        reason="Nêu rõ mức phạt tiền.",
    )

    write_chunk_screening(output_path, [record])
    records = read_chunk_screening(output_path)

    assert records == (record,)


def test_chunk_screening_reads_legacy_records_without_provenance(tmp_path: Path) -> None:
    """Version 2 screening outputs remain readable before selective migration."""
    output_path = tmp_path / "legacy_chunk_screening.jsonl"
    output_path.write_text(
        json.dumps(
            {
                "schema_version": "2.0.0",
                "chunk_id": "chunk-1",
                "has_penalty_content": True,
                "has_document_metadata": False,
                "reason": "Nêu mức phạt.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = read_chunk_screening(output_path)

    assert records[0].prompt_version is None
    assert records[0].screening_run_id is None


def test_document_metadata_round_trips_through_jsonl(tmp_path: Path) -> None:
    """A written document-metadata record reads back with identical field values."""
    output_path = tmp_path / "document_metadata.jsonl"
    record = DocumentMetadataRecord(
        document_number="168/2024/NĐ-CP",
        valid_from=date(2025, 1, 1),
        valid_to=None,
        temporal_overrides=(
            TemporalValidityOverride(
                affected_provision_paths=("Điểm m khoản 3 Điều 6",),
                valid_from=date(2026, 1, 1),
                valid_to=None,
                unresolved_basis_vi=None,
            ),
        ),
    )

    write_document_metadata(output_path, record)

    assert read_document_metadata(output_path) == record


def test_document_metadata_reads_legacy_record_without_temporal_overrides(tmp_path: Path) -> None:
    """Schema 1 metadata remains readable with an empty override collection."""
    output_path = tmp_path / "document_metadata.jsonl"
    output_path.write_text(
        '{"schema_version":"1.0.0","document_number":"168/2024/NĐ-CP","valid_from":"2025-01-01","valid_to":null}\n',
        encoding="utf-8",
    )

    record = read_document_metadata(output_path)

    assert record.temporal_overrides == ()


def test_extraction_progress_round_trips_through_jsonl(tmp_path: Path) -> None:
    """A written extraction-progress record reads back with identical field values."""
    output_path = tmp_path / "extraction_progress.jsonl"
    records = (
        ExtractionProgressRecord("chunk-1", "5.5.0", "provider/model-a", "run-a"),
        ExtractionProgressRecord("chunk-2", "5.5.1", "provider/model-b", "run-b"),
    )

    write_extraction_progress(output_path, records)

    assert read_extraction_progress(output_path) == records


def test_extraction_progress_reads_legacy_records_without_provenance(tmp_path: Path) -> None:
    """Version 1 progress remains readable before selective provenance migration."""
    output_path = tmp_path / "extraction_progress.jsonl"
    output_path.write_text('{"schema_version":"1.0.0","chunk_id":"chunk-1"}\n', encoding="utf-8")

    assert read_extraction_progress(output_path) == (ExtractionProgressRecord(chunk_id="chunk-1"),)


def test_evidence_relations_round_trip_through_jsonl(tmp_path: Path) -> None:
    """A written evidence relation reads back with identical field values."""
    output_path = tmp_path / "evidence_relations.jsonl"
    relation = EvidenceRelation(
        relation_id="relation-1",
        fact_ids=("fact-1", "fact-2"),
        bundle_kind="temporal",
        bridge="Quy định mới thay đổi quy định cũ",
        confirmed_by_model="provider/test-model",
        run_id="run-1",
    )

    write_evidence_relations(output_path, [relation])
    relations = read_evidence_relations(output_path)

    assert relations == (relation,)


def test_evidence_relations_read_legacy_prompt_provenance(tmp_path: Path) -> None:
    """Version 3 relation outputs are assigned their known prompt version during migration."""
    output_path = tmp_path / "evidence_relations.jsonl"
    output_path.write_text(
        '{"schema_version":"3.0.0","relation_id":"relation-1",'
        '"fact_ids":["fact-1","fact-2"],"bundle_kind":"multi_hop",'
        '"bridge":"Cùng vấn đề","confirmed_by_model":"provider/model","run_id":"run-1"}\n',
        encoding="utf-8",
    )

    assert read_evidence_relations(output_path)[0].prompt_version == "5.0.0"


def test_evidence_candidate_set_round_trips_through_json(tmp_path: Path) -> None:
    """A fingerprinted candidate set reads back with deterministic pair ordering."""
    output_path = tmp_path / "evidence_relation_candidates.json"
    candidate_set = EvidenceCandidateSet(
        run_id="candidate-run-1",
        pairs=(("fact-1", "fact-2"), ("fact-1", "fact-3")),
    )

    write_evidence_candidate_set(output_path, candidate_set)

    assert read_evidence_candidate_set(output_path) == candidate_set


def test_evidence_relation_progress_round_trips_through_jsonl(tmp_path: Path) -> None:
    """A written evidence-relation-progress record reads back with identical field values."""
    output_path = tmp_path / "evidence_relation_progress.jsonl"
    records = (
        EvidenceRelationProgressRecord(fact_id_a="fact-1", fact_id_b="fact-2", run_id="run-1"),
        EvidenceRelationProgressRecord(fact_id_a="fact-1", fact_id_b="fact-3", run_id="run-1"),
    )

    write_evidence_relation_progress(output_path, records)

    assert read_evidence_relation_progress(output_path) == records


def test_evidence_relation_progress_reads_legacy_prompt_provenance(tmp_path: Path) -> None:
    """Version 1.1 progress remains usable for a selective prompt migration."""
    output_path = tmp_path / "evidence_relation_progress.jsonl"
    output_path.write_text(
        '{"schema_version":"1.1.0","fact_id_a":"fact-1","fact_id_b":"fact-2","run_id":"run-1"}\n',
        encoding="utf-8",
    )

    record = read_evidence_relation_progress(output_path)[0]
    assert record.prompt_version == "5.0.0"
    assert record.confirmed_by_model is None


def test_scenarios_round_trip_through_jsonl(tmp_path: Path) -> None:
    """A written scenario reads back with identical field values."""
    output_path = tmp_path / "scenarios.jsonl"
    scenario = Scenario(
        scenario_id="scenario-1",
        query_form="specific",
        answerability="answerable",
        persona="tai_xe_ca_nhan",
        query_style="no_diacritics",
        query_length="short",
        reasoning_hops=2,
        fact_ids=("fact-1", "fact-2"),
        bridge="Cùng tình huống xử lý vi phạm",
        gold_chunk_ids=("chunk-1", "chunk-2"),
    )

    write_scenarios(output_path, [scenario])
    scenarios = read_scenarios(output_path)

    assert scenarios == (scenario,)


def _candidate_draft() -> CandidateDraft:
    """Build one candidate draft covering every nullable and tuple field."""
    return CandidateDraft(
        sample_id="sample-1",
        scenario_id="scenario-1",
        query="oto vuot den do bi phat bao nhieu",
        reference_answer="Phạt tiền 4.000.000 đồng.",
        reasoning="Theo Điều 6.",
        required_claims=(
            RequiredClaim(
                claim="Mức phạt tiền áp dụng",
                fact_id="fact-1",
                evidence=(ClaimEvidence(chunk_id="chunk-1"),),
            ),
        ),
        gold_chunk_ids=("chunk-1",),
        query_form="specific",
        persona="tai_xe_ca_nhan",
        query_style="standard",
        query_length="short",
        answerability="answerable",
        reasoning_hops=1,
        source_fact_ids=("fact-1",),
        generator_model="provider/generator-model",
        bridge="Cùng tình huống xử lý vi phạm",
    )


def test_candidates_round_trip_through_jsonl(tmp_path: Path) -> None:
    """A written candidate draft reads back with identical field values, with no gating attached."""
    output_path = tmp_path / "single_turn.jsonl"
    draft = _candidate_draft()

    write_candidates(output_path, [draft])
    drafts = read_candidates(output_path)

    assert drafts == (draft,)
