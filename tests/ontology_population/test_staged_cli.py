"""Tests for independently runnable ontology population LLM steps."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import pytest
from rdflib import Graph

from legal_rag import ontology_conduct_extraction_cli, ontology_population_binding_cli
from legal_rag.baml_client.types import (
    OntologyConsequenceEntry,
    OntologyConsequenceKind,
    OntologyEvidenceReference,
    OntologyPopulationBatchResult,
    OntologyPopulationBindingItem,
    OntologyRuleEntry,
    OntologySubjectKind,
    OntologyTargetConductBatchResult,
    OntologyTargetConductEntry,
    OntologyTargetConductInput,
    OntologyValidityMode,
)
from legal_rag.benchmark_synthesis.chunk_screening import ChunkScreeningRecord
from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.benchmark_synthesis.writer import write_chunk_screening, write_document_metadata
from legal_rag.chunker.models import Chunk
from legal_rag.chunker.writer import write_chunks
from legal_rag.io_utils import write_json_atomic
from legal_rag.ontology_population import extraction
from legal_rag.ontology_population.conduct_writer import (
    read_target_conduct_inventories,
    write_target_conduct_inventories,
)
from legal_rag.ontology_population.writer import read_population_progress, read_population_records

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _StagedPopulationClient:
    """Return separate conduct and binding outputs without external calls."""

    conduct_calls = 0
    conduct_target_ids: ClassVar[list[str]] = []
    binding_calls = 0

    def __init__(self, *_: object) -> None:
        """Accept production constructor arguments."""

    def extract_conduct_batch(
        self, items: Sequence[OntologyTargetConductInput]
    ) -> Sequence[OntologyTargetConductBatchResult]:
        """Return one locked target conduct per selected chunk."""
        type(self).conduct_calls += 1
        type(self).conduct_target_ids.extend(item.target_chunk_id for item in items)
        return tuple(
            OntologyTargetConductBatchResult(
                target_chunk_id=item.target_chunk_id,
                conducts=[
                    OntologyTargetConductEntry(
                        conduct_id="conduct_1",
                        conduct_label_vi="Điều khiển xe không gắn biển số",
                        provision_path="Điều 13 khoản 9 điểm b",
                        evidence=[OntologyEvidenceReference(evidence_id=item.evidence_sources[0].evidence_id)],
                    )
                ],
            )
            for item in items
        )

    def bind_batch(self, items: Sequence[OntologyPopulationBindingItem]) -> Sequence[OntologyPopulationBatchResult]:
        """Bind one monetary consequence to each locked conduct."""
        type(self).binding_calls += 1
        results = []
        for item in items:
            target_source = next(source for entry in item.window for source in entry.evidence_sources)
            evidence = OntologyEvidenceReference(evidence_id=target_source.evidence_id)
            results.append(
                OntologyPopulationBatchResult(
                    target_chunk_id=item.target_chunk_id,
                    rules=[
                        OntologyRuleEntry(
                            conduct_label_vi=item.target_conducts[0].conduct_label_vi,
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
                                    currency=None,
                                    point_value=None,
                                    consequence_label_vi=None,
                                    minimum_duration_months=None,
                                    maximum_duration_months=None,
                                    target_label_vi=None,
                                    responsible_role_label_vi=None,
                                    evidence=[evidence],
                                )
                            ],
                            validity_mode=OntologyValidityMode.DocumentDefault,
                            valid_from=None,
                            valid_to=None,
                            provision_path=item.target_conducts[0].provision_path,
                            evidence=[evidence],
                        )
                    ],
                )
            )
        return tuple(results)


def test_population_llm_tasks_run_as_two_resumable_cli_steps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """Conduct inventory and rule binding each call only their own LLM task."""
    chunks_path = tmp_path / "chunks.jsonl"
    screening_path = tmp_path / "screening.jsonl"
    metadata_path = tmp_path / "metadata.jsonl"
    inventories_path = tmp_path / "conducts.jsonl"
    records_path = tmp_path / "population.jsonl"
    progress_path = tmp_path / "population-progress.jsonl"
    graph_path = tmp_path / "population.ttl"
    write_chunks(chunks_path, (legal_chunk,))
    write_chunk_screening(
        screening_path,
        (ChunkScreeningRecord(chunk_id=legal_chunk.chunk_id, has_penalty_content=True, reason="Penalty rule."),),
    )
    write_document_metadata(metadata_path, document_metadata)
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setattr(ontology_conduct_extraction_cli, "find_dotenv", lambda **_: "")
    monkeypatch.setattr(ontology_population_binding_cli, "find_dotenv", lambda **_: "")
    monkeypatch.setattr(extraction, "BamlOntologyPopulationClient", _StagedPopulationClient)
    _StagedPopulationClient.conduct_calls = 0
    _StagedPopulationClient.binding_calls = 0

    conduct_argv = [
        "legal-extract-ontology-conducts",
        str(chunks_path),
        str(screening_path),
        str(metadata_path),
        str(PROJECT_ROOT / "ontology" / "population_contract.yaml"),
        str(inventories_path),
        "--max-concurrency",
        "2",
    ]
    monkeypatch.setattr(sys, "argv", conduct_argv)
    ontology_conduct_extraction_cli.main()
    ontology_conduct_extraction_cli.main()

    assert len(read_target_conduct_inventories(inventories_path)) == 1
    assert _StagedPopulationClient.conduct_calls == 1
    assert _StagedPopulationClient.binding_calls == 0

    binding_argv = [
        "legal-bind-ontology-population",
        str(chunks_path),
        str(screening_path),
        str(metadata_path),
        str(PROJECT_ROOT / "ontology" / "population_contract.yaml"),
        str(inventories_path),
        str(records_path),
        str(progress_path),
        str(graph_path),
        "--max-concurrency",
        "2",
    ]
    monkeypatch.setattr(sys, "argv", binding_argv)
    ontology_population_binding_cli.main()
    ontology_population_binding_cli.main()

    assert len(read_population_records(records_path)) == 1
    assert len(read_population_progress(progress_path)) == 1
    assert len(Graph().parse(graph_path, format="turtle")) > 0
    assert _StagedPopulationClient.conduct_calls == 1
    assert _StagedPopulationClient.binding_calls == 1


def test_conduct_step_selectively_reruns_only_reviewed_incorrect_targets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """Reviewed correct inventories retain provenance while selected targets are replaced and resumable."""
    first = replace(legal_chunk, chunk_id="chunk-1", chunk_index=0, next_chunk_id="chunk-2")
    second = replace(legal_chunk, chunk_id="chunk-2", chunk_index=1, previous_chunk_id="chunk-1")
    chunks_path = tmp_path / "chunks.jsonl"
    screening_path = tmp_path / "screening.jsonl"
    metadata_path = tmp_path / "metadata.jsonl"
    inventories_path = tmp_path / "conducts.jsonl"
    rerun_path = tmp_path / "rerun.json"
    write_chunks(chunks_path, (first, second))
    write_chunk_screening(
        screening_path,
        tuple(
            ChunkScreeningRecord(chunk_id=chunk.chunk_id, has_penalty_content=True, reason="Penalty rule.")
            for chunk in (first, second)
        ),
    )
    write_document_metadata(metadata_path, document_metadata)
    write_json_atomic(
        rerun_path,
        {
            "schema_version": "0.1.0",
            "document_number": document_metadata.document_number,
            "chunk_ids": [second.chunk_id],
        },
    )
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setattr(ontology_conduct_extraction_cli, "find_dotenv", lambda **_: "")
    monkeypatch.setattr(ontology_population_binding_cli, "find_dotenv", lambda **_: "")
    monkeypatch.setattr(extraction, "BamlOntologyPopulationClient", _StagedPopulationClient)
    base_argv = [
        "legal-extract-ontology-conducts",
        str(chunks_path),
        str(screening_path),
        str(metadata_path),
        str(PROJECT_ROOT / "ontology" / "population_contract.yaml"),
        str(inventories_path),
    ]
    monkeypatch.setattr(sys, "argv", base_argv)
    ontology_conduct_extraction_cli.main()
    initial = read_target_conduct_inventories(inventories_path)
    legacy = tuple(replace(record, run_id="reviewed-legacy-run") for record in initial)
    write_target_conduct_inventories(inventories_path, legacy)
    _StagedPopulationClient.conduct_calls = 0
    _StagedPopulationClient.conduct_target_ids = []

    monkeypatch.setattr(sys, "argv", [*base_argv, "--rerun-target-ids-path", str(rerun_path)])
    ontology_conduct_extraction_cli.main()
    rerun = read_target_conduct_inventories(inventories_path)

    assert rerun[0] == legacy[0]
    assert rerun[1].target_chunk_id == second.chunk_id
    assert rerun[1].run_id != legacy[1].run_id
    assert _StagedPopulationClient.conduct_target_ids == [second.chunk_id]

    ontology_conduct_extraction_cli.main()

    assert _StagedPopulationClient.conduct_calls == 1

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "legal-bind-ontology-population",
            str(chunks_path),
            str(screening_path),
            str(metadata_path),
            str(PROJECT_ROOT / "ontology" / "population_contract.yaml"),
            str(inventories_path),
            str(tmp_path / "population.jsonl"),
            str(tmp_path / "progress.jsonl"),
            str(tmp_path / "population.ttl"),
        ],
    )
    ontology_population_binding_cli.main()

    assert len(read_population_progress(tmp_path / "progress.jsonl")) == 2


def test_conduct_selective_rerun_rejects_unknown_targets_before_llm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """A rerun list cannot expand beyond the original population target selection."""
    chunks_path = tmp_path / "chunks.jsonl"
    screening_path = tmp_path / "screening.jsonl"
    metadata_path = tmp_path / "metadata.jsonl"
    inventories_path = tmp_path / "conducts.jsonl"
    rerun_path = tmp_path / "rerun.json"
    write_chunks(chunks_path, (legal_chunk,))
    write_chunk_screening(
        screening_path,
        (ChunkScreeningRecord(chunk_id=legal_chunk.chunk_id, has_penalty_content=True, reason="Penalty rule."),),
    )
    write_document_metadata(metadata_path, document_metadata)
    write_json_atomic(
        rerun_path,
        {
            "schema_version": "0.1.0",
            "document_number": document_metadata.document_number,
            "chunk_ids": ["unknown-chunk"],
        },
    )
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setattr(ontology_conduct_extraction_cli, "find_dotenv", lambda **_: "")
    monkeypatch.setattr(extraction, "BamlOntologyPopulationClient", _StagedPopulationClient)
    _StagedPopulationClient.conduct_calls = 0
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "legal-extract-ontology-conducts",
            str(chunks_path),
            str(screening_path),
            str(metadata_path),
            str(PROJECT_ROOT / "ontology" / "population_contract.yaml"),
            str(inventories_path),
            "--rerun-target-ids-path",
            str(rerun_path),
        ],
    )

    with pytest.raises(ValueError, match="unselected chunk"):
        ontology_conduct_extraction_cli.main()

    assert _StagedPopulationClient.conduct_calls == 0


def test_staged_binding_requires_a_complete_conduct_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """The binding step stops before LLM work when its prior checkpoint is missing."""
    chunks_path = tmp_path / "chunks.jsonl"
    screening_path = tmp_path / "screening.jsonl"
    metadata_path = tmp_path / "metadata.jsonl"
    inventories_path = tmp_path / "conducts.jsonl"
    inventories_path.write_text("", encoding="utf-8")
    write_chunks(chunks_path, (legal_chunk,))
    write_chunk_screening(
        screening_path,
        (ChunkScreeningRecord(chunk_id=legal_chunk.chunk_id, has_penalty_content=True, reason="Penalty rule."),),
    )
    write_document_metadata(metadata_path, document_metadata)
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setattr(ontology_population_binding_cli, "find_dotenv", lambda **_: "")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "legal-bind-ontology-population",
            str(chunks_path),
            str(screening_path),
            str(metadata_path),
            str(PROJECT_ROOT / "ontology" / "population_contract.yaml"),
            str(inventories_path),
            str(tmp_path / "population.jsonl"),
            str(tmp_path / "progress.jsonl"),
            str(tmp_path / "population.ttl"),
        ],
    )

    with pytest.raises(ValueError, match="inventory is incomplete"):
        ontology_population_binding_cli.main()
