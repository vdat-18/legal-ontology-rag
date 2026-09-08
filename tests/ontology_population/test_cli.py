"""Tests for the end-to-end ontology population command."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from rdflib import Graph

from legal_rag import ontology_population_cli
from legal_rag.baml_client.types import (
    FactExtractionBatchItem,
    OntologyConsequenceEntry,
    OntologyConsequenceKind,
    OntologyEvidenceReference,
    OntologyPopulationBatchResult,
    OntologyRuleEntry,
    OntologySubjectKind,
    OntologyValidityMode,
)
from legal_rag.benchmark_synthesis.chunk_screening import ChunkScreeningRecord
from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.benchmark_synthesis.writer import write_chunk_screening, write_document_metadata
from legal_rag.chunker.models import Chunk
from legal_rag.chunker.writer import write_chunks
from legal_rag.ontology_population import extraction
from legal_rag.ontology_population.evidence import fragment_evidence_sources
from legal_rag.ontology_population.writer import (
    read_population_progress,
    read_population_records,
)
from legal_rag.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _CliPopulationClient:
    """Return one typed monetary rule without calling an external model."""

    calls = 0
    settings: Settings | None = None

    def __init__(self, settings: Settings, *_: object) -> None:
        """Capture the settings passed to the production client boundary."""
        type(self).settings = settings

    def extract_batch(
        self,
        items: Sequence[FactExtractionBatchItem],
        metadata: DocumentMetadataRecord,
    ) -> Sequence[OntologyPopulationBatchResult]:
        """Return one typed rule per target chunk."""
        type(self).calls += 1
        results = []
        for item in items:
            source = fragment_evidence_sources(item.target_chunk_id, item.window[0].text)[0]
            evidence = OntologyEvidenceReference(evidence_id=source.evidence_id)
            results.append(
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
                                    currency=None,
                                    point_value=None,
                                    target_label_vi=None,
                                    responsible_role_label_vi=None,
                                    evidence=[evidence],
                                )
                            ],
                            validity_mode=OntologyValidityMode.DocumentDefault,
                            valid_from=None,
                            valid_to=None,
                            provision_path="Điều 13 khoản 9 điểm b",
                            evidence=[evidence],
                        )
                    ],
                )
            )
        return tuple(results)


def test_cli_writes_typed_records_progress_and_graph_without_runtime_shacl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    legal_chunk: Chunk,
    document_metadata: DocumentMetadataRecord,
) -> None:
    """The command executes the complete offline population pipeline."""
    chunks_path = tmp_path / "chunks.jsonl"
    screening_path = tmp_path / "screening.jsonl"
    metadata_path = tmp_path / "metadata.jsonl"
    records_path = tmp_path / "population.jsonl"
    progress_path = tmp_path / "progress.jsonl"
    graph_path = tmp_path / "population.ttl"
    preflight_path = tmp_path / "preflight.json"
    target_selection_path = tmp_path / "target-selection.json"
    dotenv_path = tmp_path / ".env"
    write_chunks(chunks_path, (legal_chunk,))
    write_chunk_screening(
        screening_path,
        (ChunkScreeningRecord(chunk_id=legal_chunk.chunk_id, has_penalty_content=True, reason="Penalty rule."),),
    )
    write_document_metadata(metadata_path, document_metadata)
    target_selection_path.write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "document_number": document_metadata.document_number,
                "chunk_ids": [legal_chunk.chunk_id],
            }
        ),
        encoding="utf-8",
    )
    dotenv_path.write_text(
        "\n".join(
            (
                "LLM_BASE_URL=https://dotenv.test/v1",
                "LLM_API_KEY=dotenv-key",
                "LLM_MODEL=dotenv-model",
                "LLM_TIMEOUT_SECONDS=45",
                'LLM_EXTRA_REQUEST_BODY={"reasoning_effort":"medium"}',
            )
        ),
        encoding="utf-8",
    )
    _CliPopulationClient.calls = 0
    _CliPopulationClient.settings = None
    concurrency: list[int] = []
    extract_population = extraction.extract_population_from_document

    def track_concurrency(*args: object, **kwargs: object) -> object:
        """Capture the CLI override while retaining the production extraction path."""
        concurrency.append(kwargs["max_concurrency"])
        return extract_population(*args, **kwargs)

    monkeypatch.setattr(extraction, "BamlOntologyPopulationClient", _CliPopulationClient)
    monkeypatch.setattr(extraction, "extract_population_from_document", track_concurrency)
    monkeypatch.setattr(ontology_population_cli, "find_dotenv", lambda **_: str(dotenv_path))
    monkeypatch.setenv("LLM_BASE_URL", "https://stale.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "stale-key")
    monkeypatch.setenv("LLM_MODEL", "stale-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("LLM_EXTRA_REQUEST_BODY", '{"reasoning_effort":"low"}')
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "legal-populate-ontology",
            str(chunks_path),
            str(screening_path),
            str(metadata_path),
            str(PROJECT_ROOT / "ontology" / "population_contract.yaml"),
            str(records_path),
            str(progress_path),
            str(graph_path),
            "--target-ids-path",
            str(target_selection_path),
            "--preflight-report-path",
            str(preflight_path),
            "--max-concurrency",
            "2",
        ],
    )

    ontology_population_cli.main()
    monkeypatch.delenv("LLM_BASE_URL")
    monkeypatch.delenv("LLM_API_KEY")
    monkeypatch.delenv("LLM_MODEL")
    monkeypatch.delenv("LLM_TIMEOUT_SECONDS")
    ontology_population_cli.main()

    assert len(read_population_records(records_path)) == 1
    progress = read_population_progress(progress_path)
    assert [record.chunk_id for record in progress] == [legal_chunk.chunk_id]
    assert all(record.run_id for record in progress)
    record = read_population_records(records_path)[0]
    assert record.ontology_version == "0.2.0"
    assert record.ontology_semantic_sha256
    assert record.population_run_id == progress[0].run_id
    assert len(Graph().parse(graph_path, format="turtle")) > 0
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["overall_status"] == "ready"
    assert preflight["ontology_semantic_sha256"] == record.ontology_semantic_sha256
    assert {item["artifact_type"] for item in preflight["artifacts"]} == {
        "chunks",
        "screening",
        "metadata",
        "population_contract",
        "target_selection",
    }
    assert _CliPopulationClient.calls == 1
    assert concurrency == [2]
    assert _CliPopulationClient.settings == Settings(
        llm_base_url="https://dotenv.test/v1",
        llm_api_key="dotenv-key",
        llm_model="dotenv-model",
        llm_timeout_seconds=45,
        llm_extra_request_body={"reasoning_effort": "medium"},
    )

    dotenv_path.write_text(
        dotenv_path.read_text(encoding="utf-8").replace("LLM_MODEL=dotenv-model", "LLM_MODEL=changed-model"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different prompt, model, input, or ontology run"):
        ontology_population_cli.main()
