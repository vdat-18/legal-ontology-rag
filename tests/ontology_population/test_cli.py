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
    OntologyCurrency,
    OntologyEvidenceEntry,
    OntologyPopulationBatchResult,
    OntologyRuleEntry,
    OntologySubjectKind,
)
from legal_rag.benchmark_synthesis.chunk_screening import ChunkScreeningRecord
from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.benchmark_synthesis.writer import write_chunk_screening, write_document_metadata
from legal_rag.chunker.models import Chunk
from legal_rag.chunker.writer import write_chunks
from legal_rag.ontology_population import extraction
from legal_rag.ontology_population.writer import read_population_progress, read_population_records
from legal_rag.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _CliPopulationClient:
    """Return one complete monetary rule without calling an external model."""

    calls = 0
    settings: Settings | None = None

    def __init__(self, settings: Settings, *_: object) -> None:
        """Capture the settings passed to the production client boundary."""
        type(self).settings = settings

    def extract_batch(self, items: Sequence[FactExtractionBatchItem]) -> Sequence[OntologyPopulationBatchResult]:
        """Return one typed rule per target chunk."""
        type(self).calls += 1
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
                                    OntologyEvidenceEntry(
                                        chunk_id=item.target_chunk_id,
                                        quoted_span=item.window[0].text,
                                    )
                                ],
                            )
                        ],
                        valid_from=None,
                        valid_to=None,
                        provision_path="Điều 13 khoản 9 điểm b",
                        evidence=[
                            OntologyEvidenceEntry(chunk_id=item.target_chunk_id, quoted_span=item.window[0].text)
                        ],
                    )
                ],
            )
            for item in items
        )


def test_cli_writes_reviewable_records_progress_and_validated_graph(
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
    dotenv_path = tmp_path / ".env"
    write_chunks(chunks_path, (legal_chunk,))
    write_chunk_screening(
        screening_path,
        (ChunkScreeningRecord(chunk_id=legal_chunk.chunk_id, has_penalty_content=True, reason="Penalty rule."),),
    )
    write_document_metadata(metadata_path, document_metadata)
    dotenv_path.write_text(
        "\n".join(
            (
                "LLM_BASE_URL=https://dotenv.test/v1",
                "LLM_API_KEY=dotenv-key",
                "LLM_MODEL=dotenv-model",
                "LLM_TIMEOUT_SECONDS=45",
            )
        ),
        encoding="utf-8",
    )
    _CliPopulationClient.calls = 0
    _CliPopulationClient.settings = None
    monkeypatch.setattr(extraction, "BamlOntologyPopulationClient", _CliPopulationClient)
    monkeypatch.setattr(ontology_population_cli, "find_dotenv", lambda **_: str(dotenv_path))
    monkeypatch.setenv("LLM_BASE_URL", "https://stale.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "stale-key")
    monkeypatch.setenv("LLM_MODEL", "stale-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
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
            "--preflight-report-path",
            str(preflight_path),
        ],
    )

    ontology_population_cli.main()
    monkeypatch.delenv("LLM_BASE_URL")
    monkeypatch.delenv("LLM_API_KEY")
    monkeypatch.delenv("LLM_MODEL")
    monkeypatch.delenv("LLM_TIMEOUT_SECONDS")
    ontology_population_cli.main()

    assert len(read_population_records(records_path)) == 1
    assert [record.chunk_id for record in read_population_progress(progress_path)] == [legal_chunk.chunk_id]
    assert len(Graph().parse(graph_path, format="turtle")) > 0
    assert json.loads(preflight_path.read_text(encoding="utf-8"))["overall_status"] == "ready"
    assert _CliPopulationClient.calls == 1
    assert _CliPopulationClient.settings == Settings(
        llm_base_url="https://dotenv.test/v1",
        llm_api_key="dotenv-key",
        llm_model="dotenv-model",
        llm_timeout_seconds=45,
    )
