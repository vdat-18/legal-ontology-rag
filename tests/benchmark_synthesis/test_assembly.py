"""Tests for simple corpus-level ground-truth packaging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.benchmark_synthesis.assembly import RUN_MANIFEST_SCHEMA_VERSION, assemble_groundtruth_corpus
from legal_rag.benchmark_synthesis.models import LEGAL_FACT_SCHEMA_VERSION, EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.writer import write_legal_facts

_DOCUMENT_A = "168_2024_ND-CP_619502"
_DOCUMENT_B = "second_document"
_SYNTHESIS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "data" / "config" / "groundtruth" / "synthesis_v1.json"


def _fact(fact_id: str = "fact-1", *, statement: str = "Nguoi dieu khien oto vuot den do bi phat tien.") -> LegalFact:
    """Build one fact with chunk-level provenance."""
    return LegalFact(
        fact_id=fact_id,
        statement=statement,
        valid_from=None,
        valid_to=None,
        document_number="168/2024/ND-CP",
        evidence=(EvidenceReference(chunk_id="chunk-1"),),
        source_chunk_id="chunk-1",
    )


def _write_inputs(
    tmp_path: Path,
    documents: tuple[tuple[str, tuple[LegalFact, ...]], ...],
) -> tuple[Path, Path, Path]:
    """Write only the per-document fact artifacts required by assembly."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    for document_id, facts in documents:
        write_legal_facts(processed_dir / f"{document_id}.legal_facts.jsonl", facts)

    config_path = tmp_path / "corpus.json"
    config_path.write_text(
        json.dumps({"documents": [{"file": f"{document_id}.docx"} for document_id, _ in documents]}),
        encoding="utf-8",
    )
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps({"snapshot_id": "snapshot-1"}), encoding="utf-8")
    return processed_dir, config_path, snapshot_path


def _assemble(tmp_path: Path, documents: tuple[tuple[str, tuple[LegalFact, ...]], ...]):
    """Package one test corpus with versioned provenance inputs."""
    processed_dir, config_path, snapshot_path = _write_inputs(tmp_path, documents)
    return assemble_groundtruth_corpus(
        processed_dir,
        config_path,
        snapshot_path,
        _SYNTHESIS_CONFIG_PATH,
        [config_path],
        llm_model="provider/test-model",
    )


def test_assembly_packages_facts_and_minimal_manifest(tmp_path: Path) -> None:
    """Assembly joins facts and records only corpus-level provenance."""
    result = _assemble(tmp_path, ((_DOCUMENT_A, (_fact(),)),))

    assert result.facts == (_fact(),)
    assert result.manifest["schema_version"] == RUN_MANIFEST_SCHEMA_VERSION
    assert result.manifest["legal_fact_schema_version"] == LEGAL_FACT_SCHEMA_VERSION
    assert result.manifest["corpus_snapshot_id"] == "snapshot-1"
    assert result.manifest["documents"] == (_DOCUMENT_A,)
    assert result.manifest["fact_count"] == 1
    assert result.manifest["document_summaries"] == [{"document_id": _DOCUMENT_A, "fact_count": 1}]
    assert result.manifest["prompt_versions"] == {
        "chunk_screening": "5.7.0",
        "document_metadata": "5.0.0",
        "fact_extraction": "5.0.0",
    }
    assert result.manifest["fact_corpus_sha256"]
    assert result.manifest["config_sha256"]


def test_assembly_preserves_document_and_fact_order(tmp_path: Path) -> None:
    """Facts retain corpus-config order and their order within each document."""
    fact_a1 = _fact("fact-a1")
    fact_a2 = _fact("fact-a2")
    fact_b1 = _fact("fact-b1")

    result = _assemble(
        tmp_path,
        ((_DOCUMENT_A, (fact_a1, fact_a2)), (_DOCUMENT_B, (fact_b1,))),
    )

    assert result.facts == (fact_a1, fact_a2, fact_b1)


def test_assembly_preserves_duplicate_and_conflicting_fact_ids(tmp_path: Path) -> None:
    """Assembly packages every input record without deduplication or conflict checks."""
    first = _fact()
    duplicate = _fact()
    conflict = _fact(statement="Noi dung khac cung fact id.")

    result = _assemble(tmp_path, ((_DOCUMENT_A, (first, duplicate, conflict)),))

    assert result.facts == (first, duplicate, conflict)
    assert result.manifest["fact_count"] == 3


def test_assembly_does_not_require_screening_metadata_or_progress_artifacts(tmp_path: Path) -> None:
    """Per-document legal facts are the only synthesis artifact assembly reads."""
    result = _assemble(tmp_path, ((_DOCUMENT_A, (_fact(),)),))

    assert len(result.facts) == 1


def test_assembly_lets_missing_fact_file_raise_naturally(tmp_path: Path) -> None:
    """A missing required fact artifact raises the reader's native file error."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    config_path = tmp_path / "corpus.json"
    config_path.write_text(json.dumps({"documents": [{"file": f"{_DOCUMENT_A}.docx"}]}), encoding="utf-8")
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps({"snapshot_id": "snapshot-1"}), encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        assemble_groundtruth_corpus(
            processed_dir,
            config_path,
            snapshot_path,
            _SYNTHESIS_CONFIG_PATH,
            [],
            llm_model="provider/test-model",
        )


def test_assembly_run_id_changes_with_ordered_fact_content(tmp_path: Path) -> None:
    """The manifest fingerprint changes when packaged fact content changes."""
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()

    first = _assemble(first_dir, ((_DOCUMENT_A, (_fact(),)),))
    second = _assemble(second_dir, ((_DOCUMENT_A, (_fact(statement="changed-content"),)),))

    assert first.manifest["run_id"] != second.manifest["run_id"]
