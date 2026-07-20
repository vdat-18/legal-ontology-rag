"""Offline tests for deterministic legal corpus snapshots."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from legal_rag.errors import DuplicateSelectedDocumentError, SelectedDocumentNotFoundError
from legal_rag.evaluation import build_corpus_snapshot, write_corpus_snapshot
from legal_rag.evaluation.schemas import CorpusSnapshot


def _write_archive(path: Path, filenames: tuple[str, ...]) -> None:
    """Create a small deterministic ZIP fixture."""
    with zipfile.ZipFile(path, "w") as archive:
        for filename in filenames:
            archive.writestr(filename, f"source for {filename}")


def _write_selection(path: Path, filenames: tuple[str, ...]) -> None:
    """Create a minimal corpus selection fixture."""
    payload = {"documents": [{"file": filename} for filename in filenames]}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_corpus_snapshot_is_stable(tmp_path: Path) -> None:
    """Unchanged inputs produce the same snapshot identifier and hashes."""
    archive_path = tmp_path / "corpus.zip"
    selection_path = tmp_path / "selection.json"
    _write_archive(archive_path, ("a.docx", "b.docx"))
    _write_selection(selection_path, ("b.docx", "a.docx"))

    first = build_corpus_snapshot(archive_path, selection_path)
    second = build_corpus_snapshot(archive_path, selection_path)

    assert first == second
    assert first.selected_documents == ("b.docx", "a.docx")
    assert first.archive.name == "corpus.zip"
    assert first.archive.size_bytes == archive_path.stat().st_size
    assert len(first.archive.sha256) == 64
    assert len(first.snapshot_id) == 64


def test_build_corpus_snapshot_rejects_duplicate_selection(tmp_path: Path) -> None:
    """A filename may appear only once in the selected corpus."""
    archive_path = tmp_path / "corpus.zip"
    selection_path = tmp_path / "selection.json"
    _write_archive(archive_path, ("a.docx",))
    _write_selection(selection_path, ("a.docx", "a.docx"))

    with pytest.raises(DuplicateSelectedDocumentError):
        build_corpus_snapshot(archive_path, selection_path)


def test_build_corpus_snapshot_rejects_missing_archive_member(tmp_path: Path) -> None:
    """Every selected filename must exist in the canonical ZIP."""
    archive_path = tmp_path / "corpus.zip"
    selection_path = tmp_path / "selection.json"
    _write_archive(archive_path, ("a.docx",))
    _write_selection(selection_path, ("missing.docx",))

    with pytest.raises(SelectedDocumentNotFoundError):
        build_corpus_snapshot(archive_path, selection_path)


def test_write_corpus_snapshot_round_trips(tmp_path: Path) -> None:
    """The committed snapshot format is valid JSON and preserves its schema."""
    archive_path = tmp_path / "corpus.zip"
    selection_path = tmp_path / "selection.json"
    output_path = tmp_path / "evaluation" / "corpus-snapshot.json"
    _write_archive(archive_path, ("a.docx",))
    _write_selection(selection_path, ("a.docx",))
    snapshot = build_corpus_snapshot(archive_path, selection_path)

    write_corpus_snapshot(snapshot, output_path)

    restored = CorpusSnapshot.model_validate_json(output_path.read_text(encoding="utf-8"))
    assert restored == snapshot
