"""Build deterministic provenance snapshots for legal evaluation corpora."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from pydantic import BaseModel, Field

from legal_rag.errors import (
    DuplicateSelectedDocumentError,
    MissingCorpusArtifactError,
    SelectedDocumentNotFoundError,
)
from legal_rag.evaluation.schemas import CorpusArtifact, CorpusSnapshot


class _SelectedDocument(BaseModel):
    """Minimal document fields required from a corpus selection file."""

    file: str = Field(min_length=1)


class _CorpusSelection(BaseModel):
    """Validated document list from the versioned selection config."""

    documents: tuple[_SelectedDocument, ...] = Field(min_length=1)


def build_corpus_snapshot(archive_path: Path, selection_path: Path) -> CorpusSnapshot:
    """Validate selected files and fingerprint the canonical corpus inputs."""
    _require_file(archive_path)
    _require_file(selection_path)

    selection = _CorpusSelection.model_validate_json(selection_path.read_text(encoding="utf-8"))
    selected_documents = tuple(document.file for document in selection.documents)
    _validate_unique(selected_documents)
    _validate_archive_members(archive_path, selected_documents)

    archive = _artifact(archive_path)
    selection_artifact = _artifact(selection_path)
    return CorpusSnapshot(
        snapshot_id=_snapshot_id(archive, selection_artifact, selected_documents),
        archive=archive,
        selection=selection_artifact,
        selected_documents=selected_documents,
    )


def write_corpus_snapshot(snapshot: CorpusSnapshot, path: Path) -> None:
    """Write a corpus snapshot as stable UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{snapshot.model_dump_json(indent=2)}\n", encoding="utf-8")


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise MissingCorpusArtifactError(path)


def _validate_unique(selected_documents: tuple[str, ...]) -> None:
    seen: set[str] = set()
    for filename in selected_documents:
        if filename in seen:
            raise DuplicateSelectedDocumentError(filename)
        seen.add(filename)


def _validate_archive_members(archive_path: Path, selected_documents: tuple[str, ...]) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        archive_members = {entry.filename for entry in archive.infolist() if not entry.is_dir()}

    for filename in selected_documents:
        if filename not in archive_members:
            raise SelectedDocumentNotFoundError(filename, archive_path.name)


def _artifact(path: Path) -> CorpusArtifact:
    return CorpusArtifact(name=path.name, sha256=_sha256(path), size_bytes=path.stat().st_size)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _snapshot_id(
    archive: CorpusArtifact,
    selection: CorpusArtifact,
    selected_documents: tuple[str, ...],
) -> str:
    identity = json.dumps(
        {
            "archive_sha256": archive.sha256,
            "selected_documents": selected_documents,
            "selection_sha256": selection.sha256,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()
