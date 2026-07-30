"""Tests for the extraction-progress backfill command."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from legal_rag import extraction_progress_backfill_cli
from legal_rag.benchmark_synthesis.chunk_screening import ChunkScreeningRecord
from legal_rag.benchmark_synthesis.fact_extraction import ExtractionProgressRecord
from legal_rag.benchmark_synthesis.writer import read_extraction_progress, write_chunk_screening


def _write_screening(tmp_path: Path) -> Path:
    """Write one screening file with two accepted chunks and one excluded chunk."""
    screening_path = tmp_path / "chunk_screening.jsonl"
    write_chunk_screening(
        screening_path,
        [
            ChunkScreeningRecord(chunk_id="chunk-1", has_penalty_content=True, reason="Contains penalty content."),
            ChunkScreeningRecord(chunk_id="chunk-2", has_penalty_content=False, reason="Table of contents."),
            ChunkScreeningRecord(chunk_id="chunk-3", has_penalty_content=True, reason="Contains penalty content."),
        ],
    )
    return screening_path


def test_backfill_records_only_the_accepted_chunks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Chunks excluded by screening were never extraction targets, so they are not marked processed."""
    screening_path = _write_screening(tmp_path)
    progress_path = tmp_path / "extraction_progress.jsonl"
    monkeypatch.setattr(sys, "argv", ["legal-backfill-extraction-progress", str(screening_path), str(progress_path)])

    extraction_progress_backfill_cli.main()

    assert read_extraction_progress(progress_path) == (
        ExtractionProgressRecord(chunk_id="chunk-1"),
        ExtractionProgressRecord(chunk_id="chunk-3"),
    )


def test_backfill_leaves_a_skipped_chunk_for_retry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A chunk named as skipped stays out of the progress file so the next run extracts it."""
    screening_path = _write_screening(tmp_path)
    progress_path = tmp_path / "extraction_progress.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "legal-backfill-extraction-progress",
            str(screening_path),
            str(progress_path),
            "--skipped-chunk-id",
            "chunk-3",
        ],
    )

    extraction_progress_backfill_cli.main()

    assert read_extraction_progress(progress_path) == (ExtractionProgressRecord(chunk_id="chunk-1"),)
