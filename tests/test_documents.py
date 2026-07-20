"""Loader tests against the real corpus zip and Docling adapter boundaries."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from legal_rag.config import Settings
from legal_rag.ingestion import iter_episodes, iter_episodes_from_config


class _FakeChunker:
    """Small Docling chunker test double."""

    def chunk(self, *, dl_doc):
        """Yield one fake chunk from a converted document."""
        return iter(
            [
                SimpleNamespace(
                    text="Khoan xu phat source",
                    meta=SimpleNamespace(headings=["Dieu 1. Pham vi"]),
                )
            ]
        )

    def contextualize(self, *, chunk):
        """Return contextual chunk text like Docling does."""
        return f"{chunk.meta.headings[-1]}\n{chunk.text}"


class _FakeConverter:
    """Small Docling converter test double."""

    def convert(self, source):
        """Return a fake converted document from a Docling stream."""
        document = SimpleNamespace(export_to_markdown=lambda: "Van ban nay co hieu luc tu ngay 01 thang 01 nam 2025")
        return SimpleNamespace(document=document, source_name=source.name)


@pytest.fixture
def settings() -> Settings:
    """Settings whose corpus source actually exists."""
    resolved = Settings.from_env()
    if not resolved.corpus_zip.exists() and not resolved.corpus_dir.exists():
        pytest.skip("corpus not available")
    return resolved


def test_yields_timestamped_episodes(settings: Settings) -> None:
    """Every episode carries a timezone-aware reference time."""
    episodes = list(iter_episodes(settings))
    assert len(episodes) > 10
    assert all(e.reference_time.tzinfo is not None for e in episodes)
    assert all(e.body.strip() for e in episodes)


def test_core_decree_effective_date(settings: Settings) -> None:
    """Decree 168/2024 episodes use the curated 2025-01-01 effective date."""
    episodes = [e for e in iter_episodes(settings) if e.name.startswith("168/2024")]
    assert episodes, "Decree 168/2024 must yield episodes"
    assert all(e.reference_time.date().isoformat() == "2025-01-01" for e in episodes)


def test_iter_episodes_from_config_uses_docling_chunks(monkeypatch, tmp_path) -> None:
    """Selected documents are converted and chunked by Docling adapters."""
    selected = tmp_path / "selection.json"
    selected.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "file": "source.docx",
                        "so_hieu": "168/2024/ND-CP",
                        "loai": "Nghi dinh",
                        "effective_date": "2025-01-01",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    corpus_dir = tmp_path / "raw"
    corpus_dir.mkdir()
    (corpus_dir / "source.docx").write_bytes(b"source bytes")
    settings = replace(Settings.from_env(), corpus_dir=corpus_dir, corpus_zip=tmp_path / "missing.zip")
    monkeypatch.setattr("legal_rag.ingestion.documents.DocumentConverter", _FakeConverter)
    monkeypatch.setattr("legal_rag.ingestion.documents.HierarchicalChunker", _FakeChunker)

    episodes = list(iter_episodes_from_config(settings, selected))

    assert len(episodes) == 1
    assert episodes[0].name == "168/2024/ND-CP · chunk 001 · Dieu 1. Pham vi"
    assert episodes[0].body == "[Văn bản: 168/2024/ND-CP] Dieu 1. Pham vi\nKhoan xu phat source"
    assert episodes[0].source_description == "Nghi dinh 168/2024/ND-CP"
    assert episodes[0].reference_time.date().isoformat() == "2025-01-01"
