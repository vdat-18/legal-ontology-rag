"""Load Vietnamese traffic-penalty documents into Graphiti episodes.

Reads a versioned corpus selection, converts each document with Docling, chunks
the converted document with Docling's chunker, and attaches the document's
effective date as the episode ``reference_time`` so Graphiti's bi-temporal model
can supersede older provisions automatically. Documents are read directly from
the distributable zip archive, or from an extracted directory when allowed.

Document parsing and chunking are delegated to Docling. Regex here is limited
to bounded Vietnamese date parsing for temporal provenance.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docling.chunking import BaseChunk, HierarchicalChunker
from docling.document_converter import DocumentConverter
from docling_core.types.io import DocumentStream
from loguru import logger

from legal_rag.config import Settings
from legal_rag.errors import ReferenceDateNotFoundError

_VN_DATE = re.compile(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})")
_EFFECTIVE = re.compile(r"có hiệu lực(?: thi hành)?\s*(?:kể từ|từ)?\s*ngày")
_SELF_EFFECTIVE = re.compile(r"(?:Nghị định|Thông tư|Luật|Quyết định|Pháp lệnh) này có hiệu lực")


@dataclass(frozen=True)
class LegalEpisode:
    """One Docling chunk ready for ``Graphiti.add_episode``."""

    name: str
    body: str
    source_description: str
    reference_time: datetime


def _read_member(settings: Settings, filename: str, *, archive_only: bool = False) -> bytes | None:
    """Read a document's bytes from the selected corpus source."""
    extracted = settings.corpus_dir / filename
    if not archive_only and extracted.exists():
        return extracted.read_bytes()
    if settings.corpus_zip.exists():
        with zipfile.ZipFile(settings.corpus_zip) as zf:
            if filename in set(zf.namelist()):
                return zf.read(filename)
    return None


def _docling_document(filename: str, data: bytes, converter: DocumentConverter) -> Any:
    """Convert source bytes to a Docling document."""
    source = DocumentStream(name=filename, stream=io.BytesIO(data))
    return converter.convert(source).document


def _docling_chunks(document: Any, chunker: HierarchicalChunker) -> list[BaseChunk]:
    """Chunk a Docling document and keep only non-empty text chunks."""
    return [chunk for chunk in chunker.chunk(dl_doc=document) if chunk.text.strip()]


def _chunk_text(chunk: BaseChunk, chunker: HierarchicalChunker) -> str:
    """Return Docling's contextual chunk text for retrieval and generation."""
    return chunker.contextualize(chunk=chunk).strip()


def _chunk_heading(chunk: BaseChunk) -> str | None:
    """Return the closest Docling heading when available."""
    headings = getattr(chunk.meta, "headings", None)
    if not headings:
        return None
    return headings[-1].strip()


def _vn_date_to_iso(text: str) -> str | None:
    """Convert 'ngày D tháng M năm Y' to ISO date, or None."""
    match = _VN_DATE.search(text)
    if not match:
        return None
    day, month, year = (int(group) for group in match.groups())
    try:
        return datetime(year, month, day, tzinfo=UTC).date().isoformat()
    except ValueError:
        return None


def _effective_date(texts: list[str]) -> str | None:
    """Find the effective date, preferring the document's self-referential clause."""
    for pattern in (_SELF_EFFECTIVE, _EFFECTIVE):
        for text in texts:
            if pattern.search(text):
                iso = _vn_date_to_iso(text)
                if iso:
                    return iso
    return None


def _first_date(texts: list[str]) -> str | None:
    """Return the first Vietnamese date found in converted document text."""
    for text in texts:
        iso = _vn_date_to_iso(text)
        if iso:
            return iso
    return None


def _resolve_reference_time(doc: dict, texts: list[str]) -> datetime:
    """Effective date from config override, else parsed from converted text."""
    iso = doc.get("effective_date") or _effective_date(texts)
    if iso is None:
        logger.warning("No effective date for {}; falling back to issue-date parse", doc["file"])
        iso = _first_date(texts)
    if iso is None:
        raise ReferenceDateNotFoundError(doc["file"])
    return datetime.fromisoformat(iso).replace(tzinfo=UTC)


def iter_episodes_from_config(
    settings: Settings,
    selection_path: Path,
    *,
    archive_only: bool = False,
    skip_missing: bool = True,
) -> Iterator[LegalEpisode]:
    """Yield Docling chunk episodes for documents selected by one config file."""
    corpus = json.loads(selection_path.read_text(encoding="utf-8"))
    converter = DocumentConverter()
    chunker = HierarchicalChunker()
    for doc in corpus["documents"]:
        data = _read_member(settings, doc["file"], archive_only=archive_only)
        if data is None:
            if not skip_missing:
                raise FileNotFoundError(doc["file"])
            logger.warning("Missing document, skipping: {}", doc["file"])
            continue

        document = _docling_document(doc["file"], data, converter)
        chunks = _docling_chunks(document, chunker)
        chunk_texts = [_chunk_text(chunk, chunker) for chunk in chunks]
        full_text = document.export_to_markdown()
        reference_time = _resolve_reference_time(doc, [full_text, *chunk_texts])
        so_hieu = doc.get("so_hieu", doc["file"])

        for index, (chunk, text) in enumerate(zip(chunks, chunk_texts, strict=True), start=1):
            heading = _chunk_heading(chunk)
            suffix = f" · {heading}" if heading else ""
            yield LegalEpisode(
                name=f"{so_hieu} · chunk {index:03d}{suffix}",
                body=f"[Văn bản: {so_hieu}] {text}",
                source_description=f"{doc.get('loai', '')} {so_hieu}".strip(),
                reference_time=reference_time,
            )
        logger.info("{}: {} Docling chunk(s), reference_time={}", so_hieu, len(chunks), reference_time.date())


def iter_episodes(settings: Settings) -> Iterator[LegalEpisode]:
    """Yield Docling chunk episodes for every in-scope penalty document."""
    yield from iter_episodes_from_config(settings, settings.penalty_config)
