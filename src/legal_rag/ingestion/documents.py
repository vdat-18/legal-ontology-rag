"""Load Vietnamese traffic-penalty documents into Graphiti episodes.

Reads the curated in-scope list (``data/config/penalty_corpus.json``), extracts
each DOCX (or PDF) into article-level episodes, and attaches the document's
effective date as the episode ``reference_time`` so Graphiti's bi-temporal model
can supersede older provisions automatically. Documents are read directly from
the distributable zip archive (or an extracted directory if present), so no
manual extraction is required.

Structure is recovered from the *typed* numbering in the text (the corpus uses
no Word auto-numbering and no heading styles); regex here is limited to bounded
lexical recognition of ``Điều`` markers and Vietnamese dates.
"""

from __future__ import annotations

import io
import json
import re
import unicodedata
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

from defusedxml.ElementTree import fromstring as xml_fromstring
from loguru import logger

from legal_rag.config import Settings
from legal_rag.errors import ReferenceDateNotFoundError, UnsupportedDocumentError

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_ARTICLE = re.compile(r"^Điều\s+(\d+)\.\s*(.*)")
_VN_DATE = re.compile(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})")
_EFFECTIVE = re.compile(r"có hiệu lực(?: thi hành)?\s*(?:kể từ|từ)?\s*ngày")
_SELF_EFFECTIVE = re.compile(r"(?:Nghị định|Thông tư|Luật|Quyết định|Pháp lệnh) này có hiệu lực")
_MAX_EPISODE_CHARS = 6000


@dataclass(frozen=True)
class LegalEpisode:
    """One article-level unit ready for ``Graphiti.add_episode``."""

    name: str
    body: str
    source_description: str
    reference_time: datetime


def _norm(text: str) -> str:
    """Normalise to Unicode NFC and collapse whitespace (Vietnamese-safe)."""
    return re.sub(r"[ \t]+", " ", unicodedata.normalize("NFC", text)).strip()


def _docx_blocks(data: bytes) -> list[tuple[str, str]]:
    """Return ordered (kind, text) blocks preserving paragraph/table order."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        root = xml_fromstring(zf.read("word/document.xml"))
    body = root.find(_W + "body")
    blocks: list[tuple[str, str]] = []
    if body is None:
        return blocks
    for el in body:
        if el.tag == _W + "p":
            text = _norm("".join(t.text or "" for t in el.iter(_W + "t")))
            if text:
                blocks.append(("p", text))
        elif el.tag == _W + "tbl":
            rows = []
            for tr in el.iter(_W + "tr"):
                cells = [_norm("".join(t.text or "" for t in tc.iter(_W + "t"))) for tc in tr.findall(_W + "tc")]
                rows.append(" | ".join(c for c in cells if c))
            table_text = "\n".join(r for r in rows if r)
            if table_text:
                blocks.append(("table", table_text))
    return blocks


def _pdf_blocks(data: bytes) -> list[tuple[str, str]]:
    """Return paragraph blocks from a PDF (fallback loader)."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    blocks: list[tuple[str, str]] = []
    for page in reader.pages:
        for line in (page.extract_text() or "").splitlines():
            text = _norm(line)
            if text:
                blocks.append(("p", text))
    return blocks


def _load_blocks(filename: str, data: bytes) -> list[tuple[str, str]]:
    """Dispatch by file suffix; raise for unsupported types."""
    suffix = filename.lower().rsplit(".", 1)[-1]
    if suffix == "docx":
        return _docx_blocks(data)
    if suffix == "pdf":
        return _pdf_blocks(data)
    raise UnsupportedDocumentError(filename)


def _read_member(settings: Settings, filename: str) -> bytes | None:
    """Read a document's bytes from the extracted dir, else from the zip archive."""
    extracted = settings.corpus_dir / filename
    if extracted.exists():
        return extracted.read_bytes()
    if settings.corpus_zip.exists():
        with zipfile.ZipFile(settings.corpus_zip) as zf:
            if filename in set(zf.namelist()):
                return zf.read(filename)
    return None


def _vn_date_to_iso(text: str) -> str | None:
    """Convert 'ngày D tháng M năm Y' to ISO date, or None."""
    match = _VN_DATE.search(text)
    if not match:
        return None
    day, month, year = (int(g) for g in match.groups())
    try:
        return datetime(year, month, day, tzinfo=UTC).date().isoformat()
    except ValueError:
        return None


def _effective_date(blocks: list[tuple[str, str]]) -> str | None:
    """Find the effective date, preferring the document's self-referential clause.

    Amendments and consolidated documents cite several dates; the clause
    '<văn bản> này có hiệu lực ...' refers to this document itself and is
    preferred over any generic 'có hiệu lực' mention.
    """
    for pattern in (_SELF_EFFECTIVE, _EFFECTIVE):
        for _, text in blocks:
            if pattern.search(text):
                iso = _vn_date_to_iso(text)
                if iso:
                    return iso
    return None


def _chunk(text: str, limit: int) -> list[str]:
    """Split text on line boundaries so each chunk stays under ``limit``."""
    if len(text) <= limit:
        return [text]
    chunks, current = [], ""
    for line in text.split("\n"):
        if current and len(current) + len(line) + 1 > limit:
            chunks.append(current)
            current = ""
        current = f"{current}\n{line}" if current else line
    if current:
        chunks.append(current)
    return chunks


def _iter_articles(blocks: list[tuple[str, str]]) -> Iterator[tuple[str, str]]:
    """Yield (heading, body) for each 'Điều N.' article in reading order."""
    heading: str | None = None
    buffer: list[str] = []
    for _, text in blocks:
        if _ARTICLE.match(text):
            if heading is not None:
                yield heading, "\n".join(buffer).strip()
            heading = text
            buffer = []
        elif heading is not None:
            buffer.append(text)
    if heading is not None:
        yield heading, "\n".join(buffer).strip()


def _resolve_reference_time(doc: dict, blocks: list[tuple[str, str]]) -> datetime:
    """Effective date from config override, else parsed from the document."""
    iso = doc.get("effective_date") or _effective_date(blocks)
    if iso is None:
        logger.warning("No effective date for {}; falling back to issue-date parse", doc["file"])
        for _, text in blocks[:6]:
            iso = _vn_date_to_iso(text)
            if iso:
                break
    if iso is None:
        raise ReferenceDateNotFoundError(doc["file"])
    return datetime.fromisoformat(iso).replace(tzinfo=UTC)


def iter_episodes(settings: Settings) -> Iterator[LegalEpisode]:
    """Yield article-level episodes for every in-scope penalty document."""
    corpus = json.loads(settings.penalty_config.read_text(encoding="utf-8"))
    for doc in corpus["documents"]:
        data = _read_member(settings, doc["file"])
        if data is None:
            logger.warning("Missing document, skipping: {}", doc["file"])
            continue
        blocks = _load_blocks(doc["file"], data)
        reference_time = _resolve_reference_time(doc, blocks)
        so_hieu = doc.get("so_hieu", doc["file"])
        count = 0
        for heading, body in _iter_articles(blocks):
            for idx, chunk in enumerate(_chunk(f"{heading}\n{body}", _MAX_EPISODE_CHARS)):
                suffix = f" (phần {idx + 1})" if idx else ""
                yield LegalEpisode(
                    name=f"{so_hieu} · {heading}{suffix}",
                    body=f"[Văn bản: {so_hieu}] {chunk}",
                    source_description=f"{doc.get('loai', '')} {so_hieu}".strip(),
                    reference_time=reference_time,
                )
                count += 1
        logger.info("{}: {} episode(s), reference_time={}", so_hieu, count, reference_time.date())
