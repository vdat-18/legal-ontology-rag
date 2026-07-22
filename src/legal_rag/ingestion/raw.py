"""Raw crawled documents — the *input* to parsing+chunking (still WIP).

Models the crawler/extractor output (``{text, meta, method, ok, note}``) and
loads mock instances from ``data/mock/raw/*.json`` so the parser can be built
and tested before the real crawler is connected. The JSON content is
illustrative MOCK data, not authoritative legal text.

Override the source directory with the ``MOCK_RAW_DIR`` environment variable.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

_DEFAULT_DIR = Path(__file__).resolve().parents[3] / "data" / "mock" / "raw"


class RawMeta(BaseModel):
    """Provenance metadata attached to a crawled legal document."""

    model_config = ConfigDict(frozen=True)

    url: str = Field(min_length=1)
    document: str = Field(min_length=1, description="Short document reference, e.g. 'ND 123/2021/ND-CP'.")
    document_number: str = Field(min_length=1, description="Official number, e.g. '123/2021/NĐ-CP'.")
    document_type: str = Field(min_length=1, description="e.g. 'nghị định', 'nghị định sửa đổi'.")
    domain: str | None = None
    issued_date: date | None = None
    effective_date: date | None = None
    expiry_date: date | None = None
    signer: str | None = None
    issuing_authority: str | None = None


class RawDocument(BaseModel):
    """One crawled document: full extracted text plus its metadata."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    meta: RawMeta
    method: str = Field(min_length=1)
    ok: bool
    note: str | None = None


def mock_raw_dir() -> Path:
    """Return the mock raw-document directory (``MOCK_RAW_DIR`` override, else default)."""
    override = os.getenv("MOCK_RAW_DIR")
    return Path(override) if override else _DEFAULT_DIR


def iter_mock_raw_documents(directory: Path | None = None) -> Iterator[RawDocument]:
    """Yield validated ``RawDocument`` records from each JSON file in the directory."""
    source = directory or mock_raw_dir()
    for path in sorted(source.glob("*.json")):
        yield RawDocument.model_validate_json(path.read_text(encoding="utf-8"))
