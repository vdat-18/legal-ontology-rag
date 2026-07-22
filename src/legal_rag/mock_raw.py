"""Mock crawled documents — fixture input for the parsing pipeline.

Models a crawler/extractor record (``{text, meta, method, ok, note}``) and loads
mock instances from ``data/mock/raw/*.json``. Uses stdlib ``json`` and frozen
dataclasses (no third-party validation) to match the repository style. The JSON
content is illustrative MOCK data, not authoritative legal text.

Override the source directory with the ``MOCK_RAW_DIR`` environment variable.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

_DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "mock" / "raw"


class MockRawValidationError(ValueError):
    """Raised when a mock raw document breaks its expected contract."""

    @classmethod
    def missing_field(cls, field_name: str) -> MockRawValidationError:
        """Build an error for a missing or empty required field."""
        return cls(f"Mock raw document is missing required field: {field_name}")

    @classmethod
    def invalid_date(cls, field_name: str, value: str) -> MockRawValidationError:
        """Build an error for a non-ISO date value."""
        return cls(f"Mock raw meta field {field_name} is not an ISO date: {value}")


@dataclass(frozen=True, slots=True)
class RawMeta:
    """Provenance metadata attached to a crawled legal document."""

    url: str
    document: str
    document_number: str
    document_type: str
    domain: str | None
    issued_date: date | None
    effective_date: date | None
    expiry_date: date | None
    signer: str | None
    issuing_authority: str | None


@dataclass(frozen=True, slots=True)
class RawDocument:
    """One crawled document: full extracted text plus its metadata."""

    text: str
    meta: RawMeta
    method: str
    ok: bool
    note: str | None


def _required_text(mapping: Mapping[str, Any], key: str) -> str:
    """Return a non-empty string field or raise a validation error."""
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MockRawValidationError.missing_field(key)
    return value


def _optional_text(mapping: Mapping[str, Any], key: str) -> str | None:
    """Return a string field if present and non-empty, else ``None``."""
    value = mapping.get(key)
    return value if isinstance(value, str) and value.strip() else None


def _optional_date(mapping: Mapping[str, Any], key: str) -> date | None:
    """Parse an ISO date field if present, else ``None``."""
    value = mapping.get(key)
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError as error:
        raise MockRawValidationError.invalid_date(key, str(value)) from error


def _meta_from_mapping(mapping: Mapping[str, Any]) -> RawMeta:
    """Build a validated ``RawMeta`` from a JSON mapping."""
    return RawMeta(
        url=_required_text(mapping, "url"),
        document=_required_text(mapping, "document"),
        document_number=_required_text(mapping, "document_number"),
        document_type=_required_text(mapping, "document_type"),
        domain=_optional_text(mapping, "domain"),
        issued_date=_optional_date(mapping, "issued_date"),
        effective_date=_optional_date(mapping, "effective_date"),
        expiry_date=_optional_date(mapping, "expiry_date"),
        signer=_optional_text(mapping, "signer"),
        issuing_authority=_optional_text(mapping, "issuing_authority"),
    )


def _document_from_mapping(mapping: Mapping[str, Any]) -> RawDocument:
    """Build a validated ``RawDocument`` from a JSON mapping."""
    meta = mapping.get("meta")
    if not isinstance(meta, Mapping):
        raise MockRawValidationError.missing_field("meta")
    return RawDocument(
        text=_required_text(mapping, "text"),
        meta=_meta_from_mapping(meta),
        method=_required_text(mapping, "method"),
        ok=bool(mapping.get("ok")),
        note=_optional_text(mapping, "note"),
    )


def mock_raw_dir() -> Path:
    """Return the mock raw-document directory (``MOCK_RAW_DIR`` override, else default)."""
    override = os.getenv("MOCK_RAW_DIR")
    return Path(override) if override else _DEFAULT_DIR


def iter_mock_raw_documents(directory: Path | None = None) -> Iterator[RawDocument]:
    """Yield validated ``RawDocument`` records from each JSON file in the directory."""
    source = directory or mock_raw_dir()
    for path in sorted(source.glob("*.json")):
        mapping = json.loads(path.read_text(encoding="utf-8"))
        yield _document_from_mapping(mapping)
