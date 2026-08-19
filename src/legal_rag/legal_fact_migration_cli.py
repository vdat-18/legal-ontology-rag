"""Command-line interface migrating legal facts from schema 1.1.0 to 1.2.0."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

from loguru import logger

SOURCE_SCHEMA_VERSION = "1.1.0"


def _parser() -> argparse.ArgumentParser:
    """Build the legal-fact migration command-line interface."""
    parser = argparse.ArgumentParser(
        description="Restamp one document's legal facts with resolved document metadata and recomputed identities."
    )
    parser.add_argument("facts_path", type=Path, help="Legal-facts JSONL path written under schema 1.1.0")
    parser.add_argument(
        "metadata_path", type=Path, help="Document-metadata JSONL path from legal-resolve-document-metadata"
    )
    return parser


def main() -> None:
    """Rewrite one document's legal facts in place under the current schema, without new extraction calls."""
    from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
    from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact, build_legal_fact
    from legal_rag.benchmark_synthesis.writer import read_document_metadata, write_legal_facts

    def _migrated_fact(payload: dict[str, Any], metadata: DocumentMetadataRecord) -> LegalFact:
        """Rebuild one stored fact with document metadata stamped and its identity recomputed."""
        if payload["schema_version"] != SOURCE_SCHEMA_VERSION:
            msg = f"Expected legal-fact schema {SOURCE_SCHEMA_VERSION}, got {payload['schema_version']}"
            raise ValueError(msg)
        return build_legal_fact(
            statement=payload["conduct"],
            conduct=payload["conduct"],
            subject=payload["subject"],
            conditions=tuple(payload["conditions"]),
            monetary_penalty=payload["monetary_penalty"],
            license_points_deducted=payload["license_points_deducted"],
            additional_penalties=tuple(payload["additional_penalties"]),
            remedial_measures=tuple(payload["remedial_measures"]),
            valid_from=date.fromisoformat(payload["valid_from"]) if payload["valid_from"] else metadata.valid_from,
            valid_to=date.fromisoformat(payload["valid_to"]) if payload["valid_to"] else metadata.valid_to,
            document_number=metadata.document_number,
            provision_path=payload["provision_path"],
            evidence=tuple(
                EvidenceReference(chunk_id=entry["chunk_id"], quoted_span=entry["quoted_span"])
                for entry in payload["evidence"]
            ),
            source_chunk_id=payload["source_chunk_id"],
        )

    arguments = _parser().parse_args()
    metadata = read_document_metadata(arguments.metadata_path)

    with arguments.facts_path.open(encoding="utf-8") as source_file:
        payloads = [json.loads(line) for line in source_file if line.strip()]
    migrated = {fact.fact_id: fact for fact in (_migrated_fact(payload, metadata) for payload in payloads)}

    write_legal_facts(arguments.facts_path, tuple(migrated.values()))
    logger.info(
        "Migrated {} facts into {} under document {}, wrote {}",
        len(payloads),
        len(migrated),
        metadata.document_number,
        arguments.facts_path,
    )


if __name__ == "__main__":
    main()
