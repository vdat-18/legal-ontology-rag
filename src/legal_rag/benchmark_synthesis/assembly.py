"""Package per-document legal facts into one corpus-level artifact."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from legal_rag.benchmark_synthesis.chunk_screening import CHUNK_SCREENING_PROMPT_VERSION
from legal_rag.benchmark_synthesis.document_metadata import DOCUMENT_METADATA_PROMPT_VERSION
from legal_rag.benchmark_synthesis.fact_extraction import FACT_EXTRACTION_PROMPT_VERSION
from legal_rag.benchmark_synthesis.models import LEGAL_FACT_SCHEMA_VERSION, LegalFact
from legal_rag.benchmark_synthesis.writer import read_legal_facts
from legal_rag.parser.hashing import sha256_json

RUN_MANIFEST_SCHEMA_VERSION = "2.0.0"


@dataclass(frozen=True, slots=True)
class AssemblyResult:
    """Corpus-level facts and their minimal provenance manifest."""

    facts: tuple[LegalFact, ...]
    manifest: dict[str, Any]


def assemble_groundtruth_corpus(
    processed_dir: Path,
    corpus_config_path: Path,
    snapshot_path: Path,
    synthesis_config_path: Path,
    config_paths: Sequence[Path],
    *,
    llm_model: str,
) -> AssemblyResult:
    """Collect configured documents' facts in order and package run provenance."""
    corpus_config = json.loads(corpus_config_path.read_text(encoding="utf-8"))
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    document_ids = tuple(Path(document["file"]).stem for document in corpus_config["documents"])

    facts: list[LegalFact] = []
    document_summaries: list[dict[str, object]] = []
    for document_id in document_ids:
        document_facts = read_legal_facts(processed_dir / f"{document_id}.legal_facts.jsonl")
        facts.extend(document_facts)
        document_summaries.append({"document_id": document_id, "fact_count": len(document_facts)})

    prompt_versions = {
        "chunk_screening": CHUNK_SCREENING_PROMPT_VERSION,
        "document_metadata": DOCUMENT_METADATA_PROMPT_VERSION,
        "fact_extraction": FACT_EXTRACTION_PROMPT_VERSION,
    }
    config_hashes = {
        path.as_posix(): _sha256_file(path) for path in (corpus_config_path, synthesis_config_path, *config_paths)
    }
    identity = {
        "config_sha256": config_hashes,
        "corpus_snapshot_id": snapshot["snapshot_id"],
        "documents": document_ids,
        "fact_corpus_sha256": sha256_json([fact.fact_id for fact in facts]),
        "legal_fact_schema_version": LEGAL_FACT_SCHEMA_VERSION,
        "llm_model": llm_model,
        "prompt_versions": prompt_versions,
    }
    manifest: dict[str, Any] = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": sha256_json(identity),
        **identity,
        "complete": True,
        "created_at": datetime.now(UTC).isoformat(),
        "document_summaries": document_summaries,
        "fact_count": len(facts),
    }
    return AssemblyResult(facts=tuple(facts), manifest=manifest)


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one configuration file."""
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
