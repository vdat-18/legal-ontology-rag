"""Build reading windows and extract legal facts from chunk text."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from functools import partial
from typing import Any, Protocol

from baml_py.logging import set_log_level

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import ChunkWindowEntry, ChunkWindowRole
from legal_rag.baml_client.types import LegalFact as BamlLegalFact
from legal_rag.baml_transport import build_client_registry, execute_baml_request
from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.benchmark_synthesis.llm_execution import (
    IncompleteLlmTasksError,
    LlmTaskExhaustedError,
    LlmTaskFailure,
    execute_llm_task,
)
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact, build_legal_fact
from legal_rag.chunker.models import Chunk
from legal_rag.settings import EvaluationSettings, Settings

# Lexical parsing, not a semantic decision: locate "Điều N" cross-references.
_ARTICLE_HEADING_PATTERN = re.compile(r"\s*Điều\s+(\d+)\b")
_ARTICLE_CITATION_PATTERN = re.compile(r"Điều\s+(\d+)\b")

DEFAULT_MAX_PRECEDING_CHUNKS = 2
DEFAULT_MAX_FOLLOWING_CHUNKS = 1
DEFAULT_EXTRACTION_MAX_RETRIES = 2

EXTRACTION_PROGRESS_SCHEMA_VERSION = "1.0.0"
FACT_EXTRACTION_PROMPT_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class ExtractionProgressRecord:
    """One chunk fact extraction has finished reading, recorded whether or not it produced facts."""

    chunk_id: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable extraction-progress record."""
        return {
            "schema_version": EXTRACTION_PROGRESS_SCHEMA_VERSION,
            "chunk_id": self.chunk_id,
        }


@dataclass(frozen=True, slots=True)
class ProvisionIndex:
    """Maps an article number to the chunk(s) whose text opens that article."""

    chunk_ids_by_article: Mapping[str, tuple[str, ...]]


def build_provision_index(chunks: Sequence[Chunk]) -> ProvisionIndex:
    """Index chunks whose text opens a numbered article, within one document."""
    chunk_ids_by_article: dict[str, list[str]] = {}
    for chunk in chunks:
        match = _ARTICLE_HEADING_PATTERN.match(chunk.text)
        if match is None:
            continue
        chunk_ids_by_article.setdefault(match.group(1), []).append(chunk.chunk_id)
    return ProvisionIndex({article: tuple(ids) for article, ids in chunk_ids_by_article.items()})


def cited_articles(chunk: Chunk) -> frozenset[str]:
    """Return article numbers cited in a chunk's text, excluding its own heading."""
    own_heading = _ARTICLE_HEADING_PATTERN.match(chunk.text)
    own_article = own_heading.group(1) if own_heading else None
    cited = {match.group(1) for match in _ARTICLE_CITATION_PATTERN.finditer(chunk.text)}
    cited.discard(own_article)
    return frozenset(cited)


def _walk_adjacent_chunks(
    start_chunk_id: str | None,
    chunks_by_id: Mapping[str, Chunk],
    entries: dict[str, ChunkWindowEntry],
    *,
    max_chunks: int,
    next_chunk_id: Callable[[Chunk], str | None],
) -> None:
    """Walk a previous/next chunk chain, adding each hop found to entries as Adjacent context."""
    chunk_id = start_chunk_id
    for _ in range(max_chunks):
        if chunk_id is None or chunk_id in entries:
            return
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            return
        entries[chunk_id] = ChunkWindowEntry(
            chunk_id=chunk_id, role=ChunkWindowRole.Adjacent, provision_path=None, text=chunk.text
        )
        chunk_id = next_chunk_id(chunk)


def build_reading_window(
    target: Chunk,
    chunks_by_id: Mapping[str, Chunk],
    provision_index: ProvisionIndex,
    *,
    max_preceding_chunks: int = DEFAULT_MAX_PRECEDING_CHUNKS,
    max_following_chunks: int = DEFAULT_MAX_FOLLOWING_CHUNKS,
) -> tuple[ChunkWindowEntry, ...]:
    """Assemble one reading window around a target chunk."""
    entries: dict[str, ChunkWindowEntry] = {
        target.chunk_id: ChunkWindowEntry(
            chunk_id=target.chunk_id, role=ChunkWindowRole.Target, provision_path=None, text=target.text
        )
    }
    _walk_adjacent_chunks(
        target.previous_chunk_id,
        chunks_by_id,
        entries,
        max_chunks=max_preceding_chunks,
        next_chunk_id=lambda chunk: chunk.previous_chunk_id,
    )
    _walk_adjacent_chunks(
        target.next_chunk_id,
        chunks_by_id,
        entries,
        max_chunks=max_following_chunks,
        next_chunk_id=lambda chunk: chunk.next_chunk_id,
    )

    for article in sorted(cited_articles(target)):
        for chunk_id in provision_index.chunk_ids_by_article.get(article, ()):
            if chunk_id in entries:
                continue
            cited_chunk = chunks_by_id[chunk_id]
            entries[chunk_id] = ChunkWindowEntry(
                chunk_id=chunk_id,
                role=ChunkWindowRole.CitedProvision,
                provision_path=f"Điều {article}",
                text=cited_chunk.text,
            )

    return tuple(entries.values())


class FactExtractionClient(Protocol):
    """Boundary implemented by a legal-fact-extraction language-model client."""

    def extract(self, window: Sequence[ChunkWindowEntry]) -> tuple[BamlLegalFact, ...]:
        """Return typed candidate facts for one reading window."""


class BamlFactExtractionClient:
    """Typed BAML client backed by a runtime OpenAI-compatible provider."""

    def __init__(self, settings: Settings, evaluation_settings: EvaluationSettings) -> None:
        """Create a runtime BAML client from typed, secret-safe configuration."""
        set_log_level("OFF")
        self._registry = build_client_registry(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_output_tokens=evaluation_settings.fact_extraction_max_output_tokens,
            temperature=evaluation_settings.fact_extraction_temperature,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        self._timeout_seconds = settings.llm_timeout_seconds
        self._max_response_bytes = evaluation_settings.max_response_bytes

    def extract(self, window: Sequence[ChunkWindowEntry]) -> tuple[BamlLegalFact, ...]:
        """Run one bounded BAML call and return its generated typed result."""
        request = b.request.ExtractLegalFacts(list(window), baml_options={"client_registry": self._registry})
        raw_output = execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)
        result = b.parse.ExtractLegalFacts(raw_output, baml_options={"client_registry": self._registry})
        return tuple(result)


def build_extracted_fact(raw: BamlLegalFact, source_chunk_id: str, metadata: DocumentMetadataRecord) -> LegalFact:
    """Assign identity to one LLM-produced fact, stamping document metadata the window cannot carry."""
    return build_legal_fact(
        conduct=raw.conduct,
        subject=raw.subject,
        conditions=tuple(raw.conditions),
        monetary_penalty=raw.monetary_penalty,
        license_points_deducted=raw.license_points_deducted,
        additional_penalties=tuple(raw.additional_penalties),
        remedial_measures=tuple(raw.remedial_measures),
        valid_from=date.fromisoformat(raw.valid_from) if raw.valid_from else metadata.valid_from,
        valid_to=date.fromisoformat(raw.valid_to) if raw.valid_to else metadata.valid_to,
        document_number=metadata.document_number,
        provision_path=raw.provision_path,
        evidence=tuple(
            EvidenceReference(chunk_id=entry.chunk_id, quoted_span=entry.quoted_span) for entry in raw.evidence
        ),
        source_chunk_id=source_chunk_id,
    )


def _extract_target_facts(
    client: FactExtractionClient,
    window: Sequence[ChunkWindowEntry],
    source_chunk_id: str,
    metadata: DocumentMetadataRecord,
) -> tuple[LegalFact, ...]:
    """Extract and fully parse one target's facts inside the retry boundary."""
    raw_facts = client.extract(window)
    return tuple(build_extracted_fact(raw_fact, source_chunk_id, metadata) for raw_fact in raw_facts)


def extract_facts_from_document(
    target_chunks: Sequence[Chunk],
    document_chunks: Sequence[Chunk],
    client: FactExtractionClient,
    metadata: DocumentMetadataRecord,
    *,
    max_preceding_chunks: int = DEFAULT_MAX_PRECEDING_CHUNKS,
    max_following_chunks: int = DEFAULT_MAX_FOLLOWING_CHUNKS,
    max_retries: int = DEFAULT_EXTRACTION_MAX_RETRIES,
    on_target_processed: Callable[[str, tuple[LegalFact, ...]], None] | None = None,
) -> tuple[LegalFact, ...]:
    """Extract legal facts from target_chunks, retrying a failed target before skipping it."""
    chunks_by_id = {chunk.chunk_id: chunk for chunk in document_chunks}
    provision_index = build_provision_index(document_chunks)

    facts: list[LegalFact] = []
    failures: list[LlmTaskFailure] = []
    for target in target_chunks:
        window = build_reading_window(
            target,
            chunks_by_id,
            provision_index,
            max_preceding_chunks=max_preceding_chunks,
            max_following_chunks=max_following_chunks,
        )

        try:
            target_facts = execute_llm_task(
                target.chunk_id,
                partial(_extract_target_facts, client, window, target.chunk_id, metadata),
                max_retries=max_retries,
            )
        except LlmTaskExhaustedError as error:
            failures.append(LlmTaskFailure(task_key=target.chunk_id, error=str(error)))
        else:
            facts.extend(target_facts)
            if on_target_processed is not None:
                on_target_processed(target.chunk_id, target_facts)
    if failures:
        incomplete_error = IncompleteLlmTasksError.for_stage("fact extraction", failures)
        raise incomplete_error
    return tuple(facts)
