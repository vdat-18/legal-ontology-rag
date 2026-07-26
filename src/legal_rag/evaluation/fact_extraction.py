"""Build bounded reading windows and extract legal facts from chunk text."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

import numpy as np
from baml_py.logging import set_log_level
from loguru import logger

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import ChunkWindowEntry, ChunkWindowRole
from legal_rag.baml_client.types import LegalFact as BamlLegalFact
from legal_rag.baml_transport import build_client_registry, execute_baml_request
from legal_rag.chunker.models import Chunk
from legal_rag.embedding.encoder import EmbeddingEncoder
from legal_rag.evaluation.models import EvidenceReference, LegalFact
from legal_rag.parser.hashing import sha256_json
from legal_rag.settings import EvaluationSettings, Settings

# Small lexical parsing task, not a semantic decision: locate "Điều N" mentions
# to resolve explicit cross-references between chunks of the same document.
_ARTICLE_HEADING_PATTERN = re.compile(r"\s*Điều\s+(\d+)\b")
_ARTICLE_CITATION_PATTERN = re.compile(r"Điều\s+(\d+)\b")

DEFAULT_MAX_EMBEDDING_CANDIDATES = 3
DEFAULT_MAX_WINDOW_CHUNKS = 6


class FactExtractionError(ValueError):
    """Raised when LLM-produced fact output fails validation against real provenance."""

    @classmethod
    def no_evidence(cls) -> FactExtractionError:
        """Build an error for a fact with no supporting evidence entries."""
        return cls("Legal fact has no evidence entries")

    @classmethod
    def evidence_outside_window(cls, chunk_id: str) -> FactExtractionError:
        """Build an error for evidence citing a chunk outside the reading window."""
        return cls(f"Evidence cites chunk_id {chunk_id!r} outside the reading window")


@dataclass(frozen=True, slots=True)
class ProvisionIndex:
    """Maps an article number to the chunk(s) whose text opens that article."""

    chunk_ids_by_article: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class FactExtractionRejection:
    """One LLM-produced fact dropped during validation, with an auditable reason."""

    target_chunk_id: str
    reason: str


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


def build_reading_window(
    target: Chunk,
    chunks_by_id: Mapping[str, Chunk],
    provision_index: ProvisionIndex,
    embedding_candidates: Sequence[Chunk] = (),
    *,
    max_window_chunks: int = DEFAULT_MAX_WINDOW_CHUNKS,
) -> tuple[ChunkWindowEntry, ...]:
    """Assemble one bounded reading window around a target chunk."""
    entries: dict[str, ChunkWindowEntry] = {
        target.chunk_id: ChunkWindowEntry(
            chunk_id=target.chunk_id, role=ChunkWindowRole.Target, provision_path=None, text=target.text
        )
    }
    for adjacent_id in (target.previous_chunk_id, target.next_chunk_id):
        if adjacent_id is None or adjacent_id in entries:
            continue
        adjacent = chunks_by_id.get(adjacent_id)
        if adjacent is not None:
            entries[adjacent_id] = ChunkWindowEntry(
                chunk_id=adjacent_id, role=ChunkWindowRole.Adjacent, provision_path=None, text=adjacent.text
            )

    for article in sorted(cited_articles(target)):
        for chunk_id in provision_index.chunk_ids_by_article.get(article, ()):
            if chunk_id in entries or len(entries) >= max_window_chunks:
                continue
            cited_chunk = chunks_by_id[chunk_id]
            entries[chunk_id] = ChunkWindowEntry(
                chunk_id=chunk_id,
                role=ChunkWindowRole.CitedProvision,
                provision_path=f"Điều {article}",
                text=cited_chunk.text,
            )

    for candidate in embedding_candidates:
        if candidate.chunk_id in entries or len(entries) >= max_window_chunks:
            continue
        entries[candidate.chunk_id] = ChunkWindowEntry(
            chunk_id=candidate.chunk_id,
            role=ChunkWindowRole.EmbeddingCandidate,
            provision_path=None,
            text=candidate.text,
        )

    return tuple(entries.values())


def find_embedding_candidates(
    target: Chunk,
    same_document_chunks: Sequence[Chunk],
    vectors_by_chunk_id: Mapping[str, np.ndarray],
    *,
    max_candidates: int = DEFAULT_MAX_EMBEDDING_CANDIDATES,
) -> tuple[Chunk, ...]:
    """Return the most similar chunks to a target within the same document."""
    target_vector = vectors_by_chunk_id[target.chunk_id]
    scored = [
        (float(np.dot(target_vector, vectors_by_chunk_id[chunk.chunk_id])), chunk)
        for chunk in same_document_chunks
        if chunk.chunk_id != target.chunk_id
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return tuple(chunk for _score, chunk in scored[:max_candidates])


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


def build_legal_fact(raw: BamlLegalFact, chunks_by_id: Mapping[str, Chunk]) -> LegalFact:
    """Validate one LLM-produced fact against real chunk provenance and assign identity.

    Raises FactExtractionError when the model cited a chunk outside the window
    or produced an unparsable date; model output is untrusted and must be
    validated before it becomes a persisted record.
    """
    if not raw.evidence:
        raise FactExtractionError.no_evidence()

    evidence: list[EvidenceReference] = []
    for entry in raw.evidence:
        chunk = chunks_by_id.get(entry.chunk_id)
        if chunk is None:
            raise FactExtractionError.evidence_outside_window(entry.chunk_id)
        evidence.append(
            EvidenceReference(
                chunk_id=entry.chunk_id,
                quoted_span=entry.quoted_span,
                unit_ids=chunk.unit_ids,
                source_refs=chunk.source_refs,
            )
        )

    valid_from = date.fromisoformat(raw.valid_from) if raw.valid_from else None
    valid_to = date.fromisoformat(raw.valid_to) if raw.valid_to else None

    identity = {
        "conduct": raw.conduct,
        "subject": raw.subject,
        "document_number": raw.document_number,
        "provision_path": raw.provision_path,
    }
    content = {
        **identity,
        "conditions": list(raw.conditions),
        "monetary_penalty": raw.monetary_penalty,
        "license_points_deducted": raw.license_points_deducted,
        "additional_penalties": list(raw.additional_penalties),
        "remedial_measures": list(raw.remedial_measures),
        "valid_from": raw.valid_from,
        "valid_to": raw.valid_to,
        "evidence": [entry.to_dict() for entry in evidence],
    }

    return LegalFact(
        fact_id=sha256_json(identity),
        fact_content_sha256=sha256_json(content),
        conduct=raw.conduct,
        subject=raw.subject,
        conditions=tuple(raw.conditions),
        monetary_penalty=raw.monetary_penalty,
        license_points_deducted=raw.license_points_deducted,
        additional_penalties=tuple(raw.additional_penalties),
        remedial_measures=tuple(raw.remedial_measures),
        valid_from=valid_from,
        valid_to=valid_to,
        document_number=raw.document_number,
        provision_path=raw.provision_path,
        evidence=tuple(evidence),
    )


def extract_facts_from_document(
    chunks: Sequence[Chunk],
    client: FactExtractionClient,
    encoder: EmbeddingEncoder,
    *,
    max_embedding_candidates: int = DEFAULT_MAX_EMBEDDING_CANDIDATES,
    max_window_chunks: int = DEFAULT_MAX_WINDOW_CHUNKS,
) -> tuple[tuple[LegalFact, ...], tuple[FactExtractionRejection, ...]]:
    """Extract validated legal facts from every chunk of one document."""
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    provision_index = build_provision_index(chunks)
    vectors = encoder.encode([chunk.text for chunk in chunks])
    vectors_by_chunk_id = {chunk.chunk_id: vectors[index] for index, chunk in enumerate(chunks)}

    facts: list[LegalFact] = []
    rejections: list[FactExtractionRejection] = []
    for target in chunks:
        embedding_candidates = find_embedding_candidates(
            target, chunks, vectors_by_chunk_id, max_candidates=max_embedding_candidates
        )
        window = build_reading_window(
            target, chunks_by_id, provision_index, embedding_candidates, max_window_chunks=max_window_chunks
        )
        for raw_fact in client.extract(window):
            try:
                facts.append(build_legal_fact(raw_fact, chunks_by_id))
            except FactExtractionError as error:
                logger.warning("Dropped candidate fact from chunk {}: {}", target.chunk_id, error)
                rejections.append(FactExtractionRejection(target_chunk_id=target.chunk_id, reason=str(error)))

    return tuple(facts), tuple(rejections)
