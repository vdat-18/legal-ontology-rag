"""Build reading windows and extract legal facts from chunk text."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

import numpy as np
from baml_py.logging import set_log_level

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import ChunkWindowEntry, ChunkWindowRole
from legal_rag.baml_client.types import LegalFact as BamlLegalFact
from legal_rag.baml_transport import build_client_registry, execute_baml_request
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.chunker.models import Chunk
from legal_rag.embedding.encoder import EmbeddingEncoder
from legal_rag.parser.hashing import sha256_json
from legal_rag.settings import EvaluationSettings, Settings

# Small lexical parsing task, not a semantic decision: locate "Điều N" mentions
# to resolve explicit cross-references between chunks of the same document.
_ARTICLE_HEADING_PATTERN = re.compile(r"\s*Điều\s+(\d+)\b")
_ARTICLE_CITATION_PATTERN = re.compile(r"Điều\s+(\d+)\b")

DEFAULT_MAX_EMBEDDING_CANDIDATES = 3
DEFAULT_MAX_PRECEDING_CHUNKS = 2


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


def build_reading_window(
    target: Chunk,
    chunks_by_id: Mapping[str, Chunk],
    provision_index: ProvisionIndex,
    embedding_candidates: Sequence[Chunk] = (),
    *,
    max_preceding_chunks: int = DEFAULT_MAX_PRECEDING_CHUNKS,
) -> tuple[ChunkWindowEntry, ...]:
    """Assemble one reading window around a target chunk."""
    entries: dict[str, ChunkWindowEntry] = {
        target.chunk_id: ChunkWindowEntry(
            chunk_id=target.chunk_id, role=ChunkWindowRole.Target, provision_path=None, text=target.text
        )
    }
    preceding_id = target.previous_chunk_id
    for _ in range(max_preceding_chunks):
        if preceding_id is None or preceding_id in entries:
            break
        preceding_chunk = chunks_by_id.get(preceding_id)
        if preceding_chunk is None:
            break
        entries[preceding_id] = ChunkWindowEntry(
            chunk_id=preceding_id, role=ChunkWindowRole.Adjacent, provision_path=None, text=preceding_chunk.text
        )
        preceding_id = preceding_chunk.previous_chunk_id

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

    for candidate in embedding_candidates:
        if candidate.chunk_id in entries:
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
    """Assign identity to one LLM-produced fact, trusting its cited provenance."""
    evidence = tuple(
        EvidenceReference(
            chunk_id=entry.chunk_id,
            quoted_span=entry.quoted_span,
            unit_ids=chunks_by_id[entry.chunk_id].unit_ids,
            source_refs=chunks_by_id[entry.chunk_id].source_refs,
        )
        for entry in raw.evidence
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
        evidence=evidence,
    )


def extract_facts_from_document(
    chunks: Sequence[Chunk],
    client: FactExtractionClient,
    encoder: EmbeddingEncoder,
    *,
    max_embedding_candidates: int = DEFAULT_MAX_EMBEDDING_CANDIDATES,
    max_preceding_chunks: int = DEFAULT_MAX_PRECEDING_CHUNKS,
) -> tuple[LegalFact, ...]:
    """Extract legal facts from every chunk of one document."""
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    provision_index = build_provision_index(chunks)
    vectors = encoder.encode([chunk.text for chunk in chunks])
    vectors_by_chunk_id = {chunk.chunk_id: vectors[index] for index, chunk in enumerate(chunks)}

    facts: list[LegalFact] = []
    for target in chunks:
        embedding_candidates = find_embedding_candidates(
            target, chunks, vectors_by_chunk_id, max_candidates=max_embedding_candidates
        )
        window = build_reading_window(
            target,
            chunks_by_id,
            provision_index,
            embedding_candidates,
            max_preceding_chunks=max_preceding_chunks,
        )
        facts.extend(build_legal_fact(raw_fact, chunks_by_id) for raw_fact in client.extract(window))

    return tuple(facts)
