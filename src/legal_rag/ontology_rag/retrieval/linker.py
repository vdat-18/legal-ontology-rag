"""Link retrieved chunks to ontology instances."""

from __future__ import annotations

from loguru import logger
from rdflib import URIRef

from legal_rag.naive_rag.models import RetrievedChunk
from legal_rag.ontology_rag.index.chunk_index import ChunkIndex
from legal_rag.ontology_rag.models import OntologyInstance


class ChunkLinker:
    """Map retrieved chunks to ontology instances."""

    def __init__(self, chunk_index: ChunkIndex) -> None:
        """Initialize with chunk index."""
        self._chunk_index = chunk_index

    def link(self, chunks: list[RetrievedChunk]) -> list[OntologyInstance]:
        """Link ranked chunks to unique, explicitly typed penalty rules."""
        source_ids_by_uri: dict[URIRef, list[str]] = {}
        for chunk in chunks:
            for chunk_id in chunk.effective_evidence_chunk_ids:
                for uri in self._chunk_index.get_instances(chunk_id):
                    source_ids_by_uri.setdefault(uri, []).append(chunk_id)
        instances = [
            OntologyInstance(
                uri=uri,
                type="PenaltyRule",
                source_chunk_ids=tuple(dict.fromkeys(source_chunk_ids)),
            )
            for uri, source_chunk_ids in source_ids_by_uri.items()
        ]
        logger.debug("Linked {} chunks to {} instances", len(chunks), len(instances))
        return instances
