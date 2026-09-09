"""Link retrieved chunks to ontology instances."""

from __future__ import annotations

from loguru import logger
from rdflib import URIRef

from legal_rag.naive_rag.retriever import RetrievedChunk
from legal_rag.ontology_rag.index.chunk_index import ChunkIndex
from legal_rag.ontology_rag.models import OntologyInstance


class ChunkLinker:
    """Map retrieved chunks to ontology instances."""

    def __init__(self, chunk_index: ChunkIndex) -> None:
        """Initialize with chunk index."""
        self._chunk_index = chunk_index

    def link(self, chunks: list[RetrievedChunk]) -> list[OntologyInstance]:
        """Link chunks to their ontology instances."""
        instances: list[OntologyInstance] = []
        for chunk in chunks:
            instance_uris = self._chunk_index.get_instances(chunk.chunk_id)
            for uri in instance_uris:
                instance = OntologyInstance(uri=uri, type=self._get_type(uri), source_chunk_ids=(chunk.chunk_id,))
                instances.append(instance)
        logger.debug("Linked {} chunks to {} instances", len(chunks), len(instances))
        return instances

    def _get_type(self, uri: URIRef) -> str:
        """Extract type from URI fragment."""
        uri_str = str(uri)
        if "PenaltyRule" in uri_str or "/rule/" in uri_str:
            return "PenaltyRule"
        if "/condition/" in uri_str:
            return "Condition"
        if "/consequence/" in uri_str:
            return "Consequence"
        return "Unknown"
