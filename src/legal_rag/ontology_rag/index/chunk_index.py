"""Index mapping chunk IDs to ontology instance URIs."""

from __future__ import annotations

from collections import defaultdict

from loguru import logger
from rdflib import Graph, URIRef

ONTOLOGY_NS = "https://w3id.org/legal-ontology-rag/ontology/source-provenance#"
DCTERMS_NS = "http://purl.org/dc/terms/"


class ChunkIndex:
    """Pre-computed mapping from chunk_id to list of instance URIs."""

    def __init__(self, graph: Graph) -> None:
        """Build chunk-to-instances index from ontology graph."""
        self._graph = graph
        self._index = self._build_index()
        logger.info(
            "Indexed {} chunks linked to {} instances", len(self._index), sum(len(v) for v in self._index.values())
        )

    def _build_index(self) -> dict[str, list[URIRef]]:
        """Query sourcedFromChunk to build the mapping."""
        query = """
        PREFIX lorsp: <https://w3id.org/legal-ontology-rag/ontology/source-provenance#>
        PREFIX dcterms: <http://purl.org/dc/terms/>

        SELECT ?instance ?chunk_id WHERE {
            ?instance lorsp:sourcedFromChunk ?chunk .
            ?chunk dcterms:identifier ?chunk_id .
        }
        """
        index: dict[str, list[URIRef]] = defaultdict(list)
        for row in self._graph.query(query):
            chunk_id = str(row.chunk_id)
            instance_uri = row.instance
            index[chunk_id].append(instance_uri)
        return dict(index)

    def get_instances(self, chunk_id: str) -> list[URIRef]:
        """Return list of instance URIs for a chunk ID."""
        return self._index.get(chunk_id, [])

    def has_instance(self, chunk_id: str) -> bool:
        """Check if a chunk has any linked ontology instances."""
        return chunk_id in self._index

    def all_chunk_ids(self) -> frozenset[str]:
        """Return all chunk IDs that have linked instances."""
        return frozenset(self._index.keys())
