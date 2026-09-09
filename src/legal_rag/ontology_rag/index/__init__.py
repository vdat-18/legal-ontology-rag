"""Ontology index modules for fast retrieval."""

from legal_rag.ontology_rag.index.chunk_index import ChunkIndex
from legal_rag.ontology_rag.index.ontology_index import OntologyIndex
from legal_rag.ontology_rag.index.temporal_index import TemporalIndex

__all__ = ["ChunkIndex", "OntologyIndex", "TemporalIndex"]
