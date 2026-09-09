"""Ontology retrieval modules."""

from legal_rag.ontology_rag.retrieval.instance_retriever import InstanceRetriever
from legal_rag.ontology_rag.retrieval.linker import ChunkLinker
from legal_rag.ontology_rag.retrieval.sparql_search import SPARQLSearch
from legal_rag.ontology_rag.retrieval.temporal_resolver import TemporalResolver

__all__ = ["ChunkLinker", "InstanceRetriever", "SPARQLSearch", "TemporalResolver"]
