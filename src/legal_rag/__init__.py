"""Temporal Ontology-Enhanced RAG for Vietnamese traffic-violation penalties.

A professional, function-oriented package built on top of ``graphiti-core``:

- ``config``      : environment-driven settings.
- ``ingestion``   : Docling document chunks -> time-stamped episodes.
- ``ontology``    : custom legal entity/edge types for LLM extraction.
- ``embeddings``  : Vietnamese (BGE-M3) embedder.
- ``evaluation``  : synthetic ground-truth candidates and evaluation support.
- ``graph``       : Graphiti client construction and ingestion orchestration.
- ``retrieval``   : point-in-time (bi-temporal) querying.
- ``cli``         : command-line entry points.
"""

__version__ = "0.1.0"
