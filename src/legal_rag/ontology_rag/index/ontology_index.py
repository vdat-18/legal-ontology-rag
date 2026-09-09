"""Load and index ontology graphs for fast retrieval."""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from rdflib import Graph, Namespace

LORSP = Namespace("https://w3id.org/legal-ontology-rag/ontology/source-provenance#")
LOROC = Namespace("https://w3id.org/legal-ontology-rag/ontology/occurrence-context#")
LORTC = Namespace("https://w3id.org/legal-ontology-rag/ontology/temporal-change#")
LORNR = Namespace("https://w3id.org/legal-ontology-rag/ontology/normative-rule#")
LORLC = Namespace("https://w3id.org/legal-ontology-rag/ontology/legal-consequence#")


class OntologyIndex:
    """Load and cache ontology graphs."""

    def __init__(self, ontology_paths: list[Path]) -> None:
        """Load ontology graphs from paths."""
        self.graph = Graph()
        for path in ontology_paths:
            if path.suffix == ".ttl":
                if not path.exists():
                    logger.warning("Ontology file does not exist: {}", path)
                    continue
                logger.info("Loading ontology graph from {}", path)
                with path.open("rb") as f:
                    self.graph.parse(f, format="turtle")
        self._bind_namespaces()
        logger.info("Loaded {} triples from {} graphs", len(self.graph), len(ontology_paths))

    def _bind_namespaces(self) -> None:
        """Bind common namespaces for SPARQL queries."""
        self.graph.bind("lorsp", LORSP)
        self.graph.bind("loroc", LOROC)
        self.graph.bind("lortc", LORTC)
        self.graph.bind("lornr", LORNR)
        self.graph.bind("lorlc", LORLC)
