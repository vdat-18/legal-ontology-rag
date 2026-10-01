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
        if not ontology_paths:
            msg = "At least one ontology Turtle file is required"
            raise ValueError(msg)

        self.graph = Graph()
        for path in ontology_paths:
            if path.suffix.lower() != ".ttl":
                msg = f"Ontology input must be a .ttl file: {path}"
                raise ValueError(msg)
            if not path.is_file():
                msg = f"Ontology file does not exist: {path}"
                raise FileNotFoundError(msg)
            logger.info("Loading ontology graph from {}", path)
            self.graph.parse(str(path.resolve()), format="turtle")
        self._bind_namespaces()
        logger.info("Loaded {} triples from {} graphs", len(self.graph), len(ontology_paths))

    def _bind_namespaces(self) -> None:
        """Bind common namespaces for SPARQL queries."""
        self.graph.bind("lorsp", LORSP)
        self.graph.bind("loroc", LOROC)
        self.graph.bind("lortc", LORTC)
        self.graph.bind("lornr", LORNR)
        self.graph.bind("lorlc", LORLC)
