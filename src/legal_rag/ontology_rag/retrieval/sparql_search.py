"""Search ontology directly by keywords via SPARQL."""

from __future__ import annotations

from loguru import logger
from rdflib import Graph

from legal_rag.ontology_rag.models import OntologyInstance


class SPARQLSearch:
    """Query ontology by keywords."""

    def __init__(self, graph: Graph) -> None:
        """Initialize with ontology graph."""
        self._graph = graph

    def search(self, keywords: list[str]) -> list[OntologyInstance]:
        """Find instances matching keywords."""
        if not keywords:
            return []

        filters = " || ".join([f'CONTAINS(LCASE(STR(?label)), "{kw}")' for kw in keywords])
        query = f"""
        PREFIX lornr: <https://w3id.org/legal-ontology-rag/ontology/normative-rule#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?instance ?label WHERE {{
            ?instance a lornr:PenaltyRule ;
                      rdfs:label ?label .
            FILTER({filters})
        }}
        """
        results = self._graph.query(query)
        instances: list[OntologyInstance] = []
        for row in results:
            instance = OntologyInstance(uri=row.instance, type="PenaltyRule", source_chunk_ids=())
            instances.append(instance)
        logger.info("SPARQL search found {} instances for keywords: {}", len(instances), keywords)
        return instances
