"""Index mapping chunk IDs to penalty-rule URIs."""

from __future__ import annotations

from collections import defaultdict

from loguru import logger
from rdflib import DCTERMS, RDF, Graph, Namespace, URIRef

LORLC = Namespace("https://w3id.org/legal-ontology-rag/ontology/legal-consequence#")
LORNR = Namespace("https://w3id.org/legal-ontology-rag/ontology/normative-rule#")
LORSP = Namespace("https://w3id.org/legal-ontology-rag/ontology/source-provenance#")


class ChunkIndex:
    """Pre-computed mapping from chunk IDs to evidenced penalty rules."""

    def __init__(self, graph: Graph) -> None:
        """Build chunk-to-instances index from ontology graph."""
        self._graph = graph
        self._index = self._build_index()
        logger.info(
            "Indexed {} chunks linked to {} instances", len(self._index), sum(len(v) for v in self._index.values())
        )

    def _build_index(self) -> dict[str, list[URIRef]]:
        """Build the mapping from explicitly typed penalty rules."""
        index: dict[str, set[URIRef]] = defaultdict(set)
        rules = sorted(set(self._graph.subjects(RDF.type, LORNR.PenaltyRule)), key=str)
        for rule in rules:
            for component in self._rule_components(rule):
                for chunk in self._graph.objects(component, LORSP.sourcedFromChunk):
                    for chunk_id in self._graph.objects(chunk, DCTERMS.identifier):
                        index[str(chunk_id)].add(rule)
        return {chunk_id: sorted(instance_uris, key=str) for chunk_id, instance_uris in sorted(index.items())}

    def _rule_components(self, rule: URIRef) -> set[URIRef]:
        """Return one rule and the evidenced components it owns."""
        components = {rule}
        predicates = (
            LORNR.hasConductSpecification,
            LORNR.hasApplicability,
            LORNR.hasException,
            LORLC.imposesConsequence,
        )
        for predicate in predicates:
            components.update(self._graph.objects(rule, predicate))

        pending = [
            root
            for applicability in self._graph.objects(rule, LORNR.hasApplicability)
            for root in self._graph.objects(applicability, LORNR.hasConditionExpression)
        ]
        visited_conditions: set[URIRef] = set()
        while pending:
            condition = pending.pop()
            if condition in visited_conditions:
                continue
            visited_conditions.add(condition)
            components.add(condition)
            pending.extend(self._graph.objects(condition, LORNR.hasOperand))
        return components

    def get_instances(self, chunk_id: str) -> list[URIRef]:
        """Return list of instance URIs for a chunk ID."""
        return self._index.get(chunk_id, [])

    def has_instance(self, chunk_id: str) -> bool:
        """Check if a chunk has any linked ontology instances."""
        return chunk_id in self._index

    def all_chunk_ids(self) -> frozenset[str]:
        """Return all chunk IDs that have linked instances."""
        return frozenset(self._index.keys())
