"""Index penalty rules by valid periods for temporal filtering."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from loguru import logger
from rdflib import Graph, URIRef


@dataclass(frozen=True, slots=True)
class TemporalEntry:
    """One rule's valid period."""

    rule_uri: URIRef
    valid_from: date
    valid_to: date | None


class TemporalIndex:
    """Penalty rules indexed by valid periods."""

    def __init__(self, graph: Graph) -> None:
        """Build temporal index from ontology graph."""
        self._graph = graph
        self._entries = self._build_index()
        logger.info("Indexed {} rules with temporal information", len(self._entries))

    def _build_index(self) -> list[TemporalEntry]:
        """Query valid periods and build sorted list."""
        query = """
        PREFIX lornr: <https://w3id.org/legal-ontology-rag/ontology/normative-rule#>
        PREFIX lortc: <https://w3id.org/legal-ontology-rag/ontology/temporal-change#>
        PREFIX time: <http://www.w3.org/2006/time#>

        SELECT ?rule ?valid_from (MAX(?valid_to) AS ?valid_to_max) WHERE {
            ?rule a lornr:PenaltyRule ;
                  lortc:validDuring ?period .
            ?period time:hasBeginning/time:inXSDDateTime ?valid_from .
            OPTIONAL {
                ?period time:hasEnd/time:inXSDDateTime ?valid_to .
            }
        }
        GROUP BY ?rule ?valid_from
        """
        entries: list[TemporalEntry] = []
        for row in self._graph.query(query):
            rule_uri = row.rule
            valid_from = self._parse_date(str(row.valid_from))
            valid_to = self._parse_date(str(row.valid_to_max)) if row.valid_to_max else None
            entries.append(TemporalEntry(rule_uri=rule_uri, valid_from=valid_from, valid_to=valid_to))
        entries.sort(key=lambda e: e.valid_from)
        return entries

    def _parse_date(self, xsd_datetime: str) -> date:
        """Parse XSD dateTime to date."""
        dt_str = xsd_datetime.rstrip("Z")
        if "T" in dt_str:
            dt_str = dt_str.split("T")[0]
        return date.fromisoformat(dt_str)

    def filter_valid(self, rules: Sequence[URIRef], query_date: date) -> list[URIRef]:
        """Return rules valid at query_date."""
        valid = []
        for entry in self._entries:
            if entry.rule_uri not in rules:
                continue
            if entry.valid_from > query_date:
                continue
            if entry.valid_to and entry.valid_to < query_date:
                continue
            valid.append(entry.rule_uri)
        return valid

    def all_rules(self) -> frozenset[URIRef]:
        """Return all rule URIs in the index."""
        return frozenset(entry.rule_uri for entry in self._entries)
