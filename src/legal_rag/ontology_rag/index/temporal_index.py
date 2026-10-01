"""Index penalty rules by valid periods for temporal filtering."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum

from loguru import logger
from rdflib import Graph, URIRef


@dataclass(frozen=True, slots=True)
class TemporalEntry:
    """One rule's valid period."""

    rule_uri: URIRef
    valid_from: date | None
    valid_to: date | None


class TemporalValidity(Enum):
    """Certainty-aware validity state for one date."""

    VALID = "valid"
    INVALID = "invalid"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True, slots=True)
class TemporalIndexResolution:
    """Rule URIs partitioned by temporal validity certainty."""

    valid: tuple[URIRef, ...]
    invalid: tuple[URIRef, ...]
    indeterminate: tuple[URIRef, ...]


def classify_validity(valid_from: date | None, valid_to: date | None, query_date: date) -> TemporalValidity:
    """Classify an interval using inclusive start and exclusive end bounds."""
    if valid_from is not None and valid_from > query_date:
        return TemporalValidity.INVALID
    if valid_to is not None and valid_to <= query_date:
        return TemporalValidity.INVALID
    if valid_from is None or valid_to is None:
        return TemporalValidity.INDETERMINATE
    return TemporalValidity.VALID


class TemporalIndex:
    """Penalty rules indexed by valid periods."""

    def __init__(self, graph: Graph) -> None:
        """Build temporal index from ontology graph."""
        self._graph = graph
        self._entries = self._build_index()
        self._entries_by_rule = {entry.rule_uri: entry for entry in self._entries}
        logger.info("Indexed {} rules with temporal information", len(self._entries))

    def _build_index(self) -> list[TemporalEntry]:
        """Query valid periods and build sorted list."""
        query = """
        PREFIX lornr: <https://w3id.org/legal-ontology-rag/ontology/normative-rule#>
        PREFIX lortc: <https://w3id.org/legal-ontology-rag/ontology/temporal-change#>
        PREFIX time: <http://www.w3.org/2006/time#>

        SELECT ?rule ?valid_from ?valid_to WHERE {
            ?rule a lornr:PenaltyRule .
            OPTIONAL {
                ?rule lortc:validDuring ?period .
                OPTIONAL {
                    ?period time:hasBeginning/time:inXSDDateTime ?valid_from .
                }
                OPTIONAL {
                    ?period time:hasEnd/time:inXSDDateTime ?valid_to .
                }
            }
        }
        ORDER BY STR(?rule)
        """
        entries_by_rule: dict[URIRef, TemporalEntry] = {}
        for row in self._graph.query(query):
            rule_uri = row.rule
            valid_from = self._parse_date(str(row.valid_from)) if row.valid_from else None
            valid_to = self._parse_date(str(row.valid_to)) if row.valid_to else None
            entry = TemporalEntry(rule_uri=rule_uri, valid_from=valid_from, valid_to=valid_to)
            previous = entries_by_rule.get(rule_uri)
            if previous is not None and previous != entry:
                msg = f"Penalty rule {rule_uri} has multiple validity intervals"
                raise ValueError(msg)
            entries_by_rule[rule_uri] = entry
        return list(entries_by_rule.values())

    def _parse_date(self, xsd_datetime: str) -> date:
        """Parse XSD dateTime to date."""
        dt_str = xsd_datetime.rstrip("Z")
        if "T" in dt_str:
            dt_str = dt_str.split("T")[0]
        return date.fromisoformat(dt_str)

    def filter_valid(self, rules: Sequence[URIRef], query_date: date) -> list[URIRef]:
        """Return rules whose validity at query_date is fully established."""
        return list(self.partition(rules, query_date).valid)

    def partition(self, rules: Sequence[URIRef], query_date: date) -> TemporalIndexResolution:
        """Partition requested rules into valid, invalid, and indeterminate groups."""
        grouped: dict[TemporalValidity, list[URIRef]] = {status: [] for status in TemporalValidity}
        for rule_uri in rules:
            entry = self._entries_by_rule.get(rule_uri)
            status = (
                TemporalValidity.INDETERMINATE
                if entry is None
                else classify_validity(entry.valid_from, entry.valid_to, query_date)
            )
            grouped[status].append(rule_uri)
        return TemporalIndexResolution(
            valid=tuple(grouped[TemporalValidity.VALID]),
            invalid=tuple(grouped[TemporalValidity.INVALID]),
            indeterminate=tuple(grouped[TemporalValidity.INDETERMINATE]),
        )

    def all_rules(self) -> frozenset[URIRef]:
        """Return all rule URIs in the index."""
        return frozenset(entry.rule_uri for entry in self._entries)
