"""Retrieve full penalty rule objects from instance URIs."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from decimal import Decimal

from loguru import logger
from rdflib import Graph, URIRef

from legal_rag.ontology_rag.models import PenaltyRule


class InstanceRetriever:
    """Fetch full PenaltyRule objects from ontology."""

    def __init__(self, graph: Graph) -> None:
        """Initialize with ontology graph."""
        self._graph = graph

    def retrieve_batch(self, uris: Sequence[URIRef]) -> list[PenaltyRule]:
        """Retrieve multiple penalty rules."""
        rules: list[PenaltyRule] = []
        for uri in uris:
            rule = self._retrieve_one(uri)
            if rule:
                rules.append(rule)
        logger.info("Retrieved {} full penalty rules", len(rules))
        return rules

    def _retrieve_one(self, uri: URIRef) -> PenaltyRule | None:
        """Retrieve one penalty rule with all properties."""
        query = f"""
        PREFIX lornr: <https://w3id.org/legal-ontology-rag/ontology/normative-rule#>
        PREFIX lorlc: <https://w3id.org/legal-ontology-rag/ontology/legal-consequence#>
        PREFIX lortc: <https://w3id.org/legal-ontology-rag/ontology/temporal-change#>
        PREFIX lorsp: <https://w3id.org/legal-ontology-rag/ontology/source-provenance#>
        PREFIX time: <http://www.w3.org/2006/time#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

        SELECT ?conduct_label ?min_amount ?max_amount ?currency ?valid_from ?valid_to
               ?point_value ?provision_path
        WHERE {{
            <{uri}> lornr:hasConductSpecification ?conductSpec .
            ?conductSpec lornr:usesConductConcept ?conduct .
            ?conduct skos:prefLabel ?conduct_label .
            <{uri}> lorsp:groundedIn ?provision .
            ?provision lorsp:structuralPath ?provision_path .
            OPTIONAL {{
                <{uri}> lorlc:imposesConsequence ?consequence .
                ?consequence a lorlc:MonetaryPenalty ;
                             lorlc:hasAmountRange ?range .
                ?range lorlc:minimumAmount ?min_amount ;
                       lorlc:maximumAmount ?max_amount ;
                       lorlc:hasCurrency ?currency .
            }}
            OPTIONAL {{
                <{uri}> lorlc:imposesConsequence ?points .
                ?points a lorlc:PointDeduction ;
                        lorlc:pointValue ?point_value .
            }}
            <{uri}> lortc:validDuring ?period .
            ?period time:hasBeginning/time:inXSDDateTime ?valid_from .
            OPTIONAL {{
                ?period time:hasEnd/time:inXSDDateTime ?valid_to .
            }}
        }}
        LIMIT 1
        """

        result = self._graph.query(query)
        if not result:
            return None

        row = next(iter(result))
        return PenaltyRule(
            uri=uri,
            conduct_label_vi=str(row.conduct_label),
            vehicle_categories=(),
            subject_kind=None,
            roles=(),
            monetary_penalty_min=Decimal(str(row.min_amount)) if row.min_amount else None,
            monetary_penalty_max=Decimal(str(row.max_amount)) if row.max_amount else None,
            currency=str(row.currency) if row.currency else None,
            point_value=int(row.point_value) if row.point_value else None,
            valid_from=self._parse_date(str(row.valid_from)),
            valid_to=self._parse_date(str(row.valid_to)) if row.valid_to else None,
            source_chunk_ids=(),
            provision_path=str(row.provision_path),
            document_number="",
        )

    def _parse_date(self, xsd_datetime: str) -> date:
        """Parse XSD dateTime to date."""
        dt_str = xsd_datetime.rstrip("Z")
        if "T" in dt_str:
            dt_str = dt_str.split("T")[0]
        return date.fromisoformat(dt_str)
