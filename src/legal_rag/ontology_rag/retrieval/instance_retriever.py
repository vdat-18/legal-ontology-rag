"""Retrieve full penalty rule objects from instance URIs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date
from decimal import Decimal
from typing import cast

from loguru import logger
from rdflib import DCTERMS, RDF, RDFS, SKOS, TIME, Graph, Literal, Namespace, URIRef

from legal_rag.ontology_rag.models import (
    ComponentProvenance,
    ConditionExpression,
    ConditionKind,
    ConsequenceKind,
    LegalConsequence,
    LegalException,
    OntologyTerm,
    PenaltyRule,
    ProvisionCitation,
)

LORSP = Namespace("https://w3id.org/legal-ontology-rag/ontology/source-provenance#")
LOROC = Namespace("https://w3id.org/legal-ontology-rag/ontology/occurrence-context#")
LORTC = Namespace("https://w3id.org/legal-ontology-rag/ontology/temporal-change#")
LORNR = Namespace("https://w3id.org/legal-ontology-rag/ontology/normative-rule#")
LORLC = Namespace("https://w3id.org/legal-ontology-rag/ontology/legal-consequence#")

_CONDITION_TYPES = {
    LORNR.AtomicCondition: "Atomic",
    LORNR.AllOf: "AllOf",
    LORNR.AnyOf: "AnyOf",
    LORNR.Not: "Not",
}
_CONSEQUENCE_TYPES = {
    LORLC.MonetaryPenalty: "MonetaryPenalty",
    LORLC.Warning: "Warning",
    LORLC.PointDeduction: "PointDeduction",
    LORLC.AdditionalSanction: "AdditionalSanction",
    LORLC.RemedialMeasure: "RemedialMeasure",
    LORLC.ReferencedPenalty: "ReferencedPenalty",
}
_SUBJECT_KINDS = {
    LOROC.naturalPerson: "NaturalPerson",
    LOROC.organization: "Organization",
}


class _OntologyContractError(ValueError):
    """Report an invalid cardinality or relationship in the RDF data contract."""

    def __init__(self, subject: URIRef, field: str, actual_count: int | None = None) -> None:
        detail = f" (found {actual_count})" if actual_count is not None else ""
        super().__init__(f"Invalid {field} for {subject}{detail}")


class InstanceRetriever:
    """Fetch full PenaltyRule objects from ontology."""

    def __init__(self, graph: Graph) -> None:
        """Initialize with ontology graph."""
        self._graph = graph

    def retrieve_batch(self, uris: Sequence[URIRef]) -> list[PenaltyRule]:
        """Retrieve multiple penalty rules, preserving first-seen URI order."""
        rules: list[PenaltyRule] = []
        for uri in dict.fromkeys(uris):
            rule = self._retrieve_one(uri)
            if rule:
                rules.append(rule)
        logger.info("Retrieved {} full penalty rules", len(rules))
        return rules

    def retrieve(self, uri: URIRef) -> PenaltyRule | None:
        """Retrieve one penalty rule."""
        return self._retrieve_one(uri)

    def _retrieve_one(self, uri: URIRef) -> PenaltyRule | None:
        """Retrieve one penalty rule without flattening multi-valued RDF relations."""
        if (uri, RDF.type, LORNR.PenaltyRule) not in self._graph:
            return None

        conduct_specifications = self._uri_objects(uri, LORNR.hasConductSpecification)
        conduct_concepts = tuple(
            concept
            for specification in conduct_specifications
            for concept in self._uri_objects(specification, LORNR.usesConductConcept)
        )
        conduct_label = next(
            (label for concept in conduct_concepts if (label := self._preferred_label(concept)) is not None),
            self._preferred_label(uri) or self._local_name(uri),
        )
        conduct_provenance = tuple(self._component_provenance(item) for item in conduct_specifications)

        applicability_profiles = self._uri_objects(uri, LORNR.hasApplicability)
        applicability_provenance = tuple(self._component_provenance(item) for item in applicability_profiles)
        subject_kinds = self._dedupe_sorted(
            self._subject_kind(value)
            for profile in applicability_profiles
            for value in self._uri_objects(profile, LORNR.requiresSubjectKind)
        )
        role_terms = self._terms(
            value for profile in applicability_profiles for value in self._uri_objects(profile, LORNR.requiresRole)
        )
        vehicle_terms = self._terms(
            value
            for profile in applicability_profiles
            for value in self._uri_objects(profile, LORNR.requiresVehicleCategory)
        )
        condition_roots = tuple(
            value
            for profile in applicability_profiles
            for value in self._uri_objects(profile, LORNR.hasConditionExpression)
        )
        condition_expressions = tuple(
            self._condition_expression(root, frozenset()) for root in self._dedupe_uris(condition_roots)
        )

        consequences = tuple(
            self._legal_consequence(value) for value in self._uri_objects(uri, LORLC.imposesConsequence)
        )
        exceptions = tuple(self._legal_exception(value) for value in self._uri_objects(uri, LORNR.hasException))
        rule_provenance = self._component_provenance(uri)
        all_provenance = [
            rule_provenance,
            *conduct_provenance,
            *applicability_provenance,
            *(item.provenance for item in consequences),
            *(item.provenance for item in exceptions),
            *(item for root in condition_expressions for item in self._condition_provenance(root)),
        ]
        source_chunk_ids = self._dedupe_sorted(
            chunk_id for provenance in all_provenance for chunk_id in provenance.source_chunk_ids
        )

        monetary = next((item for item in consequences if item.kind == "MonetaryPenalty"), None)
        points = next((item for item in consequences if item.kind == "PointDeduction"), None)
        valid_from, valid_to = self._validity_dates(uri)
        provision_path = rule_provenance.provision_paths[0] if rule_provenance.provision_paths else ""

        return PenaltyRule(
            uri=uri,
            conduct_label_vi=conduct_label,
            vehicle_categories=tuple(self._term_text(item) for item in vehicle_terms),
            subject_kind=subject_kinds[0] if subject_kinds else None,
            roles=tuple(self._term_text(item) for item in role_terms),
            monetary_penalty_min=monetary.minimum_amount if monetary else None,
            monetary_penalty_max=monetary.maximum_amount if monetary else None,
            currency=monetary.currency if monetary else None,
            point_value=points.point_value if points else None,
            valid_from=valid_from,
            valid_to=valid_to,
            source_chunk_ids=source_chunk_ids,
            provision_path=provision_path,
            document_number=self._document_number(rule_provenance.provision_uris),
            consequences=consequences,
            condition_expressions=condition_expressions,
            exceptions=exceptions,
            rule_provenance=rule_provenance,
            conduct_provenance=conduct_provenance,
            applicability_provenance=applicability_provenance,
        )

    def _condition_expression(
        self,
        uri: URIRef,
        active: frozenset[URIRef],
    ) -> ConditionExpression:
        """Build one condition subtree and reject cycles explicitly."""
        if uri in active:
            raise _OntologyContractError(uri, "cyclic condition expression")
        kinds = [kind for rdf_type, kind in _CONDITION_TYPES.items() if (uri, RDF.type, rdf_type) in self._graph]
        if len(kinds) != 1:
            raise _OntologyContractError(uri, "condition expression kind", len(kinds))
        operands = tuple(
            self._condition_expression(operand, active | {uri}) for operand in self._uri_objects(uri, LORNR.hasOperand)
        )
        return ConditionExpression(
            uri=uri,
            kind=cast(ConditionKind, kinds[0]),
            label_vi=self._preferred_label(uri),
            operands=operands,
            provenance=self._component_provenance(uri),
        )

    def _legal_consequence(self, uri: URIRef) -> LegalConsequence:
        """Build one typed consequence with its subtype-specific values."""
        kinds = [kind for rdf_type, kind in _CONSEQUENCE_TYPES.items() if (uri, RDF.type, rdf_type) in self._graph]
        if len(kinds) != 1:
            raise _OntologyContractError(uri, "legal consequence kind", len(kinds))
        kind = cast(ConsequenceKind, kinds[0])
        amount_range = self._single_uri(uri, LORLC.hasAmountRange, "amount range")
        currency_uri = self._single_uri(amount_range, LORLC.currency, "currency") if amount_range is not None else None
        return LegalConsequence(
            uri=uri,
            kind=kind,
            label_vi=self._preferred_label(uri),
            minimum_amount=self._decimal_value(amount_range, LORLC.minimumAmount),
            maximum_amount=self._decimal_value(amount_range, LORLC.maximumAmount),
            currency=self._local_name(currency_uri) if currency_uri is not None else None,
            point_value=self._integer_value(uri, LORLC.pointValue),
            minimum_duration_months=self._integer_value(uri, LORLC.minimumDurationMonths),
            maximum_duration_months=self._integer_value(uri, LORLC.maximumDurationMonths),
            affected_targets=self._terms(self._uri_objects(uri, LORLC.affectsTarget)),
            responsible_roles=self._terms(self._uri_objects(uri, LORLC.mustBePerformedBy)),
            provenance=self._component_provenance(uri),
        )

    def _legal_exception(self, uri: URIRef) -> LegalException:
        """Build one legal exception with grouped citations and redirects."""
        citations = tuple(
            self._provision_citation(value) for value in self._uri_objects(uri, LORNR.hasProvisionCitation)
        )
        excluded = self._uri_objects(uri, LORNR.citesExcludedProvision)
        return LegalException(
            uri=uri,
            citations=citations,
            excluded_provision_uris=excluded,
            excluded_provision_paths=self._provision_paths(excluded),
            redirected_rule_uris=self._uri_objects(uri, LORNR.redirectsTo),
            provenance=self._component_provenance(uri),
        )

    def _provision_citation(self, uri: URIRef) -> ProvisionCitation:
        """Build one grouped provision citation."""
        provisions = self._uri_objects(uri, LORSP.refersToProvision)
        return ProvisionCitation(
            uri=uri,
            citation_text_vi=self._preferred_literal(uri, LORSP.citationText),
            referenced_provision_uris=provisions,
            referenced_provision_paths=self._provision_paths(provisions),
        )

    def _component_provenance(self, uri: URIRef) -> ComponentProvenance:
        """Read direct provision and source-chunk links for one component."""
        provisions = self._uri_objects(uri, LORSP.groundedIn)
        chunk_ids = self._dedupe_sorted(
            str(identifier)
            for chunk in self._uri_objects(uri, LORSP.sourcedFromChunk)
            for identifier in self._graph.objects(chunk, DCTERMS.identifier)
        )
        return ComponentProvenance(
            provision_uris=provisions,
            provision_paths=self._provision_paths(provisions),
            source_chunk_ids=chunk_ids,
        )

    def _document_number(self, provisions: Sequence[URIRef]) -> str:
        """Resolve a rule's legal-resource identifier through its expression."""
        numbers = self._dedupe_sorted(
            str(identifier)
            for provision in provisions
            for expression in self._graph.subjects(LORSP.containsProvision, provision)
            for resource in self._graph.subjects(LORSP.hasExpression, expression)
            for identifier in self._graph.objects(resource, DCTERMS.identifier)
        )
        return numbers[0] if numbers else ""

    def _validity_dates(self, rule: URIRef) -> tuple[date | None, date | None]:
        """Return known interval bounds while preserving an unknown beginning."""
        period = self._single_uri(rule, LORTC.validDuring, "validity interval")
        if period is None:
            return None, None
        beginning = self._single_uri(period, TIME.hasBeginning, "validity beginning")
        end = self._single_uri(period, TIME.hasEnd, "validity end")
        return self._instant_date(beginning), self._instant_date(end)

    def _instant_date(self, instant: URIRef | None) -> date | None:
        """Read one optional OWL-Time instant."""
        if instant is None:
            return None
        value = self._single_value(instant, TIME.inXSDDateTime, "instant value")
        return self._parse_date(str(value)) if value is not None else None

    def _provision_paths(self, provisions: Iterable[URIRef]) -> tuple[str, ...]:
        """Return deterministic structural paths for provisions."""
        return self._dedupe_sorted(
            str(path) for provision in provisions for path in self._graph.objects(provision, LORSP.structuralPath)
        )

    def _terms(self, uris: Iterable[URIRef]) -> tuple[OntologyTerm, ...]:
        """Return deterministic, URI-deduplicated ontology terms."""
        return tuple(OntologyTerm(uri=uri, label_vi=self._preferred_label(uri)) for uri in self._dedupe_uris(uris))

    def _preferred_label(self, uri: URIRef) -> str | None:
        """Return a Vietnamese label when available, otherwise another label."""
        return self._preferred_literal(uri, SKOS.prefLabel) or self._preferred_literal(uri, RDFS.label)

    def _preferred_literal(self, uri: URIRef, predicate: URIRef) -> str | None:
        """Return one deterministic literal, preferring Vietnamese."""
        values = [value for value in self._graph.objects(uri, predicate) if isinstance(value, Literal)]
        if not values:
            return None
        values.sort(key=lambda value: (value.language != "vi", value.language or "", str(value)))
        return str(values[0])

    def _subject_kind(self, uri: URIRef) -> str:
        """Map controlled subject-kind individuals to the population contract."""
        return _SUBJECT_KINDS.get(uri, self._local_name(uri))

    def _term_text(self, term: OntologyTerm) -> str:
        """Return a term label with a stable URI local-name fallback."""
        return term.label_vi or self._local_name(term.uri)

    def _uri_objects(self, subject: URIRef, predicate: URIRef) -> tuple[URIRef, ...]:
        """Return deterministic URI objects for one predicate."""
        return self._dedupe_uris(
            value for value in self._graph.objects(subject, predicate) if isinstance(value, URIRef)
        )

    def _single_uri(self, subject: URIRef, predicate: URIRef, field: str) -> URIRef | None:
        """Read an optional single URI and fail clearly on malformed cardinality."""
        values = self._uri_objects(subject, predicate)
        if len(values) > 1:
            raise _OntologyContractError(subject, field, len(values))
        return values[0] if values else None

    def _single_value(self, subject: URIRef, predicate: URIRef, field: str) -> object | None:
        """Read an optional single RDF value and fail clearly on malformed cardinality."""
        values = tuple(self._graph.objects(subject, predicate))
        if len(values) > 1:
            raise _OntologyContractError(subject, field, len(values))
        return values[0] if values else None

    def _decimal_value(self, subject: URIRef | None, predicate: URIRef) -> Decimal | None:
        """Read one optional decimal value."""
        if subject is None:
            return None
        value = self._single_value(subject, predicate, self._local_name(predicate))
        return Decimal(str(value)) if value is not None else None

    def _integer_value(self, subject: URIRef, predicate: URIRef) -> int | None:
        """Read one optional integer value, preserving zero."""
        value = self._single_value(subject, predicate, self._local_name(predicate))
        return int(value) if value is not None else None

    def _condition_provenance(self, root: ConditionExpression) -> tuple[ComponentProvenance, ...]:
        """Flatten provenance from one condition tree."""
        return (
            root.provenance,
            *(item for operand in root.operands for item in self._condition_provenance(operand)),
        )

    @staticmethod
    def _dedupe_uris(values: Iterable[URIRef]) -> tuple[URIRef, ...]:
        """Return URI values once in deterministic lexical order."""
        return tuple(sorted(set(values), key=str))

    @staticmethod
    def _dedupe_sorted(values: Iterable[str]) -> tuple[str, ...]:
        """Return non-empty string values once in deterministic order."""
        return tuple(sorted({value for value in values if value}))

    @staticmethod
    def _local_name(uri: object) -> str:
        """Extract a URI local name."""
        return str(uri).rsplit("#", 1)[-1].rsplit("/", 1)[-1]

    @staticmethod
    def _parse_date(xsd_datetime: str) -> date:
        """Parse an XSD date or dateTime value."""
        date_part = xsd_datetime.rstrip("Z").split("T", maxsplit=1)[0]
        return date.fromisoformat(date_part)
