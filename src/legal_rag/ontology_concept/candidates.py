"""Build same-kind controlled-vocabulary inventories without semantic matching."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from legal_rag.ontology_concept.config import ConceptResolutionSettings
from legal_rag.ontology_concept.models import (
    ConceptCandidateInventory,
    ConceptCandidateSet,
    ConceptKind,
    ConceptOccurrence,
)
from legal_rag.ontology_population.models import OntologyPopulationRecord, PopulationEvidence, population_identity


def build_concept_candidates(
    population_records: Sequence[OntologyPopulationRecord],
    settings: ConceptResolutionSettings,
) -> ConceptCandidateSet:
    """Build one deterministic representative inventory per configured concept kind."""
    occurrences = extract_concept_occurrences(population_records, settings.included_kinds)
    by_kind: dict[str, list[ConceptOccurrence]] = defaultdict(list)
    for occurrence in occurrences:
        by_kind[occurrence.kind].append(occurrence)
    inventories = []
    for kind in settings.included_kinds:
        representatives = _label_representatives(by_kind[kind])
        if not representatives:
            continue
        if len(representatives) > settings.max_occurrences_per_inventory:
            msg = "Concept inventory occurrence count exceeds the configured bound"
            raise ValueError(msg)
        occurrence_ids = tuple(item.occurrence_id for item in representatives)
        inventories.append(
            ConceptCandidateInventory(
                candidate_id=population_identity({"kind": kind, "occurrence_ids": occurrence_ids}),
                kind=kind,
                occurrence_ids=occurrence_ids,
            )
        )
    run_id = population_identity(
        {
            "included_kinds": settings.included_kinds,
            "max_occurrences_per_inventory": settings.max_occurrences_per_inventory,
            "occurrences": [item.to_dict() for item in occurrences],
            "inventories": [item.to_dict() for item in inventories],
        }
    )
    return ConceptCandidateSet(run_id=run_id, occurrences=occurrences, inventories=tuple(inventories))


def extract_concept_occurrences(
    population_records: Sequence[OntologyPopulationRecord],
    included_kinds: Sequence[str],
) -> tuple[ConceptOccurrence, ...]:
    """Extract stable rule-scoped concept occurrences from canonical population records."""
    evidence_by_key: dict[tuple[ConceptKind, str, str], set[PopulationEvidence]] = defaultdict(set)
    records_by_key: dict[tuple[ConceptKind, str, str], OntologyPopulationRecord] = {}
    allowed = set(included_kinds)
    for record in population_records:
        values: list[tuple[ConceptKind, str, tuple[PopulationEvidence, ...]]] = []
        values.append(("RegulatedConduct", record.conduct_label_vi, record.evidence))
        values.extend(("LegalSubjectRole", label, record.evidence) for label in record.role_labels_vi)
        values.extend(("VehicleCategory", label, record.evidence) for label in record.vehicle_category_labels_vi)
        values.extend(
            ("LegalSubjectRole", consequence.responsible_role_label_vi, consequence.evidence)
            for consequence in record.consequences
            if consequence.responsible_role_label_vi is not None
        )
        for kind, label, evidence in values:
            if kind not in allowed:
                continue
            key = (kind, record.population_id, label)
            records_by_key[key] = record
            evidence_by_key[key].update(evidence)
    occurrences = []
    for (kind, population_id, label), evidence in evidence_by_key.items():
        record = records_by_key[(kind, population_id, label)]
        occurrences.append(
            ConceptOccurrence(
                occurrence_id=population_identity({"kind": kind, "population_id": population_id, "label_vi": label}),
                kind=kind,
                label_vi=label,
                population_id=population_id,
                document_number=record.document_number,
                provision_path=record.provision_path,
                evidence=tuple(sorted(evidence, key=lambda item: (item.chunk_id, item.exact_text))),
            )
        )
    return tuple(sorted(occurrences, key=lambda item: item.occurrence_id))


def _label_representatives(occurrences: Sequence[ConceptOccurrence]) -> tuple[ConceptOccurrence, ...]:
    """Choose one source-grounded representative for each exact controlled label."""
    by_label: dict[str, list[ConceptOccurrence]] = defaultdict(list)
    for occurrence in occurrences:
        by_label[occurrence.label_vi].append(occurrence)
    representatives = (
        max(
            members,
            key=lambda item: (sum(len(evidence.exact_text) for evidence in item.evidence), item.occurrence_id),
        )
        for members in by_label.values()
    )
    return tuple(sorted(representatives, key=lambda item: (item.label_vi, item.occurrence_id)))
