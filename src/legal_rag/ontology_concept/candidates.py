"""Build exhaustive same-kind concept pairs without semantic matching."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations

from legal_rag.ontology_concept.config import ConceptResolutionSettings
from legal_rag.ontology_concept.models import (
    ConceptCandidatePair,
    ConceptCandidateSet,
    ConceptKind,
    ConceptOccurrence,
)
from legal_rag.ontology_population.models import OntologyPopulationRecord, PopulationEvidence, population_identity


def build_concept_candidates(
    population_records: Sequence[OntologyPopulationRecord],
    settings: ConceptResolutionSettings,
) -> ConceptCandidateSet:
    """Build exhaustive deterministic pairs within each configured concept kind."""
    occurrences = extract_concept_occurrences(population_records, settings.included_kinds)
    by_kind: dict[str, list[ConceptOccurrence]] = defaultdict(list)
    for occurrence in occurrences:
        by_kind[occurrence.kind].append(occurrence)
    pairs = []
    for kind in settings.included_kinds:
        for left, right in combinations(sorted(by_kind[kind], key=lambda item: item.occurrence_id), 2):
            pairs.append(_candidate_pair(settings.candidate_version, left, right))
            if len(pairs) > settings.max_total_pairs:
                msg = "Concept candidate pair count exceeds the configured bound"
                raise ValueError(msg)
    ordered_pairs = tuple(sorted(pairs, key=lambda item: item.candidate_id))
    run_id = population_identity(
        {
            "candidate_version": settings.candidate_version,
            "included_kinds": settings.included_kinds,
            "max_total_pairs": settings.max_total_pairs,
            "occurrences": [item.to_dict() for item in occurrences],
        }
    )
    return ConceptCandidateSet(run_id=run_id, occurrences=occurrences, pairs=ordered_pairs)


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


def _candidate_pair(
    candidate_version: str,
    left: ConceptOccurrence,
    right: ConceptOccurrence,
) -> ConceptCandidatePair:
    """Build one stable same-kind candidate pair."""
    candidate_id = population_identity(
        {
            "candidate_version": candidate_version,
            "kind": left.kind,
            "left_occurrence_id": left.occurrence_id,
            "right_occurrence_id": right.occurrence_id,
        }
    )
    return ConceptCandidatePair(
        candidate_id=candidate_id,
        left_occurrence_id=left.occurrence_id,
        right_occurrence_id=right.occurrence_id,
    )
