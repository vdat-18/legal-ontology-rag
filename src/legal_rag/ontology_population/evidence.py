"""Deterministic source evidence for typed ontology LLM boundaries."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from legal_rag.baml_client.types import OntologyEvidenceReference, OntologyEvidenceSource
from legal_rag.ontology_population.models import PopulationEvidence


def fragment_evidence_sources(chunk_id: str, text: str) -> tuple[OntologyEvidenceSource, ...]:
    """Split one chunk into exact sentence-or-paragraph sources with stable identifiers."""
    fragments = []
    for paragraph in text.split("\n\n"):
        remainder = paragraph
        while ". " in remainder:
            sentence, remainder = remainder.split(". ", maxsplit=1)
            if sentence.strip():
                fragments.append(f"{sentence}.")
        if remainder.strip():
            fragments.append(remainder)
    return tuple(evidence_source(chunk_id, fragment) for fragment in fragments)


def population_evidence_sources(evidence: Sequence[PopulationEvidence]) -> tuple[OntologyEvidenceSource, ...]:
    """Translate persisted exact evidence into model input sources."""
    return tuple(evidence_source(item.chunk_id, item.exact_text) for item in evidence)


def evidence_source(chunk_id: str, text: str) -> OntologyEvidenceSource:
    """Build one stable source identifier without interpreting source text."""
    digest = hashlib.sha256(f"{chunk_id}\0{text}".encode()).hexdigest()[:16]
    return OntologyEvidenceSource(evidence_id=f"ev_{digest}", chunk_id=chunk_id, text=text)


def resolve_evidence_references(
    references: Sequence[OntologyEvidenceReference],
    sources: Sequence[OntologyEvidenceSource],
) -> tuple[PopulationEvidence, ...]:
    """Resolve model-selected source IDs and reject unknown or duplicate references."""
    sources_by_id: dict[str, OntologyEvidenceSource] = {}
    for source in sources:
        existing = sources_by_id.get(source.evidence_id)
        if existing is not None and (existing.chunk_id, existing.text) != (source.chunk_id, source.text):
            msg = f"Evidence source ID collision: {source.evidence_id}"
            raise ValueError(msg)
        sources_by_id[source.evidence_id] = source

    reference_keys = [(reference.evidence_id, reference.start_char, reference.end_char) for reference in references]
    if len(reference_keys) != len(set(reference_keys)):
        msg = "Evidence references must be unique within one field"
        raise ValueError(msg)
    reference_ids = [reference.evidence_id for reference in references]
    unknown_ids = set(reference_ids) - set(sources_by_id)
    if unknown_ids:
        msg = f"Unknown evidence reference: {sorted(unknown_ids)[0]}"
        raise ValueError(msg)
    resolved = []
    for reference in references:
        source = sources_by_id[reference.evidence_id]
        if (reference.start_char is None) != (reference.end_char is None):
            msg = "Evidence subspan requires both start_char and end_char"
            raise ValueError(msg)
        if reference.start_char is None:
            exact_text = source.text
        else:
            if (
                reference.start_char < 0
                or reference.start_char >= reference.end_char
                or reference.end_char > len(source.text)
            ):
                msg = "Evidence subspan offsets are outside the supplied source"
                raise ValueError(msg)
            exact_text = source.text[reference.start_char : reference.end_char]
        resolved.append(PopulationEvidence(chunk_id=source.chunk_id, exact_text=exact_text))
    return tuple(resolved)
