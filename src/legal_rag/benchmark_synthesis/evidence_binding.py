"""Bind model-cited evidence to exact source spans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact, build_legal_fact


def bind_evidence_references(
    evidence: Sequence[EvidenceReference], chunk_text_by_id: Mapping[str, str]
) -> tuple[EvidenceReference, ...]:
    """Return exact contiguous source spans, splitting only paragraph-wise omissions."""
    if not evidence:
        msg = "A legal fact must cite at least one evidence span"
        raise ValueError(msg)

    bound: list[EvidenceReference] = []
    for reference in evidence:
        chunk_text = chunk_text_by_id[reference.chunk_id]
        quoted_span = reference.quoted_span
        if not quoted_span:
            msg = f"Evidence for chunk {reference.chunk_id} must not be empty"
            raise ValueError(msg)
        if quoted_span in chunk_text:
            bound.append(reference)
            continue

        paragraphs = tuple(part for part in quoted_span.split("\n\n") if part)
        cursor = 0
        split_references: list[EvidenceReference] = []
        for paragraph in paragraphs:
            position = chunk_text.find(paragraph, cursor)
            if position < 0:
                msg = f"Evidence quoted_span is not exact source text from chunk {reference.chunk_id}"
                raise ValueError(msg)
            split_references.append(EvidenceReference(chunk_id=reference.chunk_id, quoted_span=paragraph))
            cursor = position + len(paragraph)
        if not split_references:
            msg = f"Evidence quoted_span is not exact source text from chunk {reference.chunk_id}"
            raise ValueError(msg)
        bound.extend(split_references)
    return tuple(bound)


def rebuild_fact_with_bound_evidence(fact: LegalFact, chunk_text_by_id: Mapping[str, str]) -> LegalFact:
    """Recompute one fact after binding its evidence to exact source spans."""
    evidence = bind_evidence_references(fact.evidence, chunk_text_by_id)
    if evidence == fact.evidence:
        return fact
    return build_legal_fact(
        statement=fact.statement,
        conduct=fact.conduct,
        subject=fact.subject,
        conditions=fact.conditions,
        monetary_penalty=fact.monetary_penalty,
        license_points_deducted=fact.license_points_deducted,
        additional_penalties=fact.additional_penalties,
        remedial_measures=fact.remedial_measures,
        valid_from=fact.valid_from,
        valid_to=fact.valid_to,
        document_number=fact.document_number,
        provision_path=fact.provision_path,
        evidence=evidence,
        source_chunk_id=fact.source_chunk_id,
    )
