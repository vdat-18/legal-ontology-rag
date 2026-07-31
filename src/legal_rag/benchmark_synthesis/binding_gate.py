"""Gate 1: deterministic, LLM-free validation of a candidate's evidence binding.

LLM output is untrusted by construction (AGENTS.md: "Treat model and tool
output as untrusted. Validate it before use"). A candidate failing any check
here is rejected outright — this module never raises for invalid input; a
failed check is a normal, expected outcome of gating, not a programming
error, and is reported as a value the caller inspects.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from legal_rag.benchmark_synthesis.candidate_generation import CandidateDraft, RequiredClaim
from legal_rag.benchmark_synthesis.models import LegalFact
from legal_rag.chunker.models import Chunk


@dataclass(frozen=True, slots=True)
class BindingViolation:
    """One reason a candidate failed the binding gate."""

    claim: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable violation record."""
        return {"claim": self.claim, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class BindingCheckResult:
    """Outcome of running gate 1 against one candidate."""

    passed: bool
    grounding_score: float
    violations: tuple[BindingViolation, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable binding-check record."""
        return {
            "passed": self.passed,
            "grounding_score": self.grounding_score,
            "violations": [violation.to_dict() for violation in self.violations],
        }


def check_binding(
    draft: CandidateDraft,
    facts_by_id: Mapping[str, LegalFact],
    chunks_by_id: Mapping[str, Chunk],
    *,
    span_match_threshold: float,
) -> BindingCheckResult:
    """Validate every required claim's evidence against real fact data.

    Checks, per claim: the fact exists; every referenced chunk_id belongs to
    that fact's own evidence set; every quoted_span matches the real chunk
    text at or above span_match_threshold, measured by Levenshtein partial
    ratio. grounding_score is the mean ratio across every span checked.
    """
    violations: list[BindingViolation] = []
    partial_ratios: list[float] = []
    source_fact_ids = set(draft.source_fact_ids)
    claimed_fact_ids: set[str] = set()

    for claim in draft.required_claims:
        claimed_fact_ids.add(claim.fact_id)
        claim_violations, claim_ratios = _check_claim(
            claim,
            source_fact_ids,
            facts_by_id,
            chunks_by_id,
            span_match_threshold,
        )
        violations.extend(claim_violations)
        partial_ratios.extend(claim_ratios)

    if draft.reasoning_hops > 1 and claimed_fact_ids != source_fact_ids:
        missing_fact_ids = sorted(source_fact_ids - claimed_fact_ids)
        violations.append(
            BindingViolation(
                "candidate",
                f"multi-hop candidate does not cover scenario fact_ids {missing_fact_ids!r}",
            )
        )

    grounding_score = sum(partial_ratios) / len(partial_ratios) if partial_ratios else 0.0
    return BindingCheckResult(passed=not violations, grounding_score=grounding_score, violations=tuple(violations))


def _check_claim(
    claim: RequiredClaim,
    source_fact_ids: set[str],
    facts_by_id: Mapping[str, LegalFact],
    chunks_by_id: Mapping[str, Chunk],
    span_match_threshold: float,
) -> tuple[list[BindingViolation], list[float]]:
    """Validate one required claim and return its violations and span scores."""
    fact = facts_by_id.get(claim.fact_id)
    if fact is None:
        return [BindingViolation(claim.claim, f"fact_id {claim.fact_id!r} not found")], []
    if claim.fact_id not in source_fact_ids:
        reason = f"fact_id {claim.fact_id!r} is outside the scenario fact set"
        return [BindingViolation(claim.claim, reason)], []
    if not claim.evidence:
        return [BindingViolation(claim.claim, "claim has no evidence")], []

    violations: list[BindingViolation] = []
    ratios: list[float] = []
    fact_chunk_ids = {entry.chunk_id for entry in fact.evidence}
    requested_chunk_ids = claim.requested_chunk_ids or tuple(entry.chunk_id for entry in claim.evidence)
    invalid_requested_ids = set(requested_chunk_ids) - fact_chunk_ids
    violations.extend(
        BindingViolation(
            claim.claim,
            f"requested chunk_id {chunk_id!r} is outside fact {claim.fact_id!r}'s evidence set",
        )
        for chunk_id in sorted(invalid_requested_ids)
    )
    for evidence in claim.evidence:
        if evidence.chunk_id not in fact_chunk_ids:
            reason = f"chunk_id {evidence.chunk_id!r} is outside fact {claim.fact_id!r}'s evidence set"
            violations.append(BindingViolation(claim.claim, reason))
            continue
        chunk = chunks_by_id.get(evidence.chunk_id)
        if chunk is None:
            violations.append(BindingViolation(claim.claim, f"chunk_id {evidence.chunk_id!r} not found"))
            continue
        ratio = fuzz.partial_ratio(evidence.quoted_span, chunk.text) / 100
        ratios.append(ratio)
        if ratio < span_match_threshold:
            reason = (
                f"quoted_span for chunk_id {evidence.chunk_id!r} scored {ratio:.2f}, "
                f"below threshold {span_match_threshold:.2f}"
            )
            violations.append(BindingViolation(claim.claim, reason))
    return violations, ratios
