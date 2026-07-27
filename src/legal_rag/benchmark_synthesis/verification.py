"""Gate 2: independent LLM verification and relabeling of a candidate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from baml_py.logging import set_log_level

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import ClaimReference as BamlClaimReference
from legal_rag.baml_client.types import FactContext
from legal_rag.baml_client.types import VerificationResult as BamlVerificationResult
from legal_rag.baml_transport import build_client_registry, execute_baml_request
from legal_rag.benchmark_synthesis.candidate_generation import CandidateDraft, RequiredClaim, build_fact_context
from legal_rag.benchmark_synthesis.models import LegalFact
from legal_rag.settings import EvaluationSettings, Settings

_ANSWERABILITY_BY_BAML_ENUM: dict[str, str] = {
    "Answerable": "answerable",
    "PartiallyAnswerable": "partially_answerable",
    "Underspecified": "underspecified",
    "Unanswerable": "unanswerable",
}


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Outcome of running gate 2 against one candidate, with relabeled fields."""

    passed: bool
    is_standalone: bool
    is_specific: bool
    is_question_grounded: bool
    is_answer_grounded: bool
    is_domain_relevant: bool
    is_temporal_correct: bool
    necessary_reasoning_hops: int
    observed_query_type: str
    observed_answerability: str
    reasoning: str
    verifier_model: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable verification record."""
        return {
            "passed": self.passed,
            "is_standalone": self.is_standalone,
            "is_specific": self.is_specific,
            "is_question_grounded": self.is_question_grounded,
            "is_answer_grounded": self.is_answer_grounded,
            "is_domain_relevant": self.is_domain_relevant,
            "is_temporal_correct": self.is_temporal_correct,
            "necessary_reasoning_hops": self.necessary_reasoning_hops,
            "observed_query_type": self.observed_query_type,
            "observed_answerability": self.observed_answerability,
            "reasoning": self.reasoning,
            "verifier_model": self.verifier_model,
        }


def _claim_reference(claim: RequiredClaim) -> BamlClaimReference:
    """Return the BAML-facing view of one already-generated required claim."""
    return BamlClaimReference(
        claim=claim.claim,
        fact_id=claim.fact_id,
        evidence_chunk_ids=[entry.chunk_id for entry in claim.evidence],
    )


class VerificationClient(Protocol):
    """Boundary implemented by an independent candidate-verification language-model client."""

    def verify(
        self,
        query_vi: str,
        reference_answer_vi: str,
        reasoning_vi: str,
        facts: Sequence[FactContext],
        claims: Sequence[BamlClaimReference],
        as_of_date: str | None,
        declared_reasoning_hops: int,
        query_type_catalog: Sequence[str],
    ) -> BamlVerificationResult:
        """Return one typed verification result."""


class BamlVerificationClient:
    """Typed BAML client backed by a runtime OpenAI-compatible provider, distinct from generation."""

    def __init__(self, settings: Settings, evaluation_settings: EvaluationSettings) -> None:
        """Create a runtime BAML client from typed, secret-safe configuration."""
        set_log_level("OFF")
        self._registry = build_client_registry(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_output_tokens=evaluation_settings.verification_max_output_tokens,
            temperature=evaluation_settings.verification_temperature,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        self._timeout_seconds = settings.llm_timeout_seconds
        self._max_response_bytes = evaluation_settings.max_response_bytes

    def verify(
        self,
        query_vi: str,
        reference_answer_vi: str,
        reasoning_vi: str,
        facts: Sequence[FactContext],
        claims: Sequence[BamlClaimReference],
        as_of_date: str | None,
        declared_reasoning_hops: int,
        query_type_catalog: Sequence[str],
    ) -> BamlVerificationResult:
        """Run one bounded BAML call and return its generated typed result."""
        request = b.request.VerifyCandidate(
            query_vi,
            reference_answer_vi,
            reasoning_vi,
            list(facts),
            list(claims),
            as_of_date,
            declared_reasoning_hops,
            list(query_type_catalog),
            baml_options={"client_registry": self._registry},
        )
        raw_output = execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)
        return b.parse.VerifyCandidate(raw_output, baml_options={"client_registry": self._registry})


def verify_candidate(
    draft: CandidateDraft,
    facts_by_id: Mapping[str, LegalFact],
    query_type_catalog: Sequence[str],
    client: VerificationClient,
    *,
    verifier_model: str,
) -> VerificationResult:
    """Run gate 2 against one candidate using a model distinct from the generator.

    passed is computed deterministically as the conjunction of the six
    dimensions — the verifier's own sub-judgments are trusted individually,
    but their combination into a verdict is not delegated back to the model.
    """
    facts = [facts_by_id[fact_id] for fact_id in draft.source_fact_ids]
    raw = client.verify(
        draft.query_vi,
        draft.reference_answer_vi,
        draft.reasoning_vi,
        [build_fact_context(fact) for fact in facts],
        [_claim_reference(claim) for claim in draft.required_claims],
        draft.as_of_date.isoformat() if draft.as_of_date else None,
        draft.reasoning_hops,
        query_type_catalog,
    )

    passed = (
        raw.is_standalone
        and raw.is_specific
        and raw.is_question_grounded
        and raw.is_answer_grounded
        and raw.is_domain_relevant
        and raw.is_temporal_correct
    )

    return VerificationResult(
        passed=passed,
        is_standalone=raw.is_standalone,
        is_specific=raw.is_specific,
        is_question_grounded=raw.is_question_grounded,
        is_answer_grounded=raw.is_answer_grounded,
        is_domain_relevant=raw.is_domain_relevant,
        is_temporal_correct=raw.is_temporal_correct,
        necessary_reasoning_hops=raw.necessary_reasoning_hops,
        observed_query_type=raw.observed_query_type,
        observed_answerability=_ANSWERABILITY_BY_BAML_ENUM[raw.observed_answerability.value],
        reasoning=raw.reasoning,
        verifier_model=verifier_model,
    )
