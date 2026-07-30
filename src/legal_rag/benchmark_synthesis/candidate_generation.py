"""Generate single-turn candidates from a fixed scenario and its gold evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from functools import partial
from typing import Any, Protocol

from baml_py.logging import set_log_level

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import EvolutionOperation, FactContext
from legal_rag.baml_client.types import EvolvedCandidate as BamlEvolvedCandidate
from legal_rag.baml_client.types import GeneratedSingleTurnCandidate as BamlGeneratedCandidate
from legal_rag.baml_client.types import SurfaceCritique as BamlSurfaceCritique
from legal_rag.baml_transport import build_client_registry, execute_baml_request
from legal_rag.benchmark_synthesis.llm_execution import execute_llm_task
from legal_rag.benchmark_synthesis.models import LegalFact
from legal_rag.benchmark_synthesis.scenario_sampling import Scenario
from legal_rag.parser.hashing import sha256_json
from legal_rag.settings import EvaluationSettings, Settings

CANDIDATE_SCHEMA_VERSION = "1.1.0"
CANDIDATE_GENERATION_PROMPT_VERSION = "1.1.0"

_EVOLUTION_OPERATION_BY_ID: dict[str, EvolutionOperation] = {
    "concretizing": EvolutionOperation.Concretizing,
    "reasoning": EvolutionOperation.Reasoning,
    "hypothetical": EvolutionOperation.Hypothetical,
}


@dataclass(frozen=True, slots=True)
class ClaimEvidence:
    """One quoted span backing a required claim, copied from real fact evidence."""

    chunk_id: str
    quoted_span: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable claim-evidence record."""
        return {"chunk_id": self.chunk_id, "quoted_span": self.quoted_span}


@dataclass(frozen=True, slots=True)
class RequiredClaim:
    """One claim the reference answer must support, tied to its source fact."""

    claim: str
    fact_id: str
    evidence: tuple[ClaimEvidence, ...]
    requested_chunk_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable required-claim record."""
        return {
            "claim": self.claim,
            "fact_id": self.fact_id,
            "requested_chunk_ids": list(self.requested_chunk_ids),
            "evidence": [entry.to_dict() for entry in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class FactAnswerFields:
    """Penalty fields copied mechanically from one approved fact, never LLM-generated."""

    fact_id: str
    monetary_penalty: str | None
    license_points_deducted: str | None
    additional_penalties: tuple[str, ...]
    remedial_measures: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable answer-fields record."""
        return {
            "fact_id": self.fact_id,
            "monetary_penalty": self.monetary_penalty,
            "license_points_deducted": self.license_points_deducted,
            "additional_penalties": list(self.additional_penalties),
            "remedial_measures": list(self.remedial_measures),
        }


@dataclass(frozen=True, slots=True)
class CandidateDraft:
    """One single-turn candidate before gate 1 and gate 2, still carrying declared labels."""

    sample_id: str
    scenario_id: str
    query_vi: str
    reference_answer_vi: str
    reasoning_vi: str
    required_claims: tuple[RequiredClaim, ...]
    answer_fields: tuple[FactAnswerFields, ...]
    evolutions: tuple[str, ...]
    gold_chunk_ids: tuple[str, ...]
    query_type: str
    query_form: str
    persona: str
    query_style: str
    query_length: str
    answerability: str
    reasoning_hops: int
    as_of_date: date | None
    source_fact_ids: tuple[str, ...]
    source_fact_content_sha256: tuple[str, ...]
    generator_model: str
    bundle_kind: str = "single_hop"
    bridge: tuple[str, ...] = ()
    surface_attempts: int = 0
    surface_refined: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable candidate-draft record."""
        return {
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "interaction_type": "single_turn",
            "scenario_id": self.scenario_id,
            "bundle_kind": self.bundle_kind,
            "bridge": list(self.bridge),
            "query_vi": self.query_vi,
            "reference_answer_vi": self.reference_answer_vi,
            "reasoning_vi": self.reasoning_vi,
            "required_claims": [claim.to_dict() for claim in self.required_claims],
            "answer_fields": [fields.to_dict() for fields in self.answer_fields],
            "evolutions": list(self.evolutions),
            "gold_chunk_ids": list(self.gold_chunk_ids),
            "query_type": self.query_type,
            "query_form": self.query_form,
            "persona": self.persona,
            "query_style": self.query_style,
            "query_length": self.query_length,
            "answerability": self.answerability,
            "reasoning_hops": self.reasoning_hops,
            "as_of_date": self.as_of_date.isoformat() if self.as_of_date else None,
            "source_fact_ids": list(self.source_fact_ids),
            "source_fact_content_sha256": list(self.source_fact_content_sha256),
            "generator_model": self.generator_model,
            "surface_attempts": self.surface_attempts,
            "surface_refined": self.surface_refined,
        }


def build_fact_context(fact: LegalFact) -> FactContext:
    """Return the BAML-facing view of one fact, exposing only its own evidence chunk_ids."""
    return FactContext(
        fact_id=fact.fact_id,
        conduct=fact.conduct,
        subject=fact.subject,
        conditions=list(fact.conditions),
        monetary_penalty=fact.monetary_penalty,
        license_points_deducted=fact.license_points_deducted,
        additional_penalties=list(fact.additional_penalties),
        remedial_measures=list(fact.remedial_measures),
        valid_from=fact.valid_from.isoformat() if fact.valid_from else None,
        valid_to=fact.valid_to.isoformat() if fact.valid_to else None,
        document_number=fact.document_number,
        provision_path=fact.provision_path,
        evidence_chunk_ids=[entry.chunk_id for entry in fact.evidence],
    )


def _claim_evidence(fact: LegalFact, referenced_chunk_ids: Sequence[str]) -> tuple[ClaimEvidence, ...]:
    """Return the fact's own evidence entries matching the LLM-referenced chunk_ids.

    quoted_span always comes from the fact's real evidence, never from the
    LLM's copy of it, so a claim can never carry a fabricated quote.
    """
    referenced = set(referenced_chunk_ids)
    return tuple(
        ClaimEvidence(chunk_id=entry.chunk_id, quoted_span=entry.quoted_span)
        for entry in fact.evidence
        if entry.chunk_id in referenced
    )


class CandidateGenerationClient(Protocol):
    """Boundary implemented by a candidate-generation language-model client."""

    def generate(
        self,
        facts: Sequence[FactContext],
        persona_description: str,
        query_style: str,
        query_length: str,
        query_form: str,
        query_type_description: str,
        answerability: str,
        bundle_kind: str,
        bridge: Sequence[str],
        reasoning_hops: int,
        as_of_date: str | None,
    ) -> BamlGeneratedCandidate:
        """Return one typed generated candidate."""

    def critique_surface(
        self,
        query_vi: str,
        query_style: str,
        query_length: str,
        query_form: str,
    ) -> BamlSurfaceCritique:
        """Return one typed surface critique, with a rewrite when it fails."""

    def evolve(
        self,
        query_vi: str,
        reference_answer_vi: str,
        reasoning_vi: str,
        operation: EvolutionOperation,
    ) -> BamlEvolvedCandidate:
        """Return one typed evolved query and answer."""


class BamlCandidateGenerationClient:
    """Typed BAML client backed by a runtime OpenAI-compatible provider."""

    def __init__(self, settings: Settings, evaluation_settings: EvaluationSettings) -> None:
        """Create a runtime BAML client from typed, secret-safe configuration."""
        set_log_level("OFF")
        self._registry = build_client_registry(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_output_tokens=evaluation_settings.candidate_generation_max_output_tokens,
            temperature=evaluation_settings.candidate_generation_temperature,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        self._timeout_seconds = settings.llm_timeout_seconds
        self._max_response_bytes = evaluation_settings.max_response_bytes

    def generate(
        self,
        facts: Sequence[FactContext],
        persona_description: str,
        query_style: str,
        query_length: str,
        query_form: str,
        query_type_description: str,
        answerability: str,
        bundle_kind: str,
        bridge: Sequence[str],
        reasoning_hops: int,
        as_of_date: str | None,
    ) -> BamlGeneratedCandidate:
        """Run one bounded BAML call and return its generated typed result."""
        request = b.request.GenerateSingleTurnCandidate(
            list(facts),
            persona_description,
            query_style,
            query_length,
            query_form,
            query_type_description,
            answerability,
            bundle_kind,
            list(bridge),
            reasoning_hops,
            as_of_date,
            baml_options={"client_registry": self._registry},
        )
        raw_output = execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)
        return b.parse.GenerateSingleTurnCandidate(raw_output, baml_options={"client_registry": self._registry})

    def critique_surface(
        self,
        query_vi: str,
        query_style: str,
        query_length: str,
        query_form: str,
    ) -> BamlSurfaceCritique:
        """Run one bounded BAML call and return its generated typed result."""
        request = b.request.CritiqueSurface(
            query_vi, query_style, query_length, query_form, baml_options={"client_registry": self._registry}
        )
        raw_output = execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)
        return b.parse.CritiqueSurface(raw_output, baml_options={"client_registry": self._registry})

    def evolve(
        self,
        query_vi: str,
        reference_answer_vi: str,
        reasoning_vi: str,
        operation: EvolutionOperation,
    ) -> BamlEvolvedCandidate:
        """Run one bounded BAML call and return its generated typed result."""
        request = b.request.EvolveCandidate(
            query_vi,
            reference_answer_vi,
            reasoning_vi,
            operation,
            baml_options={"client_registry": self._registry},
        )
        raw_output = execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)
        return b.parse.EvolveCandidate(raw_output, baml_options={"client_registry": self._registry})


def generate_candidate(
    scenario: Scenario,
    facts_by_id: Mapping[str, LegalFact],
    persona_description: str,
    query_type_description: str,
    client: CandidateGenerationClient,
    *,
    generator_model: str,
) -> CandidateDraft:
    """Generate one candidate from a fixed scenario and its already-chosen evidence."""
    facts = [facts_by_id[fact_id] for fact_id in scenario.fact_ids]
    raw = client.generate(
        [build_fact_context(fact) for fact in facts],
        persona_description,
        scenario.query_style,
        scenario.query_length,
        scenario.query_form,
        query_type_description,
        scenario.answerability,
        scenario.bundle_kind,
        scenario.bridge,
        scenario.reasoning_hops,
        scenario.as_of_date.isoformat() if scenario.as_of_date else None,
    )

    required_claims = tuple(
        RequiredClaim(
            claim=claim.claim,
            fact_id=claim.fact_id,
            requested_chunk_ids=tuple(claim.evidence_chunk_ids),
            evidence=(
                _claim_evidence(facts_by_id[claim.fact_id], claim.evidence_chunk_ids)
                if claim.fact_id in facts_by_id
                else ()
            ),
        )
        for claim in raw.required_claims
    )
    answer_fields = tuple(
        FactAnswerFields(
            fact_id=fact.fact_id,
            monetary_penalty=fact.monetary_penalty,
            license_points_deducted=fact.license_points_deducted,
            additional_penalties=fact.additional_penalties,
            remedial_measures=fact.remedial_measures,
        )
        for fact in facts
    )

    return _with_identity(
        CandidateDraft(
            sample_id="",
            scenario_id=scenario.scenario_id,
            bundle_kind=scenario.bundle_kind,
            bridge=scenario.bridge,
            query_vi=raw.query_vi,
            reference_answer_vi=raw.reference_answer_vi,
            reasoning_vi=raw.reasoning_vi,
            required_claims=required_claims,
            answer_fields=answer_fields,
            evolutions=(),
            gold_chunk_ids=scenario.gold_chunk_ids,
            query_type=scenario.query_type,
            query_form=scenario.query_form,
            persona=scenario.persona,
            query_style=scenario.query_style,
            query_length=scenario.query_length,
            answerability=scenario.answerability,
            reasoning_hops=scenario.reasoning_hops,
            as_of_date=scenario.as_of_date,
            source_fact_ids=scenario.fact_ids,
            source_fact_content_sha256=tuple(fact.fact_content_sha256 for fact in facts),
            generator_model=generator_model,
        )
    )


def refine_surface(
    draft: CandidateDraft,
    client: CandidateGenerationClient,
    *,
    max_retries: int,
    llm_max_retries: int = 0,
    on_attempt: Callable[[CandidateDraft], None] | None = None,
) -> CandidateDraft:
    """Rewrite only the question's surface phrasing until it matches style, bounded by max_retries.

    Claims, evidence, and answer_fields are frozen for the whole loop; only
    query_vi can change.
    """
    if draft.surface_refined:
        return draft
    query_vi = draft.query_vi
    attempts = draft.surface_attempts
    for _attempt in range(attempts, max_retries):
        critique = execute_llm_task(
            f"{draft.scenario_id}:surface:{attempts}",
            partial(client.critique_surface, query_vi, draft.query_style, draft.query_length, draft.query_form),
            max_retries=llm_max_retries,
        )
        attempts += 1
        if critique.passes:
            refined = _with_identity(replace(draft, query_vi=query_vi, surface_attempts=attempts, surface_refined=True))
            if on_attempt is not None:
                on_attempt(refined)
            return refined
        query_vi = critique.rewritten_query_vi
        draft = _with_identity(replace(draft, query_vi=query_vi, surface_attempts=attempts))
        if on_attempt is not None:
            on_attempt(draft)
    refined = _with_identity(replace(draft, query_vi=query_vi, surface_attempts=attempts, surface_refined=True))
    if on_attempt is not None:
        on_attempt(refined)
    return refined


def evolve_candidate(
    draft: CandidateDraft,
    operations: Sequence[str],
    client: CandidateGenerationClient,
    *,
    llm_max_retries: int = 0,
    on_operation_applied: Callable[[CandidateDraft], None] | None = None,
) -> CandidateDraft:
    """Apply a bounded sequence of evidence-preserving evolution operations."""
    query_vi = draft.query_vi
    reference_answer_vi = draft.reference_answer_vi
    reasoning_vi = draft.reasoning_vi
    applied = list(draft.evolutions)
    for operation_id in operations[len(applied) :]:
        result = execute_llm_task(
            f"{draft.scenario_id}:evolution:{len(applied)}:{operation_id}",
            partial(
                client.evolve,
                query_vi,
                reference_answer_vi,
                reasoning_vi,
                _EVOLUTION_OPERATION_BY_ID[operation_id],
            ),
            max_retries=llm_max_retries,
        )
        query_vi = result.query_vi
        reference_answer_vi = result.reference_answer_vi
        reasoning_vi = result.reasoning_vi
        applied.append(operation_id)
        draft = _with_identity(
            replace(
                draft,
                query_vi=query_vi,
                reference_answer_vi=reference_answer_vi,
                reasoning_vi=reasoning_vi,
                evolutions=tuple(applied),
            )
        )
        if on_operation_applied is not None:
            on_operation_applied(draft)
    return _with_identity(
        replace(
            draft,
            query_vi=query_vi,
            reference_answer_vi=reference_answer_vi,
            reasoning_vi=reasoning_vi,
            evolutions=tuple(applied),
        )
    )


def _with_identity(draft: CandidateDraft) -> CandidateDraft:
    """Return a draft whose identity reflects its current final-facing content."""
    sample_id = sha256_json(
        {
            "scenario_id": draft.scenario_id,
            "query_vi": draft.query_vi,
            "reference_answer_vi": draft.reference_answer_vi,
            "reasoning_vi": draft.reasoning_vi,
            "required_claims": [claim.to_dict() for claim in draft.required_claims],
        }
    )
    return replace(draft, sample_id=sample_id)
