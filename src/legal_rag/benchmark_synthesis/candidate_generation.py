"""Generate single-turn candidates from a fixed scenario and its gold evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Protocol

from baml_py.logging import set_log_level

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import (
    CandidateEvolutionBatchInput,
    CandidateGenerationBatchInput,
    EvolutionOperation,
    EvolvedCandidateBatchItem,
    FactContext,
    GeneratedSingleTurnCandidateBatchItem,
)
from legal_rag.baml_client.types import EvolvedCandidate as BamlEvolvedCandidate
from legal_rag.baml_client.types import GeneratedSingleTurnCandidate as BamlGeneratedCandidate
from legal_rag.baml_transport import build_client_registry, execute_baml_request
from legal_rag.benchmark_synthesis.config import SynthesisSettings
from legal_rag.benchmark_synthesis.llm_execution import execute_llm_task
from legal_rag.benchmark_synthesis.models import LegalFact
from legal_rag.benchmark_synthesis.scenario_sampling import Scenario
from legal_rag.parser.hashing import sha256_json
from legal_rag.settings import Settings

CANDIDATE_SCHEMA_VERSION = "4.0.0"
CANDIDATE_GENERATION_PROMPT_VERSION = "5.1.0"

_EVOLUTION_OPERATION_BY_ID: dict[str, EvolutionOperation] = {
    "concretizing": EvolutionOperation.Concretizing,
    "reasoning": EvolutionOperation.Reasoning,
    "hypothetical": EvolutionOperation.Hypothetical,
}


@dataclass(frozen=True, slots=True)
class ClaimEvidence:
    """One source chunk backing a required claim."""

    chunk_id: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable claim-evidence record."""
        return {"chunk_id": self.chunk_id}


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
class CandidateDraft:
    """One single-turn candidate generated and evolved, trusting LLM output as-is."""

    sample_id: str
    scenario_id: str
    query_vi: str
    reference_answer_vi: str
    reasoning_vi: str
    required_claims: tuple[RequiredClaim, ...]
    evolutions: tuple[str, ...]
    gold_chunk_ids: tuple[str, ...]
    query_form: str
    persona: str
    query_style: str
    query_length: str
    answerability: str
    reasoning_hops: int
    source_fact_ids: tuple[str, ...]
    generator_model: str
    bundle_kind: str = "single_hop"
    bridge: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable candidate-draft record."""
        return {
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "interaction_type": "single_turn",
            "scenario_id": self.scenario_id,
            "bundle_kind": self.bundle_kind,
            "bridge": self.bridge,
            "query_vi": self.query_vi,
            "reference_answer_vi": self.reference_answer_vi,
            "reasoning_vi": self.reasoning_vi,
            "required_claims": [claim.to_dict() for claim in self.required_claims],
            "evolutions": list(self.evolutions),
            "gold_chunk_ids": list(self.gold_chunk_ids),
            "query_form": self.query_form,
            "persona": self.persona,
            "query_style": self.query_style,
            "query_length": self.query_length,
            "answerability": self.answerability,
            "reasoning_hops": self.reasoning_hops,
            "source_fact_ids": list(self.source_fact_ids),
            "generator_model": self.generator_model,
        }


def build_fact_context(fact: LegalFact) -> FactContext:
    """Return the BAML-facing view of one fact, exposing only its own evidence chunk_ids."""
    return FactContext(
        fact_id=fact.fact_id,
        statement=fact.statement,
        valid_from=fact.valid_from.isoformat() if fact.valid_from else None,
        valid_to=fact.valid_to.isoformat() if fact.valid_to else None,
        document_number=fact.document_number,
        evidence_chunk_ids=[entry.chunk_id for entry in fact.evidence],
    )


def _claim_evidence(fact: LegalFact, referenced_chunk_ids: Sequence[str]) -> tuple[ClaimEvidence, ...]:
    """Return the fact's own evidence chunks matching the LLM-referenced chunk_ids."""
    referenced = set(referenced_chunk_ids)
    return tuple(ClaimEvidence(chunk_id=entry.chunk_id) for entry in fact.evidence if entry.chunk_id in referenced)


class CandidateGenerationClient(Protocol):
    """Boundary implemented by a candidate-generation language-model client."""

    def generate(
        self,
        facts: Sequence[FactContext],
        persona_description: str,
        query_style: str,
        query_length: str,
        query_form: str,
        answerability: str,
        bundle_kind: str,
        bridge: str | None,
        reasoning_hops: int,
    ) -> BamlGeneratedCandidate:
        """Return one typed generated candidate."""

    def generate_batch(
        self, items: Sequence[CandidateGenerationBatchInput]
    ) -> Sequence[GeneratedSingleTurnCandidateBatchItem]:
        """Return one typed generated candidate for every batch item."""

    def evolve(
        self,
        query_vi: str,
        reference_answer_vi: str,
        reasoning_vi: str,
        operation: EvolutionOperation,
        query_length: str,
    ) -> BamlEvolvedCandidate:
        """Return one typed evolved query and answer."""

    def evolve_batch(self, items: Sequence[CandidateEvolutionBatchInput]) -> Sequence[EvolvedCandidateBatchItem]:
        """Return one typed evolved candidate for every batch item."""


class BamlCandidateGenerationClient:
    """Typed BAML client backed by a runtime OpenAI-compatible provider."""

    def __init__(self, settings: Settings, synthesis_settings: SynthesisSettings) -> None:
        """Create a runtime BAML client from typed, secret-safe configuration."""
        set_log_level("OFF")
        self._generation_registry = build_client_registry(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_output_tokens=synthesis_settings.candidate_generation.max_output_tokens,
            temperature=synthesis_settings.candidate_generation.temperature,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        self._evolution_registry = build_client_registry(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_output_tokens=synthesis_settings.candidate_evolution.max_output_tokens,
            temperature=synthesis_settings.candidate_evolution.temperature,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        self._timeout_seconds = settings.llm_timeout_seconds
        self._max_response_bytes = synthesis_settings.max_response_bytes

    def generate(
        self,
        facts: Sequence[FactContext],
        persona_description: str,
        query_style: str,
        query_length: str,
        query_form: str,
        answerability: str,
        bundle_kind: str,
        bridge: str | None,
        reasoning_hops: int,
    ) -> BamlGeneratedCandidate:
        """Run one candidate through the batch endpoint."""
        item = CandidateGenerationBatchInput(
            task_id="single",
            facts=list(facts),
            persona_description=persona_description,
            query_style=query_style,
            query_length=query_length,
            query_form=query_form,
            answerability=answerability,
            bundle_kind=bundle_kind,
            bridge=bridge,
            reasoning_hops=reasoning_hops,
        )
        return self.generate_batch((item,))[0].candidate

    def generate_batch(
        self, items: Sequence[CandidateGenerationBatchInput]
    ) -> Sequence[GeneratedSingleTurnCandidateBatchItem]:
        """Run one bounded BAML call for a candidate batch."""
        request = b.request.GenerateSingleTurnCandidateBatch(
            list(items),
            baml_options={"client_registry": self._generation_registry},
        )
        raw_output = execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)
        return b.parse.GenerateSingleTurnCandidateBatch(
            raw_output, baml_options={"client_registry": self._generation_registry}
        )

    def evolve(
        self,
        query_vi: str,
        reference_answer_vi: str,
        reasoning_vi: str,
        operation: EvolutionOperation,
        query_length: str,
    ) -> BamlEvolvedCandidate:
        """Run one evolution through the batch endpoint."""
        item = CandidateEvolutionBatchInput(
            task_id="single",
            query_vi=query_vi,
            reference_answer_vi=reference_answer_vi,
            reasoning_vi=reasoning_vi,
            operation=operation,
            query_length=query_length,
        )
        return self.evolve_batch((item,))[0].candidate

    def evolve_batch(self, items: Sequence[CandidateEvolutionBatchInput]) -> Sequence[EvolvedCandidateBatchItem]:
        """Run one bounded BAML call for an evolution batch."""
        request = b.request.EvolveCandidateBatch(
            list(items),
            baml_options={"client_registry": self._evolution_registry},
        )
        raw_output = execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)
        return b.parse.EvolveCandidateBatch(raw_output, baml_options={"client_registry": self._evolution_registry})


def generate_candidate(
    scenario: Scenario,
    facts_by_id: Mapping[str, LegalFact],
    persona_description: str,
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
        scenario.answerability,
        scenario.bundle_kind,
        scenario.bridge,
        scenario.reasoning_hops,
    )
    return _candidate_from_generated(scenario, raw, facts_by_id, generator_model)


def generate_candidate_batch(
    tasks: Sequence[tuple[str, Scenario]],
    facts_by_id: Mapping[str, LegalFact],
    persona_descriptions: Mapping[str, str],
    client: CandidateGenerationClient,
    *,
    generator_model: str,
) -> tuple[tuple[str, CandidateDraft], ...]:
    """Generate one candidate for every task in a single LLM request."""
    items = [
        CandidateGenerationBatchInput(
            task_id=task_id,
            facts=[build_fact_context(facts_by_id[fact_id]) for fact_id in scenario.fact_ids],
            persona_description=persona_descriptions[scenario.persona],
            query_style=scenario.query_style,
            query_length=scenario.query_length,
            query_form=scenario.query_form,
            answerability=scenario.answerability,
            bundle_kind=scenario.bundle_kind,
            bridge=scenario.bridge,
            reasoning_hops=scenario.reasoning_hops,
        )
        for task_id, scenario in tasks
    ]
    generated_by_task_id = {result.task_id: result.candidate for result in client.generate_batch(items)}
    return tuple(
        (
            task_id,
            _candidate_from_generated(scenario, generated_by_task_id[task_id], facts_by_id, generator_model),
        )
        for task_id, scenario in tasks
    )


def _candidate_from_generated(
    scenario: Scenario,
    raw: BamlGeneratedCandidate,
    facts_by_id: Mapping[str, LegalFact],
    generator_model: str,
) -> CandidateDraft:
    """Attach one typed LLM result to its deterministic scenario metadata."""
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
            evolutions=(),
            gold_chunk_ids=scenario.gold_chunk_ids,
            query_form=scenario.query_form,
            persona=scenario.persona,
            query_style=scenario.query_style,
            query_length=scenario.query_length,
            answerability=scenario.answerability,
            reasoning_hops=scenario.reasoning_hops,
            source_fact_ids=scenario.fact_ids,
            generator_model=generator_model,
        )
    )


def evolve_candidate(
    draft: CandidateDraft,
    operations: Sequence[str],
    client: CandidateGenerationClient,
    *,
    llm_max_retries: int,
    llm_retry_delay_seconds: float,
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
                draft.query_length,
            ),
            max_retries=llm_max_retries,
            retry_delay_seconds=llm_retry_delay_seconds,
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


def evolve_candidate_batch(
    tasks: Sequence[tuple[str, CandidateDraft]],
    operation_id: str,
    client: CandidateGenerationClient,
) -> tuple[tuple[str, CandidateDraft], ...]:
    """Apply one evolution operation to every task in a single LLM request."""
    operation = _EVOLUTION_OPERATION_BY_ID[operation_id]
    items = [
        CandidateEvolutionBatchInput(
            task_id=task_id,
            query_vi=draft.query_vi,
            reference_answer_vi=draft.reference_answer_vi,
            reasoning_vi=draft.reasoning_vi,
            operation=operation,
            query_length=draft.query_length,
        )
        for task_id, draft in tasks
    ]
    evolved_by_task_id = {result.task_id: result.candidate for result in client.evolve_batch(items)}
    return tuple(
        (task_id, _apply_evolution(draft, operation_id, evolved_by_task_id[task_id])) for task_id, draft in tasks
    )


def _apply_evolution(draft: CandidateDraft, operation_id: str, result: BamlEvolvedCandidate) -> CandidateDraft:
    """Return one draft updated with a completed evolution operation."""
    return _with_identity(
        replace(
            draft,
            query_vi=result.query_vi,
            reference_answer_vi=result.reference_answer_vi,
            reasoning_vi=result.reasoning_vi,
            evolutions=(*draft.evolutions, operation_id),
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
