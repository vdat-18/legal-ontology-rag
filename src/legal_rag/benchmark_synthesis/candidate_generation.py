"""Generate single-turn candidates from a fixed scenario and its gold evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from baml_py.logging import set_log_level

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import (
    CandidateGenerationBatchInput,
    FactContext,
    GeneratedSingleTurnCandidateBatchItem,
)
from legal_rag.baml_client.types import GeneratedSingleTurnCandidate as BamlGeneratedCandidate
from legal_rag.baml_transport import build_client_registry, execute_baml_request
from legal_rag.benchmark_synthesis.config import SynthesisSettings
from legal_rag.benchmark_synthesis.models import LegalFact
from legal_rag.benchmark_synthesis.scenario_sampling import Scenario
from legal_rag.parser.hashing import sha256_json
from legal_rag.settings import Settings

CANDIDATE_SCHEMA_VERSION = "6.0.0"
CANDIDATE_GENERATION_PROMPT_VERSION = "7.0.0"


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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable required-claim record."""
        return {
            "claim": self.claim,
            "fact_id": self.fact_id,
            "evidence": [entry.to_dict() for entry in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class CandidateDraft:
    """One grounded single-turn candidate generated from a fixed scenario."""

    sample_id: str
    scenario_id: str
    query: str
    reference_answer: str
    reasoning: str
    required_claims: tuple[RequiredClaim, ...]
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
            "query": self.query,
            "reference_answer": self.reference_answer,
            "reasoning": self.reasoning,
            "required_claims": [claim.to_dict() for claim in self.required_claims],
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
    """Return the minimal BAML-facing view of one legal fact."""
    return FactContext(
        fact_id=fact.fact_id,
        statement=fact.statement,
        valid_from=fact.valid_from.isoformat() if fact.valid_from else None,
        valid_to=fact.valid_to.isoformat() if fact.valid_to else None,
        document_number=fact.document_number,
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


class BamlCandidateGenerationClient:
    """Typed BAML client backed by a runtime OpenAI-compatible provider."""

    def __init__(self, settings: Settings, synthesis_settings: SynthesisSettings) -> None:
        """Create a runtime BAML client from typed, secret-safe configuration."""
        set_log_level("OFF")
        self._registry = build_client_registry(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_output_tokens=synthesis_settings.candidate_generation.max_output_tokens,
            temperature=synthesis_settings.candidate_generation.temperature,
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
            baml_options={"client_registry": self._registry},
        )
        raw_output = execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)
        return b.parse.GenerateSingleTurnCandidateBatch(raw_output, baml_options={"client_registry": self._registry})


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
    scenario_facts_by_id = {fact_id: facts_by_id[fact_id] for fact_id in scenario.fact_ids}
    required_claims = tuple(
        RequiredClaim(
            claim=claim.claim,
            fact_id=claim.fact_id,
            evidence=tuple(
                ClaimEvidence(chunk_id=entry.chunk_id) for entry in scenario_facts_by_id[claim.fact_id].evidence
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
            query=raw.query,
            reference_answer=raw.reference_answer,
            reasoning=raw.reasoning,
            required_claims=required_claims,
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


def _with_identity(draft: CandidateDraft) -> CandidateDraft:
    """Return a draft whose identity reflects its current final-facing content."""
    sample_id = sha256_json(
        {
            "scenario_id": draft.scenario_id,
            "query": draft.query,
            "reference_answer": draft.reference_answer,
            "reasoning": draft.reasoning,
            "required_claims": [claim.to_dict() for claim in draft.required_claims],
        }
    )
    return replace(draft, sample_id=sample_id)
