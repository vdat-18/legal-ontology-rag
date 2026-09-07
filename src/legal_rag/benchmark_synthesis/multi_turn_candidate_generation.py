"""Generate multi-turn conversation candidates turn by turn from fixed conversation scenarios.

Turns are generated round by round across all conversations: every conversation's
turn *k* is written only after its turn *k-1* exists, because turn *k* needs the
question and reference answer of turn *k-1* as prompt history. Conversations are
independent, so turns of one round still batch and run concurrently.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Protocol

from baml_py.logging import set_log_level

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import GeneratedMultiTurnTurn as BamlGeneratedTurn
from legal_rag.baml_client.types import (
    GeneratedMultiTurnTurnBatchItem,
    MultiTurnGenerationBatchInput,
    PriorTurn,
)
from legal_rag.baml_transport import build_client_registry, execute_baml_request
from legal_rag.benchmark_synthesis.candidate_generation import (
    ClaimEvidence,
    RequiredClaim,
    build_fact_context,
)
from legal_rag.benchmark_synthesis.config import SynthesisSettings
from legal_rag.benchmark_synthesis.conversation_scenario_sampling import (
    ConversationScenario,
    TurnSpec,
)
from legal_rag.benchmark_synthesis.llm_execution import (
    LlmTaskExhaustedError,
    LlmTaskFailure,
    execute_llm_task,
)
from legal_rag.benchmark_synthesis.models import LegalFact
from legal_rag.parser.hashing import sha256_json
from legal_rag.settings import Settings

MULTI_TURN_CANDIDATE_SCHEMA_VERSION = "1.0.0"
MULTI_TURN_PROGRESS_SCHEMA_VERSION = "1.0.0"
MULTI_TURN_GENERATION_PROMPT_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class ConversationTurnDraft:
    """One generated conversation turn with its planned anchor metadata."""

    turn_number: int
    dialogue_act: str
    context_dependency: str
    answerability: str
    anchor_scenario_id: str
    query: str
    reference_answer: str
    reasoning: str
    required_claims: tuple[RequiredClaim, ...]
    gold_chunk_ids: tuple[str, ...]
    source_fact_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable conversation-turn record."""
        return {
            "turn_number": self.turn_number,
            "dialogue_act": self.dialogue_act,
            "context_dependency": self.context_dependency,
            "answerability": self.answerability,
            "anchor_scenario_id": self.anchor_scenario_id,
            "query": self.query,
            "reference_answer": self.reference_answer,
            "reasoning": self.reasoning,
            "required_claims": [claim.to_dict() for claim in self.required_claims],
            "gold_chunk_ids": list(self.gold_chunk_ids),
            "source_fact_ids": list(self.source_fact_ids),
        }


@dataclass(frozen=True, slots=True)
class MultiTurnCandidateDraft:
    """One fully generated multi-turn conversation."""

    conversation_id: str
    persona: str
    conversation_length: int
    turns: tuple[ConversationTurnDraft, ...]
    gold_chunk_ids: tuple[str, ...]
    generator_model: str
    prompt_version: str = MULTI_TURN_GENERATION_PROMPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable multi-turn candidate record."""
        return {
            "schema_version": MULTI_TURN_CANDIDATE_SCHEMA_VERSION,
            "interaction_type": "multi_turn",
            "conversation_id": self.conversation_id,
            "persona": self.persona,
            "conversation_length": self.conversation_length,
            "turns": [turn.to_dict() for turn in self.turns],
            "gold_chunk_ids": list(self.gold_chunk_ids),
            "generator_model": self.generator_model,
            "prompt_version": self.prompt_version,
        }


@dataclass(frozen=True, slots=True)
class MultiTurnProgressRecord:
    """One checkpointed conversation turn that finished generating."""

    task_id: str
    conversation_id: str
    turn_number: int
    stage: str
    draft: ConversationTurnDraft

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable multi-turn progress record."""
        return {
            "schema_version": MULTI_TURN_PROGRESS_SCHEMA_VERSION,
            "task_id": self.task_id,
            "conversation_id": self.conversation_id,
            "turn_number": self.turn_number,
            "stage": self.stage,
            "draft": self.draft.to_dict(),
        }


class MultiTurnGenerationClient(Protocol):
    """Boundary implemented by a multi-turn generation language-model client."""

    def generate_turn_batch(
        self, items: Sequence[MultiTurnGenerationBatchInput]
    ) -> Sequence[GeneratedMultiTurnTurnBatchItem]:
        """Return one typed generated turn for every batch item."""


class BamlMultiTurnGenerationClient:
    """Typed BAML client backed by a runtime OpenAI-compatible provider."""

    def __init__(self, settings: Settings, synthesis_settings: SynthesisSettings) -> None:
        """Create a runtime BAML client reusing the candidate-generation stage bounds."""
        set_log_level("OFF")
        self._registry = build_client_registry(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_output_tokens=synthesis_settings.candidate_generation.max_output_tokens,
            temperature=synthesis_settings.candidate_generation.temperature,
            timeout_seconds=settings.llm_timeout_seconds,
            reasoning_effort=synthesis_settings.candidate_generation.reasoning_effort,
        )
        self._timeout_seconds = settings.llm_timeout_seconds
        self._max_response_bytes = synthesis_settings.max_response_bytes

    def generate_turn_batch(
        self, items: Sequence[MultiTurnGenerationBatchInput]
    ) -> Sequence[GeneratedMultiTurnTurnBatchItem]:
        """Run one bounded BAML call for a conversation-turn batch."""
        request = b.request.GenerateMultiTurnTurnBatch(
            list(items),
            baml_options={"client_registry": self._registry},
        )
        raw_output = execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)
        return b.parse.GenerateMultiTurnTurnBatch(raw_output, baml_options={"client_registry": self._registry})


def turn_task_id(conversation_scenario_id: str, turn_number: int) -> str:
    """Build the stable checkpoint key for one conversation turn."""
    return f"{conversation_scenario_id}:{turn_number}"


def generate_turn_batch(
    tasks: Sequence[tuple[str, ConversationScenario, TurnSpec, Sequence[ConversationTurnDraft]]],
    facts_by_id: Mapping[str, LegalFact],
    persona_descriptions: Mapping[str, str],
    client: MultiTurnGenerationClient,
    *,
    generator_model: str,
) -> tuple[tuple[str, ConversationTurnDraft], ...]:
    """Generate one turn for every task in a single LLM request."""
    items = [
        MultiTurnGenerationBatchInput(
            task_id=task_id,
            facts=[build_fact_context(facts_by_id[fact_id]) for fact_id in turn_spec.fact_ids],
            persona_description=persona_descriptions[conversation.persona],
            dialogue_act=turn_spec.dialogue_act,
            context_dependency=turn_spec.context_dependency,
            answerability=turn_spec.answerability,
            history=[
                PriorTurn(question=prior.query, answer=prior.reference_answer) for prior in history
            ],
        )
        for task_id, conversation, turn_spec, history in tasks
    ]
    generated_by_task_id = {result.task_id: result.turn for result in client.generate_turn_batch(items)}
    return tuple(
        (
            task_id,
            _turn_from_generated(turn_spec, generated_by_task_id[task_id], facts_by_id),
        )
        for task_id, _conversation, turn_spec, _history in tasks
    )


def assemble_completed_conversations(
    conversation_scenarios: Sequence[ConversationScenario],
    progress_by_task_id: Mapping[str, MultiTurnProgressRecord],
    *,
    generator_model: str,
) -> tuple[MultiTurnCandidateDraft, ...]:
    """Assemble one candidate per conversation whose every turn finished generating."""
    conversations: list[MultiTurnCandidateDraft] = []
    for scenario in conversation_scenarios:
        drafts: list[ConversationTurnDraft] = []
        for turn_spec in scenario.turns:
            record = progress_by_task_id.get(
                turn_task_id(scenario.conversation_scenario_id, turn_spec.turn_number)
            )
            if record is None:
                break
            drafts.append(record.draft)
        if len(drafts) != len(scenario.turns):
            continue
        conversations.append(
            MultiTurnCandidateDraft(
                conversation_id=scenario.conversation_scenario_id,
                persona=scenario.persona,
                conversation_length=scenario.conversation_length,
                turns=tuple(drafts),
                gold_chunk_ids=scenario.gold_chunk_ids,
                generator_model=generator_model,
            )
        )
    return tuple(conversations)


@dataclass(frozen=True, slots=True)
class MultiTurnPipelineContext:
    """Dependencies and configured limits shared by conversation-turn work items."""

    facts_by_id: Mapping[str, LegalFact]
    persona_descriptions: Mapping[str, str]
    generation_client: MultiTurnGenerationClient
    generator_model: str
    llm_max_retries: int
    llm_retry_delay_seconds: float


def generate_multi_turn_candidates(
    conversation_scenarios: Sequence[ConversationScenario],
    progress_by_task_id: Mapping[str, MultiTurnProgressRecord],
    context: MultiTurnPipelineContext,
    checkpoint: Callable[[Sequence[MultiTurnProgressRecord]], None],
    *,
    batch_size: int,
    max_concurrency: int,
) -> tuple[LlmTaskFailure, ...]:
    """Generate pending turns round by round with bounded concurrent LLM batches.

    Round *k* only runs for conversations whose turns ``1..k-1`` are already
    checkpointed, so a turn is never prompted with a history that does not exist.
    checkpoint runs from worker threads; the caller must make it safe for
    concurrent invocation (e.g. with a lock).
    """
    max_turns = max((len(conversation.turns) for conversation in conversation_scenarios), default=0)
    failures: list[LlmTaskFailure] = []
    for turn_number in range(1, max_turns + 1):
        pending = tuple(
            (conversation, conversation.turns[turn_number - 1])
            for conversation in conversation_scenarios
            if len(conversation.turns) >= turn_number
            and progress_by_task_id.get(turn_task_id(conversation.conversation_scenario_id, turn_number)) is None
            and _prior_turns_ready(conversation, turn_number, progress_by_task_id)
        )
        if not pending:
            continue
        batches = [pending[start : start + batch_size] for start in range(0, len(pending), batch_size)]
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = {
                executor.submit(
                    _process_turn_batch, batch, progress_by_task_id, context, checkpoint
                ): batch
                for batch in batches
            }
            for future in as_completed(futures):
                batch = futures[future]
                task_ids = tuple(
                    turn_task_id(conversation.conversation_scenario_id, turn.turn_number)
                    for conversation, turn in batch
                )
                try:
                    future.result()
                except LlmTaskExhaustedError as error:
                    failures.extend(LlmTaskFailure(task_key=task_id, error=str(error)) for task_id in task_ids)
    return tuple(failures)


def _process_turn_batch(
    batch: Sequence[tuple[ConversationScenario, TurnSpec]],
    progress_by_task_id: Mapping[str, MultiTurnProgressRecord],
    context: MultiTurnPipelineContext,
    checkpoint: Callable[[Sequence[MultiTurnProgressRecord]], None],
) -> None:
    """Generate and checkpoint one conversation-turn batch."""
    tasks = tuple(
        (
            turn_task_id(conversation.conversation_scenario_id, turn_spec.turn_number),
            conversation,
            turn_spec,
            tuple(
                progress_by_task_id[
                    turn_task_id(conversation.conversation_scenario_id, prior.turn_number)
                ].draft
                for prior in conversation.turns[: turn_spec.turn_number - 1]
            ),
        )
        for conversation, turn_spec in batch
    )
    generated = execute_llm_task(
        _batch_task_key(tuple(task_id for task_id, _conversation, _turn_spec, _history in tasks)),
        lambda: generate_turn_batch(
            tasks,
            context.facts_by_id,
            context.persona_descriptions,
            context.generation_client,
            generator_model=context.generator_model,
        ),
        max_retries=context.llm_max_retries,
        retry_delay_seconds=context.llm_retry_delay_seconds,
    )
    generated_by_task_id = dict(generated)
    checkpoint(
        tuple(
            MultiTurnProgressRecord(
                task_id=task_id,
                conversation_id=conversation.conversation_scenario_id,
                turn_number=turn_spec.turn_number,
                stage="completed",
                draft=generated_by_task_id[task_id],
            )
            for task_id, conversation, turn_spec, _history in tasks
        )
    )


def _prior_turns_ready(
    conversation: ConversationScenario,
    turn_number: int,
    progress_by_task_id: Mapping[str, MultiTurnProgressRecord],
) -> bool:
    """Return whether every turn before *turn_number* is already checkpointed."""
    return all(
        turn_task_id(conversation.conversation_scenario_id, prior_number) in progress_by_task_id
        for prior_number in range(1, turn_number)
    )


def _turn_from_generated(
    turn_spec: TurnSpec,
    raw: BamlGeneratedTurn,
    facts_by_id: Mapping[str, LegalFact],
) -> ConversationTurnDraft:
    """Attach one typed LLM result to its deterministic turn metadata."""
    turn_facts_by_id = {fact_id: facts_by_id[fact_id] for fact_id in turn_spec.fact_ids}
    required_claims = tuple(
        RequiredClaim(
            claim=claim.claim,
            fact_id=claim.fact_id,
            evidence=tuple(
                ClaimEvidence(chunk_id=entry.chunk_id) for entry in turn_facts_by_id[claim.fact_id].evidence
            ),
        )
        for claim in raw.required_claims
    )
    return ConversationTurnDraft(
        turn_number=turn_spec.turn_number,
        dialogue_act=turn_spec.dialogue_act,
        context_dependency=turn_spec.context_dependency,
        answerability=turn_spec.answerability,
        anchor_scenario_id=turn_spec.anchor_scenario_id,
        query=raw.query,
        reference_answer=raw.reference_answer,
        reasoning=raw.reasoning,
        required_claims=required_claims,
        gold_chunk_ids=turn_spec.gold_chunk_ids,
        source_fact_ids=turn_spec.fact_ids,
    )


def _batch_task_key(task_ids: Sequence[str]) -> str:
    """Build one stable retry key for an ordered conversation-turn batch."""
    return f"multi-turn-batch:{sha256_json({'task_ids': list(task_ids)})}"
