"""Generate only the evaluated target for fixed multi-turn scenarios."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Protocol

from baml_py.logging import set_log_level
from loguru import logger

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import (
    GeneratedMultiTurnTarget,
    GeneratedMultiTurnTargetBatchItem,
    MultiTurnHistoryInput,
    MultiTurnTargetGenerationInput,
    MultiTurnTargetInput,
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

MULTI_TURN_CANDIDATE_SCHEMA_VERSION = "4.0.0"
MULTI_TURN_PROGRESS_SCHEMA_VERSION = "4.0.0"
MULTI_TURN_GENERATION_PROMPT_VERSION = "4.0.0"

_FACT_ALIAS_PATTERN = re.compile(r"\bT\d+F\d+\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ReviewedHistoryCandidate:
    """One reviewed single-turn record reused verbatim as conversation history."""

    scenario_id: str
    persona: str
    answerability: str
    query: str
    reference_answer: str
    gold_chunk_ids: tuple[str, ...]
    source_fact_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConversationTurnDraft:
    """One fixed history turn or generated evaluation target."""

    turn_number: int
    dialogue_act: str
    context_dependency: str
    answerability: str
    anchor_scenario_id: str
    query: str
    standalone_query: str
    reference_answer: str
    reasoning: str
    missing_information: str | None
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
            "standalone_query": self.standalone_query,
            "reference_answer": self.reference_answer,
            "reasoning": self.reasoning,
            "missing_information": self.missing_information,
            "required_claims": [claim.to_dict() for claim in self.required_claims],
            "gold_chunk_ids": list(self.gold_chunk_ids),
            "source_fact_ids": list(self.source_fact_ids),
        }


@dataclass(frozen=True, slots=True)
class MultiTurnCandidateDraft:
    """One conversation with reviewed history and one generated target."""

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
    """One checkpointed conversation that finished generating."""

    task_id: str
    conversation_id: str
    stage: str
    draft: MultiTurnCandidateDraft

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable multi-turn progress record."""
        return {
            "schema_version": MULTI_TURN_PROGRESS_SCHEMA_VERSION,
            "task_id": self.task_id,
            "conversation_id": self.conversation_id,
            "stage": self.stage,
            "draft": self.draft.to_dict(),
        }


class MultiTurnGenerationClient(Protocol):
    """Boundary implemented by a target-only language-model client."""

    def generate_target_batch(
        self,
        items: Sequence[MultiTurnTargetGenerationInput],
    ) -> Sequence[GeneratedMultiTurnTargetBatchItem]:
        """Return zero or one generated target for each batch item."""


class BamlMultiTurnGenerationClient:
    """Typed BAML client backed by a runtime OpenAI-compatible provider."""

    def __init__(self, settings: Settings, synthesis_settings: SynthesisSettings) -> None:
        """Create a runtime client using multi-turn generation bounds."""
        set_log_level("OFF")
        self._registry = build_client_registry(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_output_tokens=synthesis_settings.multi_turn_generation.max_output_tokens,
            temperature=synthesis_settings.multi_turn_generation.temperature,
            timeout_seconds=settings.llm_timeout_seconds,
            reasoning_effort=synthesis_settings.multi_turn_generation.reasoning_effort,
        )
        self._timeout_seconds = settings.llm_timeout_seconds
        self._max_response_bytes = synthesis_settings.max_response_bytes

    def generate_target_batch(
        self,
        items: Sequence[MultiTurnTargetGenerationInput],
    ) -> Sequence[GeneratedMultiTurnTargetBatchItem]:
        """Run one bounded BAML call for several final targets."""
        request = b.request.GenerateMultiTurnTargetBatch(
            list(items),
            baml_options={"client_registry": self._registry},
        )
        raw_output = execute_baml_request(
            request,
            self._timeout_seconds,
            self._max_response_bytes,
        )
        return b.parse.GenerateMultiTurnTargetBatch(
            raw_output,
            baml_options={"client_registry": self._registry},
        )


@dataclass(frozen=True, slots=True)
class MultiTurnPipelineContext:
    """Dependencies and configured limits shared by target work items."""

    facts_by_id: Mapping[str, LegalFact]
    history_candidates_by_scenario_id: Mapping[str, ReviewedHistoryCandidate]
    persona_descriptions: Mapping[str, str]
    generation_client: MultiTurnGenerationClient
    generator_model: str
    llm_max_retries: int
    llm_retry_delay_seconds: float


def generate_conversation_batch(
    conversations: Sequence[ConversationScenario],
    context: MultiTurnPipelineContext,
) -> tuple[MultiTurnCandidateDraft, ...]:
    """Generate and validate the final target for each conversation."""
    generated_by_id = _generate_raw_target_batch(conversations, context)
    return tuple(
        _conversation_from_generated(
            conversation,
            generated_by_id[conversation.conversation_scenario_id],
            context,
        )
        for conversation in conversations
    )


def assemble_completed_conversations(
    conversation_scenarios: Sequence[ConversationScenario],
    progress_by_task_id: Mapping[str, MultiTurnProgressRecord],
    *,
    generator_model: str,
) -> tuple[MultiTurnCandidateDraft, ...]:
    """Return completed conversations in scenario order."""
    del generator_model
    return tuple(
        progress_by_task_id[scenario.conversation_scenario_id].draft
        for scenario in conversation_scenarios
        if scenario.conversation_scenario_id in progress_by_task_id
    )


def generate_multi_turn_candidates(
    conversation_scenarios: Sequence[ConversationScenario],
    progress_by_task_id: Mapping[str, MultiTurnProgressRecord],
    context: MultiTurnPipelineContext,
    checkpoint: Callable[[Sequence[MultiTurnProgressRecord]], None],
    *,
    batch_size: int,
    max_concurrency: int,
) -> tuple[LlmTaskFailure, ...]:
    """Generate pending targets in bounded concurrent batches."""
    pending = tuple(
        conversation
        for conversation in conversation_scenarios
        if conversation.conversation_scenario_id not in progress_by_task_id
    )
    batches = [
        pending[start : start + batch_size]
        for start in range(0, len(pending), batch_size)
    ]
    failures: list[LlmTaskFailure] = []
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {
            executor.submit(_process_conversation_batch, batch, context, checkpoint): batch
            for batch in batches
        }
        for future in as_completed(futures):
            batch = futures[future]
            try:
                failures.extend(future.result())
            except LlmTaskExhaustedError as error:
                failures.extend(
                    LlmTaskFailure(
                        task_key=conversation.conversation_scenario_id,
                        error=str(error),
                    )
                    for conversation in batch
                )
    return tuple(failures)


def _process_conversation_batch(
    batch: Sequence[ConversationScenario],
    context: MultiTurnPipelineContext,
    checkpoint: Callable[[Sequence[MultiTurnProgressRecord]], None],
) -> tuple[LlmTaskFailure, ...]:
    """Checkpoint valid results and rerun only missing or invalid targets."""
    generated_by_id = execute_llm_task(
        _batch_task_key(
            tuple(conversation.conversation_scenario_id for conversation in batch)
        ),
        lambda: _generate_raw_target_batch(batch, context),
        max_retries=context.llm_max_retries,
        retry_delay_seconds=context.llm_retry_delay_seconds,
    )
    completed: list[MultiTurnProgressRecord] = []
    invalid: list[ConversationScenario] = []
    for conversation in batch:
        try:
            generated = generated_by_id[conversation.conversation_scenario_id]
            draft = _conversation_from_generated(conversation, generated, context)
        except (KeyError, TypeError, ValueError) as error:
            logger.warning(
                "Conversation {} failed batch validation and will rerun alone: {}",
                conversation.conversation_scenario_id,
                error,
            )
            invalid.append(conversation)
        else:
            completed.append(_progress_record(draft))
    if completed:
        checkpoint(completed)

    failures: list[LlmTaskFailure] = []
    for conversation in invalid:
        task_id = conversation.conversation_scenario_id
        try:
            draft = execute_llm_task(
                f"multi-turn-target:{task_id}",
                lambda conversation=conversation: generate_conversation_batch(
                    (conversation,), context
                )[0],
                max_retries=context.llm_max_retries,
                retry_delay_seconds=context.llm_retry_delay_seconds,
            )
        except LlmTaskExhaustedError as error:
            failures.append(LlmTaskFailure(task_key=task_id, error=str(error)))
        else:
            checkpoint((_progress_record(draft),))
    return tuple(failures)


def _progress_record(draft: MultiTurnCandidateDraft) -> MultiTurnProgressRecord:
    """Build one completed conversation checkpoint."""
    return MultiTurnProgressRecord(
        task_id=draft.conversation_id,
        conversation_id=draft.conversation_id,
        stage="completed",
        draft=draft,
    )


def _generate_raw_target_batch(
    conversations: Sequence[ConversationScenario],
    context: MultiTurnPipelineContext,
) -> dict[str, GeneratedMultiTurnTarget]:
    """Call the client and salvage every uniquely identified expected result."""
    items = [_generation_input(conversation, context) for conversation in conversations]
    results = tuple(context.generation_client.generate_target_batch(items))
    expected_ids = {
        conversation.conversation_scenario_id for conversation in conversations
    }
    counts = Counter(result.task_id for result in results)
    generated_by_id = {
        result.task_id: result.target
        for result in results
        if result.task_id in expected_ids and counts[result.task_id] == 1
    }
    omitted = expected_ids - generated_by_id.keys()
    unexpected = set(counts) - expected_ids
    if omitted or unexpected:
        logger.warning(
            "Multi-turn target batch omitted/duplicated {} expected task(s) and returned {} unexpected task(s)",
            len(omitted),
            len(unexpected),
        )
    return generated_by_id


def _generation_input(
    conversation: ConversationScenario,
    context: MultiTurnPipelineContext,
) -> MultiTurnTargetGenerationInput:
    """Build one prompt item from reviewed history and novel target facts."""
    history = [
        _history_candidate(turn, context) for turn in conversation.turns[:-1]
    ]
    if any(candidate.persona != conversation.persona for candidate in history):
        msg = "Reviewed history persona must match the conversation persona"
        raise ValueError(msg)
    target = conversation.turns[-1]
    return MultiTurnTargetGenerationInput(
        task_id=conversation.conversation_scenario_id,
        persona_description=context.persona_descriptions[conversation.persona],
        history=[
            MultiTurnHistoryInput(
                turn_number=turn.turn_number,
                question=candidate.query,
                answer=candidate.reference_answer,
            )
            for turn, candidate in zip(conversation.turns[:-1], history, strict=True)
        ],
        target=MultiTurnTargetInput(
            turn_number=target.turn_number,
            reference_turn_number=(
                1 if target.dialogue_act == "return" else target.turn_number - 1
            ),
            facts=[
                build_fact_context(context.facts_by_id[fact_id]).model_copy(
                    update={"fact_id": _fact_alias(target.turn_number, index)}
                )
                for index, fact_id in enumerate(target.fact_ids, start=1)
            ],
            dialogue_act=target.dialogue_act,
            context_dependency=target.context_dependency,
            answerability=target.answerability,
        ),
    )


def _history_candidate(
    turn_spec: TurnSpec,
    context: MultiTurnPipelineContext,
) -> ReviewedHistoryCandidate:
    """Resolve and validate one reviewed history record."""
    candidate = context.history_candidates_by_scenario_id[turn_spec.anchor_scenario_id]
    if not candidate.query.strip() or not candidate.reference_answer.strip():
        msg = "Reviewed history must contain a question and answer"
        raise ValueError(msg)
    return candidate


def _conversation_from_generated(
    scenario: ConversationScenario,
    generated: GeneratedMultiTurnTarget,
    context: MultiTurnPipelineContext,
) -> MultiTurnCandidateDraft:
    """Combine reviewed history with one generated final target."""
    target_spec = scenario.turns[-1]
    if generated.turn_number != target_spec.turn_number:
        msg = "Generated target turn number must match the planned final turn"
        raise ValueError(msg)
    context_turns = tuple(
        _context_turn_from_reviewed(spec, _history_candidate(spec, context))
        for spec in scenario.turns[:-1]
    )
    history_fact_ids = {
        fact_id for turn in context_turns for fact_id in turn.source_fact_ids
    }
    if history_fact_ids & set(target_spec.fact_ids):
        msg = "Target facts must not appear in reviewed history"
        raise ValueError(msg)
    aliases = _target_fact_ids_by_alias(target_spec)
    target = generated.model_copy(
        update={
            "required_claims": [
                claim.model_copy(
                    update={"fact_id": aliases.get(claim.fact_id, claim.fact_id)}
                )
                for claim in generated.required_claims
            ]
        }
    )
    target_turn = _target_turn_from_generated(target_spec, target, context.facts_by_id)
    return MultiTurnCandidateDraft(
        conversation_id=scenario.conversation_scenario_id,
        persona=scenario.persona,
        conversation_length=scenario.conversation_length,
        turns=(*context_turns, target_turn),
        gold_chunk_ids=target_turn.gold_chunk_ids,
        generator_model=context.generator_model,
    )


def _context_turn_from_reviewed(
    turn_spec: TurnSpec,
    candidate: ReviewedHistoryCandidate,
) -> ConversationTurnDraft:
    """Copy one reviewed single-turn candidate into history without regeneration."""
    return ConversationTurnDraft(
        turn_number=turn_spec.turn_number,
        dialogue_act=turn_spec.dialogue_act,
        context_dependency=turn_spec.context_dependency,
        answerability=candidate.answerability,
        anchor_scenario_id=turn_spec.anchor_scenario_id,
        query=candidate.query,
        standalone_query=candidate.query,
        reference_answer=candidate.reference_answer,
        reasoning="",
        missing_information=None,
        required_claims=(),
        gold_chunk_ids=candidate.gold_chunk_ids,
        source_fact_ids=candidate.source_fact_ids,
    )


def _target_turn_from_generated(
    turn_spec: TurnSpec,
    raw: GeneratedMultiTurnTarget,
    facts_by_id: Mapping[str, LegalFact],
) -> ConversationTurnDraft:
    """Validate target grounding while leaving surface wording flexible."""
    if not raw.query.strip() or not raw.standalone_query.strip():
        msg = "Target query and standalone query must be non-empty"
        raise ValueError(msg)
    if " ".join(raw.query.split()).casefold() == " ".join(raw.standalone_query.split()).casefold():
        msg = "Target query must depend on history"
        raise ValueError(msg)
    if not raw.reference_answer.strip() or not raw.reasoning.strip():
        msg = "Target answer and reasoning must be non-empty"
        raise ValueError(msg)
    if not raw.required_claims:
        msg = "A retrieval-required target must contain required claims"
        raise ValueError(msg)
    _validate_natural_language_output(raw)

    missing_information = (
        raw.missing_information.strip() if raw.missing_information else None
    )
    if turn_spec.answerability == "answerable" and missing_information is not None:
        msg = "Answerable targets must not declare missing information"
        raise ValueError(msg)
    if turn_spec.answerability == "partially_answerable" and missing_information is None:
        msg = "Partially answerable targets must declare missing information"
        raise ValueError(msg)

    turn_facts_by_id = {
        fact_id: facts_by_id[fact_id] for fact_id in turn_spec.fact_ids
    }
    allowed_fact_ids = set(turn_spec.fact_ids)
    returned_claim_fact_ids = [claim.fact_id for claim in raw.required_claims]
    returned_fact_ids = set(returned_claim_fact_ids)
    if not returned_fact_ids <= allowed_fact_ids:
        msg = "Required claims must only reference novel target facts"
        raise ValueError(msg)
    if len(returned_claim_fact_ids) != len(returned_fact_ids):
        msg = "Each target fact may support only one required claim"
        raise ValueError(msg)
    if len({(claim.claim, claim.fact_id) for claim in raw.required_claims}) != len(raw.required_claims):
        msg = "Required claims must be unique"
        raise ValueError(msg)

    required_claims = tuple(
        RequiredClaim(
            claim=claim.claim,
            fact_id=claim.fact_id,
            evidence=tuple(
                ClaimEvidence(chunk_id=entry.chunk_id)
                for entry in turn_facts_by_id[claim.fact_id].evidence
            ),
        )
        for claim in raw.required_claims
    )
    selected_fact_ids = tuple(
        fact_id for fact_id in turn_spec.fact_ids if fact_id in returned_fact_ids
    )
    gold_chunk_ids = tuple(
        dict.fromkeys(
            entry.chunk_id
            for fact_id in selected_fact_ids
            for entry in turn_facts_by_id[fact_id].evidence
        )
    )
    return ConversationTurnDraft(
        turn_number=turn_spec.turn_number,
        dialogue_act=turn_spec.dialogue_act,
        context_dependency=turn_spec.context_dependency,
        answerability=turn_spec.answerability,
        anchor_scenario_id=turn_spec.anchor_scenario_id,
        query=raw.query,
        standalone_query=raw.standalone_query,
        reference_answer=raw.reference_answer,
        reasoning=raw.reasoning,
        missing_information=missing_information,
        required_claims=required_claims,
        gold_chunk_ids=gold_chunk_ids,
        source_fact_ids=selected_fact_ids,
    )


def _validate_natural_language_output(raw: GeneratedMultiTurnTarget) -> None:
    """Reject model-facing fact aliases in user-visible text."""
    text_fields = (
        raw.query,
        raw.standalone_query,
        raw.reference_answer,
        raw.reasoning,
        raw.missing_information or "",
        *(claim.claim for claim in raw.required_claims),
    )
    if any(_FACT_ALIAS_PATTERN.search(value) for value in text_fields):
        msg = "Natural-language output must not expose fact aliases"
        raise ValueError(msg)


def _target_fact_ids_by_alias(turn_spec: TurnSpec) -> dict[str, str]:
    """Map short aliases returned by the model back to persisted fact IDs."""
    return {
        _fact_alias(turn_spec.turn_number, index): fact_id
        for index, fact_id in enumerate(turn_spec.fact_ids, start=1)
    }


def _fact_alias(turn_number: int, fact_number: int) -> str:
    """Return one short fact alias unique inside a conversation."""
    return f"T{turn_number}F{fact_number}"


def _batch_task_key(conversation_ids: Sequence[str]) -> str:
    """Build one stable retry key for an ordered target batch."""
    return f"multi-turn-target-batch:{sha256_json({'conversation_ids': list(conversation_ids)})}"
