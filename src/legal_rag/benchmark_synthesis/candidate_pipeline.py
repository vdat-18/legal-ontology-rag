"""Checkpointed orchestration for one single-turn candidate work item."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from legal_rag.benchmark_synthesis.binding_gate import check_binding
from legal_rag.benchmark_synthesis.candidate_generation import (
    CandidateDraft,
    CandidateGenerationClient,
    evolve_candidate,
    generate_candidate,
    refine_surface,
)
from legal_rag.benchmark_synthesis.candidate_record import CandidateProgressRecord
from legal_rag.benchmark_synthesis.llm_execution import execute_llm_task
from legal_rag.benchmark_synthesis.models import LegalFact
from legal_rag.benchmark_synthesis.scenario_sampling import Scenario
from legal_rag.benchmark_synthesis.verification import VerificationClient, verify_candidate
from legal_rag.chunker.models import Chunk


@dataclass(frozen=True, slots=True)
class CandidatePipelineContext:
    """Dependencies and configured limits shared by candidate work items."""

    facts_by_id: Mapping[str, LegalFact]
    chunks_by_id: Mapping[str, Chunk]
    persona_descriptions: Mapping[str, str]
    query_type_descriptions: Mapping[str, str]
    query_type_catalog: Sequence[str]
    evolution_operations: Sequence[str]
    generation_client: CandidateGenerationClient
    verification_client: VerificationClient
    generator_model: str
    verifier_model: str
    span_match_threshold: float
    max_quality_retries: int
    llm_max_retries: int
    llm_retry_delay_seconds: float


def process_candidate_task(
    scenario: Scenario,
    attempt_index: int,
    progress: CandidateProgressRecord | None,
    context: CandidatePipelineContext,
    checkpoint: Callable[[CandidateProgressRecord], None],
) -> CandidateProgressRecord:
    """Resume one scenario attempt from its latest persisted stage and finish it."""
    task_id = f"{scenario.scenario_id}:{attempt_index}"
    if progress is None:
        draft = execute_llm_task(
            f"{task_id}:generation",
            lambda: generate_candidate(
                scenario,
                context.facts_by_id,
                context.persona_descriptions[scenario.persona],
                context.query_type_descriptions[scenario.query_type],
                context.generation_client,
                generator_model=context.generator_model,
            ),
            max_retries=context.llm_max_retries,
            retry_delay_seconds=context.llm_retry_delay_seconds,
        )
        progress = CandidateProgressRecord(task_id, scenario.scenario_id, attempt_index, "generated", draft)
        checkpoint(progress)

    if not progress.draft.surface_refined:
        progress = _refine_with_checkpoints(progress, scenario, context, checkpoint)

    if len(progress.draft.evolutions) < len(context.evolution_operations):
        progress = _evolve_with_checkpoints(progress, scenario, context, checkpoint)

    if progress.binding is None:
        binding = check_binding(
            progress.draft,
            context.facts_by_id,
            context.chunks_by_id,
            span_match_threshold=context.span_match_threshold,
        )
        progress = CandidateProgressRecord(
            task_id,
            scenario.scenario_id,
            attempt_index,
            "binding_checked",
            progress.draft,
            binding,
        )
        checkpoint(progress)

    if not progress.binding.passed:
        completed = CandidateProgressRecord(
            task_id,
            scenario.scenario_id,
            attempt_index,
            "completed",
            progress.draft,
            progress.binding,
        )
        checkpoint(completed)
        return completed

    verification = progress.verification or execute_llm_task(
        f"{task_id}:verification",
        lambda: verify_candidate(
            progress.draft,
            context.facts_by_id,
            context.query_type_catalog,
            context.verification_client,
            verifier_model=context.verifier_model,
        ),
        max_retries=context.llm_max_retries,
        retry_delay_seconds=context.llm_retry_delay_seconds,
    )
    completed = CandidateProgressRecord(
        task_id,
        scenario.scenario_id,
        attempt_index,
        "completed",
        progress.draft,
        progress.binding,
        verification,
    )
    checkpoint(completed)
    return completed


def _refine_with_checkpoints(
    progress: CandidateProgressRecord,
    scenario: Scenario,
    context: CandidatePipelineContext,
    checkpoint: Callable[[CandidateProgressRecord], None],
) -> CandidateProgressRecord:
    """Refine surface text while persisting every successful critic call."""
    current = progress

    def _checkpoint_draft(draft: CandidateDraft) -> None:
        nonlocal current
        current = CandidateProgressRecord(
            progress.task_id,
            scenario.scenario_id,
            progress.attempt_index,
            "surface",
            draft,
        )
        checkpoint(current)

    draft = refine_surface(
        current.draft,
        context.generation_client,
        max_retries=context.max_quality_retries,
        llm_max_retries=context.llm_max_retries,
        llm_retry_delay_seconds=context.llm_retry_delay_seconds,
        on_attempt=_checkpoint_draft,
    )
    current = CandidateProgressRecord(
        progress.task_id,
        scenario.scenario_id,
        progress.attempt_index,
        "surface_refined",
        draft,
    )
    checkpoint(current)
    return current


def _evolve_with_checkpoints(
    progress: CandidateProgressRecord,
    scenario: Scenario,
    context: CandidatePipelineContext,
    checkpoint: Callable[[CandidateProgressRecord], None],
) -> CandidateProgressRecord:
    """Apply remaining evolutions while persisting every successful call."""
    current = progress

    def _checkpoint_draft(draft: CandidateDraft) -> None:
        nonlocal current
        current = CandidateProgressRecord(
            progress.task_id,
            scenario.scenario_id,
            progress.attempt_index,
            "evolution",
            draft,
        )
        checkpoint(current)

    draft = evolve_candidate(
        current.draft,
        context.evolution_operations,
        context.generation_client,
        llm_max_retries=context.llm_max_retries,
        llm_retry_delay_seconds=context.llm_retry_delay_seconds,
        on_operation_applied=_checkpoint_draft,
    )
    current = CandidateProgressRecord(
        progress.task_id,
        scenario.scenario_id,
        progress.attempt_index,
        "evolved",
        draft,
    )
    checkpoint(current)
    return current
