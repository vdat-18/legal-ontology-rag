"""Tests for bounded-concurrency candidate generation and resume."""

from __future__ import annotations

import threading
import time
from datetime import date

from legal_rag.baml_client.types import (
    CandidateEvolutionBatchInput,
    CandidateGenerationBatchInput,
    EvolvedCandidateBatchItem,
    GeneratedSingleTurnCandidateBatchItem,
)
from legal_rag.baml_client.types import ClaimReference as BamlClaimReference
from legal_rag.baml_client.types import EvolvedCandidate as BamlEvolvedCandidate
from legal_rag.baml_client.types import GeneratedSingleTurnCandidate as BamlGeneratedCandidate
from legal_rag.benchmark_synthesis.candidate_generation import CandidateDraft
from legal_rag.benchmark_synthesis.candidate_pipeline import CandidatePipelineContext, generate_candidates
from legal_rag.benchmark_synthesis.candidate_record import CandidateProgressRecord
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.scenario_sampling import Scenario


def _fact(fact_id: str = "fact-1") -> LegalFact:
    """Build one minimal legal fact for pipeline tests."""
    return LegalFact(
        fact_id=fact_id,
        fact_content_sha256=f"content-{fact_id}",
        statement="Người điều khiển ô tô vượt đèn đỏ bị phạt tiền.",
        conduct="Vượt đèn đỏ",
        subject="Người điều khiển ô tô",
        conditions=(),
        monetary_penalty="4.000.000 đồng",
        license_points_deducted=None,
        additional_penalties=(),
        remedial_measures=(),
        valid_from=None,
        valid_to=None,
        document_number="168/2024/NĐ-CP",
        provision_path="Điều 6",
        evidence=(EvidenceReference(chunk_id="chunk-1"),),
    )


def _scenario(scenario_id: str) -> Scenario:
    """Build one minimal scenario for pipeline tests."""
    return Scenario(
        scenario_id=scenario_id,
        query_type="penalty_for_conduct",
        query_form="specific",
        answerability="answerable",
        persona="tai_xe_ca_nhan",
        query_style="standard",
        query_length="short",
        reasoning_hops=1,
        as_of_date=None,
        fact_ids=("fact-1",),
        bridge=(),
        gold_chunk_ids=("chunk-1",),
    )


def _context(generation_client: object, evolution_operations: tuple[str, ...] = ()) -> CandidatePipelineContext:
    """Build a pipeline context with no evolution rounds, for fast deterministic tests."""
    return CandidatePipelineContext(
        facts_by_id={"fact-1": _fact()},
        persona_descriptions={"tai_xe_ca_nhan": "Tài xế cá nhân"},
        query_type_descriptions={"penalty_for_conduct": "Mức phạt của một hành vi"},
        evolution_operations=evolution_operations,
        generation_client=generation_client,
        generator_model="m",
        llm_max_retries=0,
        llm_retry_delay_seconds=0,
    )


def _generated() -> BamlGeneratedCandidate:
    """Build one fixed generated candidate."""
    return BamlGeneratedCandidate(
        query_vi="q",
        reference_answer_vi="a",
        reasoning_vi="r",
        required_claims=[
            BamlClaimReference(claim="c", fact_id="fact-1", evidence_chunk_ids=["chunk-1"]),
        ],
    )


class _FakeGenerationClient:
    """Return a fixed generated candidate; evolve is unused since no rounds are configured."""

    def generate(self, *args: object, **kwargs: object) -> BamlGeneratedCandidate:
        """Return the fixed generated candidate."""
        return _generated()

    def generate_batch(self, items: list[CandidateGenerationBatchInput]) -> list[GeneratedSingleTurnCandidateBatchItem]:
        """Generate one fixed result for every batch item."""
        return [
            GeneratedSingleTurnCandidateBatchItem(task_id=item.task_id, candidate=self.generate()) for item in items
        ]

    def evolve(self, *args: object, **kwargs: object) -> BamlEvolvedCandidate:
        """Unused when evolution_operations is empty."""
        raise NotImplementedError

    def evolve_batch(self, items: list[CandidateEvolutionBatchInput]) -> list[EvolvedCandidateBatchItem]:
        """Evolve every batch item through the single-item fake."""
        return [
            EvolvedCandidateBatchItem(
                task_id=item.task_id,
                candidate=self.evolve(
                    item.query_vi,
                    item.reference_answer_vi,
                    item.reasoning_vi,
                    item.operation,
                    item.query_length,
                ),
            )
            for item in items
        ]


def _draft(scenario_id: str) -> CandidateDraft:
    """Build one completed candidate draft for resume tests."""
    return CandidateDraft(
        sample_id=f"sample-{scenario_id}",
        scenario_id=scenario_id,
        query_vi="already done",
        reference_answer_vi="a",
        reasoning_vi="r",
        required_claims=(),
        answer_fields=(),
        evolutions=(),
        gold_chunk_ids=("chunk-1",),
        query_type="penalty_for_conduct",
        query_form="specific",
        persona="tai_xe_ca_nhan",
        query_style="standard",
        query_length="short",
        answerability="answerable",
        reasoning_hops=1,
        as_of_date=date(2025, 1, 1),
        source_fact_ids=("fact-1",),
        source_fact_content_sha256=("content-fact-1",),
        generator_model="m",
    )


def test_generate_candidates_skips_already_completed_tasks() -> None:
    """A task already marked completed in progress is not resubmitted for work."""
    scenarios = [_scenario("scenario-1"), _scenario("scenario-2")]
    progress_by_task_id = {
        "scenario-1:0": CandidateProgressRecord("scenario-1:0", "scenario-1", 0, "completed", _draft("scenario-1")),
    }
    calls: list[str] = []

    class _CountingClient(_FakeGenerationClient):
        def generate(self, *args: object, **kwargs: object) -> BamlGeneratedCandidate:
            calls.append("generate")
            return _generated()

    failures = generate_candidates(
        scenarios,
        1,
        progress_by_task_id,
        _context(_CountingClient()),
        lambda record: progress_by_task_id.__setitem__(record.task_id, record),
        batch_size=1,
        max_concurrency=1,
    )

    assert failures == ()
    assert calls == ["generate"]
    assert progress_by_task_id["scenario-1:0"].draft.query_vi == "already done"
    assert progress_by_task_id["scenario-2:0"].completed is True


def test_generate_candidates_respects_candidates_per_scenario() -> None:
    """Each scenario gets exactly candidates_per_scenario independent attempts."""
    scenarios = [_scenario("scenario-1")]
    progress_by_task_id: dict[str, CandidateProgressRecord] = {}

    failures = generate_candidates(
        scenarios,
        3,
        progress_by_task_id,
        _context(_FakeGenerationClient()),
        lambda record: progress_by_task_id.__setitem__(record.task_id, record),
        batch_size=2,
        max_concurrency=2,
    )

    assert failures == ()
    completed_task_ids = {task_id for task_id, record in progress_by_task_id.items() if record.completed}
    assert completed_task_ids == {"scenario-1:0", "scenario-1:1", "scenario-1:2"}


def test_generate_candidates_reports_a_task_that_fails_every_attempt_as_a_failure() -> None:
    """A scenario whose generation keeps raising is reported as a failure, others still complete.

    max_concurrency=1 makes the single worker execute submitted tasks in strict FIFO order, so
    the first generate() call deterministically corresponds to the first submitted scenario.
    """
    scenarios = [_scenario("scenario-1"), _scenario("scenario-2")]
    progress_by_task_id: dict[str, CandidateProgressRecord] = {}
    call_count = 0

    class _FailsOnFirstCallClient(_FakeGenerationClient):
        def generate(self, *args: object, **kwargs: object) -> BamlGeneratedCandidate:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                msg = "provider timed out"
                raise TimeoutError(msg)
            return _generated()

    failures = generate_candidates(
        scenarios,
        1,
        progress_by_task_id,
        _context(_FailsOnFirstCallClient()),
        lambda record: progress_by_task_id.__setitem__(record.task_id, record),
        batch_size=1,
        max_concurrency=1,
    )

    assert len(failures) == 1
    assert failures[0].task_key == "scenario-1:0"
    assert "scenario-1:0" not in progress_by_task_id
    assert progress_by_task_id["scenario-2:0"].completed is True


def test_generate_candidates_bounds_concurrent_generation_calls() -> None:
    """At most max_concurrency generation calls run at once, and it is actually reached."""
    scenarios = [_scenario(f"scenario-{i}") for i in range(4)]
    progress_by_task_id: dict[str, CandidateProgressRecord] = {}
    in_flight = 0
    max_observed = 0
    lock = threading.Lock()

    class _TrackingClient(_FakeGenerationClient):
        def generate(self, *args: object, **kwargs: object) -> BamlGeneratedCandidate:
            nonlocal in_flight, max_observed
            with lock:
                in_flight += 1
                max_observed = max(max_observed, in_flight)
            time.sleep(0.05)
            with lock:
                in_flight -= 1
            return _generated()

    failures = generate_candidates(
        scenarios,
        1,
        progress_by_task_id,
        _context(_TrackingClient()),
        lambda record: progress_by_task_id.__setitem__(record.task_id, record),
        batch_size=1,
        max_concurrency=2,
    )

    assert failures == ()
    assert max_observed == 2
    assert len(progress_by_task_id) == 4


def test_generate_candidates_batches_generation_and_evolution_calls() -> None:
    """Both LLM stages receive at most batch_size tasks per request."""
    scenarios = [_scenario(f"scenario-{index}") for index in range(12)]
    progress_by_task_id: dict[str, CandidateProgressRecord] = {}
    generation_batch_sizes: list[int] = []
    evolution_batch_sizes: list[int] = []

    class _BatchTrackingClient(_FakeGenerationClient):
        def generate_batch(
            self, items: list[CandidateGenerationBatchInput]
        ) -> list[GeneratedSingleTurnCandidateBatchItem]:
            generation_batch_sizes.append(len(items))
            return super().generate_batch(items)

        def evolve(self, *args: object, **kwargs: object) -> BamlEvolvedCandidate:
            return BamlEvolvedCandidate(query_vi="evolved-q", reference_answer_vi="evolved-a", reasoning_vi="evolved-r")

        def evolve_batch(self, items: list[CandidateEvolutionBatchInput]) -> list[EvolvedCandidateBatchItem]:
            evolution_batch_sizes.append(len(items))
            return super().evolve_batch(items)

    failures = generate_candidates(
        scenarios,
        1,
        progress_by_task_id,
        _context(_BatchTrackingClient(), ("concretizing",)),
        lambda record: progress_by_task_id.__setitem__(record.task_id, record),
        batch_size=5,
        max_concurrency=1,
    )

    assert failures == ()
    assert generation_batch_sizes == [5, 5, 2]
    assert evolution_batch_sizes == [5, 5, 2]
    assert all(record.completed for record in progress_by_task_id.values())
