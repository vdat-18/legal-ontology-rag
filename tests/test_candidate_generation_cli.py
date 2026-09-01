"""Tests for the end-to-end single-turn candidate generation command."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

from legal_rag import candidate_generation_cli
from legal_rag.baml_client.types import (
    CandidateGenerationBatchInput,
    GeneratedSingleTurnCandidateBatchItem,
)
from legal_rag.baml_client.types import ClaimReference as BamlClaimReference
from legal_rag.baml_client.types import GeneratedSingleTurnCandidate as BamlGeneratedCandidate
from legal_rag.benchmark_synthesis import candidate_generation, writer
from legal_rag.benchmark_synthesis.candidate_generation import (
    CANDIDATE_GENERATION_PROMPT_VERSION,
    LEGACY_CANDIDATE_GENERATION_PROMPT_VERSION,
    LEGACY_CANDIDATE_SCHEMA_VERSION,
)
from legal_rag.benchmark_synthesis.config import SynthesisSettings
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.scenario_sampling import Scenario, load_persona_descriptions
from legal_rag.benchmark_synthesis.stage_cache import (
    build_candidate_generation_fingerprint,
    stage_cache_path,
    write_stage_cache_fingerprint,
)
from legal_rag.benchmark_synthesis.writer import (
    read_candidate_progress,
    read_candidates,
    write_candidate_progress,
    write_candidates,
    write_legal_facts,
    write_scenarios,
)

_REPO_CONFIG_DIR = Path(__file__).resolve().parent.parent / "data" / "config" / "groundtruth"
_REFERENCE_ANSWER = "Người điều khiển ô tô vượt đèn đỏ bị phạt tiền."


def _fact() -> LegalFact:
    """Build one approved-ready legal fact for CLI tests."""
    return LegalFact(
        fact_id="fact-1",
        statement="Người điều khiển ô tô vượt đèn đỏ bị phạt tiền.",
        valid_from=None,
        valid_to=None,
        document_number="168/2024/NĐ-CP",
        evidence=(EvidenceReference(chunk_id="chunk-1"),),
    )


def _scenario() -> Scenario:
    """Build one scenario referencing the approved fact."""
    return Scenario(
        scenario_id="scenario-1",
        query_form="specific",
        answerability="answerable",
        persona="tai_xe_ca_nhan",
        query_style="standard",
        query_length="short",
        reasoning_hops=1,
        fact_ids=("fact-1",),
        bridge=None,
        gold_chunk_ids=("chunk-1",),
    )


class _FakeGenerationClient:
    """Return a fixed candidate grounded in the source fact."""

    def __init__(self, settings: object, synthesis_settings: object) -> None:
        return None

    def generate(self, *args: object, **kwargs: object) -> BamlGeneratedCandidate:
        """Return one candidate grounded in the real chunk text."""
        return BamlGeneratedCandidate(
            query="oto vuot den do bi phat bao nhieu",
            reference_answer=_REFERENCE_ANSWER,
            reasoning="Theo Điều 6.",
            required_claims=[BamlClaimReference(claim="Mức phạt tiền áp dụng", fact_id="fact-1")],
        )

    def generate_batch(self, items: list[CandidateGenerationBatchInput]) -> list[GeneratedSingleTurnCandidateBatchItem]:
        """Generate one candidate for every batch item."""
        return [
            GeneratedSingleTurnCandidateBatchItem(task_id=item.task_id, candidate=self.generate()) for item in items
        ]


def _configure_cli(
    monkeypatch: pytest.MonkeyPatch,
    facts_path: Path,
    scenarios_path: Path,
    candidates_output_path: Path,
    *,
    candidates_per_scenario: int,
) -> None:
    """Set deterministic CLI arguments, environment, and fake LLM clients."""
    from legal_rag.benchmark_synthesis import llm_execution

    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:20128/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "provider/test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setattr(llm_execution.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(candidate_generation, "BamlCandidateGenerationClient", _FakeGenerationClient)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "legal-generate-candidates",
            str(facts_path),
            str(scenarios_path),
            str(candidates_output_path),
            "--personas",
            str(_REPO_CONFIG_DIR / "personas.json"),
            "--candidates-per-scenario",
            str(candidates_per_scenario),
            "--synthesis-config",
            str(_REPO_CONFIG_DIR / "synthesis_v1.json"),
        ],
    )


def test_candidate_generation_cli_produces_candidate_from_scenario(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A generated candidate is written through end to end with no gating checks."""
    facts_path = tmp_path / "legal_facts.jsonl"
    scenarios_path = tmp_path / "scenarios.jsonl"
    candidates_path = tmp_path / "single_turn.jsonl"

    write_legal_facts(facts_path, [_fact()])
    write_scenarios(scenarios_path, [_scenario()])
    _configure_cli(monkeypatch, facts_path, scenarios_path, candidates_path, candidates_per_scenario=1)

    candidate_generation_cli.main()

    drafts = read_candidates(candidates_path)
    assert len(drafts) == 1
    assert drafts[0].answerability == "answerable"
    assert drafts[0].query == "oto vuot den do bi phat bao nhieu"
    assert stage_cache_path(candidates_path).exists()


def test_candidate_generation_cli_generates_multiple_attempts_per_scenario(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """candidates_per_scenario controls how many draws are made per scenario."""
    facts_path = tmp_path / "legal_facts.jsonl"
    scenarios_path = tmp_path / "scenarios.jsonl"
    candidates_path = tmp_path / "single_turn.jsonl"

    write_legal_facts(facts_path, [_fact()])
    write_scenarios(scenarios_path, [_scenario()])
    _configure_cli(monkeypatch, facts_path, scenarios_path, candidates_path, candidates_per_scenario=3)

    candidate_generation_cli.main()

    drafts = read_candidates(candidates_path)
    assert len(drafts) == 3


def test_candidate_generation_cli_rejects_progress_from_a_different_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Candidate progress cannot be resumed after a prompt-facing runtime input changes."""
    facts_path = tmp_path / "legal_facts.jsonl"
    scenarios_path = tmp_path / "scenarios.jsonl"
    candidates_path = tmp_path / "single_turn.jsonl"
    write_legal_facts(facts_path, [_fact()])
    write_scenarios(scenarios_path, [_scenario()])
    _configure_cli(monkeypatch, facts_path, scenarios_path, candidates_path, candidates_per_scenario=1)

    candidate_generation_cli.main()
    monkeypatch.setenv("LLM_MODEL", "provider/changed-model")

    with pytest.raises(ValueError, match="different synthesis run"):
        candidate_generation_cli.main()


def test_candidate_generation_cli_rebuilds_candidates_from_progress_after_output_write_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Completed progress prevents an LLM replay if the derived candidates file was not written."""
    facts_path = tmp_path / "legal_facts.jsonl"
    scenarios_path = tmp_path / "scenarios.jsonl"
    candidates_path = tmp_path / "single_turn.jsonl"
    progress_path = tmp_path / "single_turn.progress.jsonl"
    write_legal_facts(facts_path, [_fact()])
    write_scenarios(scenarios_path, [_scenario()])
    _configure_cli(monkeypatch, facts_path, scenarios_path, candidates_path, candidates_per_scenario=1)
    original_write_candidates = writer.write_candidates

    def _crash_candidates(*_args: object, **_kwargs: object) -> None:
        raise SystemExit

    monkeypatch.setattr(writer, "write_candidates", _crash_candidates)

    with pytest.raises(SystemExit):
        candidate_generation_cli.main()

    assert read_candidate_progress(progress_path)[0].completed is True
    monkeypatch.setattr(writer, "write_candidates", original_write_candidates)

    class _NoReplayClient(_FakeGenerationClient):
        def generate_batch(
            self, items: list[CandidateGenerationBatchInput]
        ) -> list[GeneratedSingleTurnCandidateBatchItem]:
            pytest.fail("Completed candidate progress must bypass generation")

    monkeypatch.setattr(candidate_generation, "BamlCandidateGenerationClient", _NoReplayClient)

    candidate_generation_cli.main()

    assert len(read_candidates(candidates_path)) == 1


def test_candidate_generation_cli_selectively_regenerates_legacy_prompt_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A prompt update replaces only explicitly selected completed tasks."""
    facts_path = tmp_path / "legal_facts.jsonl"
    scenarios_path = tmp_path / "scenarios.jsonl"
    candidates_path = tmp_path / "single_turn.jsonl"
    progress_path = tmp_path / "single_turn.progress.jsonl"
    write_legal_facts(facts_path, [_fact()])
    write_scenarios(scenarios_path, [_scenario()])
    _configure_cli(monkeypatch, facts_path, scenarios_path, candidates_path, candidates_per_scenario=1)
    candidate_generation_cli.main()

    progress = read_candidate_progress(progress_path)
    legacy_draft = replace(progress[0].draft, prompt_version=LEGACY_CANDIDATE_GENERATION_PROMPT_VERSION)
    legacy_progress = replace(progress[0], draft=legacy_draft)
    write_candidate_progress(progress_path, (legacy_progress,))
    write_candidates(candidates_path, (legacy_draft,))
    synthesis_settings = SynthesisSettings.from_path(_REPO_CONFIG_DIR / "synthesis_v1.json")
    write_stage_cache_fingerprint(
        stage_cache_path(candidates_path),
        build_candidate_generation_fingerprint(
            (_fact(),),
            (_scenario(),),
            load_persona_descriptions(_REPO_CONFIG_DIR / "personas.json"),
            synthesis_settings,
            candidates_per_scenario=1,
            llm_base_url="http://localhost:20128/v1",
            llm_model="provider/test-model",
            prompt_version=LEGACY_CANDIDATE_GENERATION_PROMPT_VERSION,
            candidate_schema_version=LEGACY_CANDIDATE_SCHEMA_VERSION,
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [*sys.argv, "--regenerate-task-id", "scenario-1:0"],
    )

    candidate_generation_cli.main()

    regenerated = read_candidates(candidates_path)
    assert len(regenerated) == 1
    assert regenerated[0].prompt_version == CANDIDATE_GENERATION_PROMPT_VERSION


def test_candidate_generation_cli_migrates_only_replaced_scenarios(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Scenario replacement preserves retained progress and generates only new tasks."""
    facts_path = tmp_path / "legal_facts.jsonl"
    scenarios_path = tmp_path / "scenarios.jsonl"
    candidates_path = tmp_path / "single_turn.jsonl"
    write_legal_facts(facts_path, [_fact()])
    first = _scenario()
    retained = replace(_scenario(), scenario_id="scenario-2", query_style="formal")
    replacement = replace(_scenario(), scenario_id="scenario-3", query_style="polite")
    write_scenarios(scenarios_path, (first, retained))
    _configure_cli(monkeypatch, facts_path, scenarios_path, candidates_path, candidates_per_scenario=1)
    candidate_generation_cli.main()
    retained_sample_id = {draft.scenario_id: draft.sample_id for draft in read_candidates(candidates_path)}[
        retained.scenario_id
    ]

    write_scenarios(scenarios_path, (retained, replacement))
    generated_task_ids: list[str] = []

    class _ReplacementClient(_FakeGenerationClient):
        def generate_batch(
            self, items: list[CandidateGenerationBatchInput]
        ) -> list[GeneratedSingleTurnCandidateBatchItem]:
            generated_task_ids.extend(item.task_id for item in items)
            return super().generate_batch(items)

    monkeypatch.setattr(candidate_generation, "BamlCandidateGenerationClient", _ReplacementClient)
    monkeypatch.setattr(
        sys,
        "argv",
        [*sys.argv, "--replace-scenario-id", first.scenario_id],
    )

    candidate_generation_cli.main()

    drafts = read_candidates(candidates_path)
    assert [draft.scenario_id for draft in drafts] == [retained.scenario_id, replacement.scenario_id]
    assert drafts[0].sample_id == retained_sample_id
    assert generated_task_ids == [f"{replacement.scenario_id}:0"]
