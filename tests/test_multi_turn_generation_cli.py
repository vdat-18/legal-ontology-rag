"""Tests for the end-to-end multi-turn candidate generation command."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from legal_rag import multi_turn_generation_cli
from legal_rag.baml_client.types import ClaimReference as BamlClaimReference
from legal_rag.baml_client.types import (
    GeneratedMultiTurnTurn,
    GeneratedMultiTurnTurnBatchItem,
    MultiTurnGenerationBatchInput,
)
from legal_rag.benchmark_synthesis import multi_turn_candidate_generation
from legal_rag.benchmark_synthesis.conversation_scenario_sampling import (
    ConversationScenario,
    TurnSpec,
)
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.writer import (
    read_multi_turn_candidates,
    read_multi_turn_progress,
    write_conversation_scenarios,
    write_legal_facts,
    write_multi_turn_progress,
)

_REPO_CONFIG_DIR = Path(__file__).resolve().parent.parent / "data" / "config" / "groundtruth"

_SEEN_ITEMS: list[MultiTurnGenerationBatchInput] = []


class _FakeMultiTurnGenerationClient:
    """Record batch items and return one grounded turn for every task."""

    def __init__(self, settings: object, synthesis_settings: object) -> None:
        return None

    def generate_turn_batch(
        self, items: list[MultiTurnGenerationBatchInput]
    ) -> list[GeneratedMultiTurnTurnBatchItem]:
        """Generate one turn per item, grounded in the item's first fact."""
        _SEEN_ITEMS.extend(items)
        results = []
        for item in items:
            fact_id = item.facts[0].fact_id
            results.append(
                GeneratedMultiTurnTurnBatchItem(
                    task_id=item.task_id,
                    turn=GeneratedMultiTurnTurn(
                        query=f"Câu hỏi cho {item.task_id}",
                        reference_answer=f"Đáp án cho {item.task_id}",
                        reasoning="Theo fact của lượt.",
                        required_claims=[BamlClaimReference(claim="ý bắt buộc", fact_id=fact_id)],
                    ),
                )
            )
        return results


def _fact(fact_id: str, chunk_id: str) -> LegalFact:
    """Build one legal fact anchored on one chunk."""
    return LegalFact(
        fact_id=fact_id,
        statement=f"Statement của {fact_id}.",
        valid_from=None,
        valid_to=None,
        document_number="168/2024/NĐ-CP",
        evidence=(EvidenceReference(chunk_id=chunk_id),),
    )


def _conversation(conversation_id: str, length: int) -> ConversationScenario:
    """Build one planned conversation with one fact per turn."""
    turns = [
        TurnSpec(
            turn_number=turn_number,
            dialogue_act="inquiry" if turn_number == 1 else "continuity",
            context_dependency="standalone" if turn_number == 1 else "anaphora",
            answerability="answerable",
            anchor_scenario_id=f"scenario-{turn_number}",
            fact_ids=(f"fact-{turn_number}",),
            gold_chunk_ids=(f"chunk-{turn_number}",),
        )
        for turn_number in range(1, length + 1)
    ]
    return ConversationScenario(
        conversation_scenario_id=conversation_id,
        persona="tai_xe_ca_nhan",
        conversation_length=length,
        turns=tuple(turns),
        gold_chunk_ids=tuple(f"chunk-{turn_number}" for turn_number in range(1, length + 1)),
    )


def _configure_cli(
    monkeypatch: pytest.MonkeyPatch,
    facts_path: Path,
    conversations_path: Path,
    candidates_output_path: Path,
) -> None:
    """Set deterministic CLI arguments, environment, and a fake LLM client."""
    from legal_rag.benchmark_synthesis import llm_execution

    _SEEN_ITEMS.clear()
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:20128/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "provider/test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setattr(llm_execution.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        multi_turn_candidate_generation,
        "BamlMultiTurnGenerationClient",
        _FakeMultiTurnGenerationClient,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "legal-generate-multi-turn-candidates",
            str(facts_path),
            str(conversations_path),
            str(candidates_output_path),
            "--personas",
            str(_REPO_CONFIG_DIR / "personas.json"),
            "--synthesis-config",
            str(_REPO_CONFIG_DIR / "synthesis_v1.json"),
        ],
    )


def _prepare_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """Write facts and two planned conversations (2 and 3 turns)."""
    facts_path = tmp_path / "facts.jsonl"
    write_legal_facts(facts_path, tuple(_fact(f"fact-{index}", f"chunk-{index}") for index in (1, 2, 3)))
    conversations_path = tmp_path / "conversations.jsonl"
    write_conversation_scenarios(
        conversations_path,
        (_conversation("conv-2-turns", 2), _conversation("conv-3-turns", 3)),
    )
    return facts_path, conversations_path


def test_multi_turn_generation_cli_generates_all_conversations(tmp_path: Path) -> None:
    """The command writes every conversation with ordered turns and per-turn evidence."""
    facts_path, conversations_path = _prepare_inputs(tmp_path)
    candidates_output_path = tmp_path / "multi_turn_candidates.jsonl"
    _configure_cli(monkeypatch := pytest.MonkeyPatch(), facts_path, conversations_path, candidates_output_path)
    with monkeypatch.context():
        multi_turn_generation_cli.main()

    conversations = read_multi_turn_candidates(candidates_output_path)
    assert [conversation.conversation_id for conversation in conversations] == [
        "conv-2-turns",
        "conv-3-turns",
    ]
    for conversation in conversations:
        assert [turn.turn_number for turn in conversation.turns] == list(
            range(1, conversation.conversation_length + 1)
        )
        for turn in conversation.turns:
            (claim,) = turn.required_claims
            assert claim.fact_id == turn.source_fact_ids[0]
            assert [entry.chunk_id for entry in claim.evidence] == list(turn.gold_chunk_ids)
            assert turn.query == f"Câu hỏi cho {conversation.conversation_id}:{turn.turn_number}"
    progress_output_path = candidates_output_path.with_suffix(".progress.jsonl")
    assert len(read_multi_turn_progress(progress_output_path)) == 5


def test_multi_turn_generation_cli_passes_prior_turns_as_history(tmp_path: Path) -> None:
    """Later-turn items receive the generated question and answer of earlier turns."""
    facts_path, conversations_path = _prepare_inputs(tmp_path)
    candidates_output_path = tmp_path / "multi_turn_candidates.jsonl"
    _configure_cli(monkeypatch := pytest.MonkeyPatch(), facts_path, conversations_path, candidates_output_path)
    with monkeypatch.context():
        multi_turn_generation_cli.main()

    items_by_task_id = {item.task_id: item for item in _SEEN_ITEMS}
    second_turn = items_by_task_id["conv-3-turns:2"]
    assert len(second_turn.history) == 1
    assert second_turn.history[0].question == "Câu hỏi cho conv-3-turns:1"
    assert second_turn.history[0].answer == "Đáp án cho conv-3-turns:1"
    third_turn = items_by_task_id["conv-3-turns:3"]
    assert [prior.question for prior in third_turn.history] == [
        "Câu hỏi cho conv-3-turns:1",
        "Câu hỏi cho conv-3-turns:2",
    ]


def test_multi_turn_generation_cli_resumes_missing_turns_only(tmp_path: Path) -> None:
    """A rerun regenerates only turns missing from the checkpoint file."""
    facts_path, conversations_path = _prepare_inputs(tmp_path)
    candidates_output_path = tmp_path / "multi_turn_candidates.jsonl"
    _configure_cli(monkeypatch := pytest.MonkeyPatch(), facts_path, conversations_path, candidates_output_path)
    with monkeypatch.context():
        multi_turn_generation_cli.main()

    progress_output_path = candidates_output_path.with_suffix(".progress.jsonl")
    retained_progress = tuple(
        record for record in read_multi_turn_progress(progress_output_path) if record.task_id != "conv-3-turns:3"
    )
    write_multi_turn_progress(progress_output_path, retained_progress)

    _configure_cli(monkeypatch, facts_path, conversations_path, candidates_output_path)
    with monkeypatch.context():
        multi_turn_generation_cli.main()

    assert [item.task_id for item in _SEEN_ITEMS] == ["conv-3-turns:3"]
    assert len(read_multi_turn_candidates(candidates_output_path)) == 2
