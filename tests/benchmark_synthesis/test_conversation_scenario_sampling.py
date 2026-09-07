"""Tests for multi-turn conversation scenario sampling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.benchmark_synthesis.conversation_scenario_sampling import (
    CONVERSATION_DISTRIBUTION_SCHEMA_VERSION,
    ConversationDistribution,
    load_conversation_distribution,
    sample_conversation_scenarios,
)
from legal_rag.benchmark_synthesis.evidence_graph import EvidenceRelation
from legal_rag.benchmark_synthesis.scenario_sampling import Scenario

_REPO_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "config" / "groundtruth"


def _scenario(scenario_id: str, fact_ids: tuple[str, ...], gold_chunk_ids: tuple[str, ...]) -> Scenario:
    """Build one anchor scenario with a fixed evidence anchor."""
    return Scenario(
        scenario_id=scenario_id,
        query_form="specific",
        answerability="answerable",
        persona="tai_xe_ca_nhan",
        query_style="standard",
        query_length="short",
        reasoning_hops=len(fact_ids),
        fact_ids=fact_ids,
        bridge=None,
        gold_chunk_ids=gold_chunk_ids,
    )


def _relation(relation_id: str, left: str, right: str) -> EvidenceRelation:
    """Build one confirmed relation connecting two facts."""
    return EvidenceRelation(
        relation_id=relation_id,
        fact_ids=(left, right),
        bundle_kind="multi_hop",
        bridge="bridge text",
        confirmed_by_model="test-model",
        run_id="run-1",
    )


def _pool() -> tuple[Scenario, ...]:
    """Build one scenario pool with shared-fact and relation-linked neighbors."""
    return (
        _scenario("scenario-1", ("fact-1", "fact-2"), ("chunk-1", "chunk-2")),
        _scenario("scenario-2", ("fact-2", "fact-3"), ("chunk-3",)),
        _scenario("scenario-3", ("fact-4",), ("chunk-4",)),
        _scenario("scenario-4", ("fact-5",), ("chunk-5",)),
    )


def _distribution_payload(**overrides: object) -> dict[str, object]:
    """Build one valid conversation-distribution payload with optional overrides."""
    payload: dict[str, object] = {
        "schema_version": CONVERSATION_DISTRIBUTION_SCHEMA_VERSION,
        "conversation_length_weights": {"3": 1.0},
        "turn_flow_weights": {"continuity": 10.0, "switch": 5.0, "clarification": 0.1, "underspecified": 0.1},
        "turn_context_dependency_weights": {"anaphora": 1.0, "ellipsis": 1.0},
        "turn_answerability_weights": {"answerable": 1.0, "partially_answerable": 0.1},
    }
    payload.update(overrides)
    return payload


def _write_distribution(tmp_path: Path, payload: dict[str, object]) -> Path:
    """Persist one distribution payload beside the tests."""
    config_path = tmp_path / "conversation_distribution.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def test_load_conversation_distribution_reads_repo_config() -> None:
    """The shipped repository config loads with every axis populated."""
    distribution = load_conversation_distribution(_REPO_CONFIG_DIR / "conversation_distribution.json")

    assert {length for length, _weight in distribution.conversation_length_weights} == {3, 5, 7, 10}
    assert {act for act, _weight in distribution.turn_flow_weights} == {
        "continuity",
        "switch",
        "return",
        "clarification",
        "underspecified",
    }


def test_load_conversation_distribution_rejects_unknown_schema(tmp_path: Path) -> None:
    """An unknown schema version fails clearly instead of silently misreading the file."""
    payload = _distribution_payload(schema_version="0.0.1")
    config_path = _write_distribution(tmp_path, payload)

    with pytest.raises(ValueError, match="Unsupported conversation-distribution schema"):
        load_conversation_distribution(config_path)


def test_distribution_rejects_unknown_values_and_bad_weights() -> None:
    """Unknown axis values and non-positive weights are rejected before sampling."""
    with pytest.raises(ValueError, match="turn_flow_weights"):
        ConversationDistribution(
            conversation_length_weights=((3, 1.0),),
            turn_flow_weights=(("unknown_act", 1.0),),
            turn_context_dependency_weights=(("anaphora", 1.0),),
            turn_answerability_weights=(("answerable", 1.0),),
        )
    with pytest.raises(ValueError, match="turn_flow_weights"):
        ConversationDistribution(
            conversation_length_weights=((3, 1.0),),
            turn_flow_weights=(("continuity", 0.0),),
            turn_context_dependency_weights=(("anaphora", 1.0),),
            turn_answerability_weights=(("answerable", 1.0),),
        )
    with pytest.raises(ValueError, match="conversation_length_weights"):
        ConversationDistribution(
            conversation_length_weights=((1, 1.0),),
            turn_flow_weights=(("continuity", 1.0),),
            turn_context_dependency_weights=(("anaphora", 1.0),),
            turn_answerability_weights=(("answerable", 1.0),),
        )


def test_first_turn_is_standalone_inquiry_on_pool_scenario(tmp_path: Path) -> None:
    """Turn 1 is always a standalone inquiry anchored on one pool scenario."""
    distribution = load_conversation_distribution(_write_distribution(tmp_path, _distribution_payload()))

    conversations = sample_conversation_scenarios(
        _pool(),
        (),
        distribution,
        total_conversations=2,
    )

    for conversation in conversations:
        first = conversation.turns[0]
        assert first.turn_number == 1
        assert first.dialogue_act == "inquiry"
        assert first.context_dependency == "standalone"
        anchor = next(s for s in _pool() if s.scenario_id == first.anchor_scenario_id)
        assert first.fact_ids == anchor.fact_ids
        assert first.gold_chunk_ids == anchor.gold_chunk_ids


def test_continuity_anchor_shares_fact_and_switch_reaches_new_scenario(tmp_path: Path) -> None:
    """Follow-up turns anchor on related scenarios while keeping per-turn evidence."""
    distribution = load_conversation_distribution(_write_distribution(tmp_path, _distribution_payload()))

    (conversation,) = sample_conversation_scenarios(
        _pool(),
        (_relation("relation-1", "fact-1", "fact-4"),),
        distribution,
        total_conversations=1,
    )

    second, third = conversation.turns[1], conversation.turns[2]
    first = conversation.turns[0]
    assert second.dialogue_act == "continuity"
    assert set(second.fact_ids) & set(first.fact_ids)
    assert third.anchor_scenario_id != first.anchor_scenario_id
    assert third.gold_chunk_ids
    assert set(conversation.gold_chunk_ids) == {
        chunk_id for turn in conversation.turns for chunk_id in turn.gold_chunk_ids
    }


def test_return_only_in_long_conversations(tmp_path: Path) -> None:
    """Topic return needs a conversation of at least six turns and an earlier anchor."""
    payload = _distribution_payload(
        conversation_length_weights={"6": 1.0},
        turn_flow_weights={"return": 10.0, "continuity": 0.1, "switch": 0.1, "clarification": 0.1, "underspecified": 0.1},
    )
    distribution = load_conversation_distribution(_write_distribution(tmp_path, payload))

    (conversation,) = sample_conversation_scenarios(
        _pool(),
        (_relation("relation-1", "fact-1", "fact-4"), _relation("relation-2", "fact-3", "fact-5")),
        distribution,
        total_conversations=1,
    )

    assert conversation.conversation_length == 6
    returns = [turn for turn in conversation.turns[1:] if turn.dialogue_act == "return"]
    assert returns
    for turn in returns:
        earlier_anchors = [prior.anchor_scenario_id for prior in conversation.turns[: turn.turn_number - 2]]
        assert turn.anchor_scenario_id in earlier_anchors


def test_underspecified_turns_force_partially_answerable(tmp_path: Path) -> None:
    """Every underspecified turn is partially answerable by construction."""
    payload = _distribution_payload(turn_flow_weights={"underspecified": 100.0})
    distribution = load_conversation_distribution(_write_distribution(tmp_path, payload))

    (conversation,) = sample_conversation_scenarios(
        _pool(),
        (),
        distribution,
        total_conversations=1,
    )

    follow_ups = conversation.turns[1:]
    assert follow_ups
    for turn in follow_ups:
        assert turn.dialogue_act == "underspecified"
        assert turn.answerability == "partially_answerable"


def test_clarification_reuses_previous_anchor(tmp_path: Path) -> None:
    """A clarification turn keeps the immediately previous anchor scenario."""
    payload = _distribution_payload(turn_flow_weights={"clarification": 100.0})
    distribution = load_conversation_distribution(_write_distribution(tmp_path, payload))

    (conversation,) = sample_conversation_scenarios(
        _pool(),
        (),
        distribution,
        total_conversations=1,
    )

    previous = conversation.turns[0]
    for turn in conversation.turns[1:]:
        assert turn.dialogue_act == "clarification"
        assert turn.anchor_scenario_id == previous.anchor_scenario_id
        previous = turn


def test_single_scenario_pool_falls_back_to_same_anchor_acts(tmp_path: Path) -> None:
    """A one-scenario pool still yields conversations through clarification turns."""
    distribution = load_conversation_distribution(_write_distribution(tmp_path, _distribution_payload()))

    (conversation,) = sample_conversation_scenarios(
        (_scenario("scenario-1", ("fact-1",), ("chunk-1",)),),
        (),
        distribution,
        total_conversations=1,
    )

    assert len(conversation.turns) == 3
    assert all(turn.anchor_scenario_id == "scenario-1" for turn in conversation.turns)
    assert all(turn.dialogue_act in {"inquiry", "clarification", "underspecified"} for turn in conversation.turns)


def test_sampling_is_deterministic_and_balances_lengths(tmp_path: Path) -> None:
    """Identical inputs produce identical conversations and lengths follow weights."""
    payload = _distribution_payload(conversation_length_weights={"3": 1.0, "5": 1.0})
    distribution = load_conversation_distribution(_write_distribution(tmp_path, payload))

    first_run = sample_conversation_scenarios(_pool(), (), distribution, total_conversations=10)
    second_run = sample_conversation_scenarios(_pool(), (), distribution, total_conversations=10)

    assert first_run == second_run
    length_counts = {3: 0, 5: 0}
    for conversation in first_run:
        length_counts[conversation.conversation_length] += 1
    assert length_counts == {3: 5, 5: 5}
