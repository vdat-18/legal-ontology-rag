"""Tests for target-first multi-turn conversation sampling."""

from __future__ import annotations

import json
from collections import Counter
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

_REPO_CONFIG_DIR = Path(__file__).resolve().parents[2] / "data" / "config" / "groundtruth"


def _scenario(
    scenario_id: str,
    fact_ids: tuple[str, ...],
    *,
    answerability: str = "answerable",
    persona: str = "tai_xe_ca_nhan",
) -> Scenario:
    """Build one reviewed scenario."""
    return Scenario(
        scenario_id=scenario_id,
        query_form="specific",
        answerability=answerability,
        persona=persona,
        query_style="standard",
        query_length="short",
        reasoning_hops=len(fact_ids),
        fact_ids=fact_ids,
        bridge=None,
        gold_chunk_ids=tuple(f"chunk-{fact_id}" for fact_id in fact_ids),
    )


def _relation(relation_id: str, left: str, right: str) -> EvidenceRelation:
    """Build one confirmed fact relation."""
    return EvidenceRelation(
        relation_id=relation_id,
        fact_ids=(left, right),
        bundle_kind="multi_hop",
        bridge="bridge",
        confirmed_by_model="model",
        run_id="run",
    )


def _pool() -> tuple[Scenario, ...]:
    """Build enough same-persona scenarios for long non-leaking history."""
    return (
        _scenario("target-a", ("shared-a", "new-a")),
        _scenario("context-a", ("shared-a", "context-a")),
        _scenario("target-p", ("shared-p", "new-p"), answerability="partially_answerable"),
        _scenario("context-p", ("shared-p", "context-p")),
        *tuple(_scenario(f"filler-{index}", (f"filler-fact-{index}",)) for index in range(10)),
    )


def _distribution_payload(**overrides: object) -> dict[str, object]:
    """Build a valid target distribution payload."""
    payload: dict[str, object] = {
        "schema_version": CONVERSATION_DISTRIBUTION_SCHEMA_VERSION,
        "conversation_length_weights": {"3": 1.0},
        "turn_flow_weights": {"continuity": 1.0, "deepening": 1.0, "return": 1.0},
        "turn_context_dependency_weights": {"anaphora": 1.0, "ellipsis": 1.0},
        "turn_answerability_weights": {"answerable": 1.0, "partially_answerable": 1.0},
    }
    payload.update(overrides)
    return payload


def _write_distribution(tmp_path: Path, payload: dict[str, object]) -> Path:
    config_path = tmp_path / "conversation_distribution.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def test_repo_distribution_contains_only_retrieval_required_targets() -> None:
    """The shipped distribution excludes history-only and unverified targets."""
    distribution = load_conversation_distribution(
        _REPO_CONFIG_DIR / "conversation_distribution.json"
    )

    assert {value for value, _ in distribution.turn_flow_weights} == {
        "continuity",
        "deepening",
        "return",
    }
    assert {value for value, _ in distribution.turn_answerability_weights} == {
        "answerable",
        "partially_answerable",
    }


def test_distribution_rejects_old_unanswerable_and_switch_targets() -> None:
    """Unsupported target classes fail before producing misleading labels."""
    with pytest.raises(ValueError, match="turn_flow_weights"):
        ConversationDistribution(
            conversation_length_weights=((3, 1.0),),
            turn_flow_weights=(("switch", 1.0),),
            turn_context_dependency_weights=(("anaphora", 1.0),),
            turn_answerability_weights=(("answerable", 1.0),),
        )
    with pytest.raises(ValueError, match="turn_answerability_weights"):
        ConversationDistribution(
            conversation_length_weights=((3, 1.0),),
            turn_flow_weights=(("continuity", 1.0),),
            turn_context_dependency_weights=(("anaphora", 1.0),),
            turn_answerability_weights=(("unanswerable", 1.0),),
        )


def test_sampling_keeps_target_facts_out_of_history(tmp_path: Path) -> None:
    """Every final target contains only facts absent from all earlier turns."""
    distribution = load_conversation_distribution(
        _write_distribution(tmp_path, _distribution_payload())
    )
    conversations = sample_conversation_scenarios(
        _pool(), (), distribution, total_conversations=12
    )

    for conversation in conversations:
        target = conversation.turns[-1]
        history_fact_ids = {
            fact_id for turn in conversation.turns[:-1] for fact_id in turn.fact_ids
        }
        assert target.fact_ids
        assert not history_fact_ids & set(target.fact_ids)
        assert target.dialogue_act in {"continuity", "deepening", "return"}
        assert target.context_dependency in {"anaphora", "ellipsis"}
        assert all(turn.context_dependency == "standalone" for turn in conversation.turns[:-1])


def test_continuity_and_deepening_use_a_shared_fact_anchor(tmp_path: Path) -> None:
    """Adjacent follow-ups stay on a topic established by the referenced history."""
    distribution = load_conversation_distribution(
        _write_distribution(
            tmp_path,
            _distribution_payload(
                turn_flow_weights={"continuity": 1.0, "deepening": 1.0}
            ),
        )
    )
    scenarios = _pool()
    scenarios_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    conversations = sample_conversation_scenarios(
        scenarios, (), distribution, total_conversations=4
    )

    for conversation in conversations:
        target_anchor = scenarios_by_id[conversation.turns[-1].anchor_scenario_id]
        context_anchor = scenarios_by_id[conversation.turns[-2].anchor_scenario_id]
        assert set(target_anchor.fact_ids) & set(context_anchor.fact_ids)


def test_target_distribution_is_balanced_independently(tmp_path: Path) -> None:
    """Answerability, dialogue act, and dependency counts apply to final turns."""
    distribution = load_conversation_distribution(
        _write_distribution(tmp_path, _distribution_payload())
    )
    conversations = sample_conversation_scenarios(
        _pool(), (), distribution, total_conversations=12
    )
    targets = [conversation.turns[-1] for conversation in conversations]

    assert Counter(turn.answerability for turn in targets) == {
        "answerable": 6,
        "partially_answerable": 6,
    }
    assert Counter(turn.dialogue_act for turn in targets) == {
        "continuity": 4,
        "deepening": 4,
        "return": 4,
    }
    assert Counter(turn.context_dependency for turn in targets) == {
        "anaphora": 6,
        "ellipsis": 6,
    }


def test_return_places_related_context_before_fillers(tmp_path: Path) -> None:
    """A return target points back past intervening standalone history."""
    payload = _distribution_payload(
        conversation_length_weights={"5": 1.0},
        turn_flow_weights={"return": 1.0},
    )
    distribution = load_conversation_distribution(_write_distribution(tmp_path, payload))
    (conversation,) = sample_conversation_scenarios(
        _pool(), (), distribution, total_conversations=1
    )

    first_fact_ids = set(conversation.turns[0].fact_ids)
    target_anchor = next(
        scenario for scenario in _pool() if scenario.scenario_id == conversation.turns[-1].anchor_scenario_id
    )
    assert first_fact_ids & set(target_anchor.fact_ids)
    assert conversation.turns[-2].anchor_scenario_id.startswith("filler-")


def test_sampling_is_deterministic_and_balances_lengths(tmp_path: Path) -> None:
    """Identical inputs produce identical plans and configured length counts."""
    payload = _distribution_payload(conversation_length_weights={"3": 1.0, "5": 1.0})
    distribution = load_conversation_distribution(_write_distribution(tmp_path, payload))

    first = sample_conversation_scenarios(_pool(), (), distribution, total_conversations=10)
    second = sample_conversation_scenarios(_pool(), (), distribution, total_conversations=10)

    assert first == second
    assert Counter(item.conversation_length for item in first) == {3: 5, 5: 5}
    assert len({item.conversation_scenario_id for item in first}) == len(first)


def test_sampling_requires_a_related_non_leaking_pair(tmp_path: Path) -> None:
    """An isolated scenario cannot silently become a fake multi-turn target."""
    distribution = load_conversation_distribution(
        _write_distribution(tmp_path, _distribution_payload())
    )

    with pytest.raises(ValueError, match="related scenarios with novel target facts"):
        sample_conversation_scenarios(
            (_scenario("only", ("fact",)),),
            (),
            distribution,
            total_conversations=1,
        )


def test_confirmed_relations_can_connect_disjoint_fact_sets(tmp_path: Path) -> None:
    """Related disjoint scenarios remain valid target/history pairs."""
    scenarios = (
        _scenario("context", ("context-fact",)),
        _scenario("target", ("target-fact",)),
        _scenario("filler", ("filler-fact",)),
    )
    distribution = load_conversation_distribution(
        _write_distribution(
            tmp_path,
            _distribution_payload(turn_flow_weights={"return": 1.0}),
        )
    )

    (conversation,) = sample_conversation_scenarios(
        scenarios,
        (_relation("relation", "context-fact", "target-fact"),),
        distribution,
        total_conversations=1,
    )

    assert set(conversation.turns[-1].fact_ids) in ({"context-fact"}, {"target-fact"})
    assert not set(conversation.turns[0].fact_ids) & set(conversation.turns[-1].fact_ids)
