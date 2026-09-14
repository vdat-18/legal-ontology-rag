"""Plan target-first multi-turn scenarios from reviewed single-turn scenarios."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from legal_rag.benchmark_synthesis.evidence_graph import EvidenceRelation
from legal_rag.benchmark_synthesis.scenario_sampling import Scenario
from legal_rag.parser.hashing import sha256_json

CONVERSATION_SCENARIO_SCHEMA_VERSION = "2.1.0"
CONVERSATION_DISTRIBUTION_SCHEMA_VERSION = "2.0.0"

TurnDialogueAct = str
TurnContextDependency = str

_TARGET_DIALOGUE_ACTS = ("continuity", "deepening", "return")
_TARGET_CONTEXT_DEPENDENCIES = ("anaphora", "ellipsis")
_ANSWERABILITY_VALUES = ("answerable", "partially_answerable")


@dataclass(frozen=True, slots=True)
class ConversationDistribution:
    """Versioned weighting axes for target-first conversation sampling."""

    conversation_length_weights: tuple[tuple[int, float], ...]
    turn_flow_weights: tuple[tuple[TurnDialogueAct, float], ...]
    turn_context_dependency_weights: tuple[tuple[TurnContextDependency, float], ...]
    turn_answerability_weights: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        """Reject weights that cannot produce retrieval-required targets."""
        if not self.conversation_length_weights or any(
            length < 2 or weight <= 0
            for length, weight in self.conversation_length_weights
        ):
            msg = "conversation_length_weights must pair lengths >= 2 with positive weights"
            raise ValueError(msg)
        for name, weights, allowed in (
            ("turn_flow_weights", self.turn_flow_weights, _TARGET_DIALOGUE_ACTS),
            (
                "turn_context_dependency_weights",
                self.turn_context_dependency_weights,
                _TARGET_CONTEXT_DEPENDENCIES,
            ),
            ("turn_answerability_weights", self.turn_answerability_weights, _ANSWERABILITY_VALUES),
        ):
            if not weights or any(value not in allowed or weight <= 0 for value, weight in weights):
                msg = f"{name} must only use known values with positive weights"
                raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class TurnSpec:
    """One fixed history turn or retrieval-required target turn."""

    turn_number: int
    dialogue_act: TurnDialogueAct
    context_dependency: TurnContextDependency
    answerability: str
    anchor_scenario_id: str
    fact_ids: tuple[str, ...]
    gold_chunk_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the structural contract used by target generation."""
        if self.turn_number < 1:
            msg = "turn_number must be positive"
            raise ValueError(msg)
        if not self.fact_ids:
            msg = "A conversation turn must reference at least one fact"
            raise ValueError(msg)
        if self.answerability not in _ANSWERABILITY_VALUES:
            msg = f"Unsupported answerability: {self.answerability}"
            raise ValueError(msg)
        if self.turn_number == 1:
            if self.dialogue_act != "inquiry" or self.context_dependency != "standalone":
                msg = "The first conversation turn must be a standalone inquiry"
                raise ValueError(msg)
            return
        if self.dialogue_act == "switch":
            if self.context_dependency != "standalone":
                msg = "A copied context switch must remain standalone"
                raise ValueError(msg)
            return
        if self.dialogue_act not in _TARGET_DIALOGUE_ACTS:
            msg = f"Unsupported dialogue act: {self.dialogue_act}"
            raise ValueError(msg)
        if self.context_dependency not in _TARGET_CONTEXT_DEPENDENCIES:
            msg = f"Unsupported context dependency: {self.context_dependency}"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable turn-spec record."""
        return {
            "turn_number": self.turn_number,
            "dialogue_act": self.dialogue_act,
            "context_dependency": self.context_dependency,
            "answerability": self.answerability,
            "anchor_scenario_id": self.anchor_scenario_id,
            "fact_ids": list(self.fact_ids),
            "gold_chunk_ids": list(self.gold_chunk_ids),
        }


@dataclass(frozen=True, slots=True)
class ConversationScenario:
    """One conversation plan whose final turn contains only novel target facts."""

    conversation_scenario_id: str
    persona: str
    conversation_length: int
    turns: tuple[TurnSpec, ...]
    gold_chunk_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate ordering and guarantee that the target cannot leak from history."""
        if self.conversation_length < 2 or len(self.turns) != self.conversation_length:
            msg = "conversation_length must match at least two turns"
            raise ValueError(msg)
        if [turn.turn_number for turn in self.turns] != list(range(1, self.conversation_length + 1)):
            msg = "Conversation turns must be numbered consecutively from 1"
            raise ValueError(msg)
        if not self.persona:
            msg = "A conversation scenario must have a persona"
            raise ValueError(msg)
        history_fact_ids = {
            fact_id for turn in self.turns[:-1] for fact_id in turn.fact_ids
        }
        if history_fact_ids & set(self.turns[-1].fact_ids):
            msg = "Target facts must not appear in conversation history"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable conversation-scenario record."""
        return {
            "schema_version": CONVERSATION_SCENARIO_SCHEMA_VERSION,
            "conversation_scenario_id": self.conversation_scenario_id,
            "persona": self.persona,
            "conversation_length": self.conversation_length,
            "turns": [turn.to_dict() for turn in self.turns],
            "gold_chunk_ids": list(self.gold_chunk_ids),
        }


@dataclass(frozen=True, slots=True)
class _ScenarioAdjacency:
    """Scenario reachability derived from shared facts and confirmed relations."""

    shared_fact_neighbors: Mapping[str, frozenset[str]]
    relation_neighbors: Mapping[str, frozenset[str]]
    scenario_ids_by_persona: Mapping[str, frozenset[str]]


@dataclass(frozen=True, slots=True)
class _TargetPair:
    """A target and context anchor connected without sharing every target fact."""

    target_id: str
    context_id: str
    shares_fact: bool


class _ConversationUsage:
    """Deterministic balancing state for evaluated target turns."""

    def __init__(self) -> None:
        self.lengths = Counter()
        self.acts = Counter()
        self.dependencies = Counter()
        self.answerabilities = Counter()
        self.targets = Counter()
        self.contexts = Counter()


_T = TypeVar("_T")


def _pick_least_used_weighted(values: Sequence[tuple[_T, float]], counts: Counter) -> _T:
    """Pick the value minimizing usage over weight, preserving configured tie order."""
    return min(values, key=lambda entry: counts[entry[0]] / entry[1])[0]


def load_conversation_distribution(config_path: Path) -> ConversationDistribution:
    """Load the versioned target distribution configuration."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["schema_version"] != CONVERSATION_DISTRIBUTION_SCHEMA_VERSION:
        msg = f"Unsupported conversation-distribution schema: {config['schema_version']}"
        raise ValueError(msg)
    return ConversationDistribution(
        conversation_length_weights=tuple(
            (int(length), float(weight))
            for length, weight in config["conversation_length_weights"].items()
        ),
        turn_flow_weights=tuple(
            (act, float(weight)) for act, weight in config["turn_flow_weights"].items()
        ),
        turn_context_dependency_weights=tuple(
            (dependency, float(weight))
            for dependency, weight in config["turn_context_dependency_weights"].items()
        ),
        turn_answerability_weights=tuple(
            (value, float(weight))
            for value, weight in config["turn_answerability_weights"].items()
        ),
    )


def sample_conversation_scenarios(
    scenarios: Sequence[Scenario],
    relations: Sequence[EvidenceRelation],
    distribution: ConversationDistribution,
    *,
    total_conversations: int,
) -> tuple[ConversationScenario, ...]:
    """Sample conversations around balanced final targets, then attach fixed history."""
    if total_conversations <= 0:
        return ()
    if not scenarios:
        msg = "Conversation sampling requires reviewed single-turn scenarios"
        raise ValueError(msg)
    scenarios_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    adjacency = _scenario_adjacency(scenarios, relations)
    pairs = _eligible_target_pairs(scenarios, adjacency, scenarios_by_id)
    configured_answerabilities = {
        value for value, _weight in distribution.turn_answerability_weights
    }
    pairs = tuple(
        pair
        for pair in pairs
        if scenarios_by_id[pair.target_id].answerability in configured_answerabilities
    )
    if not pairs:
        msg = "Conversation sampling requires related scenarios with novel target facts"
        raise ValueError(msg)

    usage = _ConversationUsage()
    conversations: list[ConversationScenario] = []
    conversation_ids: set[str] = set()
    answerability_weights = dict(distribution.turn_answerability_weights)
    for _ in range(total_conversations):
        length = _pick_least_used_weighted(distribution.conversation_length_weights, usage.lengths)
        usage.lengths[length] += 1
        act = _pick_least_used_weighted(distribution.turn_flow_weights, usage.acts)
        usage.acts[act] += 1
        dependency = _pick_least_used_weighted(
            distribution.turn_context_dependency_weights,
            usage.dependencies,
        )
        usage.dependencies[dependency] += 1
        eligible_pairs = (
            pairs if act == "return" else tuple(pair for pair in pairs if pair.shares_fact)
        )
        if not eligible_pairs:
            msg = f"No shared-fact target pairs support dialogue act {act}"
            raise ValueError(msg)
        pair = min(
            eligible_pairs,
            key=lambda item: (
                usage.answerabilities[scenarios_by_id[item.target_id].answerability]
                / answerability_weights[scenarios_by_id[item.target_id].answerability],
                usage.targets[item.target_id],
                usage.contexts[item.context_id],
                item.target_id,
                item.context_id,
            ),
        )
        target = scenarios_by_id[pair.target_id]
        context_anchor = scenarios_by_id[pair.context_id]
        usage.answerabilities[target.answerability] += 1
        usage.targets[target.scenario_id] += 1
        usage.contexts[context_anchor.scenario_id] += 1
        for filler_offset in range(len(scenarios)):
            conversation = _build_conversation(
                target,
                context_anchor,
                act,
                dependency,
                length,
                scenarios,
                usage.contexts,
                filler_offset=filler_offset,
            )
            if conversation.conversation_scenario_id not in conversation_ids:
                break
        else:
            msg = "Unable to build another unique conversation scenario"
            raise ValueError(msg)
        conversation_ids.add(conversation.conversation_scenario_id)
        for turn in conversation.turns[:-1]:
            if turn.anchor_scenario_id != context_anchor.scenario_id:
                usage.contexts[turn.anchor_scenario_id] += 1
        conversations.append(conversation)
    return tuple(conversations)


def _scenario_adjacency(
    scenarios: Sequence[Scenario],
    relations: Sequence[EvidenceRelation],
) -> _ScenarioAdjacency:
    """Index shared-fact and confirmed-relation neighbors."""
    scenario_ids_by_fact: dict[str, set[str]] = {}
    scenario_ids_by_persona: dict[str, set[str]] = {}
    for scenario in scenarios:
        scenario_ids_by_persona.setdefault(scenario.persona, set()).add(scenario.scenario_id)
        for fact_id in scenario.fact_ids:
            scenario_ids_by_fact.setdefault(fact_id, set()).add(scenario.scenario_id)

    shared = {scenario.scenario_id: set() for scenario in scenarios}
    for peer_ids in scenario_ids_by_fact.values():
        for scenario_id in peer_ids:
            shared[scenario_id].update(peer_ids - {scenario_id})
    related = {scenario.scenario_id: set() for scenario in scenarios}
    for relation in relations:
        for left_id in scenario_ids_by_fact.get(relation.fact_ids[0], ()):
            for right_id in scenario_ids_by_fact.get(relation.fact_ids[1], ()):
                if left_id != right_id:
                    related[left_id].add(right_id)
                    related[right_id].add(left_id)
    return _ScenarioAdjacency(
        shared_fact_neighbors={key: frozenset(value) for key, value in shared.items()},
        relation_neighbors={
            key: frozenset(related[key] - shared[key]) for key in related
        },
        scenario_ids_by_persona={
            persona: frozenset(ids) for persona, ids in scenario_ids_by_persona.items()
        },
    )


def _eligible_target_pairs(
    scenarios: Sequence[Scenario],
    adjacency: _ScenarioAdjacency,
    scenarios_by_id: Mapping[str, Scenario],
) -> tuple[_TargetPair, ...]:
    """Return same-persona related pairs whose target contributes new facts."""
    pairs: list[_TargetPair] = []
    for target in scenarios:
        if target.answerability not in _ANSWERABILITY_VALUES:
            continue
        shared_neighbors = adjacency.shared_fact_neighbors[target.scenario_id]
        neighbors = (shared_neighbors | adjacency.relation_neighbors[target.scenario_id]) & (
            adjacency.scenario_ids_by_persona[target.persona]
        )
        for context_id in sorted(neighbors):
            context = scenarios_by_id[context_id]
            if set(target.fact_ids) - set(context.fact_ids):
                pairs.append(
                    _TargetPair(
                        target.scenario_id,
                        context_id,
                        shares_fact=context_id in shared_neighbors,
                    )
                )
    return tuple(pairs)


def _build_conversation(
    target: Scenario,
    context_anchor: Scenario,
    dialogue_act: str,
    dependency: str,
    length: int,
    scenarios: Sequence[Scenario],
    context_usage: Counter,
    *,
    filler_offset: int = 0,
) -> ConversationScenario:
    """Place related history and keep only facts absent from every history turn."""
    novel_target_fact_ids = tuple(
        fact_id for fact_id in target.fact_ids if fact_id not in context_anchor.fact_ids
    )
    filler_count = length - 2
    fillers = sorted(
        (
            scenario
            for scenario in scenarios
            if scenario.persona == target.persona
            and scenario.scenario_id not in {target.scenario_id, context_anchor.scenario_id}
            and not set(scenario.fact_ids) & set(novel_target_fact_ids)
        ),
        key=lambda scenario: (context_usage[scenario.scenario_id], scenario.scenario_id),
    )
    if len(fillers) < filler_count:
        msg = f"Not enough non-leaking history scenarios for target {target.scenario_id}"
        raise ValueError(msg)
    rotated_fillers = fillers[filler_offset:] + fillers[:filler_offset]
    selected_fillers = rotated_fillers[:filler_count]
    history = (
        [context_anchor, *selected_fillers]
        if dialogue_act == "return"
        else [*selected_fillers, context_anchor]
    )
    turns = [
        TurnSpec(
            turn_number=index,
            dialogue_act="inquiry" if index == 1 else "switch",
            context_dependency="standalone",
            answerability=scenario.answerability,
            anchor_scenario_id=scenario.scenario_id,
            fact_ids=scenario.fact_ids,
            gold_chunk_ids=scenario.gold_chunk_ids,
        )
        for index, scenario in enumerate(history, start=1)
    ]
    turns.append(
        TurnSpec(
            turn_number=length,
            dialogue_act=dialogue_act,
            context_dependency=dependency,
            answerability=target.answerability,
            anchor_scenario_id=target.scenario_id,
            fact_ids=novel_target_fact_ids,
            gold_chunk_ids=target.gold_chunk_ids,
        )
    )
    conversation_id = sha256_json(
        {
            "persona": target.persona,
            "conversation_length": length,
            "turns": [turn.to_dict() for turn in turns],
        }
    )
    return ConversationScenario(
        conversation_scenario_id=conversation_id,
        persona=target.persona,
        conversation_length=length,
        turns=tuple(turns),
        gold_chunk_ids=target.gold_chunk_ids,
    )
