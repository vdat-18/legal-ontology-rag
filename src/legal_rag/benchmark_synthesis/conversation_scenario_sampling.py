"""Sample multi-turn conversation scenarios from reviewed single-turn scenarios.

A conversation is a deterministic walk over the reviewed scenario pool: turn 1
anchors on one scenario, and every later turn anchors on a scenario reached
through shared facts, a confirmed evidence relation, a return to an earlier
anchor, or a clarification of the previous turn. Each turn keeps its own fact
and gold-chunk anchor before any question text exists, mirroring how
single-turn scenarios separate the sampling plan from candidate generation.
"""

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

CONVERSATION_SCENARIO_SCHEMA_VERSION = "1.0.0"
CONVERSATION_DISTRIBUTION_SCHEMA_VERSION = "1.0.0"

TurnDialogueAct = str
TurnContextDependency = str

_TURN_DIALOGUE_ACTS = ("continuity", "deepening", "switch", "return", "clarification", "underspecified")
_TURN_CONTEXT_DEPENDENCIES = ("anaphora", "ellipsis")
_TURN_ANSWERABILITY_VALUES = ("answerable", "partially_answerable", "unanswerable")
_TOPIC_RETURN_MIN_CONVERSATION_LENGTH = 6


@dataclass(frozen=True, slots=True)
class ConversationDistribution:
    """Versioned weighting axes for multi-turn conversation sampling."""

    conversation_length_weights: tuple[tuple[int, float], ...]
    turn_flow_weights: tuple[tuple[TurnDialogueAct, float], ...]
    turn_context_dependency_weights: tuple[tuple[TurnContextDependency, float], ...]
    turn_answerability_weights: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        """Reject weights that would make deterministic balancing impossible."""
        if not self.conversation_length_weights or any(
            length < 2 or weight <= 0 for length, weight in self.conversation_length_weights
        ):
            msg = "conversation_length_weights must pair lengths >= 2 with positive weights"
            raise ValueError(msg)
        for name, weights, allowed in (
            ("turn_flow_weights", self.turn_flow_weights, _TURN_DIALOGUE_ACTS),
            ("turn_context_dependency_weights", self.turn_context_dependency_weights, _TURN_CONTEXT_DEPENDENCIES),
            ("turn_answerability_weights", self.turn_answerability_weights, _TURN_ANSWERABILITY_VALUES),
        ):
            if not weights or any(act not in allowed or weight <= 0 for act, weight in weights):
                msg = f"{name} must only use known values with positive weights"
                raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class TurnSpec:
    """One planned conversation turn, anchored on one reviewed scenario."""

    turn_number: int
    dialogue_act: TurnDialogueAct
    context_dependency: TurnContextDependency
    answerability: str
    anchor_scenario_id: str
    fact_ids: tuple[str, ...]
    gold_chunk_ids: tuple[str, ...]

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
    """One planned multi-turn conversation before any question text exists."""

    conversation_scenario_id: str
    persona: str
    conversation_length: int
    turns: tuple[TurnSpec, ...]
    gold_chunk_ids: tuple[str, ...]

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


def load_conversation_distribution(config_path: Path) -> ConversationDistribution:
    """Load the versioned conversation-distribution configuration."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["schema_version"] != CONVERSATION_DISTRIBUTION_SCHEMA_VERSION:
        msg = f"Unsupported conversation-distribution schema: {config['schema_version']}"
        raise ValueError(msg)
    return ConversationDistribution(
        conversation_length_weights=tuple(
            (int(length), float(weight)) for length, weight in config["conversation_length_weights"].items()
        ),
        turn_flow_weights=tuple((act, float(weight)) for act, weight in config["turn_flow_weights"].items()),
        turn_context_dependency_weights=tuple(
            (dependency, float(weight)) for dependency, weight in config["turn_context_dependency_weights"].items()
        ),
        turn_answerability_weights=tuple(
            (value, float(weight)) for value, weight in config["turn_answerability_weights"].items()
        ),
    )


@dataclass(frozen=True, slots=True)
class _ScenarioAdjacency:
    """Scenario reachability derived from shared facts and confirmed relations."""

    shared_fact_neighbors: Mapping[str, frozenset[str]]
    relation_neighbors: Mapping[str, frozenset[str]]


class _ConversationUsage:
    """Deterministic least-used balancing state shared across sampled conversations."""

    def __init__(self, distribution: ConversationDistribution) -> None:
        """Initialize usage counters for every configured sampling axis."""
        self.lengths = Counter()
        self.acts = Counter()
        self.dependencies = Counter()
        self.answerabilities = Counter()
        self.seeds = Counter()

    def pick_length(self, distribution: ConversationDistribution) -> int:
        """Pick the least-used conversation length by configured weight."""
        return _pick_least_used_weighted(distribution.conversation_length_weights, self.lengths)

    def pick_act(self, feasible: Sequence[TurnDialogueAct], weights: Sequence[tuple[TurnDialogueAct, float]]) -> str:
        """Pick the least-used feasible dialogue act by configured weight."""
        feasible_weights = tuple((act, weight) for act, weight in weights if act in feasible)
        return _pick_least_used_weighted(feasible_weights, self.acts)

    def pick_dependency(self, distribution: ConversationDistribution) -> str:
        """Pick the least-used context dependency by configured weight."""
        return _pick_least_used_weighted(distribution.turn_context_dependency_weights, self.dependencies)

    def pick_answerability(self, distribution: ConversationDistribution) -> str:
        """Pick the least-used turn answerability by configured weight."""
        return _pick_least_used_weighted(distribution.turn_answerability_weights, self.answerabilities)


_T = TypeVar("_T")


def _pick_least_used_weighted(values: Sequence[tuple[_T, float]], counts: Counter) -> _T:
    """Pick the value minimizing usage over weight, preferring configured order on ties."""
    return min(values, key=lambda entry: counts[entry[0]] / entry[1])[0]


def sample_conversation_scenarios(
    scenarios: Sequence[Scenario],
    relations: Sequence[EvidenceRelation],
    distribution: ConversationDistribution,
    *,
    total_conversations: int,
) -> tuple[ConversationScenario, ...]:
    """Sample conversation scenarios as deterministic walks over the scenario pool."""
    if total_conversations <= 0:
        return ()
    if not scenarios:
        msg = "Conversation sampling requires at least one anchor scenario"
        raise ValueError(msg)
    scenarios_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    adjacency = _scenario_adjacency(scenarios, relations)
    usage = _ConversationUsage(distribution)

    conversations: list[ConversationScenario] = []
    for _ in range(total_conversations):
        length = usage.pick_length(distribution)
        usage.lengths[length] += 1
        seed = min(scenarios, key=lambda scenario: (usage.seeds[scenario.scenario_id], scenario.scenario_id))
        usage.seeds[seed.scenario_id] += 1
        conversations.append(_sample_conversation(seed, length, scenarios_by_id, adjacency, distribution, usage))
    return tuple(conversations)


def _scenario_adjacency(scenarios: Sequence[Scenario], relations: Sequence[EvidenceRelation]) -> _ScenarioAdjacency:
    """Index scenario reachability through shared facts and confirmed evidence relations."""
    scenario_ids_by_fact: dict[str, set[str]] = {}
    for scenario in scenarios:
        for fact_id in scenario.fact_ids:
            scenario_ids_by_fact.setdefault(fact_id, set()).add(scenario.scenario_id)

    shared: dict[str, set[str]] = {scenario.scenario_id: set() for scenario in scenarios}
    for peer_ids in scenario_ids_by_fact.values():
        for scenario_id in peer_ids:
            shared[scenario_id].update(peer_ids)

    related: dict[str, set[str]] = {scenario.scenario_id: set() for scenario in scenarios}
    for relation in relations:
        for left_id in scenario_ids_by_fact.get(relation.fact_ids[0], ()):
            for right_id in scenario_ids_by_fact.get(relation.fact_ids[1], ()):
                if left_id != right_id:
                    related[left_id].add(right_id)

    return _ScenarioAdjacency(
        shared_fact_neighbors={scenario_id: frozenset(peers - {scenario_id}) for scenario_id, peers in shared.items()},
        relation_neighbors={
            scenario_id: frozenset(related[scenario_id] - shared[scenario_id] - {scenario_id}) for scenario_id in shared
        },
    )


def _sample_conversation(
    seed: Scenario,
    length: int,
    scenarios_by_id: Mapping[str, Scenario],
    adjacency: _ScenarioAdjacency,
    distribution: ConversationDistribution,
    usage: _ConversationUsage,
) -> ConversationScenario:
    """Plan every turn of one conversation, anchoring each turn on one scenario."""
    anchors: list[str] = []
    anchor_usage: Counter[str] = Counter()
    turns: list[TurnSpec] = []

    opening_answerability = usage.pick_answerability(distribution)
    usage.answerabilities[opening_answerability] += 1
    turns.append(
        TurnSpec(
            turn_number=1,
            dialogue_act="inquiry",
            context_dependency="standalone",
            answerability=opening_answerability,
            anchor_scenario_id=seed.scenario_id,
            fact_ids=seed.fact_ids,
            gold_chunk_ids=seed.gold_chunk_ids,
        )
    )
    anchors.append(seed.scenario_id)
    anchor_usage[seed.scenario_id] += 1

    for turn_number in range(2, length + 1):
        dialogue_act = usage.pick_act(
            _feasible_acts(length, anchors, adjacency, scenarios_by_id),
            distribution.turn_flow_weights,
        )
        usage.acts[dialogue_act] += 1
        anchor_id = _resolve_anchor(dialogue_act, anchors, adjacency, scenarios_by_id, anchor_usage)
        anchor_usage[anchor_id] += 1
        anchors.append(anchor_id)
        anchor = scenarios_by_id[anchor_id]
        dependency = usage.pick_dependency(distribution)
        usage.dependencies[dependency] += 1
        answerability = (
            "partially_answerable" if dialogue_act == "underspecified" else usage.pick_answerability(distribution)
        )
        usage.answerabilities[answerability] += 1
        turns.append(
            TurnSpec(
                turn_number=turn_number,
                dialogue_act=dialogue_act,
                context_dependency=dependency,
                answerability=answerability,
                anchor_scenario_id=anchor.scenario_id,
                fact_ids=anchor.fact_ids,
                gold_chunk_ids=anchor.gold_chunk_ids,
            )
        )

    return ConversationScenario(
        conversation_scenario_id=sha256_json(
            {
                "persona": seed.persona,
                "conversation_length": length,
                "turns": [turn.to_dict() for turn in turns],
            }
        ),
        persona=seed.persona,
        conversation_length=length,
        turns=tuple(turns),
        gold_chunk_ids=tuple(sorted({chunk_id for turn in turns for chunk_id in turn.gold_chunk_ids})),
    )


def _feasible_acts(
    conversation_length: int,
    anchors: Sequence[str],
    adjacency: _ScenarioAdjacency,
    scenarios_by_id: Mapping[str, Scenario],
) -> tuple[TurnDialogueAct, ...]:
    """Return every dialogue act whose anchor requirement can still be satisfied."""
    previous_anchor = anchors[-1]
    unused_exists = len(set(anchors)) < len(scenarios_by_id)
    feasible = ["clarification", "underspecified", "deepening"]
    if unused_exists:
        feasible.append("switch")
    if adjacency.shared_fact_neighbors[previous_anchor] - set(anchors):
        feasible.append("continuity")
    if conversation_length >= _TOPIC_RETURN_MIN_CONVERSATION_LENGTH and anchors[:-1]:
        feasible.append("return")
    return tuple(feasible)


def _resolve_anchor(
    dialogue_act: TurnDialogueAct,
    anchors: Sequence[str],
    adjacency: _ScenarioAdjacency,
    scenarios_by_id: Mapping[str, Scenario],
    anchor_usage: Counter[str],
) -> str:
    """Resolve the anchor scenario id for one planned dialogue act."""
    previous_anchor = anchors[-1]
    if dialogue_act in {"clarification", "underspecified", "deepening"}:
        return previous_anchor
    if dialogue_act == "return":
        candidates = tuple(dict.fromkeys(anchors[:-1]))
        return min(candidates, key=lambda anchor_id: (anchor_usage[anchor_id], anchor_id))
    if dialogue_act == "continuity":
        candidates = adjacency.shared_fact_neighbors[previous_anchor] - set(anchors)
        return min(candidates, key=lambda anchor_id: (anchor_usage[anchor_id], anchor_id))
    candidates = adjacency.relation_neighbors[previous_anchor] - set(anchors)
    if not candidates:
        candidates = frozenset(scenarios_by_id) - set(anchors)
    return min(candidates, key=lambda anchor_id: (anchor_usage[anchor_id], anchor_id))
