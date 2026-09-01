"""Load versioned scenario configuration and sample diverse scenarios.

Scenario generation only covers bundles built from real evidence relations,
so it can only ever produce `answerable` and `partially_answerable`
scenarios. `underspecified` and `unanswerable` scenarios are not derived
from a bundle at all — the corpus has no adequate evidence by definition —
and need a separate generation path, out of scope here.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from legal_rag.benchmark_synthesis.evidence_graph import BundleKind, EvidenceBundle
from legal_rag.benchmark_synthesis.models import LegalFact
from legal_rag.parser.hashing import sha256_json

SCENARIO_SCHEMA_VERSION = "4.0.0"
SCENARIO_DISTRIBUTION_SCHEMA_VERSION = "2.0.0"


@dataclass(frozen=True, slots=True)
class PersonaCatalog:
    """Versioned catalog of persona identifiers."""

    persona_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScenarioDistribution:
    """Versioned catalog of surface-diversity axis values."""

    query_styles: tuple[str, ...]
    query_lengths: tuple[str, ...]
    query_forms: tuple[str, ...]
    bundle_answerability_values: tuple[str, ...]
    bundle_kind_weights: tuple[tuple[BundleKind, float], ...]


@dataclass(frozen=True, slots=True)
class ScenarioSamplingConfig:
    """Combined configuration needed to sample scenarios."""

    personas: PersonaCatalog
    distribution: ScenarioDistribution


@dataclass(frozen=True, slots=True)
class Scenario:
    """One structured scenario, generated before any question text exists."""

    scenario_id: str
    query_form: str
    answerability: str
    persona: str
    query_style: str
    query_length: str
    reasoning_hops: int
    fact_ids: tuple[str, ...]
    bridge: str | None
    gold_chunk_ids: tuple[str, ...]
    bundle_kind: BundleKind = "single_hop"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable scenario record."""
        return {
            "schema_version": SCENARIO_SCHEMA_VERSION,
            "scenario_id": self.scenario_id,
            "bundle_kind": self.bundle_kind,
            "query_form": self.query_form,
            "answerability": self.answerability,
            "persona": self.persona,
            "query_style": self.query_style,
            "query_length": self.query_length,
            "reasoning_hops": self.reasoning_hops,
            "fact_ids": list(self.fact_ids),
            "bridge": self.bridge,
            "gold_chunk_ids": list(self.gold_chunk_ids),
        }


@dataclass(frozen=True, slots=True)
class ScenarioReplacementResult:
    """A repaired scenario set and the evidence bundles removed from sampling."""

    scenarios: tuple[Scenario, ...]
    replaced_scenario_ids: frozenset[str]
    excluded_bundle_ids: frozenset[str]


def load_personas(config_path: Path) -> PersonaCatalog:
    """Load the versioned persona configuration."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return PersonaCatalog(persona_ids=tuple(entry["id"] for entry in config["personas"]))


def load_scenario_distribution(config_path: Path) -> ScenarioDistribution:
    """Load the versioned surface-diversity axis configuration."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["schema_version"] != SCENARIO_DISTRIBUTION_SCHEMA_VERSION:
        msg = f"Unsupported scenario-distribution schema: {config['schema_version']}"
        raise ValueError(msg)
    return ScenarioDistribution(
        query_styles=tuple(config["query_styles"]),
        query_lengths=tuple(config["query_lengths"]),
        query_forms=tuple(config["query_forms"]),
        bundle_answerability_values=tuple(config["bundle_answerability_values"]),
        bundle_kind_weights=tuple(
            (bundle_kind, float(weight)) for bundle_kind, weight in config["bundle_kind_weights"].items()
        ),
    )


def load_persona_descriptions(config_path: Path) -> Mapping[str, str]:
    """Load a persona_id to human-readable Vietnamese description mapping."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {entry["id"]: entry["description_vi"] for entry in config["personas"]}


def sample_scenarios(
    bundles: Sequence[EvidenceBundle],
    facts_by_id: Mapping[str, LegalFact],
    config: ScenarioSamplingConfig,
    *,
    total_samples: int,
) -> tuple[Scenario, ...]:
    """Sample diverse scenarios with weighted bundle-kind and surface-axis coverage."""
    return _sample_scenarios(
        bundles,
        facts_by_id,
        config,
        total_samples=total_samples,
        retained_scenarios=(),
    )


def replace_scenarios(
    bundles: Sequence[EvidenceBundle],
    facts_by_id: Mapping[str, LegalFact],
    config: ScenarioSamplingConfig,
    existing_scenarios: Sequence[Scenario],
    replace_scenario_ids: frozenset[str],
    *,
    total_samples: int,
) -> ScenarioReplacementResult:
    """Replace explicitly rejected scenarios while preserving every unaffected scenario."""
    existing_by_id = {scenario.scenario_id: scenario for scenario in existing_scenarios}
    if len(existing_by_id) != len(existing_scenarios):
        msg = "Existing scenarios contain duplicate scenario IDs"
        raise ValueError(msg)
    unknown_ids = replace_scenario_ids - existing_by_id.keys()
    if unknown_ids:
        msg = f"Unknown scenarios requested for replacement: {sorted(unknown_ids)}"
        raise ValueError(msg)

    bundles_by_key = _bundles_by_scenario_key(bundles)
    scenario_bundle_ids = {
        scenario.scenario_id: _scenario_bundle_id(scenario, bundles_by_key) for scenario in existing_scenarios
    }
    excluded_bundle_ids = frozenset(scenario_bundle_ids[scenario_id] for scenario_id in replace_scenario_ids)
    expanded_replacement_ids = frozenset(
        scenario_id for scenario_id, bundle_id in scenario_bundle_ids.items() if bundle_id in excluded_bundle_ids
    )
    retained_scenarios = tuple(
        scenario for scenario in existing_scenarios if scenario.scenario_id not in expanded_replacement_ids
    )
    retained_bundle_ids = frozenset(scenario_bundle_ids[scenario.scenario_id] for scenario in retained_scenarios)
    eligible_bundles = tuple(bundle for bundle in bundles if bundle.bundle_id in retained_bundle_ids)
    if not eligible_bundles:
        msg = "Scenario replacement requires at least one retained evidence bundle"
        raise ValueError(msg)
    scenarios = _sample_scenarios(
        eligible_bundles,
        facts_by_id,
        config,
        total_samples=total_samples,
        retained_scenarios=retained_scenarios,
    )
    if len(scenarios) != total_samples:
        msg = f"Scenario replacement produced {len(scenarios)} of {total_samples} requested scenarios"
        raise ValueError(msg)
    return ScenarioReplacementResult(
        scenarios=scenarios,
        replaced_scenario_ids=expanded_replacement_ids,
        excluded_bundle_ids=excluded_bundle_ids,
    )


def _sample_scenarios(
    bundles: Sequence[EvidenceBundle],
    facts_by_id: Mapping[str, LegalFact],
    config: ScenarioSamplingConfig,
    *,
    total_samples: int,
    retained_scenarios: Sequence[Scenario],
) -> tuple[Scenario, ...]:
    """Sample scenarios from current usage counts, retaining approved records unchanged."""
    if not bundles or total_samples <= 0:
        return tuple(retained_scenarios[:total_samples])
    if len(retained_scenarios) > total_samples:
        msg = "Retained scenario count exceeds requested total"
        raise ValueError(msg)

    bundles_by_kind: dict[BundleKind, dict[int, list[tuple[EvidenceBundle, tuple[LegalFact, ...]]]]] = {}
    for bundle in bundles:
        facts = tuple(facts_by_id[fact_id] for fact_id in bundle.fact_ids)
        bundles_by_kind.setdefault(bundle.kind, {}).setdefault(len(bundle.fact_ids), []).append((bundle, facts))
    for bundles_by_hops in bundles_by_kind.values():
        for hop_bundles in bundles_by_hops.values():
            hop_bundles.sort(key=lambda item: item[0].bundle_id)
    weighted_kinds = tuple(
        (kind, weight)
        for kind, weight in config.distribution.bundle_kind_weights
        if weight > 0 and bundles_by_kind.get(kind)
    )
    if not weighted_kinds:
        return ()

    kind_order = {kind: index for index, (kind, _weight) in enumerate(weighted_kinds)}
    weights = dict(weighted_kinds)

    bundles_by_key = _bundles_by_scenario_key(bundles)
    scenarios = list(retained_scenarios)
    (
        used_persona_bundle_pairs,
        usage_counts,
        kind_usage,
        hop_usage,
        bundle_offsets,
        bundle_attempts,
    ) = _initial_sampling_state(retained_scenarios, bundles_by_key)

    while len(scenarios) < total_samples:
        bundle_kind = min(
            weights,
            key=lambda kind: (kind_usage[kind] / weights[kind], kind_order[kind]),
        )
        bundles_by_hops = bundles_by_kind[bundle_kind]
        unused_hops = tuple(
            hops for hops, hop_bundles in bundles_by_hops.items() if hop_usage[(bundle_kind, hops)] < len(hop_bundles)
        )
        reasoning_hops = min(
            unused_hops or bundles_by_hops,
            key=lambda hops: (hop_usage[(bundle_kind, hops)], hops),
        )
        hop_bundles = bundles_by_hops[reasoning_hops]
        offset_key = (bundle_kind, reasoning_hops)
        unused_bundle_count = sum(bundle_attempts[bundle.bundle_id] == 0 for bundle, _facts in hop_bundles)
        while True:
            bundle, facts = hop_bundles[bundle_offsets[offset_key] % len(hop_bundles)]
            bundle_offsets[offset_key] += 1
            if unused_bundle_count == 0 or bundle_attempts[bundle.bundle_id] == 0:
                break
        attempt_index = bundle_attempts[bundle.bundle_id]
        bundle_attempts[bundle.bundle_id] += 1

        persona = _pick_persona(
            config.personas.persona_ids,
            bundle.bundle_id,
            used_persona_bundle_pairs,
            usage_counts["persona"],
        )
        query_style = _pick_least_used(config.distribution.query_styles, usage_counts["query_style"])
        query_length = _pick_least_used(config.distribution.query_lengths, usage_counts["query_length"])
        query_form = _pick_least_used(config.distribution.query_forms, usage_counts["query_form"])
        answerability = _pick_least_used(config.distribution.bundle_answerability_values, usage_counts["answerability"])

        used_persona_bundle_pairs.add((persona, bundle.bundle_id))
        kind_usage[bundle_kind] += 1
        hop_usage[offset_key] += 1
        usage_counts["persona"][persona] += 1
        usage_counts["query_style"][query_style] += 1
        usage_counts["query_length"][query_length] += 1
        usage_counts["query_form"][query_form] += 1
        usage_counts["answerability"][answerability] += 1

        scenarios.append(
            Scenario(
                scenario_id=sha256_json(
                    {
                        "answerability": answerability,
                        "attempt_index": attempt_index,
                        "bridge": bundle.bridge,
                        "bundle_id": bundle.bundle_id,
                        "bundle_kind": bundle.kind,
                        "facts": [fact.fact_id for fact in facts],
                        "persona": persona,
                        "query_form": query_form,
                        "query_length": query_length,
                        "query_style": query_style,
                        "reasoning_hops": len(bundle.fact_ids),
                    }
                ),
                bundle_kind=bundle.kind,
                query_form=query_form,
                answerability=answerability,
                persona=persona,
                query_style=query_style,
                query_length=query_length,
                reasoning_hops=len(bundle.fact_ids),
                fact_ids=bundle.fact_ids,
                bridge=bundle.bridge,
                gold_chunk_ids=tuple(sorted({evidence.chunk_id for fact in facts for evidence in fact.evidence})),
            )
        )
    return tuple(scenarios)


def _initial_sampling_state(
    retained_scenarios: Sequence[Scenario],
    bundles_by_key: Mapping[tuple[BundleKind, tuple[str, ...], str | None], EvidenceBundle],
) -> tuple[
    set[tuple[str, str]],
    dict[str, Counter[str]],
    Counter[BundleKind],
    Counter[tuple[BundleKind, int]],
    Counter[tuple[BundleKind, int]],
    Counter[str],
]:
    """Build sampler usage counts from scenarios that must remain unchanged."""
    used_persona_bundle_pairs: set[tuple[str, str]] = set()
    usage_counts: dict[str, Counter[str]] = {
        "persona": Counter(),
        "query_style": Counter(),
        "query_length": Counter(),
        "query_form": Counter(),
        "answerability": Counter(),
    }
    kind_usage: Counter[BundleKind] = Counter()
    hop_usage: Counter[tuple[BundleKind, int]] = Counter()
    bundle_attempts: Counter[str] = Counter()
    for scenario in retained_scenarios:
        bundle_id = _scenario_bundle_id(scenario, bundles_by_key)
        offset_key = (scenario.bundle_kind, scenario.reasoning_hops)
        used_persona_bundle_pairs.add((scenario.persona, bundle_id))
        kind_usage[scenario.bundle_kind] += 1
        hop_usage[offset_key] += 1
        bundle_attempts[bundle_id] += 1
        usage_counts["persona"][scenario.persona] += 1
        usage_counts["query_style"][scenario.query_style] += 1
        usage_counts["query_length"][scenario.query_length] += 1
        usage_counts["query_form"][scenario.query_form] += 1
        usage_counts["answerability"][scenario.answerability] += 1
    return (
        used_persona_bundle_pairs,
        usage_counts,
        kind_usage,
        hop_usage,
        Counter(hop_usage),
        bundle_attempts,
    )


def _bundles_by_scenario_key(
    bundles: Sequence[EvidenceBundle],
) -> dict[tuple[BundleKind, tuple[str, ...], str | None], EvidenceBundle]:
    """Index bundles by the fields persisted on a scenario."""
    bundles_by_key: dict[tuple[BundleKind, tuple[str, ...], str | None], EvidenceBundle] = {}
    for bundle in bundles:
        key = (bundle.kind, bundle.fact_ids, bundle.bridge)
        if key in bundles_by_key:
            msg = f"Evidence bundles are ambiguous for scenario key: {key}"
            raise ValueError(msg)
        bundles_by_key[key] = bundle
    return bundles_by_key


def _scenario_bundle_id(
    scenario: Scenario,
    bundles_by_key: Mapping[tuple[BundleKind, tuple[str, ...], str | None], EvidenceBundle],
) -> str:
    """Resolve the evidence bundle backing one persisted scenario."""
    key = (scenario.bundle_kind, scenario.fact_ids, scenario.bridge)
    bundle = bundles_by_key.get(key)
    if bundle is None:
        msg = f"Scenario {scenario.scenario_id} does not match an eligible evidence bundle"
        raise ValueError(msg)
    return bundle.bundle_id


def _pick_persona(
    persona_ids: Sequence[str],
    bundle_id: str,
    used_pairs: set[tuple[str, str]],
    counts: Counter[str],
) -> str:
    """Pick the least-used persona not yet paired with this bundle."""
    unused = tuple(persona_id for persona_id in persona_ids if (persona_id, bundle_id) not in used_pairs)
    return _pick_least_used(unused or persona_ids, counts)


def _pick_least_used(values: Sequence[str], counts: Counter[str]) -> str:
    """Pick the value used least so far, preferring the first on ties."""
    return min(values, key=lambda value: counts[value])
