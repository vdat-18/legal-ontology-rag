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
    if not bundles or total_samples <= 0:
        return ()

    used_persona_bundle_pairs: set[tuple[str, str]] = set()
    usage_counts: dict[str, Counter[str]] = {
        "persona": Counter(),
        "query_style": Counter(),
        "query_length": Counter(),
        "query_form": Counter(),
        "answerability": Counter(),
    }
    bundles_by_kind: dict[BundleKind, list[tuple[EvidenceBundle, tuple[LegalFact, ...]]]] = {}
    for bundle in bundles:
        facts = tuple(facts_by_id[fact_id] for fact_id in bundle.fact_ids)
        bundles_by_kind.setdefault(bundle.kind, []).append((bundle, facts))
    for kind_bundles in bundles_by_kind.values():
        kind_bundles.sort(key=lambda item: item[0].bundle_id)
    weighted_kinds = tuple(
        (kind, weight)
        for kind, weight in config.distribution.bundle_kind_weights
        if weight > 0 and bundles_by_kind.get(kind)
    )
    if not weighted_kinds:
        return ()

    scenarios: list[Scenario] = []
    kind_usage: Counter[BundleKind] = Counter()
    bundle_offsets: Counter[BundleKind] = Counter()
    bundle_attempts: Counter[str] = Counter()
    kind_order = {kind: index for index, (kind, _weight) in enumerate(weighted_kinds)}
    weights = dict(weighted_kinds)

    while len(scenarios) < total_samples:
        bundle_kind = min(
            weights,
            key=lambda kind: (kind_usage[kind] / weights[kind], kind_order[kind]),
        )
        kind_bundles = bundles_by_kind[bundle_kind]
        bundle, facts = kind_bundles[bundle_offsets[bundle_kind] % len(kind_bundles)]
        bundle_offsets[bundle_kind] += 1
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
