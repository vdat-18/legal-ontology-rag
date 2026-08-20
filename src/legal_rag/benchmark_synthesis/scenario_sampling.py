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
from datetime import date
from pathlib import Path
from typing import Any

from legal_rag.benchmark_synthesis.evidence_graph import BundleKind, EvidenceBundle
from legal_rag.benchmark_synthesis.models import LegalFact
from legal_rag.parser.hashing import sha256_json

SCENARIO_SCHEMA_VERSION = "1.0.0"
SCENARIO_DISTRIBUTION_SCHEMA_VERSION = "1.1.0"


@dataclass(frozen=True, slots=True)
class QueryTypeDefinition:
    """One query type and the evidence-bundle shapes capable of supporting it."""

    query_type_id: str
    bundle_kinds: tuple[BundleKind, ...]
    minimum_facts: int


@dataclass(frozen=True, slots=True)
class QueryTaxonomy:
    """Versioned catalog of query type identifiers."""

    definitions: tuple[QueryTypeDefinition, ...]

    @property
    def query_type_ids(self) -> tuple[str, ...]:
        """Return query type identifiers in configured order."""
        return tuple(definition.query_type_id for definition in self.definitions)


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

    query_taxonomy: QueryTaxonomy
    personas: PersonaCatalog
    distribution: ScenarioDistribution


@dataclass(frozen=True, slots=True)
class EvolutionConfig:
    """Versioned catalog of evidence-preserving evolution operations."""

    operation_ids: tuple[str, ...]
    max_rounds: int


@dataclass(frozen=True, slots=True)
class Scenario:
    """One structured scenario, generated before any question text exists."""

    scenario_id: str
    query_type: str
    query_form: str
    answerability: str
    persona: str
    query_style: str
    query_length: str
    reasoning_hops: int
    as_of_date: date | None
    fact_ids: tuple[str, ...]
    bridge: tuple[str, ...]
    gold_chunk_ids: tuple[str, ...]
    bundle_kind: BundleKind = "single_hop"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable scenario record."""
        return {
            "schema_version": SCENARIO_SCHEMA_VERSION,
            "scenario_id": self.scenario_id,
            "bundle_kind": self.bundle_kind,
            "query_type": self.query_type,
            "query_form": self.query_form,
            "answerability": self.answerability,
            "persona": self.persona,
            "query_style": self.query_style,
            "query_length": self.query_length,
            "reasoning_hops": self.reasoning_hops,
            "as_of_date": self.as_of_date.isoformat() if self.as_of_date else None,
            "fact_ids": list(self.fact_ids),
            "bridge": list(self.bridge),
            "gold_chunk_ids": list(self.gold_chunk_ids),
        }


def load_query_taxonomy(config_path: Path) -> QueryTaxonomy:
    """Load the versioned query-taxonomy configuration."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return QueryTaxonomy(
        definitions=tuple(
            QueryTypeDefinition(
                query_type_id=entry["id"],
                bundle_kinds=tuple(entry["bundle_kinds"]),
                minimum_facts=entry.get("minimum_facts", 1),
            )
            for entry in config["query_types"]
            if entry.get("mvp_enabled", True)
        )
    )


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


def load_evolution_config(config_path: Path) -> EvolutionConfig:
    """Load the versioned evolution-operation configuration."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return EvolutionConfig(
        operation_ids=tuple(entry["id"] for entry in config["operations"]),
        max_rounds=config["max_rounds"],
    )


def load_query_type_descriptions(config_path: Path) -> Mapping[str, str]:
    """Load a query_type_id to human-readable Vietnamese description mapping."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {entry["id"]: entry["description_vi"] for entry in config["query_types"]}


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
        "query_style": Counter(),
        "query_length": Counter(),
        "query_form": Counter(),
        "query_type": Counter(),
        "answerability": Counter(),
    }
    bundles_by_kind: dict[BundleKind, list[tuple[EvidenceBundle, tuple[LegalFact, ...], tuple[str, ...]]]] = {}
    for bundle in bundles:
        facts = tuple(facts_by_id[fact_id] for fact_id in bundle.fact_ids)
        eligible_query_types = _eligible_query_types(bundle, facts, config.query_taxonomy)
        if eligible_query_types:
            bundles_by_kind.setdefault(bundle.kind, []).append((bundle, facts, eligible_query_types))
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
        bundle, facts, eligible_query_types = kind_bundles[bundle_offsets[bundle_kind] % len(kind_bundles)]
        bundle_offsets[bundle_kind] += 1
        attempt_index = bundle_attempts[bundle.bundle_id]
        bundle_attempts[bundle.bundle_id] += 1

        persona = _pick_persona(config.personas.persona_ids, bundle.bundle_id, used_persona_bundle_pairs, attempt_index)
        query_style = _pick_least_used(config.distribution.query_styles, usage_counts["query_style"])
        query_length = _pick_least_used(config.distribution.query_lengths, usage_counts["query_length"])
        query_form = _pick_least_used(config.distribution.query_forms, usage_counts["query_form"])
        query_type = _pick_least_used(eligible_query_types, usage_counts["query_type"])
        answerability = _pick_least_used(config.distribution.bundle_answerability_values, usage_counts["answerability"])
        as_of_date = _pick_shared_as_of_date(facts)

        used_persona_bundle_pairs.add((persona, bundle.bundle_id))
        kind_usage[bundle_kind] += 1
        usage_counts["query_style"][query_style] += 1
        usage_counts["query_length"][query_length] += 1
        usage_counts["query_form"][query_form] += 1
        usage_counts["query_type"][query_type] += 1
        usage_counts["answerability"][answerability] += 1

        scenarios.append(
            Scenario(
                scenario_id=sha256_json(
                    {
                        "answerability": answerability,
                        "as_of_date": as_of_date.isoformat() if as_of_date else None,
                        "attempt_index": attempt_index,
                        "bridge": list(bundle.bridge),
                        "bundle_id": bundle.bundle_id,
                        "bundle_kind": bundle.kind,
                        "facts": [fact.fact_id for fact in facts],
                        "persona": persona,
                        "query_form": query_form,
                        "query_length": query_length,
                        "query_style": query_style,
                        "query_type": query_type,
                        "reasoning_hops": len(bundle.fact_ids),
                    }
                ),
                bundle_kind=bundle.kind,
                query_type=query_type,
                query_form=query_form,
                answerability=answerability,
                persona=persona,
                query_style=query_style,
                query_length=query_length,
                reasoning_hops=len(bundle.fact_ids),
                as_of_date=as_of_date,
                fact_ids=bundle.fact_ids,
                bridge=bundle.bridge,
                gold_chunk_ids=tuple(sorted({evidence.chunk_id for fact in facts for evidence in fact.evidence})),
            )
        )
    return tuple(scenarios)


def _eligible_query_types(
    bundle: EvidenceBundle,
    facts: Sequence[LegalFact],
    taxonomy: QueryTaxonomy,
) -> tuple[str, ...]:
    """Return configured query types whose structural requirements match a bundle."""
    eligible: list[str] = []
    for definition in taxonomy.definitions:
        if bundle.kind not in definition.bundle_kinds or len(facts) < definition.minimum_facts:
            continue
        eligible.append(definition.query_type_id)
    return tuple(eligible)


def _pick_persona(
    persona_ids: Sequence[str],
    bundle_id: str,
    used_pairs: set[tuple[str, str]],
    attempt_index: int,
) -> str:
    """Pick a persona not yet paired with this bundle, cycling once exhausted."""
    for persona_id in persona_ids:
        if (persona_id, bundle_id) not in used_pairs:
            return persona_id
    return persona_ids[attempt_index % len(persona_ids)]


def _pick_least_used(values: Sequence[str], counts: Counter[str]) -> str:
    """Pick the value used least so far, preferring the first on ties."""
    return min(values, key=lambda value: counts[value])


def _pick_shared_as_of_date(facts: Sequence[LegalFact]) -> date | None:
    """Return a date inside every fact's validity interval, or None when no shared date exists."""
    valid_froms = [fact.valid_from for fact in facts if fact.valid_from is not None]
    if not valid_froms:
        return None
    candidate = max(valid_froms)
    valid_tos = [fact.valid_to for fact in facts if fact.valid_to is not None]
    if valid_tos and candidate > min(valid_tos):
        return None
    return candidate
