"""Load versioned scenario configuration and greedily sample diverse scenarios.

Scenario generation only covers bundles built from real evidence relations,
so it can only ever produce `answerable` and `partially_answerable`
scenarios. `underspecified` and `unanswerable` scenarios are not derived
from a bundle at all — the corpus has no adequate evidence by definition —
and need a separate generation path, out of scope here.
"""

from __future__ import annotations

import json
import math
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


@dataclass(frozen=True, slots=True)
class QueryTypeDefinition:
    """One query type and the evidence-bundle shapes capable of supporting it."""

    query_type_id: str
    bundle_kinds: tuple[BundleKind, ...]
    minimum_facts: int
    required_all_fields: tuple[str, ...]
    required_any_fields: tuple[str, ...]


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
                required_all_fields=tuple(entry.get("required_all_fields", ())),
                required_any_fields=tuple(entry.get("required_any_fields", ())),
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
    return ScenarioDistribution(
        query_styles=tuple(config["query_styles"]),
        query_lengths=tuple(config["query_lengths"]),
        query_forms=tuple(config["query_forms"]),
        bundle_answerability_values=tuple(config["bundle_answerability_values"]),
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
    """Greedily sample diverse scenarios from evidence bundles.

    Quota per bundle is `ceil(total_samples / len(bundles))`. Within each
    bundle, persona prioritizes a persona-bundle pairing not yet seen; every
    other axis prioritizes the value least used so far across the whole run.
    """
    if not bundles:
        return ()
    samples_per_bundle = math.ceil(total_samples / len(bundles))

    used_persona_bundle_pairs: set[tuple[str, str]] = set()
    usage_counts: dict[str, Counter[str]] = {
        "query_style": Counter(),
        "query_length": Counter(),
        "query_form": Counter(),
        "query_type": Counter(),
        "answerability": Counter(),
    }

    scenarios: list[Scenario] = []
    for bundle in bundles:
        facts = [facts_by_id[fact_id] for fact_id in bundle.fact_ids]
        for attempt_index in range(samples_per_bundle):
            persona = _pick_persona(
                config.personas.persona_ids, bundle.bundle_id, used_persona_bundle_pairs, attempt_index
            )
            query_style = _pick_least_used(config.distribution.query_styles, usage_counts["query_style"])
            query_length = _pick_least_used(config.distribution.query_lengths, usage_counts["query_length"])
            query_form = _pick_least_used(config.distribution.query_forms, usage_counts["query_form"])
            eligible_query_types = _eligible_query_types(bundle, facts, config.query_taxonomy)
            if not eligible_query_types:
                break
            query_type = _pick_least_used(eligible_query_types, usage_counts["query_type"])
            answerability = _pick_least_used(
                config.distribution.bundle_answerability_values, usage_counts["answerability"]
            )

            used_persona_bundle_pairs.add((persona, bundle.bundle_id))
            usage_counts["query_style"][query_style] += 1
            usage_counts["query_length"][query_length] += 1
            usage_counts["query_form"][query_form] += 1
            usage_counts["query_type"][query_type] += 1
            usage_counts["answerability"][answerability] += 1

            scenarios.append(
                Scenario(
                    scenario_id=sha256_json(
                        {
                            "bundle_id": bundle.bundle_id,
                            "query_type": query_type,
                            "query_form": query_form,
                            "answerability": answerability,
                            "persona": persona,
                            "query_style": query_style,
                            "query_length": query_length,
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
                    as_of_date=_pick_as_of_date(facts),
                    fact_ids=bundle.fact_ids,
                    bridge=bundle.bridge,
                    gold_chunk_ids=tuple(sorted({evidence.chunk_id for fact in facts for evidence in fact.evidence})),
                )
            )
            if len(scenarios) == total_samples:
                return tuple(scenarios)
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
        if any(not any(getattr(fact, field) for fact in facts) for field in definition.required_all_fields):
            continue
        if definition.required_any_fields and not any(
            getattr(fact, field) for fact in facts for field in definition.required_any_fields
        ):
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


def _pick_as_of_date(facts: Sequence[LegalFact]) -> date | None:
    """Return the latest valid_from among the bundle's facts, or None if none is set."""
    valid_froms = [fact.valid_from for fact in facts if fact.valid_from is not None]
    return max(valid_froms) if valid_froms else None
