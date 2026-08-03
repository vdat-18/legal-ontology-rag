"""Tests for scenario configuration loading and diverse sampling."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from legal_rag.benchmark_synthesis.evidence_graph import EvidenceBundle
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.scenario_sampling import (
    PersonaCatalog,
    QueryTaxonomy,
    QueryTypeDefinition,
    Scenario,
    ScenarioDistribution,
    ScenarioSamplingConfig,
    load_evolution_config,
    load_persona_descriptions,
    load_personas,
    load_query_taxonomy,
    load_query_type_descriptions,
    load_scenario_distribution,
    sample_scenarios,
)

_REPO_CONFIG_DIR = Path(__file__).resolve().parents[2] / "data" / "config" / "groundtruth"


def _fact(
    fact_id: str,
    *,
    content_sha256: str | None = None,
    valid_from: date | None = None,
    valid_to: date | None = None,
) -> LegalFact:
    """Build one minimal legal fact for scenario-sampling tests."""
    return LegalFact(
        fact_id=fact_id,
        fact_content_sha256=content_sha256 or f"content-{fact_id}",
        conduct="Vượt đèn đỏ",
        subject="Người điều khiển ô tô",
        conditions=(),
        monetary_penalty="4.000.000 đồng",
        license_points_deducted=None,
        additional_penalties=(),
        remedial_measures=(),
        valid_from=valid_from,
        valid_to=valid_to,
        document_number="168/2024/NĐ-CP",
        provision_path="Điều 6",
        evidence=(EvidenceReference(chunk_id=f"chunk-{fact_id}", quoted_span=""),),
    )


def _bundle(
    bundle_id: str,
    fact_ids: tuple[str, ...],
    *,
    kind: str = "single_hop",
    bridge: tuple[str, ...] = ("Điều 6",),
) -> EvidenceBundle:
    """Build one evidence bundle for scenario-sampling tests."""
    return EvidenceBundle(bundle_id=bundle_id, kind=kind, fact_ids=fact_ids, bridge=bridge)


def _config(
    *,
    query_type_ids: tuple[str, ...] = ("penalty_for_conduct",),
    persona_ids: tuple[str, ...] = ("tai_xe_ca_nhan", "luat_su"),
    query_styles: tuple[str, ...] = ("standard", "no_diacritics"),
    query_lengths: tuple[str, ...] = ("short", "long"),
    query_forms: tuple[str, ...] = ("specific", "abstract"),
) -> ScenarioSamplingConfig:
    """Build one scenario sampling configuration for tests."""
    return ScenarioSamplingConfig(
        query_taxonomy=QueryTaxonomy(
            definitions=tuple(
                QueryTypeDefinition(
                    query_type_id=query_type_id,
                    bundle_kinds=("single_hop", "synthesis", "multi_hop", "temporal"),
                    minimum_facts=1,
                    required_all_fields=(),
                    required_any_fields=(),
                )
                for query_type_id in query_type_ids
            )
        ),
        personas=PersonaCatalog(persona_ids=persona_ids),
        distribution=ScenarioDistribution(
            query_styles=query_styles,
            query_lengths=query_lengths,
            query_forms=query_forms,
            bundle_answerability_values=("answerable", "partially_answerable"),
            bundle_kind_weights=(
                ("single_hop", 0.4),
                ("synthesis", 0.2),
                ("multi_hop", 0.2),
                ("temporal", 0.2),
            ),
        ),
    )


def test_load_query_taxonomy_reads_repo_config() -> None:
    """The versioned query-taxonomy file in the repo loads without error."""
    taxonomy = load_query_taxonomy(_REPO_CONFIG_DIR / "query_taxonomy.json")

    assert "penalty_for_conduct" in taxonomy.query_type_ids
    assert "out_of_scope" not in taxonomy.query_type_ids


def test_load_personas_reads_repo_config() -> None:
    """The versioned persona file in the repo loads without error."""
    personas = load_personas(_REPO_CONFIG_DIR / "personas.json")

    assert "tai_xe_ca_nhan" in personas.persona_ids


def test_load_scenario_distribution_reads_repo_config() -> None:
    """The versioned scenario-distribution file in the repo loads without error."""
    distribution = load_scenario_distribution(_REPO_CONFIG_DIR / "scenario_distribution.json")

    assert "no_diacritics" in distribution.query_styles
    assert distribution.bundle_answerability_values == ("answerable", "partially_answerable")
    assert dict(distribution.bundle_kind_weights) == {
        "single_hop": 0.4,
        "synthesis": 0.2,
        "multi_hop": 0.2,
        "temporal": 0.2,
    }


def test_load_evolution_config_reads_repo_config() -> None:
    """The versioned evolution file in the repo loads without error."""
    config = load_evolution_config(_REPO_CONFIG_DIR / "evolution.json")

    assert "concretizing" in config.operation_ids
    assert config.max_rounds == 2


def test_load_query_type_descriptions_reads_repo_config() -> None:
    """Every query_type_id in the repo config maps to a non-empty description."""
    descriptions = load_query_type_descriptions(_REPO_CONFIG_DIR / "query_taxonomy.json")

    assert descriptions["penalty_for_conduct"] == "Mức phạt của một hành vi"


def test_load_persona_descriptions_reads_repo_config() -> None:
    """Every persona_id in the repo config maps to a non-empty description."""
    descriptions = load_persona_descriptions(_REPO_CONFIG_DIR / "personas.json")

    assert descriptions["tai_xe_ca_nhan"] == "Tài xế cá nhân"


def test_sample_scenarios_returns_empty_for_no_bundles() -> None:
    """Sampling with no bundles produces no scenarios, not an error."""
    scenarios = sample_scenarios([], {}, _config(), total_samples=10)

    assert scenarios == ()


def test_sample_scenarios_stops_at_the_requested_total() -> None:
    """Ceiling quotas distribute coverage without exceeding the requested total."""
    facts_by_id = {"a": _fact("a"), "b": _fact("b")}
    bundles = [_bundle("bundle-a", ("a",)), _bundle("bundle-b", ("b",))]

    scenarios = sample_scenarios(bundles, facts_by_id, _config(), total_samples=3)

    assert len(scenarios) == 3


def test_sample_scenarios_balances_bundle_kinds_for_small_totals() -> None:
    """A small sample includes each available bundle kind before repeating one."""
    facts_by_id = {fact_id: _fact(fact_id) for fact_id in ("a", "b", "c", "d", "e", "f")}
    bundles = [
        _bundle("single", ("a",)),
        _bundle("synthesis", ("b", "c"), kind="synthesis"),
        _bundle("multi-hop", ("d", "e"), kind="multi_hop"),
        _bundle("temporal", ("e", "f"), kind="temporal"),
    ]

    scenarios = sample_scenarios(bundles, facts_by_id, _config(), total_samples=4)

    assert {scenario.bundle_kind for scenario in scenarios} == {
        "single_hop",
        "synthesis",
        "multi_hop",
        "temporal",
    }


def test_sample_scenarios_sets_reasoning_hops_to_bundle_size() -> None:
    """A two-fact bundle produces scenarios with reasoning_hops=2."""
    facts_by_id = {"a": _fact("a"), "b": _fact("b")}
    bundle = _bundle("bundle-ab", ("a", "b"), kind="synthesis")

    scenarios = sample_scenarios([bundle], facts_by_id, _config(), total_samples=1)

    assert scenarios[0].reasoning_hops == 2


def test_sample_scenarios_collects_deduplicated_sorted_gold_chunk_ids() -> None:
    """gold_chunk_ids is the sorted union of evidence chunk_ids across the bundle's facts."""
    facts_by_id = {"a": _fact("a"), "b": _fact("b")}
    bundle = _bundle("bundle-ab", ("a", "b"), kind="synthesis")

    scenarios = sample_scenarios([bundle], facts_by_id, _config(), total_samples=1)

    assert scenarios[0].gold_chunk_ids == ("chunk-a", "chunk-b")


def test_sample_scenarios_picks_date_inside_shared_validity_interval() -> None:
    """as_of_date is the latest valid_from when it falls inside every validity interval."""
    facts_by_id = {
        "a": _fact("a", valid_from=date(2024, 1, 1), valid_to=date(2025, 12, 31)),
        "b": _fact("b", valid_from=date(2025, 1, 1), valid_to=date(2026, 12, 31)),
    }
    bundle = _bundle("bundle-ab", ("a", "b"), kind="synthesis")

    scenarios = sample_scenarios([bundle], facts_by_id, _config(), total_samples=1)

    assert scenarios[0].as_of_date == date(2025, 1, 1)


def test_sample_scenarios_as_of_date_is_none_for_disjoint_validity_intervals() -> None:
    """as_of_date is None when the facts have no shared validity date."""
    facts_by_id = {
        "a": _fact("a", valid_from=date(2024, 1, 1), valid_to=date(2024, 12, 31)),
        "b": _fact("b", valid_from=date(2025, 1, 1), valid_to=date(2025, 12, 31)),
    }
    bundle = _bundle("bundle-ab", ("a", "b"), kind="temporal")

    scenarios = sample_scenarios([bundle], facts_by_id, _config(), total_samples=1)

    assert scenarios[0].as_of_date is None


def test_sample_scenarios_as_of_date_is_none_without_any_valid_from() -> None:
    """as_of_date is None when no fact in the bundle has a valid_from."""
    facts_by_id = {"a": _fact("a")}
    bundle = _bundle("bundle-a", ("a",))

    scenarios = sample_scenarios([bundle], facts_by_id, _config(), total_samples=1)

    assert scenarios[0].as_of_date is None


def test_sample_scenarios_prioritizes_unused_persona_bundle_pairing() -> None:
    """Within one bundle's quota, each sample gets a different persona until exhausted."""
    facts_by_id = {"a": _fact("a")}
    bundle = _bundle("bundle-a", ("a",))

    scenarios = sample_scenarios([bundle], facts_by_id, _config(), total_samples=2)

    assert {scenario.persona for scenario in scenarios} == {"tai_xe_ca_nhan", "luat_su"}


def test_sample_scenarios_rotates_least_used_query_style() -> None:
    """Across scenarios, each configured query_style is used before any repeats."""
    facts_by_id = {"a": _fact("a"), "b": _fact("b")}
    bundles = [_bundle("bundle-a", ("a",)), _bundle("bundle-b", ("b",))]

    scenarios = sample_scenarios(bundles, facts_by_id, _config(), total_samples=2)

    assert {scenario.query_style for scenario in scenarios} == {"standard", "no_diacritics"}


def test_sample_scenarios_scenario_id_is_deterministic() -> None:
    """Two runs with identical inputs produce identical scenario_id values."""
    facts_by_id = {"a": _fact("a")}
    bundle = _bundle("bundle-a", ("a",))

    first = sample_scenarios([bundle], facts_by_id, _config(), total_samples=1)
    second = sample_scenarios([bundle], facts_by_id, _config(), total_samples=1)

    assert first[0].scenario_id == second[0].scenario_id


def test_sample_scenarios_is_invariant_to_bundle_input_order() -> None:
    """Reordering equivalent bundle input does not change sampled scenarios."""
    facts_by_id = {fact_id: _fact(fact_id) for fact_id in ("a", "b")}
    bundles = [_bundle("bundle-b", ("b",)), _bundle("bundle-a", ("a",))]

    first = sample_scenarios(bundles, facts_by_id, _config(), total_samples=2)
    second = sample_scenarios(list(reversed(bundles)), facts_by_id, _config(), total_samples=2)

    assert first == second


def test_sample_scenarios_scenario_id_changes_with_bridge() -> None:
    """Changing relation semantics changes scenario identity even when the bundle ID is unchanged."""
    facts_by_id = {"a": _fact("a")}
    first_bundle = _bundle("bundle-a", ("a",), bridge=("same_subject",))
    second_bundle = _bundle("bundle-a", ("a",), bridge=("supersedes",))

    first = sample_scenarios([first_bundle], facts_by_id, _config(), total_samples=1)
    second = sample_scenarios([second_bundle], facts_by_id, _config(), total_samples=1)

    assert first[0].scenario_id != second[0].scenario_id


def test_sample_scenarios_scenario_id_changes_with_fact_content() -> None:
    """Changing a fact's content hash invalidates scenario identity."""
    bundle = _bundle("bundle-a", ("a",))
    first_facts = {"a": _fact("a", content_sha256="first-content")}
    second_facts = {"a": _fact("a", content_sha256="second-content")}

    first = sample_scenarios([bundle], first_facts, _config(), total_samples=1)
    second = sample_scenarios([bundle], second_facts, _config(), total_samples=1)

    assert first[0].scenario_id != second[0].scenario_id


def test_scenario_to_dict_serializes_stable_output_schema() -> None:
    """Scenario tuples and dates serialize to JSON-compatible values."""
    scenario = Scenario(
        scenario_id="scenario-1",
        query_type="penalty_for_conduct",
        query_form="specific",
        answerability="answerable",
        persona="tai_xe_ca_nhan",
        query_style="no_diacritics",
        query_length="short",
        reasoning_hops=2,
        as_of_date=date(2025, 2, 10),
        fact_ids=("fact-1", "fact-2"),
        bridge=("Điều 6",),
        gold_chunk_ids=("chunk-1", "chunk-2"),
    )

    record = scenario.to_dict()

    assert record["as_of_date"] == "2025-02-10"
    assert record["fact_ids"] == ["fact-1", "fact-2"]
