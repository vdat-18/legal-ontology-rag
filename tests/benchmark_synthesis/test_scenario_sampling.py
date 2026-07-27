"""Tests for scenario configuration loading and diverse sampling."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from legal_rag.benchmark_synthesis.evidence_graph import EvidenceBundle
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.scenario_sampling import (
    PersonaCatalog,
    QueryTaxonomy,
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


def _fact(fact_id: str, *, valid_from: date | None = None) -> LegalFact:
    """Build one minimal legal fact for scenario-sampling tests."""
    return LegalFact(
        fact_id=fact_id,
        fact_content_sha256=f"content-{fact_id}",
        conduct="Vượt đèn đỏ",
        subject="Người điều khiển ô tô",
        conditions=(),
        monetary_penalty="4.000.000 đồng",
        license_points_deducted=None,
        additional_penalties=(),
        remedial_measures=(),
        valid_from=valid_from,
        valid_to=None,
        document_number="168/2024/NĐ-CP",
        provision_path="Điều 6",
        evidence=(EvidenceReference(chunk_id=f"chunk-{fact_id}", quoted_span="", unit_ids=(), source_refs=()),),
    )


def _bundle(bundle_id: str, fact_ids: tuple[str, ...], *, kind: str = "single_hop") -> EvidenceBundle:
    """Build one evidence bundle for scenario-sampling tests."""
    return EvidenceBundle(bundle_id=bundle_id, kind=kind, fact_ids=fact_ids, bridge=("Điều 6",))


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
        query_taxonomy=QueryTaxonomy(query_type_ids=query_type_ids),
        personas=PersonaCatalog(persona_ids=persona_ids),
        distribution=ScenarioDistribution(
            query_styles=query_styles,
            query_lengths=query_lengths,
            query_forms=query_forms,
            bundle_answerability_values=("answerable", "partially_answerable"),
        ),
    )


def test_load_query_taxonomy_reads_repo_config() -> None:
    """The versioned query-taxonomy file in the repo loads without error."""
    taxonomy = load_query_taxonomy(_REPO_CONFIG_DIR / "query_taxonomy.json")

    assert "penalty_for_conduct" in taxonomy.query_type_ids
    assert "out_of_scope" in taxonomy.query_type_ids


def test_load_personas_reads_repo_config() -> None:
    """The versioned persona file in the repo loads without error."""
    personas = load_personas(_REPO_CONFIG_DIR / "personas.json")

    assert "tai_xe_ca_nhan" in personas.persona_ids


def test_load_scenario_distribution_reads_repo_config() -> None:
    """The versioned scenario-distribution file in the repo loads without error."""
    distribution = load_scenario_distribution(_REPO_CONFIG_DIR / "scenario_distribution.json")

    assert "no_diacritics" in distribution.query_styles
    assert distribution.bundle_answerability_values == ("answerable", "partially_answerable")


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


def test_sample_scenarios_applies_ceiling_quota_per_bundle() -> None:
    """Total samples split across bundles round up, matching ceil(total/bundles)."""
    facts_by_id = {"a": _fact("a"), "b": _fact("b")}
    bundles = [_bundle("bundle-a", ("a",)), _bundle("bundle-b", ("b",))]

    scenarios = sample_scenarios(bundles, facts_by_id, _config(), total_samples=3)

    # ceil(3 / 2) = 2 per bundle, so 4 total, not 3.
    assert len(scenarios) == 4


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


def test_sample_scenarios_picks_latest_valid_from_as_of_date() -> None:
    """as_of_date is the latest valid_from among the bundle's facts."""
    facts_by_id = {
        "a": _fact("a", valid_from=date(2024, 1, 1)),
        "b": _fact("b", valid_from=date(2025, 1, 1)),
    }
    bundle = _bundle("bundle-ab", ("a", "b"), kind="synthesis")

    scenarios = sample_scenarios([bundle], facts_by_id, _config(), total_samples=1)

    assert scenarios[0].as_of_date == date(2025, 1, 1)


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
