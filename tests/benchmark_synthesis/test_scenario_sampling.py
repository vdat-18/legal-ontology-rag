"""Tests for scenario configuration loading and diverse sampling."""

from __future__ import annotations

from pathlib import Path

from legal_rag.benchmark_synthesis.evidence_graph import EvidenceBundle
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.scenario_sampling import (
    PersonaCatalog,
    Scenario,
    ScenarioDistribution,
    ScenarioSamplingConfig,
    load_evolution_config,
    load_persona_descriptions,
    load_personas,
    load_scenario_distribution,
    sample_scenarios,
)

_REPO_CONFIG_DIR = Path(__file__).resolve().parents[2] / "data" / "config" / "groundtruth"


def _fact(fact_id: str) -> LegalFact:
    """Build one minimal legal fact for scenario-sampling tests."""
    return LegalFact(
        fact_id=fact_id,
        statement="Người điều khiển ô tô vượt đèn đỏ bị phạt tiền.",
        valid_from=None,
        valid_to=None,
        document_number="168/2024/NĐ-CP",
        evidence=(EvidenceReference(chunk_id=f"chunk-{fact_id}"),),
    )


def _bundle(
    bundle_id: str,
    fact_ids: tuple[str, ...],
    *,
    kind: str = "single_hop",
    bridge: str | None = "Cùng tình huống xử lý vi phạm",
) -> EvidenceBundle:
    """Build one evidence bundle for scenario-sampling tests."""
    return EvidenceBundle(bundle_id=bundle_id, kind=kind, fact_ids=fact_ids, bridge=bridge)


def _config(
    *,
    persona_ids: tuple[str, ...] = ("tai_xe_ca_nhan", "luat_su"),
    query_styles: tuple[str, ...] = ("standard", "no_diacritics"),
    query_lengths: tuple[str, ...] = ("short", "long"),
    query_forms: tuple[str, ...] = ("specific", "abstract"),
) -> ScenarioSamplingConfig:
    """Build one scenario sampling configuration for tests."""
    return ScenarioSamplingConfig(
        personas=PersonaCatalog(persona_ids=persona_ids),
        distribution=ScenarioDistribution(
            query_styles=query_styles,
            query_lengths=query_lengths,
            query_forms=query_forms,
            bundle_answerability_values=("answerable", "partially_answerable"),
            bundle_kind_weights=(("single_hop", 0.4), ("multi_hop", 0.4), ("temporal", 0.2)),
        ),
    )


def _sample_scenarios(
    bundles: list[EvidenceBundle],
    facts_by_id: dict[str, LegalFact],
    config: ScenarioSamplingConfig,
    *,
    total_samples: int,
) -> tuple[Scenario, ...]:
    """Sample scenarios directly from evidence bundles."""
    return sample_scenarios(bundles, facts_by_id, config, total_samples=total_samples)


def test_load_personas_reads_repo_config() -> None:
    """The versioned persona file in the repo loads without error."""
    personas = load_personas(_REPO_CONFIG_DIR / "personas.json")

    assert "tai_xe_ca_nhan" in personas.persona_ids


def test_load_scenario_distribution_reads_repo_config() -> None:
    """The versioned scenario-distribution file in the repo loads without error."""
    distribution = load_scenario_distribution(_REPO_CONFIG_DIR / "scenario_distribution.json")

    assert "no_diacritics" in distribution.query_styles
    assert distribution.bundle_answerability_values == ("answerable", "partially_answerable")
    assert dict(distribution.bundle_kind_weights) == {"single_hop": 0.4, "multi_hop": 0.4, "temporal": 0.2}


def test_load_evolution_config_reads_repo_config() -> None:
    """The versioned evolution file in the repo loads without error."""
    config = load_evolution_config(_REPO_CONFIG_DIR / "evolution.json")

    assert "concretizing" in config.operation_ids
    assert config.max_rounds == 1


def test_load_persona_descriptions_reads_repo_config() -> None:
    """Every persona ID in the repo config maps to a non-empty description."""
    descriptions = load_persona_descriptions(_REPO_CONFIG_DIR / "personas.json")

    assert descriptions["tai_xe_ca_nhan"] == "Tài xế cá nhân hỏi ngắn gọn về tình huống thực tế của mình"


def test_sample_scenarios_returns_empty_for_no_bundles() -> None:
    """Sampling with no bundles produces no scenarios, not an error."""
    assert _sample_scenarios([], {}, _config(), total_samples=10) == ()


def test_sample_scenarios_stops_at_the_requested_total() -> None:
    """Sampling never exceeds the requested total."""
    facts_by_id = {"a": _fact("a"), "b": _fact("b")}
    bundles = [_bundle("bundle-a", ("a",)), _bundle("bundle-b", ("b",))]

    scenarios = _sample_scenarios(bundles, facts_by_id, _config(), total_samples=3)

    assert len(scenarios) == 3


def test_sample_scenarios_balances_bundle_kinds_for_small_totals() -> None:
    """A small sample includes each available bundle kind before repeating one."""
    facts_by_id = {fact_id: _fact(fact_id) for fact_id in ("a", "b", "c", "d", "e")}
    bundles = [
        _bundle("single", ("a",)),
        _bundle("multi-hop", ("b", "c"), kind="multi_hop"),
        _bundle("temporal", ("d", "e"), kind="temporal"),
    ]

    scenarios = _sample_scenarios(bundles, facts_by_id, _config(), total_samples=3)

    assert {scenario.bundle_kind for scenario in scenarios} == {"single_hop", "multi_hop", "temporal"}


def test_sample_scenarios_uses_every_bundle_without_semantic_planning() -> None:
    """Every evidence bundle remains eligible without a query taxonomy or planner result."""
    facts_by_id = {"a": _fact("a"), "b": _fact("b")}
    bundles = [_bundle("bundle-a", ("a",)), _bundle("bundle-b", ("b",))]

    scenarios = _sample_scenarios(bundles, facts_by_id, _config(), total_samples=2)

    assert {scenario.fact_ids for scenario in scenarios} == {("a",), ("b",)}


def test_sample_scenarios_sets_reasoning_hops_and_gold_chunks_from_bundle() -> None:
    """Scenario reasoning and provenance come directly from the selected bundle."""
    facts_by_id = {"a": _fact("a"), "b": _fact("b")}
    bundle = _bundle("bundle-ab", ("a", "b"), kind="multi_hop")

    scenario = _sample_scenarios([bundle], facts_by_id, _config(), total_samples=1)[0]

    assert scenario.reasoning_hops == 2
    assert scenario.gold_chunk_ids == ("chunk-a", "chunk-b")


def test_sample_scenarios_prioritizes_unused_persona_bundle_pairing() -> None:
    """Each sample gets a different persona for one bundle until personas are exhausted."""
    facts_by_id = {"a": _fact("a")}
    bundle = _bundle("bundle-a", ("a",))

    scenarios = _sample_scenarios([bundle], facts_by_id, _config(), total_samples=2)

    assert {scenario.persona for scenario in scenarios} == {"tai_xe_ca_nhan", "luat_su"}


def test_sample_scenarios_rotates_least_used_surface_axes() -> None:
    """Configured surface values are used before any value repeats."""
    facts_by_id = {"a": _fact("a"), "b": _fact("b")}
    bundles = [_bundle("bundle-a", ("a",)), _bundle("bundle-b", ("b",))]

    scenarios = _sample_scenarios(bundles, facts_by_id, _config(), total_samples=2)

    assert {scenario.query_style for scenario in scenarios} == {"standard", "no_diacritics"}
    assert {scenario.query_form for scenario in scenarios} == {"specific", "abstract"}


def test_sample_scenarios_is_deterministic_and_input_order_invariant() -> None:
    """Equivalent inputs produce identical scenarios regardless of bundle order."""
    facts_by_id = {fact_id: _fact(fact_id) for fact_id in ("a", "b")}
    bundles = [_bundle("bundle-b", ("b",)), _bundle("bundle-a", ("a",))]

    first = _sample_scenarios(bundles, facts_by_id, _config(), total_samples=2)
    second = _sample_scenarios(list(reversed(bundles)), facts_by_id, _config(), total_samples=2)

    assert first == second


def test_sample_scenarios_scenario_id_changes_with_evidence() -> None:
    """Changing relation context or fact identity invalidates scenario identity."""
    first = _sample_scenarios(
        [_bundle("bundle-a", ("a",), bridge="Cùng chủ thể")], {"a": _fact("a")}, _config(), total_samples=1
    )
    second = _sample_scenarios(
        [_bundle("bundle-a", ("b",), bridge="Quy định thay thế")], {"b": _fact("b")}, _config(), total_samples=1
    )

    assert first[0].scenario_id != second[0].scenario_id


def test_scenario_to_dict_serializes_stable_output_schema() -> None:
    """Scenario tuples serialize to JSON-compatible values without semantic planning fields."""
    scenario = Scenario(
        scenario_id="scenario-1",
        query_form="specific",
        answerability="answerable",
        persona="tai_xe_ca_nhan",
        query_style="no_diacritics",
        query_length="short",
        reasoning_hops=2,
        fact_ids=("fact-1", "fact-2"),
        bridge="Cùng tình huống xử lý vi phạm",
        gold_chunk_ids=("chunk-1", "chunk-2"),
    )

    record = scenario.to_dict()

    assert "query_type" not in record
    assert "as_of_date" not in record
    assert record["fact_ids"] == ["fact-1", "fact-2"]
    assert record["bridge"] == "Cùng tình huống xử lý vi phạm"
