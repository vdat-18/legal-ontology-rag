"""Tests for semantic scenario planning over evidence bundles."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from legal_rag.baml_client.types import (
    PlannedScenarioQueryType,
    ScenarioPlanningBatchItem,
    ScenarioPlanningBatchResult,
)
from legal_rag.benchmark_synthesis import scenario_planning
from legal_rag.benchmark_synthesis.config import LlmStageSettings, SynthesisSettings
from legal_rag.benchmark_synthesis.evidence_graph import EvidenceBundle
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.scenario_planning import (
    BamlScenarioPlanningClient,
    build_scenario_planning_item,
    build_scenario_planning_run_id,
    plan_scenario_bundles,
    structurally_supported_query_types,
)
from legal_rag.benchmark_synthesis.scenario_sampling import (
    QueryTaxonomy,
    QueryTypeDefinition,
    ScenarioBundlePlan,
    ScenarioQueryTypePlan,
)
from legal_rag.benchmark_synthesis.writer import read_scenario_plans, write_scenario_plans
from legal_rag.settings import Settings

_SYNTHESIS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "data" / "config" / "groundtruth" / "synthesis_v1.json"


def _fact(fact_id: str, statement: str, *, valid_from: date | None = None) -> LegalFact:
    """Build one statement-only fact for scenario-planning tests."""
    return LegalFact(
        fact_id=fact_id,
        statement=statement,
        valid_from=valid_from,
        valid_to=None,
        document_number="168/2024/NĐ-CP",
        evidence=(EvidenceReference(chunk_id=f"chunk-{fact_id}"),),
    )


def _bundle(*fact_ids: str, kind: str = "single_hop") -> EvidenceBundle:
    """Build one deterministic evidence bundle."""
    return EvidenceBundle(bundle_id="bundle-1", kind=kind, fact_ids=tuple(fact_ids), bridge=("vượt đèn đỏ",))


def _taxonomy() -> QueryTaxonomy:
    """Build structural query-type candidates for tests."""
    return QueryTaxonomy(
        definitions=(
            QueryTypeDefinition("legal_rule", ("single_hop", "multi_hop", "temporal"), 1),
            QueryTypeDefinition("penalty_for_conduct", ("single_hop", "multi_hop", "temporal"), 1),
            QueryTypeDefinition("comparison", ("multi_hop", "temporal"), 2),
        )
    )


def _descriptions() -> dict[str, str]:
    """Return prompt-facing query-type descriptions."""
    return {
        "legal_rule": "Hỏi quy định pháp lý",
        "penalty_for_conduct": "Hỏi mức phạt của hành vi",
        "comparison": "So sánh hai quy định",
    }


def _settings() -> Settings:
    """Build runtime settings for an offline BAML request."""
    return Settings(
        llm_base_url="http://localhost:20128/v1",
        llm_api_key="provider-test-key",
        llm_model="provider/test-model",
        llm_timeout_seconds=30,
    )


def test_structural_candidates_only_filter_bundle_shape() -> None:
    """Structural filtering leaves semantic compatibility to the planner LLM."""
    assert structurally_supported_query_types(_bundle("fact-a"), _taxonomy()) == (
        "legal_rule",
        "penalty_for_conduct",
    )


def test_build_scenario_planning_item_sends_statements_candidates_and_dates() -> None:
    """Planning input contains complete statements and only configured candidate dates."""
    fact = _fact("fact-a", "Vượt đèn đỏ bị phạt tiền.", valid_from=date(2025, 1, 1))

    item = build_scenario_planning_item(_bundle("fact-a"), {fact.fact_id: fact}, _taxonomy(), _descriptions())

    assert item.facts[0].statement == fact.statement
    assert [candidate.query_type_id for candidate in item.query_type_candidates] == [
        "legal_rule",
        "penalty_for_conduct",
    ]
    assert item.candidate_as_of_dates == ["2025-01-01"]


class _PenaltyOnlyClient:
    """Approve only the penalty query type for every bundle."""

    def plan_batch(self, items: Sequence[ScenarioPlanningBatchItem]) -> list[ScenarioPlanningBatchResult]:
        """Return one supported type using the provided candidate date."""
        return [
            ScenarioPlanningBatchResult(
                bundle_id=item.bundle_id,
                supported_query_types=[
                    PlannedScenarioQueryType(
                        query_type_id="penalty_for_conduct",
                        as_of_date=item.candidate_as_of_dates[0],
                    )
                ],
            )
            for item in items
        ]


def test_plan_scenario_bundles_persists_only_llm_supported_types() -> None:
    """The semantic plan carries the LLM-approved query type and selected date."""
    fact = _fact("fact-a", "Vượt đèn đỏ bị phạt tiền.", valid_from=date(2025, 1, 1))
    bundle = _bundle("fact-a")

    plans, failures = plan_scenario_bundles(
        [bundle],
        {fact.fact_id: fact},
        _taxonomy(),
        _descriptions(),
        _PenaltyOnlyClient(),
        run_id="run-1",
        batch_size=10,
        max_concurrency=1,
        max_retries=0,
        retry_delay_seconds=0,
    )

    assert failures == ()
    assert plans[0].query_types[0].query_type_id == "penalty_for_conduct"
    assert plans[0].query_types[0].as_of_date == date(2025, 1, 1)


def test_plan_scenario_bundles_rejects_unavailable_query_type() -> None:
    """A model cannot introduce a query type absent from the typed request."""
    fact = _fact("fact-a", "Quyết định xử phạt phải được gửi đúng hạn.")

    class _InvalidClient:
        def plan_batch(self, items: Sequence[ScenarioPlanningBatchItem]) -> list[ScenarioPlanningBatchResult]:
            return [
                ScenarioPlanningBatchResult(
                    bundle_id=item.bundle_id,
                    supported_query_types=[PlannedScenarioQueryType(query_type_id="invented_type", as_of_date=None)],
                )
                for item in items
            ]

    plans, failures = plan_scenario_bundles(
        [_bundle("fact-a")],
        {fact.fact_id: fact},
        _taxonomy(),
        _descriptions(),
        _InvalidClient(),
        run_id="run-1",
        batch_size=10,
        max_concurrency=1,
        max_retries=0,
        retry_delay_seconds=0,
    )

    assert plans == ()
    assert len(failures) == 1
    assert failures[0].task_key.startswith("scenario-planning-batch:")


def test_scenario_plan_writer_round_trips_empty_and_supported_plans(tmp_path: Path) -> None:
    """Scenario plans persist empty capability sets as completed work units."""
    plans = (
        ScenarioBundlePlan(
            bundle_id="bundle-empty",
            query_types=(),
            run_id="run-1",
        ),
        ScenarioBundlePlan(
            bundle_id="bundle-supported",
            query_types=(
                ScenarioQueryTypePlan(
                    query_type_id="penalty_for_conduct",
                    as_of_date=date(2025, 1, 1),
                ),
            ),
            run_id="run-1",
        ),
    )
    output_path = tmp_path / "scenario_plans.jsonl"

    write_scenario_plans(output_path, plans)

    assert read_scenario_plans(output_path) == plans


def test_scenario_planning_run_id_changes_with_query_type_description() -> None:
    """Changing prompt-facing taxonomy text invalidates resumable plans."""
    fact = _fact("fact-a", "Vượt đèn đỏ bị phạt tiền.")
    bundle = _bundle("fact-a")
    synthesis_settings = SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH)
    first = build_scenario_planning_run_id(
        [bundle],
        {fact.fact_id: fact},
        _taxonomy(),
        _descriptions(),
        synthesis_settings,
        llm_base_url="http://localhost:20128/v1",
        llm_model="provider/test-model",
    )
    changed_descriptions = {**_descriptions(), "penalty_for_conduct": "Mô tả đã thay đổi"}
    second = build_scenario_planning_run_id(
        [bundle],
        {fact.fact_id: fact},
        _taxonomy(),
        changed_descriptions,
        synthesis_settings,
        llm_base_url="http://localhost:20128/v1",
        llm_model="provider/test-model",
    )

    assert first != second


def test_scenario_planning_run_id_changes_with_fact_statement() -> None:
    """Changing planner-visible fact content invalidates resumable plans."""
    bundle = _bundle("fact-a")
    synthesis_settings = SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH)
    first_fact = _fact("fact-a", "Vượt đèn đỏ bị phạt tiền.")
    second_fact = _fact("fact-a", "Quyết định xử phạt phải được gửi đúng hạn.")

    first = build_scenario_planning_run_id(
        [bundle],
        {first_fact.fact_id: first_fact},
        _taxonomy(),
        _descriptions(),
        synthesis_settings,
        llm_base_url="http://localhost:20128/v1",
        llm_model="provider/test-model",
    )
    second = build_scenario_planning_run_id(
        [bundle],
        {second_fact.fact_id: second_fact},
        _taxonomy(),
        _descriptions(),
        synthesis_settings,
        llm_base_url="http://localhost:20128/v1",
        llm_model="provider/test-model",
    )

    assert first != second


def test_scenario_planning_client_uses_dedicated_stage_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario planning uses its own configured token and temperature bounds."""
    configured_tokens: list[int] = []

    def _capture_registry(**kwargs: object) -> object:
        configured_tokens.append(int(kwargs["max_output_tokens"]))
        return object()

    synthesis_settings = replace(
        SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH),
        scenario_planning=LlmStageSettings(303, 0.0),
    )
    monkeypatch.setattr(scenario_planning, "build_client_registry", _capture_registry)

    BamlScenarioPlanningClient(_settings(), synthesis_settings)

    assert configured_tokens == [303]
