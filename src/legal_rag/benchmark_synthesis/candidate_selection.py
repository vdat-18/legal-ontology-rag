"""Parse and validate explicit candidate-generation rerun selections."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from legal_rag.benchmark_synthesis.candidate_record import CandidateProgressRecord
from legal_rag.benchmark_synthesis.scenario_sampling import Scenario


def load_regenerate_task_ids(inline_ids: Sequence[str], ids_path: Path | None) -> frozenset[str]:
    """Return task IDs supplied inline or through a JSON string array."""
    task_ids = list(inline_ids)
    if ids_path is not None:
        payload = json.loads(ids_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(value, str) for value in payload):
            msg = f"Regenerate task id file must contain a JSON string array: {ids_path}"
            raise ValueError(msg)
        task_ids.extend(payload)
    if any(not task_id.strip() for task_id in task_ids):
        msg = "Regenerate task IDs must be non-empty strings"
        raise ValueError(msg)
    return frozenset(task_ids)


def validate_selective_generation(
    task_ids: frozenset[str],
    completed_ids: frozenset[str],
    requested_ids: frozenset[str],
) -> None:
    """Require known selections and prior progress for every unselected task."""
    unknown_ids = requested_ids - task_ids
    if unknown_ids:
        msg = f"Unknown candidate-generation tasks requested for regeneration: {sorted(unknown_ids)}"
        raise ValueError(msg)
    if task_ids - requested_ids - completed_ids:
        msg = "Selective candidate generation requires prior progress for every unselected task"
        raise ValueError(msg)


def prior_scenarios_from_progress(records: Sequence[CandidateProgressRecord]) -> tuple[Scenario, ...]:
    """Reconstruct the prior ordered scenario input from persisted candidate progress."""
    scenarios_by_id: dict[str, Scenario] = {}
    for record in records:
        draft = record.draft
        scenario = Scenario(
            scenario_id=draft.scenario_id,
            bundle_kind=draft.bundle_kind,
            query_form=draft.query_form,
            answerability=draft.answerability,
            persona=draft.persona,
            query_style=draft.query_style,
            query_length=draft.query_length,
            reasoning_hops=draft.reasoning_hops,
            fact_ids=draft.source_fact_ids,
            bridge=draft.bridge,
            gold_chunk_ids=draft.gold_chunk_ids,
        )
        existing = scenarios_by_id.setdefault(scenario.scenario_id, scenario)
        if existing != scenario:
            msg = f"Candidate progress disagrees about scenario {scenario.scenario_id}"
            raise ValueError(msg)
    return tuple(scenarios_by_id.values())


def validate_scenario_replacement(
    prior_scenarios: Sequence[Scenario],
    current_scenarios: Sequence[Scenario],
    replace_scenario_ids: frozenset[str],
) -> frozenset[str]:
    """Validate an explicit same-size scenario replacement and return new scenario IDs."""
    prior_by_id = {scenario.scenario_id: scenario for scenario in prior_scenarios}
    prior_ids = frozenset(scenario.scenario_id for scenario in prior_scenarios)
    current_ids = frozenset(scenario.scenario_id for scenario in current_scenarios)
    if len(prior_ids) != len(prior_scenarios) or len(current_ids) != len(current_scenarios):
        msg = "Scenario replacement requires unique scenario IDs"
        raise ValueError(msg)
    unknown_ids = replace_scenario_ids - prior_ids
    if unknown_ids:
        msg = f"Unknown scenarios requested for replacement: {sorted(unknown_ids)}"
        raise ValueError(msg)
    rejected_keys = {_scenario_evidence_key(prior_by_id[scenario_id]) for scenario_id in replace_scenario_ids}
    expected_removed_ids = frozenset(
        scenario.scenario_id
        for scenario in prior_scenarios
        if _scenario_evidence_key(scenario) in rejected_keys
    )
    removed_ids = prior_ids - current_ids
    added_ids = current_ids - prior_ids
    if removed_ids != expected_removed_ids:
        msg = (
            "Scenario replacement removed unexpected IDs: "
            f"expected {sorted(expected_removed_ids)}, got {sorted(removed_ids)}"
        )
        raise ValueError(msg)
    if len(added_ids) != len(removed_ids) or len(current_scenarios) != len(prior_scenarios):
        msg = "Scenario replacement must preserve the total scenario count"
        raise ValueError(msg)
    return added_ids


def _scenario_evidence_key(scenario: Scenario) -> tuple[str, tuple[str, ...], str | None]:
    """Identify scenarios backed by the same evidence bundle."""
    return scenario.bundle_kind, scenario.fact_ids, scenario.bridge
