"""Plan semantically supported query types for evidence bundles."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date
from functools import partial
from typing import Protocol

from baml_py.logging import set_log_level

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import (
    ScenarioPlanningBatchItem,
    ScenarioPlanningBatchResult,
    ScenarioPlanningFact,
    ScenarioQueryTypeCandidate,
)
from legal_rag.baml_transport import build_client_registry, execute_baml_request
from legal_rag.benchmark_synthesis.config import SynthesisSettings
from legal_rag.benchmark_synthesis.evidence_graph import EvidenceBundle
from legal_rag.benchmark_synthesis.llm_execution import LlmTaskExhaustedError, LlmTaskFailure, execute_llm_task
from legal_rag.benchmark_synthesis.models import LegalFact
from legal_rag.benchmark_synthesis.scenario_sampling import (
    QueryTaxonomy,
    ScenarioBundlePlan,
    ScenarioQueryTypePlan,
)
from legal_rag.parser.hashing import sha256_json
from legal_rag.settings import Settings

SCENARIO_PLANNING_PROMPT_VERSION = "1.1.0"


class ScenarioPlanningClient(Protocol):
    """Boundary implemented by a scenario-planning language-model client."""

    def plan_batch(self, items: Sequence[ScenarioPlanningBatchItem]) -> Sequence[ScenarioPlanningBatchResult]:
        """Return one semantic plan for every bundle in the batch."""


class BamlScenarioPlanningClient:
    """Typed BAML client backed by a runtime OpenAI-compatible provider."""

    def __init__(self, settings: Settings, synthesis_settings: SynthesisSettings) -> None:
        """Create a runtime BAML client from typed, secret-safe configuration."""
        set_log_level("OFF")
        self._registry = build_client_registry(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_output_tokens=synthesis_settings.scenario_planning.max_output_tokens,
            temperature=synthesis_settings.scenario_planning.temperature,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        self._timeout_seconds = settings.llm_timeout_seconds
        self._max_response_bytes = synthesis_settings.max_response_bytes

    def plan_batch(self, items: Sequence[ScenarioPlanningBatchItem]) -> Sequence[ScenarioPlanningBatchResult]:
        """Run one bounded BAML call over a batch of evidence bundles."""
        request = b.request.PlanScenarioBundles(list(items), baml_options={"client_registry": self._registry})
        raw_output = execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)
        return b.parse.PlanScenarioBundles(raw_output, baml_options={"client_registry": self._registry})


def build_scenario_planning_run_id(
    bundles: Sequence[EvidenceBundle],
    facts_by_id: Mapping[str, LegalFact],
    taxonomy: QueryTaxonomy,
    query_type_descriptions: Mapping[str, str],
    synthesis_settings: SynthesisSettings,
    *,
    llm_base_url: str,
    llm_model: str,
) -> str:
    """Hash every input that can change semantic scenario planning."""
    referenced_fact_ids = sorted({fact_id for bundle in bundles for fact_id in bundle.fact_ids})
    return sha256_json(
        {
            "bundles": [bundle.to_dict() for bundle in bundles],
            "facts": [
                {
                    "document_number": facts_by_id[fact_id].document_number,
                    "fact_id": fact_id,
                    "statement": facts_by_id[fact_id].statement,
                    "valid_from": (
                        facts_by_id[fact_id].valid_from.isoformat() if facts_by_id[fact_id].valid_from else None
                    ),
                    "valid_to": facts_by_id[fact_id].valid_to.isoformat() if facts_by_id[fact_id].valid_to else None,
                }
                for fact_id in referenced_fact_ids
            ],
            "llm_base_url": llm_base_url,
            "llm_model": llm_model,
            "prompt_version": SCENARIO_PLANNING_PROMPT_VERSION,
            "query_types": [
                {
                    "bundle_kinds": list(definition.bundle_kinds),
                    "description_vi": query_type_descriptions[definition.query_type_id],
                    "minimum_facts": definition.minimum_facts,
                    "query_type_id": definition.query_type_id,
                }
                for definition in taxonomy.definitions
            ],
            "stage": asdict(synthesis_settings.scenario_planning),
            "batch_size": synthesis_settings.scenario_planning_batch_size,
        }
    )


def structurally_supported_query_types(bundle: EvidenceBundle, taxonomy: QueryTaxonomy) -> tuple[str, ...]:
    """Return query types whose configured bundle shape matches the evidence bundle."""
    return tuple(
        definition.query_type_id
        for definition in taxonomy.definitions
        if bundle.kind in definition.bundle_kinds and len(bundle.fact_ids) >= definition.minimum_facts
    )


def build_scenario_planning_item(
    bundle: EvidenceBundle,
    facts_by_id: Mapping[str, LegalFact],
    taxonomy: QueryTaxonomy,
    query_type_descriptions: Mapping[str, str],
) -> ScenarioPlanningBatchItem:
    """Build one typed scenario-planning request from a bundle and its facts."""
    facts = tuple(facts_by_id[fact_id] for fact_id in bundle.fact_ids)
    query_type_ids = structurally_supported_query_types(bundle, taxonomy)
    candidate_dates = sorted({fact.valid_from.isoformat() for fact in facts if fact.valid_from is not None})
    return ScenarioPlanningBatchItem(
        bundle_id=bundle.bundle_id,
        bundle_kind=bundle.kind,
        bridge=bundle.bridge,
        facts=[
            ScenarioPlanningFact(
                fact_id=fact.fact_id,
                statement=fact.statement,
                document_number=fact.document_number,
                valid_from=fact.valid_from.isoformat() if fact.valid_from else None,
                valid_to=fact.valid_to.isoformat() if fact.valid_to else None,
            )
            for fact in facts
        ],
        query_type_candidates=[
            ScenarioQueryTypeCandidate(
                query_type_id=query_type_id,
                description_vi=query_type_descriptions[query_type_id],
            )
            for query_type_id in query_type_ids
        ],
        candidate_as_of_dates=candidate_dates,
    )


def plan_scenario_bundles(
    bundles: Sequence[EvidenceBundle],
    facts_by_id: Mapping[str, LegalFact],
    taxonomy: QueryTaxonomy,
    query_type_descriptions: Mapping[str, str],
    client: ScenarioPlanningClient,
    *,
    run_id: str,
    batch_size: int,
    max_concurrency: int,
    max_retries: int,
    retry_delay_seconds: float,
    on_batch_completed: Callable[[Sequence[ScenarioBundlePlan]], None] | None = None,
) -> tuple[tuple[ScenarioBundlePlan, ...], tuple[LlmTaskFailure, ...]]:
    """Plan evidence bundles concurrently and checkpoint each successful batch."""
    items = [build_scenario_planning_item(bundle, facts_by_id, taxonomy, query_type_descriptions) for bundle in bundles]
    batches = [items[start : start + batch_size] for start in range(0, len(items), batch_size)]
    plans: list[ScenarioBundlePlan] = []
    failures: list[LlmTaskFailure] = []
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {
            executor.submit(
                execute_llm_task,
                _batch_task_key(batch),
                partial(_plan_batch, client, batch, run_id),
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds,
            ): batch
            for batch in batches
        }
        for future in as_completed(futures):
            batch = futures[future]
            try:
                batch_plans = future.result()
            except LlmTaskExhaustedError as error:
                failures.append(LlmTaskFailure(task_key=_batch_task_key(batch), error=str(error)))
                continue
            plans.extend(batch_plans)
            if on_batch_completed is not None:
                on_batch_completed(batch_plans)
    return tuple(plans), tuple(failures)


def _plan_batch(
    client: ScenarioPlanningClient,
    items: Sequence[ScenarioPlanningBatchItem],
    run_id: str,
) -> tuple[ScenarioBundlePlan, ...]:
    """Plan one batch and translate typed LLM results into persisted records."""
    results = client.plan_batch(items)
    requested_ids = [item.bundle_id for item in items]
    result_ids = [result.bundle_id for result in results]
    if len(result_ids) != len(set(result_ids)) or set(result_ids) != set(requested_ids):
        msg = "Scenario planning response must contain exactly one result for every requested bundle"
        raise ValueError(msg)
    results_by_bundle_id = {result.bundle_id: result for result in results}
    items_by_bundle_id = {item.bundle_id: item for item in items}
    return tuple(
        _build_plan(results_by_bundle_id[bundle_id], items_by_bundle_id[bundle_id], run_id)
        for bundle_id in requested_ids
    )


def _build_plan(
    result: ScenarioPlanningBatchResult,
    item: ScenarioPlanningBatchItem,
    run_id: str,
) -> ScenarioBundlePlan:
    """Build one persisted plan while enforcing structural references from the typed request."""
    candidate_ids = {candidate.query_type_id for candidate in item.query_type_candidates}
    candidate_dates = set(item.candidate_as_of_dates)
    planned_ids = [planned.query_type_id for planned in result.supported_query_types]
    if len(planned_ids) != len(set(planned_ids)):
        msg = f"Scenario plan for {item.bundle_id} contains duplicate query types"
        raise ValueError(msg)
    query_types: list[ScenarioQueryTypePlan] = []
    for planned in result.supported_query_types:
        if planned.query_type_id not in candidate_ids:
            msg = f"Scenario plan for {item.bundle_id} references an unavailable query type"
            raise ValueError(msg)
        if planned.as_of_date is not None and planned.as_of_date not in candidate_dates:
            msg = f"Scenario plan for {item.bundle_id} references an unavailable as_of_date"
            raise ValueError(msg)
        query_types.append(
            ScenarioQueryTypePlan(
                query_type_id=planned.query_type_id,
                as_of_date=date.fromisoformat(planned.as_of_date) if planned.as_of_date else None,
            )
        )
    return ScenarioBundlePlan(bundle_id=item.bundle_id, query_types=tuple(query_types), run_id=run_id)


def _batch_task_key(items: Sequence[ScenarioPlanningBatchItem]) -> str:
    """Build one deterministic retry key from the ordered bundle IDs in a batch."""
    bundle_ids = "\0".join(item.bundle_id for item in items)
    return f"scenario-planning-batch:{hashlib.sha256(bundle_ids.encode('utf-8')).hexdigest()}"
