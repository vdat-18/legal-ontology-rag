"""Typed BAML planner for one-shot adaptive retrieval."""

from __future__ import annotations

import unicodedata
from typing import Protocol

from loguru import logger

from legal_rag.adaptive_rag.models import AdaptiveQueryPlan, AdaptiveRoute
from legal_rag.baml_client.sync_client import b
from legal_rag.baml_transport import BamlCaller
from legal_rag.settings import AdaptiveRagSettings, Settings


class AdaptivePlanner(Protocol):
    """Boundary implemented by adaptive query planners."""

    def plan(self, query: str) -> AdaptiveQueryPlan:
        """Return one validated retrieval plan for *query*."""


class BamlAdaptivePlanner:
    """Create and validate an adaptive retrieval plan with BAML."""

    def __init__(self, settings: Settings, adaptive_settings: AdaptiveRagSettings) -> None:
        """Initialize the bounded planner client."""
        self._settings = adaptive_settings
        self._caller = BamlCaller.from_settings(
            settings,
            max_output_tokens=adaptive_settings.planner_max_output_tokens,
            temperature=adaptive_settings.planner_temperature,
            max_response_bytes=adaptive_settings.planner_max_response_bytes,
        )

    def plan(self, query: str) -> AdaptiveQueryPlan:
        """Generate, parse, normalize, and validate one retrieval plan."""
        request = b.request.PlanAdaptiveQuery(
            query,
            self._settings.max_subqueries,
            baml_options=self._caller.options,
        )
        raw_output = self._caller.execute(request)
        result = b.parse.PlanAdaptiveQuery(raw_output, baml_options=self._caller.options)
        route = AdaptiveRoute.DIRECT if result.route.value == "Direct" else AdaptiveRoute.MULTI_HOP
        return _normalize_plan(route, result.subqueries, query, self._settings.max_subqueries)


def _normalize_plan(
    route: AdaptiveRoute,
    subqueries: list[str],
    original_query: str,
    max_subqueries: int,
) -> AdaptiveQueryPlan:
    """Normalize planner strings and enforce the route contract."""
    if route == AdaptiveRoute.DIRECT:
        if subqueries:
            logger.warning(
                "Adaptive planner returned {} subqueries for a direct route; ignoring them",
                len(subqueries),
            )
        return AdaptiveQueryPlan(route=route)

    original_key = _normalized_query_key(original_query)
    normalized_subqueries: list[str] = []
    seen: set[str] = {original_key}
    for raw_subquery in subqueries:
        subquery = " ".join(raw_subquery.split())
        if not subquery:
            msg = "adaptive subqueries must not be empty"
            raise ValueError(msg)
        subquery_key = _normalized_query_key(subquery)
        if subquery_key in seen:
            continue
        seen.add(subquery_key)
        normalized_subqueries.append(subquery)

    if not 2 <= len(normalized_subqueries) <= max_subqueries:
        msg = f"multi_hop adaptive plans must contain 2 to {max_subqueries} unique subqueries"
        raise ValueError(msg)
    return AdaptiveQueryPlan(route=route, subqueries=tuple(normalized_subqueries))


def _normalized_query_key(query: str) -> str:
    """Return a stable Unicode and whitespace-normalized deduplication key."""
    return unicodedata.normalize("NFKC", " ".join(query.split())).casefold()
