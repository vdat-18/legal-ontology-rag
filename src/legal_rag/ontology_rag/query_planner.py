"""Typed query planning for temporal ontology retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Protocol

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import ConversationTurn as BamlConversationTurn
from legal_rag.baml_transport import BamlCaller
from legal_rag.settings import Settings


class TemporalQueryMode(Enum):
    """How a question constrains legal-rule validity."""

    CURRENT = "current"
    AT_DATE = "at_date"
    COMPARE_DATES = "compare_dates"


@dataclass(frozen=True, slots=True)
class OntologyQueryPlan:
    """Standalone retrieval query and its explicit temporal targets."""

    search_query: str
    temporal_mode: TemporalQueryMode
    query_dates: tuple[date, ...] = ()


class OntologyQueryPlanner(Protocol):
    """Boundary implemented by ontology query planners."""

    def plan(
        self,
        query: str,
        history: Sequence[tuple[str, str]],
        reference_date: date,
    ) -> OntologyQueryPlan:
        """Return one validated retrieval and temporal plan."""


class BamlOntologyQueryPlanner:
    """Plan a standalone ontology query with one bounded BAML call."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the typed planner client."""
        self._caller = BamlCaller.from_settings(
            settings,
            max_output_tokens=4096,
            temperature=0.0,
            max_response_bytes=1024 * 1024,
        )

    def plan(
        self,
        query: str,
        history: Sequence[tuple[str, str]],
        reference_date: date,
    ) -> OntologyQueryPlan:
        """Generate and validate one ontology query plan."""
        baml_history = [BamlConversationTurn(question=question, answer=answer) for question, answer in history]
        request = b.request.PlanOntologyQuery(
            query,
            baml_history,
            reference_date.isoformat(),
            baml_options=self._caller.options,
        )
        raw_output = self._caller.execute(request)
        result = b.parse.PlanOntologyQuery(raw_output, baml_options=self._caller.options)
        mode = {
            "Current": TemporalQueryMode.CURRENT,
            "AtDate": TemporalQueryMode.AT_DATE,
            "CompareDates": TemporalQueryMode.COMPARE_DATES,
        }[result.temporal_mode.value]
        return normalize_query_plan(result.query_vi, mode, result.query_dates)


def normalize_query_plan(
    search_query: str,
    temporal_mode: TemporalQueryMode,
    raw_dates: Sequence[str],
) -> OntologyQueryPlan:
    """Normalize planner output and enforce the temporal-mode contract."""
    normalized_query = " ".join(search_query.split())
    if not normalized_query:
        msg = "Ontology query planner returned an empty search query"
        raise ValueError(msg)

    parsed_dates: list[date] = []
    for raw_date in raw_dates:
        try:
            parsed = date.fromisoformat(raw_date)
        except ValueError as error:
            msg = f"Ontology query planner returned an invalid ISO date: {raw_date!r}"
            raise ValueError(msg) from error
        if parsed.isoformat() != raw_date:
            msg = f"Ontology query planner returned an invalid ISO date: {raw_date!r}"
            raise ValueError(msg)
        if parsed not in parsed_dates:
            parsed_dates.append(parsed)

    expected_count = {
        TemporalQueryMode.CURRENT: (0, 0),
        TemporalQueryMode.AT_DATE: (1, 1),
        TemporalQueryMode.COMPARE_DATES: (2, 8),
    }[temporal_mode]
    if not expected_count[0] <= len(parsed_dates) <= expected_count[1]:
        msg = (
            f"{temporal_mode.value} ontology plans require {expected_count[0]} to "
            f"{expected_count[1]} unique query dates"
        )
        raise ValueError(msg)
    return OntologyQueryPlan(
        search_query=normalized_query,
        temporal_mode=temporal_mode,
        query_dates=tuple(parsed_dates),
    )
