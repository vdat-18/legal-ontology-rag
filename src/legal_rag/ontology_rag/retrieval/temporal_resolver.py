"""Filter penalty rules by temporal validity."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from loguru import logger

from legal_rag.ontology_rag.index.temporal_index import TemporalValidity, classify_validity
from legal_rag.ontology_rag.models import PenaltyRule


@dataclass(frozen=True, slots=True)
class TemporalResolution:
    """Penalty rules partitioned by temporal validity certainty."""

    valid: tuple[PenaltyRule, ...]
    invalid: tuple[PenaltyRule, ...]
    indeterminate: tuple[PenaltyRule, ...]


class TemporalResolver:
    """Filter rules by query date."""

    def filter_valid(self, rules: Sequence[PenaltyRule], query_date: date) -> list[PenaltyRule]:
        """Return rules whose validity at query_date is fully established."""
        return list(self.partition(rules, query_date).valid)

    def partition(self, rules: Sequence[PenaltyRule], query_date: date) -> TemporalResolution:
        """Partition rules into valid, invalid, and indeterminate groups."""
        grouped: dict[TemporalValidity, list[PenaltyRule]] = {status: [] for status in TemporalValidity}
        for rule in rules:
            status = classify_validity(rule.valid_from, rule.valid_to, query_date)
            grouped[status].append(rule)
        result = TemporalResolution(
            valid=tuple(grouped[TemporalValidity.VALID]),
            invalid=tuple(grouped[TemporalValidity.INVALID]),
            indeterminate=tuple(grouped[TemporalValidity.INDETERMINATE]),
        )
        logger.debug(
            "Resolved {} rules at {}: {} valid, {} invalid, {} indeterminate",
            len(rules),
            query_date,
            len(result.valid),
            len(result.invalid),
            len(result.indeterminate),
        )
        return result
