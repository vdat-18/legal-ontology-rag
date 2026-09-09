"""Filter penalty rules by temporal validity."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from loguru import logger

from legal_rag.ontology_rag.models import PenaltyRule


class TemporalResolver:
    """Filter rules by query date."""

    def filter_valid(self, rules: Sequence[PenaltyRule], query_date: date) -> list[PenaltyRule]:
        """Return rules valid at query_date."""
        valid_rules: list[PenaltyRule] = []
        for rule in rules:
            if rule.valid_from > query_date:
                continue
            if rule.valid_to and rule.valid_to < query_date:
                continue
            valid_rules.append(rule)
        logger.debug("Filtered {} rules to {} valid at {}", len(rules), len(valid_rules), query_date)
        return valid_rules
