"""Build structured context from penalty rules for LLM generation."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Any

from loguru import logger

from legal_rag.naive_rag.models import RetrievedChunk
from legal_rag.ontology_rag.models import PenaltyRule


class ContextBuilder:
    """Build structured JSON context from ontology instances and chunks."""

    def build(self, rules: Sequence[PenaltyRule], chunks: Sequence[RetrievedChunk]) -> dict[str, Any]:
        """Build structured context with typed rules and fallback chunks."""
        rule_chunk_ids = {chunk_id for rule in rules for chunk_id in rule.source_chunk_ids}

        context: dict[str, Any] = {
            "valid_penalty_rules": [self._build_rule(rule) for rule in rules],
            "fallback_chunks": [
                {"chunk_id": chunk.chunk_id, "text": chunk.text[:500], "source_name": chunk.source_name}
                for chunk in chunks
                if chunk.chunk_id not in rule_chunk_ids
            ],
        }

        logger.debug("Built context with {} rules and {} fallback chunks", len(rules), len(context["fallback_chunks"]))
        return context

    def _build_rule(self, rule: PenaltyRule) -> dict[str, Any]:
        """Build one rule context dict."""
        rule_dict: dict[str, Any] = {
            "uri": str(rule.uri),
            "conduct": rule.conduct_label_vi,
            "vehicle_categories": list(rule.vehicle_categories),
            "subject_kind": rule.subject_kind,
            "roles": list(rule.roles),
            "consequences": [],
            "valid_period": {
                "valid_from": rule.valid_from.isoformat(),
                "valid_to": rule.valid_to.isoformat() if rule.valid_to else "present",
            },
            "provision_path": rule.provision_path,
        }

        if rule.monetary_penalty_min is not None and rule.monetary_penalty_max is not None:
            rule_dict["consequences"].append(
                {
                    "type": "monetary_penalty",
                    "min_amount": self._format_currency(rule.monetary_penalty_min),
                    "max_amount": self._format_currency(rule.monetary_penalty_max),
                    "currency": rule.currency or "VND",
                }
            )

        if rule.point_value is not None:
            rule_dict["consequences"].append({"type": "point_deduction", "value": rule.point_value})

        return rule_dict

    def _format_currency(self, amount: Decimal) -> str:
        """Format decimal amount as string."""
        return str(amount)
