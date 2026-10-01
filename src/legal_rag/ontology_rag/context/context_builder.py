"""Build structured, source-grounded context for ontology answer generation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

from loguru import logger

from legal_rag.naive_rag.models import RetrievedChunk
from legal_rag.ontology_rag.models import ConditionExpression, LegalConsequence, LegalException, PenaltyRule


class ContextBuilder:
    """Build one faithful context from selected ontology rules and their source chunks."""

    def build(
        self,
        rules: Sequence[PenaltyRule],
        chunks: Sequence[RetrievedChunk],
        temporal_status: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Build structured rules plus source chunks not represented by a selected rule."""
        statuses = temporal_status or {}
        rule_dicts = [self._build_rule(rule, statuses.get(str(rule.uri), "unknown")) for rule in rules]
        rule_chunk_ids = {chunk_id for rule in rules for chunk_id in rule.source_chunk_ids}
        unique_chunks = self._unique_chunks(chunks)
        context: dict[str, Any] = {
            "selected_penalty_rules": rule_dicts,
            # Retain the former key while callers migrate to the more accurate name.
            "valid_penalty_rules": rule_dicts,
            "fallback_chunks": [
                {"chunk_id": chunk.chunk_id, "text": chunk.text, "source_name": chunk.source_name}
                for chunk in unique_chunks
                if chunk.chunk_id not in rule_chunk_ids
            ],
        }
        logger.debug(
            "Built context with {} rules and {} fallback chunks",
            len(rules),
            len(context["fallback_chunks"]),
        )
        return context

    def build_answer_contexts(
        self,
        rules: Sequence[PenaltyRule],
        chunks: Sequence[RetrievedChunk],
        temporal_status: Mapping[str, str] | None = None,
        *,
        max_characters: int | None = None,
    ) -> list[RetrievedChunk]:
        """Render bounded rule contexts while retaining exact source text per rule."""
        if max_characters is not None and max_characters < 1:
            msg = f"max_characters must be positive, got {max_characters}"
            raise ValueError(msg)
        statuses = temporal_status or {}
        unique_chunks = self._unique_chunks(chunks)
        chunks_by_id = {chunk.chunk_id: chunk for chunk in unique_chunks}
        rule_chunk_ids = {chunk_id for rule in rules for chunk_id in rule.source_chunk_ids}
        source_ids_by_rule = [
            tuple(chunk_id for chunk_id in rule.source_chunk_ids if chunk_id in chunks_by_id) for rule in rules
        ]
        missing_source_rules = sum(not source_ids for source_ids in source_ids_by_rule)
        if missing_source_rules:
            logger.warning("{} selected rules have no exact source chunk available", missing_source_rules)

        fallback_chunks = [
            chunk for chunk in unique_chunks if rule_chunk_ids.isdisjoint(chunk.effective_evidence_chunk_ids)
        ]
        if max_characters is None:
            answer_contexts = [
                self._rule_context(
                    rule,
                    chunks_by_id,
                    statuses.get(str(rule.uri), "unknown"),
                    source_ids_by_rule[index],
                )
                for index, rule in enumerate(rules)
            ]
            answer_contexts.extend(fallback_chunks)
            return answer_contexts

        answer_contexts, used_characters, omitted_source_chunks = self._bounded_rule_contexts(
            rules,
            chunks_by_id,
            statuses,
            source_ids_by_rule,
            max_characters,
        )
        used_characters, omitted_fallback_chunks = self._append_bounded_fallbacks(
            answer_contexts,
            fallback_chunks,
            used_characters,
            max_characters,
        )
        if omitted_source_chunks or omitted_fallback_chunks:
            logger.info(
                "Generation context used {}/{} characters and omitted {} rule-source chunks and {} fallback chunks",
                used_characters,
                max_characters,
                omitted_source_chunks,
                omitted_fallback_chunks,
            )
        return answer_contexts

    def _bounded_rule_contexts(
        self,
        rules: Sequence[PenaltyRule],
        chunks_by_id: Mapping[str, RetrievedChunk],
        statuses: Mapping[str, str],
        source_ids_by_rule: Sequence[tuple[str, ...]],
        max_characters: int,
    ) -> tuple[list[RetrievedChunk], int, int]:
        """Reserve one exact source per rule, then add further sources while they fit."""
        included_source_ids = [source_ids[:1] for source_ids in source_ids_by_rule]
        contexts = [
            self._rule_context(
                rule,
                chunks_by_id,
                statuses.get(str(rule.uri), "unknown"),
                included_source_ids[index],
            )
            for index, rule in enumerate(rules)
        ]
        used_characters = sum(self._context_characters(context) for context in contexts)
        if used_characters > max_characters:
            msg = (
                f"Generation context requires {used_characters} characters for the minimum rule evidence, "
                f"exceeding the {max_characters}-character budget"
            )
            raise ValueError(msg)

        omitted_source_chunks = 0
        for index, rule in enumerate(rules):
            for chunk_id in source_ids_by_rule[index][1:]:
                candidate_source_ids = (*included_source_ids[index], chunk_id)
                candidate = self._rule_context(
                    rule,
                    chunks_by_id,
                    statuses.get(str(rule.uri), "unknown"),
                    candidate_source_ids,
                )
                candidate_characters = self._context_characters(candidate)
                existing_characters = self._context_characters(contexts[index])
                if used_characters - existing_characters + candidate_characters > max_characters:
                    omitted_source_chunks += 1
                    continue
                included_source_ids[index] = candidate_source_ids
                contexts[index] = candidate
                used_characters += candidate_characters - existing_characters
        return contexts, used_characters, omitted_source_chunks

    def _append_bounded_fallbacks(
        self,
        contexts: list[RetrievedChunk],
        fallback_chunks: Sequence[RetrievedChunk],
        used_characters: int,
        max_characters: int,
    ) -> tuple[int, int]:
        """Append fallback chunks in rank order while they fit the remaining budget."""
        omitted_count = 0
        for chunk in fallback_chunks:
            chunk_characters = self._context_characters(chunk)
            if used_characters + chunk_characters > max_characters:
                omitted_count += 1
                continue
            contexts.append(chunk)
            used_characters += chunk_characters
        return used_characters, omitted_count

    def _rule_context(
        self,
        rule: PenaltyRule,
        chunks_by_id: Mapping[str, RetrievedChunk],
        temporal_status: str,
        source_chunk_ids: Sequence[str] | None = None,
    ) -> RetrievedChunk:
        """Render one rule without presenting normalized fields as verbatim legal text."""
        structured = json.dumps(
            self._build_rule(rule, temporal_status),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        included_source_ids = tuple(
            chunk_id
            for chunk_id in (rule.source_chunk_ids if source_chunk_ids is None else source_chunk_ids)
            if chunk_id in chunks_by_id
        )
        source_sections = [
            f"[{chunks_by_id[chunk_id].source_name}]\n{chunks_by_id[chunk_id].text}" for chunk_id in included_source_ids
        ]
        text_parts = [
            "DỮ LIỆU ONTOLOGY ĐÃ CHUẨN HOÁ (không phải nguyên văn để trích dẫn):",
            structured,
        ]
        if source_sections:
            text_parts.extend(("VĂN BẢN NGUỒN DÙNG ĐỂ TRÍCH DẪN NGUYÊN VĂN:", "\n\n".join(source_sections)))
        else:
            text_parts.append("Không tìm thấy văn bản nguồn tương ứng trong collection retrieval.")
        rule_id = str(rule.uri)
        return RetrievedChunk(
            chunk_id=rule_id,
            text="\n".join(text_parts),
            source_name=self._rule_source_name(rule),
            kind="ontology_rule",
            score=1.0,
            unit_id=rule_id,
            unit_kind="ontology_rule",
            evidence_chunk_ids=included_source_ids,
        )

    def _build_rule(self, rule: PenaltyRule, temporal_status: str) -> dict[str, Any]:
        """Build one JSON-serializable rule without inventing missing values."""
        return {
            "uri": str(rule.uri),
            "conduct": rule.conduct_label_vi,
            "vehicle_categories": list(rule.vehicle_categories),
            "subject_kind": rule.subject_kind,
            "roles": list(rule.roles),
            "conditions": [self._build_condition(condition) for condition in rule.condition_expressions],
            "consequences": [self._build_consequence(consequence) for consequence in rule.consequences]
            or self._legacy_consequences(rule),
            "exceptions": [self._build_exception(exception) for exception in rule.exceptions],
            "valid_period": {
                "valid_from": rule.valid_from.isoformat() if rule.valid_from is not None else None,
                "valid_to_exclusive": rule.valid_to.isoformat() if rule.valid_to is not None else None,
                "status_for_query": temporal_status,
            },
            "document_number": rule.document_number or None,
            "provision_path": rule.provision_path or None,
            "source_chunk_ids": list(rule.source_chunk_ids),
        }

    def _build_condition(self, condition: ConditionExpression) -> dict[str, Any]:
        """Render one condition-expression subtree."""
        return {
            "uri": str(condition.uri),
            "kind": condition.kind,
            "label": condition.label_vi,
            "operands": [self._build_condition(operand) for operand in condition.operands],
            "source_chunk_ids": list(condition.provenance.source_chunk_ids),
        }

    def _build_consequence(self, consequence: LegalConsequence) -> dict[str, Any]:
        """Render one independently evidenced consequence."""
        return {
            "uri": str(consequence.uri),
            "type": consequence.kind,
            "label": consequence.label_vi,
            "min_amount": self._format_decimal(consequence.minimum_amount),
            "max_amount": self._format_decimal(consequence.maximum_amount),
            "currency": consequence.currency,
            "point_value": consequence.point_value,
            "minimum_duration_months": consequence.minimum_duration_months,
            "maximum_duration_months": consequence.maximum_duration_months,
            "affected_targets": [term.label_vi or str(term.uri) for term in consequence.affected_targets],
            "responsible_roles": [term.label_vi or str(term.uri) for term in consequence.responsible_roles],
            "source_chunk_ids": list(consequence.provenance.source_chunk_ids),
        }

    def _legacy_consequences(self, rule: PenaltyRule) -> list[dict[str, Any]]:
        """Keep hand-built legacy rules usable while canonical graphs expose typed consequences."""
        consequences: list[dict[str, Any]] = []
        if rule.monetary_penalty_min is not None or rule.monetary_penalty_max is not None:
            consequences.append(
                {
                    "type": "MonetaryPenalty",
                    "min_amount": self._format_decimal(rule.monetary_penalty_min),
                    "max_amount": self._format_decimal(rule.monetary_penalty_max),
                    "currency": rule.currency,
                }
            )
        if rule.point_value is not None:
            consequences.append({"type": "PointDeduction", "point_value": rule.point_value})
        return consequences

    def _build_exception(self, exception: LegalException) -> dict[str, Any]:
        """Render one exception, its citations, and verified redirect targets."""
        return {
            "uri": str(exception.uri),
            "citations": [
                {
                    "text": citation.citation_text_vi,
                    "referenced_provisions": list(citation.referenced_provision_paths),
                }
                for citation in exception.citations
            ],
            "excluded_provisions": list(exception.excluded_provision_paths),
            "redirected_rule_uris": [str(uri) for uri in exception.redirected_rule_uris],
            "source_chunk_ids": list(exception.provenance.source_chunk_ids),
        }

    @staticmethod
    def _format_decimal(amount: Decimal | None) -> str | None:
        """Return an exact decimal string without treating zero as missing."""
        return str(amount) if amount is not None else None

    @staticmethod
    def _rule_source_name(rule: PenaltyRule) -> str:
        """Build the human-facing legal source label supplied to generation."""
        parts = [part for part in (rule.provision_path, rule.document_number) if part]
        return " — ".join(parts) if parts else str(rule.uri)

    @staticmethod
    def _unique_chunks(chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        """Keep the first occurrence of each source chunk in rank order."""
        seen: set[str] = set()
        unique: list[RetrievedChunk] = []
        for chunk in chunks:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            unique.append(chunk)
        return unique

    @staticmethod
    def _context_characters(chunk: RetrievedChunk) -> int:
        """Count the text-bearing fields sent through the generation boundary."""
        return len(chunk.chunk_id) + len(chunk.source_name) + len(chunk.text)
