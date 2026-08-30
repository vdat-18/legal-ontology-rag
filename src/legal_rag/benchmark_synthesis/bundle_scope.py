"""LLM confirmation that evidence bundles match the benchmark domain."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Protocol

from baml_py.logging import set_log_level

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import (
    BundleScopeBatchItem,
    BundleScopeBatchResult,
    BundleScopeFact,
)
from legal_rag.baml_transport import build_client_registry, execute_baml_request
from legal_rag.benchmark_synthesis.config import SynthesisSettings
from legal_rag.benchmark_synthesis.evidence_graph import EvidenceBundle
from legal_rag.benchmark_synthesis.llm_execution import (
    LlmTaskExhaustedError,
    LlmTaskFailure,
    execute_llm_task,
)
from legal_rag.benchmark_synthesis.models import LegalFact
from legal_rag.parser.hashing import sha256_json
from legal_rag.settings import Settings

BUNDLE_SCOPE_SCHEMA_VERSION = "1.0.0"
BUNDLE_SCOPE_PROMPT_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class BundleScopeRecord:
    """One persisted domain-scope decision for an evidence bundle."""

    bundle_id: str
    is_in_scope: bool
    reason: str
    run_id: str
    prompt_version: str
    confirmed_by_model: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable scope decision."""
        return {
            "schema_version": BUNDLE_SCOPE_SCHEMA_VERSION,
            "bundle_id": self.bundle_id,
            "is_in_scope": self.is_in_scope,
            "reason": self.reason,
            "run_id": self.run_id,
            "prompt_version": self.prompt_version,
            "confirmed_by_model": self.confirmed_by_model,
        }


def build_bundle_scope_run_id(
    bundles: Sequence[EvidenceBundle],
    facts: Sequence[LegalFact],
    synthesis_settings: SynthesisSettings,
    *,
    llm_base_url: str,
    llm_model: str,
) -> str:
    """Build an identity for scope decisions while excluding execution-only concurrency."""
    facts_by_id = {fact.fact_id: fact for fact in facts}
    return sha256_json(
        {
            "bundles": [
                {
                    "bundle_id": bundle.bundle_id,
                    "kind": bundle.kind,
                    "fact_ids": list(bundle.fact_ids),
                    "bridge": bundle.bridge,
                    "statements": [facts_by_id[fact_id].statement for fact_id in bundle.fact_ids],
                }
                for bundle in bundles
            ],
            "generation": asdict(synthesis_settings.bundle_scope),
            "llm_base_url": llm_base_url,
            "llm_model": llm_model,
            "prompt_version": BUNDLE_SCOPE_PROMPT_VERSION,
            "schema_version": BUNDLE_SCOPE_SCHEMA_VERSION,
        }
    )


class BundleScopeClient(Protocol):
    """Boundary implemented by a typed bundle-scope language-model client."""

    def confirm_batch(
        self,
        bundles: Sequence[EvidenceBundle],
        facts_by_id: Mapping[str, LegalFact],
    ) -> Sequence[BundleScopeBatchResult]:
        """Return one scope decision for every input bundle."""


class BamlBundleScopeClient:
    """Typed BAML client backed by the configured OpenAI-compatible provider."""

    def __init__(self, settings: Settings, synthesis_settings: SynthesisSettings) -> None:
        """Create a runtime client with the configured bundle-scope limits."""
        set_log_level("OFF")
        self._registry = build_client_registry(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_output_tokens=synthesis_settings.bundle_scope.max_output_tokens,
            temperature=synthesis_settings.bundle_scope.temperature,
            timeout_seconds=settings.llm_timeout_seconds,
            reasoning_effort=synthesis_settings.bundle_scope.reasoning_effort,
        )
        self._timeout_seconds = settings.llm_timeout_seconds
        self._max_response_bytes = synthesis_settings.max_response_bytes

    def confirm_batch(
        self,
        bundles: Sequence[EvidenceBundle],
        facts_by_id: Mapping[str, LegalFact],
    ) -> Sequence[BundleScopeBatchResult]:
        """Confirm one bounded batch through BAML."""
        items = [
            BundleScopeBatchItem(
                bundle_id=bundle.bundle_id,
                bundle_kind=bundle.kind,
                bridge=bundle.bridge,
                facts=[_fact_context(facts_by_id[fact_id]) for fact_id in bundle.fact_ids],
            )
            for bundle in bundles
        ]
        request = b.request.ConfirmBundleScopeBatch(items, baml_options={"client_registry": self._registry})
        raw_output = execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)
        return b.parse.ConfirmBundleScopeBatch(raw_output, baml_options={"client_registry": self._registry})


def confirm_bundle_scopes(
    bundles: Sequence[EvidenceBundle],
    facts: Sequence[LegalFact],
    client: BundleScopeClient,
    *,
    run_id: str,
    model: str,
    batch_size: int,
    max_concurrency: int,
    max_retries: int,
    retry_delay_seconds: float,
    on_batch_confirmed: Callable[[Sequence[BundleScopeRecord]], None] | None = None,
) -> tuple[tuple[BundleScopeRecord, ...], tuple[LlmTaskFailure, ...]]:
    """Confirm pending bundles concurrently and checkpoint every completed batch."""
    facts_by_id = {fact.fact_id: fact for fact in facts}
    batches = [bundles[start : start + batch_size] for start in range(0, len(bundles), batch_size)]
    records: list[BundleScopeRecord] = []
    failures: list[LlmTaskFailure] = []
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {
            executor.submit(
                execute_llm_task,
                _batch_task_key(batch),
                partial(_confirm_batch, batch, facts_by_id, client, run_id=run_id, model=model),
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds,
            ): batch
            for batch in batches
        }
        for future in as_completed(futures):
            batch = futures[future]
            try:
                batch_records = future.result()
            except LlmTaskExhaustedError as error:
                failures.append(LlmTaskFailure(task_key=_batch_task_key(batch), error=str(error)))
                continue
            records.extend(batch_records)
            if on_batch_confirmed is not None:
                on_batch_confirmed(batch_records)
    return tuple(records), tuple(failures)


def _fact_context(fact: LegalFact) -> BundleScopeFact:
    """Return the minimal fact representation needed for scope confirmation."""
    return BundleScopeFact(
        fact_id=fact.fact_id,
        statement=fact.statement,
        valid_from=fact.valid_from.isoformat() if fact.valid_from else None,
        valid_to=fact.valid_to.isoformat() if fact.valid_to else None,
        document_number=fact.document_number,
    )


def _confirm_batch(
    bundles: Sequence[EvidenceBundle],
    facts_by_id: Mapping[str, LegalFact],
    client: BundleScopeClient,
    *,
    run_id: str,
    model: str,
) -> tuple[BundleScopeRecord, ...]:
    """Validate one model response and attach reproducibility metadata."""
    results = client.confirm_batch(bundles, facts_by_id)
    expected_ids = tuple(bundle.bundle_id for bundle in bundles)
    result_ids = tuple(result.bundle_id for result in results)
    if len(results) != len(bundles) or len(set(result_ids)) != len(result_ids) or set(result_ids) != set(expected_ids):
        msg = "Bundle-scope batch response must contain exactly one result for every bundle"
        raise ValueError(msg)
    results_by_id = {result.bundle_id: result for result in results}
    return tuple(
        BundleScopeRecord(
            bundle_id=bundle_id,
            is_in_scope=results_by_id[bundle_id].decision.is_in_scope,
            reason=results_by_id[bundle_id].decision.reason,
            run_id=run_id,
            prompt_version=BUNDLE_SCOPE_PROMPT_VERSION,
            confirmed_by_model=model,
        )
        for bundle_id in expected_ids
    )


def _batch_task_key(bundles: Sequence[EvidenceBundle]) -> str:
    """Build one deterministic retry key for a bundle batch."""
    bundle_ids = "\0".join(bundle.bundle_id for bundle in bundles)
    return f"bundle-scope-batch:{hashlib.sha256(bundle_ids.encode('utf-8')).hexdigest()}"
