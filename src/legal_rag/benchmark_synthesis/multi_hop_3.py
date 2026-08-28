"""Build and confirm three-fact multi-hop evidence bundles."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import partial
from itertools import combinations
from typing import Any, Protocol

from baml_py.logging import set_log_level

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import (
    EvidenceConnectionSummary,
    LegalFactSummary,
    MultiHop3BundleBatchItem,
)
from legal_rag.baml_transport import build_client_registry, execute_baml_request
from legal_rag.benchmark_synthesis.config import SynthesisSettings
from legal_rag.benchmark_synthesis.evidence_graph import EvidenceBundle, EvidenceRelation
from legal_rag.benchmark_synthesis.llm_execution import (
    LlmTaskExhaustedError,
    LlmTaskFailure,
    execute_llm_task,
)
from legal_rag.benchmark_synthesis.models import LegalFact
from legal_rag.parser.hashing import sha256_json
from legal_rag.settings import Settings

MULTI_HOP_3_CANDIDATE_SELECTION_VERSION = "1.0.0"
MULTI_HOP_3_CANDIDATE_SET_SCHEMA_VERSION = "1.0.0"
MULTI_HOP_3_BUNDLE_SCHEMA_VERSION = "1.0.0"
MULTI_HOP_3_PROGRESS_SCHEMA_VERSION = "1.0.0"
MULTI_HOP_3_PROMPT_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class MultiHop3CandidateSet:
    """Fingerprint-validated three-fact candidates selected from the pair graph."""

    run_id: str
    triples: tuple[tuple[str, str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable candidate set."""
        return {
            "schema_version": MULTI_HOP_3_CANDIDATE_SET_SCHEMA_VERSION,
            "run_id": self.run_id,
            "triples": [list(triple) for triple in self.triples],
        }


@dataclass(frozen=True, slots=True)
class ConfirmedMultiHop3Bundle:
    """One LLM-confirmed bundle whose answer requires all three facts."""

    bundle_id: str
    fact_ids: tuple[str, str, str]
    bridge: str
    confirmed_by_model: str
    run_id: str
    prompt_version: str = MULTI_HOP_3_PROMPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable confirmed bundle."""
        return {
            "schema_version": MULTI_HOP_3_BUNDLE_SCHEMA_VERSION,
            "bundle_id": self.bundle_id,
            "fact_ids": list(self.fact_ids),
            "bridge": self.bridge,
            "confirmed_by_model": self.confirmed_by_model,
            "run_id": self.run_id,
            "prompt_version": self.prompt_version,
        }

    def to_evidence_bundle(self) -> EvidenceBundle:
        """Return the shared scenario-sampling view of this bundle."""
        return EvidenceBundle(
            bundle_id=self.bundle_id,
            kind="multi_hop",
            fact_ids=self.fact_ids,
            bridge=self.bridge,
        )


@dataclass(frozen=True, slots=True)
class MultiHop3ProgressRecord:
    """One three-fact candidate whose confirmation has completed."""

    candidate_id: str
    run_id: str
    prompt_version: str = MULTI_HOP_3_PROMPT_VERSION
    confirmed_by_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable progress record."""
        return {
            "schema_version": MULTI_HOP_3_PROGRESS_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "run_id": self.run_id,
            "prompt_version": self.prompt_version,
            "confirmed_by_model": self.confirmed_by_model,
        }


def multi_hop_3_candidate_id(fact_ids: Sequence[str]) -> str:
    """Return the content identity of one canonical three-fact candidate."""
    canonical = tuple(sorted(fact_ids))
    if len(canonical) != 3 or len(set(canonical)) != 3:
        msg = "A multi-hop-3 candidate must contain exactly three distinct fact IDs"
        raise ValueError(msg)
    return sha256_json({"fact_ids": list(canonical)})


def build_multi_hop_3_candidate_run_id(
    relations: Sequence[EvidenceRelation],
    candidate_limit: int,
) -> str:
    """Fingerprint pair-graph semantics and candidate-selection settings."""
    return sha256_json(
        {
            "selection_version": MULTI_HOP_3_CANDIDATE_SELECTION_VERSION,
            "candidate_limit": candidate_limit,
            "relations": [
                {
                    "bridge": relation.bridge,
                    "bundle_kind": relation.bundle_kind,
                    "fact_ids": list(relation.fact_ids),
                }
                for relation in sorted(relations, key=lambda item: item.fact_ids)
            ],
        }
    )


def build_multi_hop_3_run_id(
    candidate_run_id: str,
    synthesis_settings: SynthesisSettings,
    *,
    llm_base_url: str,
    llm_model: str,
) -> str:
    """Fingerprint every input that changes three-fact confirmation."""
    return sha256_json(
        {
            "candidate_run_id": candidate_run_id,
            "llm_base_url": llm_base_url,
            "llm_model": llm_model,
            "prompt_version": MULTI_HOP_3_PROMPT_VERSION,
            "generation": asdict(synthesis_settings.multi_hop_3),
            "batch_size": synthesis_settings.multi_hop_3_batch_size,
        }
    )


def find_multi_hop_3_candidates(
    relations: Sequence[EvidenceRelation],
    *,
    limit: int,
) -> tuple[tuple[str, str, str], ...]:
    """Select diverse connected three-fact paths from confirmed multi-hop relations."""
    if limit < 1:
        return ()
    neighbors: dict[str, set[str]] = defaultdict(set)
    for relation in relations:
        if relation.bundle_kind != "multi_hop":
            continue
        fact_id_a, fact_id_b = relation.fact_ids
        neighbors[fact_id_a].add(fact_id_b)
        neighbors[fact_id_b].add(fact_id_a)

    candidates_by_pivot = {
        pivot: tuple(
            sorted(
                {
                    tuple(sorted((neighbor_a, pivot, neighbor_b)))
                    for neighbor_a, neighbor_b in combinations(sorted(pivot_neighbors), 2)
                }
            )
        )
        for pivot, pivot_neighbors in neighbors.items()
        if len(pivot_neighbors) >= 2
    }
    selected: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    offset = 0
    pivots = sorted(candidates_by_pivot)
    while len(selected) < limit:
        added = False
        for pivot in pivots:
            pivot_candidates = candidates_by_pivot[pivot]
            if offset >= len(pivot_candidates):
                continue
            added = True
            candidate = pivot_candidates[offset]
            if candidate in seen:
                continue
            seen.add(candidate)
            selected.append(candidate)
            if len(selected) == limit:
                break
        if not added:
            break
        offset += 1
    return tuple(selected)


def candidate_connections(
    fact_ids: tuple[str, str, str],
    relations_by_pair: Mapping[tuple[str, str], EvidenceRelation],
) -> tuple[EvidenceRelation, ...]:
    """Return every confirmed multi-hop edge contained in a candidate triple."""
    return tuple(
        relations_by_pair[pair]
        for pair in combinations(fact_ids, 2)
        if pair in relations_by_pair and relations_by_pair[pair].bundle_kind == "multi_hop"
    )


class MultiHop3Client(Protocol):
    """Boundary implemented by a three-fact bundle confirmation client."""

    def confirm_batch(
        self,
        triples: Sequence[tuple[tuple[LegalFact, LegalFact, LegalFact], tuple[EvidenceRelation, ...]]],
    ) -> Sequence[ConfirmedMultiHop3Bundle | None]:
        """Return one confirmed bundle or None per candidate, in input order."""


class BamlMultiHop3Client:
    """Typed BAML client for confirming three-fact evidence bundles."""

    def __init__(self, settings: Settings, synthesis_settings: SynthesisSettings, *, run_id: str) -> None:
        """Create a runtime client from typed, secret-safe configuration."""
        set_log_level("OFF")
        self._registry = build_client_registry(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_output_tokens=synthesis_settings.multi_hop_3.max_output_tokens,
            temperature=synthesis_settings.multi_hop_3.temperature,
            timeout_seconds=settings.llm_timeout_seconds,
            reasoning_effort=synthesis_settings.multi_hop_3.reasoning_effort,
        )
        self._timeout_seconds = settings.llm_timeout_seconds
        self._max_response_bytes = synthesis_settings.max_response_bytes
        self._model = settings.llm_model
        self._run_id = run_id

    def confirm_batch(
        self,
        triples: Sequence[tuple[tuple[LegalFact, LegalFact, LegalFact], tuple[EvidenceRelation, ...]]],
    ) -> Sequence[ConfirmedMultiHop3Bundle | None]:
        """Confirm one bounded batch and translate typed results in input order."""
        items = [
            MultiHop3BundleBatchItem(
                candidate_id=multi_hop_3_candidate_id([fact.fact_id for fact in facts]),
                facts=[_summarize(fact) for fact in facts],
                confirmed_connections=[
                    EvidenceConnectionSummary(fact_ids=list(relation.fact_ids), bridge=relation.bridge)
                    for relation in connections
                ],
            )
            for facts, connections in triples
        ]
        request = b.request.ConfirmMultiHop3BundleBatch(items, baml_options={"client_registry": self._registry})
        raw_output = execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)
        results = b.parse.ConfirmMultiHop3BundleBatch(raw_output, baml_options={"client_registry": self._registry})
        requested_ids = [item.candidate_id for item in items]
        result_ids = [result.candidate_id for result in results]
        if len(result_ids) != len(set(result_ids)) or set(result_ids) != set(requested_ids):
            msg = "Multi-hop-3 batch response must contain exactly one result for every candidate"
            raise ValueError(msg)
        results_by_id = {result.candidate_id: result.bundle for result in results}
        return [
            _build_bundle(results_by_id[item.candidate_id], facts, self._model, self._run_id)
            for item, (facts, _connections) in zip(items, triples, strict=True)
        ]


def _summarize(fact: LegalFact) -> LegalFactSummary:
    """Return the BAML-facing legal-fact summary."""
    return LegalFactSummary(
        fact_id=fact.fact_id,
        statement=fact.statement,
        valid_from=fact.valid_from.isoformat() if fact.valid_from else None,
        valid_to=fact.valid_to.isoformat() if fact.valid_to else None,
        document_number=fact.document_number,
    )


def _build_bundle(
    result: Any,
    facts: tuple[LegalFact, LegalFact, LegalFact],
    model: str,
    run_id: str,
) -> ConfirmedMultiHop3Bundle | None:
    """Translate one BAML verdict into a persisted bundle, or None."""
    if not result.is_valid:
        return None
    if result.bridge is None:
        msg = "A valid multi-hop-3 bundle must declare its bridge"
        raise ValueError(msg)
    fact_ids = tuple(sorted(fact.fact_id for fact in facts))
    identity = {
        "bridge": result.bridge,
        "fact_ids": list(fact_ids),
        "kind": "multi_hop",
        "run_id": run_id,
    }
    return ConfirmedMultiHop3Bundle(
        bundle_id=sha256_json(identity),
        fact_ids=fact_ids,
        bridge=result.bridge,
        confirmed_by_model=model,
        run_id=run_id,
    )


def _batch_task_key(triples: Sequence[tuple[str, str, str]]) -> str:
    """Build a deterministic retry key for one candidate batch."""
    candidate_ids = "\0".join(multi_hop_3_candidate_id(triple) for triple in triples)
    return f"multi-hop-3-batch:{hashlib.sha256(candidate_ids.encode('utf-8')).hexdigest()}"


def _confirm_batch(
    client: MultiHop3Client,
    facts_by_id: Mapping[str, LegalFact],
    relations_by_pair: Mapping[tuple[str, str], EvidenceRelation],
    triples: Sequence[tuple[str, str, str]],
) -> tuple[tuple[tuple[str, str, str], ConfirmedMultiHop3Bundle | None], ...]:
    """Confirm one batch and pair each verdict with its input triple."""
    inputs = [
        (
            tuple(facts_by_id[fact_id] for fact_id in triple),
            candidate_connections(triple, relations_by_pair),
        )
        for triple in triples
    ]
    bundles = client.confirm_batch(inputs)
    if len(bundles) != len(triples):
        msg = f"Multi-hop-3 batch returned {len(bundles)} results for {len(triples)} candidates"
        raise ValueError(msg)
    return tuple(zip(triples, bundles, strict=True))


def confirm_multi_hop_3_bundles(
    facts: Sequence[LegalFact],
    relations: Sequence[EvidenceRelation],
    candidates: Sequence[tuple[str, str, str]],
    client: MultiHop3Client,
    *,
    batch_size: int,
    max_concurrency: int,
    max_retries: int,
    retry_delay_seconds: float,
    on_batch_confirmed: Callable[
        [Sequence[tuple[tuple[str, str, str], ConfirmedMultiHop3Bundle | None]]],
        None,
    ]
    | None = None,
) -> tuple[tuple[ConfirmedMultiHop3Bundle, ...], tuple[LlmTaskFailure, ...]]:
    """Confirm candidate triples concurrently and checkpoint completed batches."""
    facts_by_id = {fact.fact_id: fact for fact in facts}
    relations_by_pair = {relation.fact_ids: relation for relation in relations}
    batches = [candidates[start : start + batch_size] for start in range(0, len(candidates), batch_size)]
    bundles: list[ConfirmedMultiHop3Bundle] = []
    failures: list[LlmTaskFailure] = []
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {
            executor.submit(
                execute_llm_task,
                _batch_task_key(batch),
                partial(_confirm_batch, client, facts_by_id, relations_by_pair, batch),
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds,
            ): batch
            for batch in batches
        }
        for future in as_completed(futures):
            batch = futures[future]
            try:
                batch_results = future.result()
            except LlmTaskExhaustedError as error:
                failures.append(LlmTaskFailure(task_key=_batch_task_key(batch), error=str(error)))
                continue
            bundles.extend(bundle for _triple, bundle in batch_results if bundle is not None)
            if on_batch_confirmed is not None:
                on_batch_confirmed(batch_results)
    return tuple(bundles), tuple(failures)
