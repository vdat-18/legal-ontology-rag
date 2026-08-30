"""Build and confirm five-fact multi-hop evidence bundles."""

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
from legal_rag.baml_client.types import EvidenceConnectionSummary, LegalFactSummary, MultiHop5BundleBatchItem
from legal_rag.baml_transport import build_client_registry, execute_baml_request
from legal_rag.benchmark_synthesis.config import SynthesisSettings
from legal_rag.benchmark_synthesis.evidence_graph import EvidenceBundle, EvidenceRelation
from legal_rag.benchmark_synthesis.llm_execution import (
    LlmTaskExhaustedError,
    LlmTaskFailure,
    execute_llm_task,
)
from legal_rag.benchmark_synthesis.models import LegalFact
from legal_rag.benchmark_synthesis.multi_hop_4 import ConfirmedMultiHop4Bundle
from legal_rag.parser.hashing import sha256_json
from legal_rag.settings import Settings

MULTI_HOP_5_CANDIDATE_SELECTION_VERSION = "1.0.0"
MULTI_HOP_5_CANDIDATE_SET_SCHEMA_VERSION = "1.0.0"
MULTI_HOP_5_BUNDLE_SCHEMA_VERSION = "1.0.0"
MULTI_HOP_5_PROGRESS_SCHEMA_VERSION = "1.0.0"
MULTI_HOP_5_PROMPT_VERSION = "1.3.0"

FactId5 = tuple[str, str, str, str, str]
FactTuple5 = tuple[LegalFact, LegalFact, LegalFact, LegalFact, LegalFact]


@dataclass(frozen=True, slots=True)
class MultiHop5Candidate:
    """One five-fact extension of an already confirmed four-fact bundle."""

    fact_ids: FactId5
    source_bundle_id: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable candidate."""
        return {"fact_ids": list(self.fact_ids), "source_bundle_id": self.source_bundle_id}


@dataclass(frozen=True, slots=True)
class MultiHop5CandidateSet:
    """Fingerprint-validated five-fact candidates selected from four-fact seeds."""

    run_id: str
    candidates: tuple[MultiHop5Candidate, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable candidate set."""
        return {
            "schema_version": MULTI_HOP_5_CANDIDATE_SET_SCHEMA_VERSION,
            "run_id": self.run_id,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class ConfirmedMultiHop5Bundle:
    """One LLM-confirmed bundle whose answer requires all five facts."""

    bundle_id: str
    fact_ids: FactId5
    bridge: str
    confirmed_by_model: str
    run_id: str
    prompt_version: str = MULTI_HOP_5_PROMPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable confirmed bundle."""
        return {
            "schema_version": MULTI_HOP_5_BUNDLE_SCHEMA_VERSION,
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
class MultiHop5ProgressRecord:
    """One five-fact candidate whose confirmation has completed."""

    candidate_id: str
    run_id: str
    prompt_version: str = MULTI_HOP_5_PROMPT_VERSION
    confirmed_by_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable progress record."""
        return {
            "schema_version": MULTI_HOP_5_PROGRESS_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "run_id": self.run_id,
            "prompt_version": self.prompt_version,
            "confirmed_by_model": self.confirmed_by_model,
        }


def multi_hop_5_candidate_id(fact_ids: Sequence[str]) -> str:
    """Return the content identity of one canonical five-fact candidate."""
    canonical = tuple(sorted(fact_ids))
    if len(canonical) != 5 or len(set(canonical)) != 5:
        msg = "A multi-hop-5 candidate must contain exactly five distinct fact IDs"
        raise ValueError(msg)
    return sha256_json({"fact_ids": list(canonical)})


def build_multi_hop_5_candidate_run_id(
    source_bundles: Sequence[ConfirmedMultiHop4Bundle],
    relations: Sequence[EvidenceRelation],
    candidate_limit: int,
) -> str:
    """Fingerprint four-fact seeds, pair-graph semantics, and selection settings."""
    return sha256_json(
        {
            "selection_version": MULTI_HOP_5_CANDIDATE_SELECTION_VERSION,
            "candidate_limit": candidate_limit,
            "source_bundles": [
                {"bridge": bundle.bridge, "bundle_id": bundle.bundle_id, "fact_ids": list(bundle.fact_ids)}
                for bundle in sorted(source_bundles, key=lambda item: item.bundle_id)
            ],
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


def build_multi_hop_5_run_id(
    candidate_run_id: str,
    synthesis_settings: SynthesisSettings,
    *,
    llm_base_url: str,
    llm_model: str,
) -> str:
    """Fingerprint every input that changes five-fact confirmation."""
    return sha256_json(
        {
            "candidate_run_id": candidate_run_id,
            "llm_base_url": llm_base_url,
            "llm_model": llm_model,
            "prompt_version": MULTI_HOP_5_PROMPT_VERSION,
            "generation": asdict(synthesis_settings.multi_hop_5),
            "batch_size": synthesis_settings.multi_hop_5_batch_size,
        }
    )


def find_multi_hop_5_candidates(
    source_bundles: Sequence[ConfirmedMultiHop4Bundle],
    relations: Sequence[EvidenceRelation],
    *,
    limit: int,
) -> tuple[MultiHop5Candidate, ...]:
    """Extend confirmed four-fact bundles through one current multi-hop edge."""
    if limit < 1:
        return ()
    neighbors, relation_pairs = _multi_hop_graph(relations)
    candidates_by_seed = _candidate_extensions_by_seed(source_bundles, neighbors, relation_pairs)
    return _round_robin_candidates(candidates_by_seed, limit)


def _multi_hop_graph(
    relations: Sequence[EvidenceRelation],
) -> tuple[dict[str, set[str]], set[tuple[str, str]]]:
    """Build the undirected graph formed by confirmed multi-hop pair relations."""
    neighbors: dict[str, set[str]] = defaultdict(set)
    relation_pairs: set[tuple[str, str]] = set()
    for relation in relations:
        if relation.bundle_kind != "multi_hop":
            continue
        fact_id_a, fact_id_b = relation.fact_ids
        relation_pairs.add(relation.fact_ids)
        neighbors[fact_id_a].add(fact_id_b)
        neighbors[fact_id_b].add(fact_id_a)
    return neighbors, relation_pairs


def _candidate_extensions_by_seed(
    source_bundles: Sequence[ConfirmedMultiHop4Bundle],
    neighbors: Mapping[str, set[str]],
    relation_pairs: set[tuple[str, str]],
) -> dict[str, tuple[MultiHop5Candidate, ...]]:
    """Build canonical one-fact extensions for every connected source bundle."""
    candidates_by_seed: dict[str, tuple[MultiHop5Candidate, ...]] = {}
    for bundle in sorted(source_bundles, key=lambda item: item.bundle_id):
        seed_ids = frozenset(bundle.fact_ids)
        seed_connection_count = sum(tuple(sorted(pair)) in relation_pairs for pair in combinations(seed_ids, 2))
        if seed_connection_count < 3:
            continue
        extension_ids = sorted(set().union(*(neighbors[fact_id] for fact_id in seed_ids)) - seed_ids)
        candidates_by_seed[bundle.bundle_id] = tuple(
            MultiHop5Candidate(
                fact_ids=tuple(sorted((*bundle.fact_ids, extension_id))),
                source_bundle_id=bundle.bundle_id,
            )
            for extension_id in extension_ids
        )
    return candidates_by_seed


def _round_robin_candidates(
    candidates_by_seed: Mapping[str, tuple[MultiHop5Candidate, ...]],
    limit: int,
) -> tuple[MultiHop5Candidate, ...]:
    """Select unique fact sets without allowing one seed to dominate."""
    selected: list[MultiHop5Candidate] = []
    seen_fact_ids: set[FactId5] = set()
    offset = 0
    seed_ids = sorted(candidates_by_seed)
    while len(selected) < limit:
        added = False
        for seed_id in seed_ids:
            seed_candidates = candidates_by_seed[seed_id]
            if offset >= len(seed_candidates):
                continue
            added = True
            candidate = seed_candidates[offset]
            if candidate.fact_ids in seen_fact_ids:
                continue
            seen_fact_ids.add(candidate.fact_ids)
            selected.append(candidate)
            if len(selected) == limit:
                break
        if not added:
            break
        offset += 1
    return tuple(selected)


def candidate_connections(
    fact_ids: FactId5,
    relations_by_pair: Mapping[tuple[str, str], EvidenceRelation],
) -> tuple[EvidenceRelation, ...]:
    """Return every confirmed multi-hop edge contained in a candidate."""
    return tuple(
        relations_by_pair[pair]
        for pair in combinations(fact_ids, 2)
        if pair in relations_by_pair and relations_by_pair[pair].bundle_kind == "multi_hop"
    )


MultiHop5Input = tuple[FactTuple5, ConfirmedMultiHop4Bundle, tuple[EvidenceRelation, ...]]


class MultiHop5Client(Protocol):
    """Boundary implemented by a five-fact bundle confirmation client."""

    def confirm_batch(self, candidates: Sequence[MultiHop5Input]) -> Sequence[ConfirmedMultiHop5Bundle | None]:
        """Return one confirmed bundle or None per candidate, in input order."""


class BamlMultiHop5Client:
    """Typed BAML client for confirming five-fact evidence bundles."""

    def __init__(self, settings: Settings, synthesis_settings: SynthesisSettings, *, run_id: str) -> None:
        """Create a runtime client from typed, secret-safe configuration."""
        set_log_level("OFF")
        self._registry = build_client_registry(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_output_tokens=synthesis_settings.multi_hop_5.max_output_tokens,
            temperature=synthesis_settings.multi_hop_5.temperature,
            timeout_seconds=settings.llm_timeout_seconds,
            reasoning_effort=synthesis_settings.multi_hop_5.reasoning_effort,
        )
        self._timeout_seconds = settings.llm_timeout_seconds
        self._max_response_bytes = synthesis_settings.max_response_bytes
        self._model = settings.llm_model
        self._run_id = run_id

    def confirm_batch(self, candidates: Sequence[MultiHop5Input]) -> Sequence[ConfirmedMultiHop5Bundle | None]:
        """Confirm one bounded batch and translate typed results in input order."""
        items = [
            MultiHop5BundleBatchItem(
                candidate_id=multi_hop_5_candidate_id([fact.fact_id for fact in facts]),
                source_fact_ids=list(source_bundle.fact_ids),
                source_bridge=source_bundle.bridge,
                facts=[_summarize(fact) for fact in facts],
                confirmed_connections=[
                    EvidenceConnectionSummary(fact_ids=list(relation.fact_ids), bridge=relation.bridge)
                    for relation in connections
                ],
            )
            for facts, source_bundle, connections in candidates
        ]
        request = b.request.ConfirmMultiHop5BundleBatch(items, baml_options={"client_registry": self._registry})
        raw_output = execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)
        results = b.parse.ConfirmMultiHop5BundleBatch(raw_output, baml_options={"client_registry": self._registry})
        requested_ids = [item.candidate_id for item in items]
        result_ids = [result.candidate_id for result in results]
        if len(result_ids) != len(set(result_ids)) or set(result_ids) != set(requested_ids):
            msg = "Multi-hop-5 batch response must contain exactly one result for every candidate"
            raise ValueError(msg)
        results_by_id = {result.candidate_id: result.bundle for result in results}
        return [
            _build_bundle(results_by_id[item.candidate_id], facts, self._model, self._run_id)
            for item, (facts, _source_bundle, _connections) in zip(items, candidates, strict=True)
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


def _build_bundle(result: Any, facts: FactTuple5, model: str, run_id: str) -> ConfirmedMultiHop5Bundle | None:
    """Translate one BAML verdict into a persisted bundle, or None."""
    if not result.is_valid:
        return None
    if result.bridge is None:
        msg = "A valid multi-hop-5 bundle must declare its bridge"
        raise ValueError(msg)
    fact_ids = tuple(sorted(fact.fact_id for fact in facts))
    identity = {"bridge": result.bridge, "fact_ids": list(fact_ids), "kind": "multi_hop", "run_id": run_id}
    return ConfirmedMultiHop5Bundle(
        bundle_id=sha256_json(identity),
        fact_ids=fact_ids,
        bridge=result.bridge,
        confirmed_by_model=model,
        run_id=run_id,
    )


def _batch_task_key(candidates: Sequence[MultiHop5Candidate]) -> str:
    """Build a deterministic retry key for one candidate batch."""
    candidate_ids = "\0".join(multi_hop_5_candidate_id(candidate.fact_ids) for candidate in candidates)
    return f"multi-hop-5-batch:{hashlib.sha256(candidate_ids.encode('utf-8')).hexdigest()}"


def _confirm_batch(
    client: MultiHop5Client,
    facts_by_id: Mapping[str, LegalFact],
    source_bundles_by_id: Mapping[str, ConfirmedMultiHop4Bundle],
    relations_by_pair: Mapping[tuple[str, str], EvidenceRelation],
    candidates: Sequence[MultiHop5Candidate],
) -> tuple[tuple[MultiHop5Candidate, ConfirmedMultiHop5Bundle | None], ...]:
    """Confirm one batch and pair each verdict with its candidate."""
    inputs: list[MultiHop5Input] = [
        (
            tuple(facts_by_id[fact_id] for fact_id in candidate.fact_ids),
            source_bundles_by_id[candidate.source_bundle_id],
            candidate_connections(candidate.fact_ids, relations_by_pair),
        )
        for candidate in candidates
    ]
    bundles = client.confirm_batch(inputs)
    if len(bundles) != len(candidates):
        msg = f"Multi-hop-5 batch returned {len(bundles)} results for {len(candidates)} candidates"
        raise ValueError(msg)
    return tuple(zip(candidates, bundles, strict=True))


def confirm_multi_hop_5_bundles(
    facts: Sequence[LegalFact],
    relations: Sequence[EvidenceRelation],
    source_bundles: Sequence[ConfirmedMultiHop4Bundle],
    candidates: Sequence[MultiHop5Candidate],
    client: MultiHop5Client,
    *,
    batch_size: int,
    max_concurrency: int,
    max_retries: int,
    retry_delay_seconds: float,
    on_batch_confirmed: Callable[
        [Sequence[tuple[MultiHop5Candidate, ConfirmedMultiHop5Bundle | None]]],
        None,
    ]
    | None = None,
) -> tuple[tuple[ConfirmedMultiHop5Bundle, ...], tuple[LlmTaskFailure, ...]]:
    """Confirm five-fact candidates concurrently and checkpoint completed batches."""
    facts_by_id = {fact.fact_id: fact for fact in facts}
    source_bundles_by_id = {bundle.bundle_id: bundle for bundle in source_bundles}
    relations_by_pair = {relation.fact_ids: relation for relation in relations}
    batches = [candidates[start : start + batch_size] for start in range(0, len(candidates), batch_size)]
    bundles: list[ConfirmedMultiHop5Bundle] = []
    failures: list[LlmTaskFailure] = []
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {
            executor.submit(
                execute_llm_task,
                _batch_task_key(batch),
                partial(_confirm_batch, client, facts_by_id, source_bundles_by_id, relations_by_pair, batch),
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
            bundles.extend(bundle for _candidate, bundle in batch_results if bundle is not None)
            if on_batch_confirmed is not None:
                on_batch_confirmed(batch_results)
    return tuple(bundles), tuple(failures)
