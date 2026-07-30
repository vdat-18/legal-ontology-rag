"""Tests for evidence-relation confirmation and bundle sampling."""

from __future__ import annotations

import numpy as np
import pytest

from legal_rag.baml_client.types import ConfirmedRelation as BamlConfirmedRelation
from legal_rag.baml_client.types import RelationType as BamlRelationType
from legal_rag.benchmark_synthesis import evidence_graph
from legal_rag.benchmark_synthesis.evidence_graph import (
    BamlEvidenceRelationClient,
    EvidenceRelation,
    classify_bundle_kind,
    confirm_evidence_relations,
    find_candidate_fact_pairs,
    sample_bundles,
)
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.settings import EvaluationSettings, Settings


def _fact(
    fact_id: str,
    *,
    document_number: str = "168/2024/NĐ-CP",
    conduct: str = "Vượt đèn đỏ",
    subject: str = "Người điều khiển ô tô",
    provision_path: str = "Điều 6",
) -> LegalFact:
    """Build one minimal legal fact for evidence-graph tests."""
    return LegalFact(
        fact_id=fact_id,
        fact_content_sha256=f"content-{fact_id}",
        conduct=conduct,
        subject=subject,
        conditions=(),
        monetary_penalty="4.000.000 đồng",
        license_points_deducted=None,
        additional_penalties=(),
        remedial_measures=(),
        valid_from=None,
        valid_to=None,
        document_number=document_number,
        provision_path=provision_path,
        evidence=(EvidenceReference(chunk_id="chunk-1", quoted_span=""),),
    )


class _FakeEncoder:
    """Return pre-programmed vectors keyed by embedding text."""

    def __init__(self, vectors_by_text: dict[str, np.ndarray]) -> None:
        self._vectors_by_text = vectors_by_text

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return the vector configured for each input text."""
        return np.array([self._vectors_by_text[text] for text in texts])


def test_find_candidate_fact_pairs_orders_and_deduplicates_symmetric_pairs() -> None:
    """A pair is returned once, sorted, regardless of which fact was the query anchor."""
    fact_a = _fact("fact-a", conduct="Vượt đèn đỏ")
    fact_b = _fact("fact-b", conduct="Không đội mũ bảo hiểm")
    encoder = _FakeEncoder(
        {
            evidence_graph._fact_embedding_text(fact_a): np.array([1.0, 0.0]),
            evidence_graph._fact_embedding_text(fact_b): np.array([1.0, 0.0]),
        }
    )

    pairs = find_candidate_fact_pairs([fact_a, fact_b], encoder, min_similarity=0.5)

    assert pairs == (("fact-a", "fact-b"),)


def test_find_candidate_fact_pairs_excludes_pairs_missed_by_both_legs() -> None:
    """A pair that is neither dense-similar nor lexically overlapping never becomes a candidate."""
    fact_a = _fact("fact-a", conduct="Vượt đèn đỏ", subject="Ô tô", provision_path="Điều 6")
    fact_b = _fact("fact-b", conduct="Chở hàng cồng kềnh", subject="Xe máy", provision_path="Khoản 2")
    encoder = _FakeEncoder(
        {
            evidence_graph._fact_embedding_text(fact_a): np.array([1.0, 0.0]),
            evidence_graph._fact_embedding_text(fact_b): np.array([0.0, 1.0]),
        }
    )

    pairs = find_candidate_fact_pairs([fact_a, fact_b], encoder, min_similarity=0.5)

    assert pairs == ()


def test_find_candidate_fact_pairs_recovers_a_lexical_match_the_dense_leg_rejects() -> None:
    """Near-verbatim restatement is proposed even when the dense leg scores it below min_similarity."""
    original = _fact("fact-original", conduct="Không chấp hành hiệu lệnh của đèn tín hiệu giao thông")
    amending = _fact("fact-amending", conduct="Không chấp hành hiệu lệnh của đèn tín hiệu giao thông")
    encoder = _FakeEncoder(
        {
            evidence_graph._fact_embedding_text(original): np.array([1.0, 0.0]),
            evidence_graph._fact_embedding_text(amending): np.array([0.0, 1.0]),
        }
    )

    pairs = find_candidate_fact_pairs([original, amending], encoder, min_similarity=0.9)

    assert pairs == (("fact-amending", "fact-original"),)


def test_find_candidate_fact_pairs_respects_max_candidates_per_fact() -> None:
    """Only the top-K most similar partners are kept per fact."""
    anchor = _fact("anchor", conduct="Vượt đèn đỏ")
    close = _fact("close", conduct="Không chấp hành hiệu lệnh đèn tín hiệu")
    farther = _fact("farther", conduct="Không đội mũ bảo hiểm")
    vectors = {
        evidence_graph._fact_embedding_text(anchor): np.array([1.0, 0.0]),
        evidence_graph._fact_embedding_text(close): np.array([0.99, 0.14]),
        evidence_graph._fact_embedding_text(farther): np.array([0.9, 0.44]),
    }
    encoder = _FakeEncoder(vectors)

    pairs = find_candidate_fact_pairs([anchor, close, farther], encoder, max_candidates_per_fact=1, min_similarity=0.0)

    assert ("anchor", "close") in pairs
    assert ("anchor", "farther") not in pairs


class _FakeRelationClient:
    """Return pre-programmed relations keyed by sorted fact_id pair."""

    def __init__(self, relations_by_pair: dict[tuple[str, str], EvidenceRelation | None]) -> None:
        self._relations_by_pair = relations_by_pair

    def confirm(self, fact_a: LegalFact, fact_b: LegalFact) -> EvidenceRelation | None:
        """Return the relation configured for this fact pair."""
        return self._relations_by_pair[tuple(sorted((fact_a.fact_id, fact_b.fact_id)))]


def _relation(
    from_fact_id: str = "fact-a",
    to_fact_id: str = "fact-b",
    relation_type: str = "conduct_to_penalty",
) -> EvidenceRelation:
    """Build one confirmed evidence relation for tests."""
    return EvidenceRelation(
        relation_id="relation-1",
        relation_type=relation_type,
        from_fact_id=from_fact_id,
        to_fact_id=to_fact_id,
        bridge=("Điều 6",),
        confirmed_by_model="provider/test-model",
    )


def test_confirm_evidence_relations_drops_unconfirmed_pairs() -> None:
    """Only pairs the client confirms as related survive."""
    fact_a, fact_b, fact_c = _fact("fact-a"), _fact("fact-b"), _fact("fact-c")
    client = _FakeRelationClient(
        {
            ("fact-a", "fact-b"): _relation("fact-a", "fact-b"),
            ("fact-a", "fact-c"): None,
        }
    )

    relations = confirm_evidence_relations(
        [fact_a, fact_b, fact_c], [("fact-a", "fact-b"), ("fact-a", "fact-c")], client
    )

    assert relations == (_relation("fact-a", "fact-b"),)


def test_confirm_evidence_relations_calls_on_pair_confirmed_after_each_pair() -> None:
    """on_pair_confirmed receives the pair and its relation (or None), once per candidate pair."""
    fact_a, fact_b, fact_c = _fact("fact-a"), _fact("fact-b"), _fact("fact-c")
    client = _FakeRelationClient(
        {
            ("fact-a", "fact-b"): _relation("fact-a", "fact-b"),
            ("fact-a", "fact-c"): None,
        }
    )
    calls: list[tuple[tuple[str, str], EvidenceRelation | None]] = []

    confirm_evidence_relations(
        [fact_a, fact_b, fact_c],
        [("fact-a", "fact-b"), ("fact-a", "fact-c")],
        client,
        on_pair_confirmed=lambda pair, relation: calls.append((pair, relation)),
    )

    assert calls == [
        (("fact-a", "fact-b"), _relation("fact-a", "fact-b")),
        (("fact-a", "fact-c"), None),
    ]


def test_confirm_evidence_relations_retries_a_failing_pair_then_succeeds() -> None:
    """A pair whose confirmation call raises is retried, and a later success still yields its relation."""
    fact_a, fact_b = _fact("fact-a"), _fact("fact-b")
    attempts = 0

    class _FlakyClient:
        def confirm(self, fact_x: LegalFact, fact_y: LegalFact) -> EvidenceRelation | None:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                msg = "provider timed out"
                raise TimeoutError(msg)
            return _relation("fact-a", "fact-b")

    relations = confirm_evidence_relations([fact_a, fact_b], [("fact-a", "fact-b")], _FlakyClient())

    assert attempts == 2
    assert relations == (_relation("fact-a", "fact-b"),)


def test_confirm_evidence_relations_skips_a_pair_that_fails_every_attempt() -> None:
    """A pair that keeps raising is skipped, but other pairs are still confirmed."""
    fact_a, fact_b, fact_c = _fact("fact-a"), _fact("fact-b"), _fact("fact-c")

    class _FailsForOnePairClient:
        def confirm(self, fact_x: LegalFact, fact_y: LegalFact) -> EvidenceRelation | None:
            if {fact_x.fact_id, fact_y.fact_id} == {"fact-a", "fact-b"}:
                msg = "provider timed out"
                raise TimeoutError(msg)
            return _relation(fact_x.fact_id, fact_y.fact_id)

    relations = confirm_evidence_relations(
        [fact_a, fact_b, fact_c],
        [("fact-a", "fact-b"), ("fact-a", "fact-c")],
        _FailsForOnePairClient(),
    )

    assert [(r.from_fact_id, r.to_fact_id) for r in relations] == [("fact-a", "fact-c")]


def test_classify_bundle_kind_amended_by_is_always_temporal() -> None:
    """An amendment relation is temporal even within the same document."""
    facts_by_id = {"a": _fact("a", document_number="168/2024/NĐ-CP"), "b": _fact("b", document_number="168/2024/NĐ-CP")}
    relation = _relation("a", "b", relation_type="amended_by")

    assert classify_bundle_kind(relation, facts_by_id) == "temporal"


def test_classify_bundle_kind_cross_document_is_multi_hop() -> None:
    """A non-amendment relation across two documents is multi-hop."""
    facts_by_id = {"a": _fact("a", document_number="168/2024/NĐ-CP"), "b": _fact("b", document_number="238/2026/NĐ-CP")}
    relation = _relation("a", "b", relation_type="conduct_to_penalty")

    assert classify_bundle_kind(relation, facts_by_id) == "multi_hop"


def test_classify_bundle_kind_same_document_is_synthesis() -> None:
    """A non-amendment relation within one document is synthesis, not multi-hop."""
    facts_by_id = {"a": _fact("a", document_number="168/2024/NĐ-CP"), "b": _fact("b", document_number="168/2024/NĐ-CP")}
    relation = _relation("a", "b", relation_type="conduct_to_penalty")

    assert classify_bundle_kind(relation, facts_by_id) == "synthesis"


def test_sample_bundles_includes_one_single_hop_bundle_per_fact() -> None:
    """Every fact contributes exactly one single-hop bundle regardless of relations."""
    facts = [_fact("a"), _fact("b")]

    bundles = sample_bundles(facts, [], overlap_threshold=0.5)

    single_hop = [bundle for bundle in bundles if bundle.kind == "single_hop"]
    assert {bundle.fact_ids for bundle in single_hop} == {("a",), ("b",)}


def test_sample_bundles_excludes_relation_bundle_exceeding_overlap_threshold() -> None:
    """A relation bundle that duplicates an already-selected bundle is dropped."""
    facts = [_fact("a"), _fact("b")]
    duplicate_relations = [_relation("a", "b"), _relation("a", "b", relation_type="conduct_to_subject")]

    bundles = sample_bundles(facts, duplicate_relations, overlap_threshold=0.5)

    pair_bundles = [bundle for bundle in bundles if bundle.kind != "single_hop"]
    assert len(pair_bundles) == 1


def _settings() -> Settings:
    """Build runtime settings for an offline BAML request."""
    return Settings(
        llm_base_url="http://localhost:20128/v1",
        llm_api_key="provider-test-key",
        llm_model="provider/test-model",
        llm_timeout_seconds=30,
    )


def _evaluation_settings() -> EvaluationSettings:
    """Build ground-truth synthesis settings for an offline BAML request."""
    return EvaluationSettings(
        embedding_profile="qwen3_0_6b",
        span_match_threshold=0.85,
        bundle_overlap_threshold=0.5,
        max_quality_retries=2,
    )


def test_baml_evidence_relation_client_translates_confirmed_relation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A confirmed relation from the generated client becomes a persisted EvidenceRelation."""
    client = BamlEvidenceRelationClient(_settings(), _evaluation_settings())
    monkeypatch.setattr(
        evidence_graph,
        "execute_baml_request",
        lambda *args, **kwargs: '{"is_related":true,"relation_type":"AmendedBy","bridge":["Điều 6 khoản 9 điểm b"]}',
    )

    relation = client.confirm(_fact("fact-a"), _fact("fact-b"))

    assert relation is not None
    assert relation.relation_type == "amended_by"
    assert relation.from_fact_id == "fact-a"
    assert relation.to_fact_id == "fact-b"
    assert relation.bridge == ("Điều 6 khoản 9 điểm b",)
    assert relation.confirmed_by_model == "provider/test-model"


def test_baml_evidence_relation_client_returns_none_when_unrelated(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unrelated verdict from the model returns None, not a fabricated relation."""
    client = BamlEvidenceRelationClient(_settings(), _evaluation_settings())
    monkeypatch.setattr(
        evidence_graph,
        "execute_baml_request",
        lambda *args, **kwargs: '{"is_related":false,"relation_type":null,"bridge":[]}',
    )

    relation = client.confirm(_fact("fact-a"), _fact("fact-b"))

    assert relation is None


def test_build_relation_is_deterministic_for_identical_inputs() -> None:
    """Re-confirming the same pair and type produces the same relation_id."""
    result = BamlConfirmedRelation(is_related=True, relation_type=BamlRelationType.AmendedBy, bridge=["Điều 6"])

    first = evidence_graph._build_relation(result, "fact-a", "fact-b", "provider/test-model")
    second = evidence_graph._build_relation(result, "fact-a", "fact-b", "provider/test-model")

    assert first.relation_id == second.relation_id
