"""Tests for evidence-relation confirmation and bundle sampling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from unicodedata import normalize

import numpy as np
import pytest

from legal_rag.baml_client.types import ConfirmedRelation as BamlConfirmedRelation
from legal_rag.baml_client.types import RelationType as BamlRelationType
from legal_rag.benchmark_synthesis import evidence_graph
from legal_rag.benchmark_synthesis.config import EvidenceRetrievalSettings, LlmStageSettings, SynthesisSettings
from legal_rag.benchmark_synthesis.evidence_graph import (
    BamlEvidenceRelationClient,
    EvidenceRelation,
    classify_bundle_kind,
    confirm_evidence_relations,
    find_candidate_fact_pairs,
    sample_bundles,
)
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.embedding.profiles import load_configured_embedding_profile
from legal_rag.settings import Settings

_SYNTHESIS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "data" / "config" / "groundtruth" / "synthesis_v1.json"


def _fact(
    fact_id: str,
    *,
    document_number: str = "168/2024/NĐ-CP",
    conduct: str = "Vượt đèn đỏ",
    subject: str = "Người điều khiển ô tô",
    provision_path: str = "Điều 6",
    conditions: tuple[str, ...] = (),
) -> LegalFact:
    """Build one minimal legal fact for evidence-graph tests."""
    return LegalFact(
        fact_id=fact_id,
        fact_content_sha256=f"content-{fact_id}",
        conduct=conduct,
        subject=subject,
        conditions=conditions,
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


def _retrieval_settings(**changes: float | int) -> EvidenceRetrievalSettings:
    """Return checked-in evidence retrieval settings with explicit test overrides."""
    settings = SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH).evidence_retrieval
    return replace(settings, **changes)


class _FakeEncoder:
    """Return pre-programmed vectors keyed by embedding text."""

    def __init__(self, vectors_by_text: dict[str, np.ndarray]) -> None:
        self._vectors_by_text = vectors_by_text

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return the vector configured for each input text."""
        return np.array([self._vectors_by_text[text] for text in texts])


class _FakeWordSegmenter:
    """Return deterministic lexical units without starting VnCoreNLP."""

    def segment(self, texts: Sequence[str]) -> tuple[tuple[str, ...], ...]:
        """Split each test text into plain alphanumeric units."""
        return tuple(tuple(evidence_graph._lexical_units(text)) for text in texts)


def test_vietnamese_legal_tokens_preserve_syllables_and_add_segmented_terms() -> None:
    """BM25 keeps exact accented syllables while adding Vietnamese legal words and phrase bigrams."""
    tokens = evidence_graph._vietnamese_legal_tokens(
        "Không chấp hành quyết định xử phạt",
        ("Không", "chấp_hành", "quyết_định", "xử_phạt"),
    )

    assert "s:không" in tokens
    assert "s:chấp" in tokens
    assert "w:chấp_hành" in tokens
    assert "w:quyết_định" in tokens
    assert "w:xử_phạt" in tokens
    assert "b:chấp_hành|quyết_định" in tokens
    assert "s:khong" not in tokens


def test_vietnamese_legal_tokens_normalize_unicode_and_stop_bigrams_at_punctuation() -> None:
    """Equivalent Unicode forms match, and phrase features never cross sentence punctuation."""
    decomposed = normalize("NFD", "Không chấp hành. Ô tô")
    tokens = evidence_graph._vietnamese_legal_tokens(
        decomposed,
        ("Không", "chấp_hành", ".", "Ô_tô"),
    )

    assert "s:không" in tokens
    assert "w:chấp_hành" in tokens
    assert "w:ô_tô" in tokens
    assert "b:chấp_hành|ô_tô" not in tokens


def test_fact_lexical_corpora_keep_fact_fields_isolated() -> None:
    """Conduct, subject, and authority terms are scored by separate BM25 indexes."""
    fact = _fact("fact-a", conduct="Xử phạt", subject="Ô tô", provision_path="Điều 6")

    corpora = evidence_graph._lexical_corpora(
        [fact],
        [evidence_graph._fact_retrieval_texts(fact)],
        _FakeWordSegmenter(),
    )

    assert "s:xử" in corpora["core"][0]
    assert "s:điều" not in corpora["core"][0]
    assert "s:điều" in corpora["authority"][0]
    assert "s:xử" not in corpora["authority"][0]


def test_authority_tokens_canonicalize_document_and_citation_structure() -> None:
    """Equivalent legal authority fields expose exact and order-independent structural tokens."""
    fact = _fact(
        "fact-a",
        document_number="168/2024/NĐ-CP",
        provision_path="điểm c khoản 2 Điều 30",
    )

    tokens = evidence_graph._authority_atomic_tokens(fact)

    assert "a:document:168_2024_nđ_cp" in tokens
    assert "a:article:30" in tokens
    assert "a:clause:2" in tokens
    assert "a:point:c" in tokens
    assert "a:citation:article_30|clause_2|point_c" in tokens


def test_candidate_fingerprint_changes_with_embedding_provider_or_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate caches become stale when either dense or lexical retrieval identity changes."""
    facts = [_fact("fact-a"), _fact("fact-b")]
    settings = SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH)
    profile = load_configured_embedding_profile(settings.embedding_profile)
    original = evidence_graph.build_evidence_candidate_run_id(
        facts,
        profile,
        settings,
        embedding_base_url="https://embedding.test/v1",
        embedding_model="provider/bge-m3",
    )
    changed_model = evidence_graph.build_evidence_candidate_run_id(
        facts,
        profile,
        settings,
        embedding_base_url="https://embedding.test/v1",
        embedding_model="provider/other-model",
    )
    monkeypatch.setattr(evidence_graph, "VIETNAMESE_LEXICAL_TOKENIZER_VERSION", "vncorenlp_hybrid_v3")
    changed_tokenizer = evidence_graph.build_evidence_candidate_run_id(
        facts,
        profile,
        settings,
        embedding_base_url="https://embedding.test/v1",
        embedding_model="provider/bge-m3",
    )

    assert len({original, changed_model, changed_tokenizer}) == 3


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

    pairs = find_candidate_fact_pairs(
        [fact_a, fact_b],
        encoder,
        _FakeWordSegmenter(),
        _retrieval_settings(),
    )

    assert pairs == (("fact-a", "fact-b"),)


@pytest.mark.parametrize("facts", [[], [_fact("fact-a")]])
def test_find_candidate_fact_pairs_returns_empty_for_fewer_than_two_facts(facts: list[LegalFact]) -> None:
    """Candidate selection handles empty and one-fact corpora without constructing BM25."""
    assert (
        find_candidate_fact_pairs(
            facts,
            _FakeEncoder({}),
            _FakeWordSegmenter(),
            _retrieval_settings(),
        )
        == ()
    )


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

    pairs = find_candidate_fact_pairs(
        [fact_a, fact_b],
        encoder,
        _FakeWordSegmenter(),
        _retrieval_settings(
            lexical_conditions_weight=0.0,
            lexical_consequences_weight=0.0,
            lexical_authority_weight=0.0,
        ),
    )

    assert pairs == ()


def test_find_candidate_fact_pairs_recovers_a_lexical_match_the_dense_leg_rejects() -> None:
    """Near-verbatim restatement is proposed even when the dense leg scores it below min_similarity."""
    original = _fact(
        "fact-original",
        conduct="Không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
        provision_path="Điều 6",
    )
    amending = _fact(
        "fact-amending",
        document_number="336/2025/NĐ-CP",
        conduct="Không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
        provision_path="Điều 4",
    )
    encoder = _FakeEncoder(
        {
            evidence_graph._fact_embedding_text(original): np.array([1.0, 0.0]),
            evidence_graph._fact_embedding_text(amending): np.array([0.0, 1.0]),
        }
    )

    pairs = find_candidate_fact_pairs(
        [original, amending],
        encoder,
        _FakeWordSegmenter(),
        _retrieval_settings(
            dense_weight=0.0,
            lexical_conditions_weight=0.0,
            lexical_consequences_weight=0.0,
            lexical_authority_weight=0.0,
        ),
    )

    assert pairs == (("fact-amending", "fact-original"),)


def test_find_candidate_fact_pairs_recovers_shared_conditions_from_the_condition_leg() -> None:
    """Exception conditions can propose a pair even when core and dense retrieval reject it."""
    condition = ("Trong trường hợp phương tiện đang thực hiện nhiệm vụ khẩn cấp",)
    fact_a = _fact(
        "fact-a",
        conduct="Không chấp hành biển báo",
        subject="Người đi bộ",
        provision_path="Điều 1",
        conditions=condition,
    )
    fact_b = _fact(
        "fact-b",
        conduct="Đi vào đường cấm",
        subject="Người điều khiển xe chuyên dùng",
        provision_path="Điều 99",
        conditions=condition,
    )
    facts = [
        fact_a,
        fact_b,
        _fact("noise-1", conduct="Dừng xe sai vị trí", conditions=("Khi trời mưa lớn",)),
        _fact("noise-2", conduct="Chở quá số người", conditions=("Trên tuyến đường nội bộ",)),
        _fact("noise-3", conduct="Không có gương", conditions=("Trong khu vực bến cảng",)),
    ]
    encoder = _FakeEncoder(
        {
            evidence_graph._fact_embedding_text(fact): vector
            for fact, vector in zip(facts, np.eye(len(facts)), strict=True)
        }
    )

    pairs = find_candidate_fact_pairs(
        facts,
        encoder,
        _FakeWordSegmenter(),
        _retrieval_settings(
            dense_weight=0.0,
            lexical_core_weight=0.0,
            lexical_conditions_weight=1.0,
            lexical_consequences_weight=0.0,
            lexical_authority_weight=0.0,
        ),
    )

    assert ("fact-a", "fact-b") in pairs


def test_find_candidate_fact_pairs_recovers_shared_consequences_from_the_consequence_leg() -> None:
    """Shared penalties can propose a pair independently of conduct and authority wording."""
    fact_a = _fact(
        "fact-a",
        conduct="Không chấp hành biển báo",
        subject="Người đi bộ",
        provision_path="Điều 1",
    )
    fact_b = _fact(
        "fact-b",
        conduct="Đi vào đường cấm",
        subject="Người điều khiển xe chuyên dùng",
        provision_path="Điều 99",
    )
    facts = [
        fact_a,
        fact_b,
        replace(_fact("noise-1", conduct="Dừng xe sai vị trí"), monetary_penalty="200.000 đồng"),
        replace(_fact("noise-2", conduct="Chở quá số người"), monetary_penalty="600.000 đồng"),
        replace(_fact("noise-3", conduct="Không có gương"), monetary_penalty="1.500.000 đồng"),
    ]
    encoder = _FakeEncoder(
        {
            evidence_graph._fact_embedding_text(fact): vector
            for fact, vector in zip(facts, np.eye(len(facts)), strict=True)
        }
    )

    pairs = find_candidate_fact_pairs(
        facts,
        encoder,
        _FakeWordSegmenter(),
        _retrieval_settings(
            dense_weight=0.0,
            lexical_core_weight=0.0,
            lexical_conditions_weight=0.0,
            lexical_consequences_weight=1.0,
            lexical_authority_weight=0.0,
        ),
    )

    assert ("fact-a", "fact-b") in pairs


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

    pairs = find_candidate_fact_pairs(
        [anchor, close, farther],
        encoder,
        _FakeWordSegmenter(),
        _retrieval_settings(
            max_candidates_per_fact=1,
            min_similarity=0.0,
            lexical_core_weight=0.0,
            lexical_conditions_weight=0.0,
            lexical_consequences_weight=0.0,
            lexical_authority_weight=0.0,
        ),
    )

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
        run_id="run-1",
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
        [fact_a, fact_b, fact_c],
        [("fact-a", "fact-b"), ("fact-a", "fact-c")],
        client,
        max_retries=2,
        retry_delay_seconds=0,
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
        max_retries=2,
        retry_delay_seconds=0,
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

    relations = confirm_evidence_relations(
        [fact_a, fact_b],
        [("fact-a", "fact-b")],
        _FlakyClient(),
        max_retries=2,
        retry_delay_seconds=0,
    )

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

    persisted: list[tuple[str, str]] = []
    with pytest.raises(RuntimeError, match="evidence relation confirmation left 1 incomplete"):
        confirm_evidence_relations(
            [fact_a, fact_b, fact_c],
            [("fact-a", "fact-b"), ("fact-a", "fact-c")],
            _FailsForOnePairClient(),
            max_retries=2,
            retry_delay_seconds=0,
            on_pair_confirmed=lambda pair, _relation: persisted.append(pair),
        )

    assert persisted == [("fact-a", "fact-c")]


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


def _synthesis_settings() -> SynthesisSettings:
    """Build ground-truth synthesis settings for an offline BAML request."""
    return SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH)


def test_evidence_relation_client_uses_relation_stage_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Relation confirmation no longer inherits fact-extraction generation bounds."""
    configured_tokens: list[int] = []

    def _capture_registry(**kwargs: object) -> object:
        configured_tokens.append(int(kwargs["max_output_tokens"]))
        return object()

    settings = replace(
        _synthesis_settings(),
        fact_extraction=LlmStageSettings(101, 0.0),
        evidence_relation=LlmStageSettings(202, 0.0),
    )
    monkeypatch.setattr(evidence_graph, "build_client_registry", _capture_registry)

    BamlEvidenceRelationClient(_settings(), settings, run_id="run-1")

    assert configured_tokens == [202]


def test_baml_evidence_relation_client_translates_confirmed_relation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A confirmed relation from the generated client becomes a persisted EvidenceRelation."""
    client = BamlEvidenceRelationClient(_settings(), _synthesis_settings(), run_id="run-1")
    monkeypatch.setattr(
        evidence_graph,
        "execute_baml_request",
        lambda *args, **kwargs: (
            '{"is_related":true,"relation_type":"AmendedBy",'
            '"from_fact_id":"fact-b","to_fact_id":"fact-a",'
            '"bridge":["Điều 6 khoản 9 điểm b"]}'
        ),
    )

    relation = client.confirm(_fact("fact-a"), _fact("fact-b"))

    assert relation is not None
    assert relation.relation_type == "amended_by"
    assert relation.from_fact_id == "fact-b"
    assert relation.to_fact_id == "fact-a"
    assert relation.bridge == ("Điều 6 khoản 9 điểm b",)
    assert relation.confirmed_by_model == "provider/test-model"
    assert relation.run_id == "run-1"


def test_baml_evidence_relation_client_returns_none_when_unrelated(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unrelated verdict from the model returns None, not a fabricated relation."""
    client = BamlEvidenceRelationClient(_settings(), _synthesis_settings(), run_id="run-1")
    monkeypatch.setattr(
        evidence_graph,
        "execute_baml_request",
        lambda *args, **kwargs: (
            '{"is_related":false,"relation_type":null,"from_fact_id":null,"to_fact_id":null,"bridge":[]}'
        ),
    )

    relation = client.confirm(_fact("fact-a"), _fact("fact-b"))

    assert relation is None


def test_baml_evidence_relation_client_sends_conditions_and_exact_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Relation confirmation sends applicability conditions and exact quoted evidence to BAML."""
    captured_summaries: list[object] = []
    request = object()

    def _capture_request(fact_a: object, fact_b: object, **kwargs: object) -> object:
        captured_summaries.extend((fact_a, fact_b))
        return request

    fact_a = replace(
        _fact("fact-a"),
        conditions=("Khi chuyển hướng tại nơi đường giao nhau",),
        evidence=(EvidenceReference(chunk_id="chunk-a", quoted_span="Trích nguyên văn fact A"),),
    )
    fact_b = replace(
        _fact("fact-b"),
        conditions=("Trừ trường hợp xe ưu tiên",),
        evidence=(EvidenceReference(chunk_id="chunk-b", quoted_span="Trích nguyên văn fact B"),),
    )
    monkeypatch.setattr(evidence_graph.b.request, "ConfirmEvidenceRelation", _capture_request)
    monkeypatch.setattr(
        evidence_graph,
        "execute_baml_request",
        lambda received_request, *args, **kwargs: (
            '{"is_related":false,"relation_type":null,"from_fact_id":null,"to_fact_id":null,"bridge":[]}'
            if received_request is request
            else pytest.fail("BAML request object was not forwarded to the transport")
        ),
    )

    relation = BamlEvidenceRelationClient(_settings(), _synthesis_settings(), run_id="run-1").confirm(fact_a, fact_b)

    assert relation is None
    assert len(captured_summaries) == 2
    summary_a, summary_b = captured_summaries
    assert summary_a.conditions == ["Khi chuyển hướng tại nơi đường giao nhau"]
    assert summary_a.evidence[0].chunk_id == "chunk-a"
    assert summary_a.evidence[0].quoted_span == "Trích nguyên văn fact A"
    assert summary_b.conditions == ["Trừ trường hợp xe ưu tiên"]
    assert summary_b.evidence[0].chunk_id == "chunk-b"
    assert summary_b.evidence[0].quoted_span == "Trích nguyên văn fact B"


def test_build_relation_is_deterministic_for_identical_inputs() -> None:
    """Re-confirming the same pair and type produces the same relation_id."""
    fact_a, fact_b = _fact("fact-a"), _fact("fact-b")
    result = BamlConfirmedRelation(
        is_related=True,
        relation_type=BamlRelationType.AmendedBy,
        from_fact_id="fact-a",
        to_fact_id="fact-b",
        bridge=["Điều 6"],
    )

    first = evidence_graph._build_relation(result, fact_a, fact_b, "provider/test-model", "run-1")
    second = evidence_graph._build_relation(result, fact_a, fact_b, "provider/test-model", "run-1")

    assert first is not None
    assert second is not None
    assert first.relation_id == second.relation_id


@pytest.mark.parametrize(
    ("from_fact_id", "to_fact_id"),
    [("fact-a", "fact-a"), ("fact-a", "fact-c")],
)
def test_build_relation_rejects_invalid_direction(from_fact_id: str, to_fact_id: str) -> None:
    """A confirmed direction must contain each input fact exactly once."""
    result = BamlConfirmedRelation(
        is_related=True,
        relation_type=BamlRelationType.AmendedBy,
        from_fact_id=from_fact_id,
        to_fact_id=to_fact_id,
        bridge=["Điều 6"],
    )

    with pytest.raises(ValueError, match="direction must contain each input fact_id exactly once"):
        evidence_graph._build_relation(
            result,
            _fact("fact-a"),
            _fact("fact-b"),
            "provider/test-model",
            "run-1",
        )
