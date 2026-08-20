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
    conduct: str | None = "Vượt đèn đỏ",
    subject: str | None = "Người điều khiển ô tô",
    statement: str | None = None,
) -> LegalFact:
    """Build one minimal legal fact for evidence-graph tests."""
    return LegalFact(
        fact_id=fact_id,
        statement=statement or ". ".join(part for part in (conduct, subject) if part),
        valid_from=None,
        valid_to=None,
        document_number=document_number,
        evidence=(EvidenceReference(chunk_id="chunk-1"),),
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


def test_vietnamese_legal_tokens_use_segmented_words_and_bigrams() -> None:
    """BM25 indexes normalized Vietnamese words and adjacent phrase bigrams without syllable duplication."""
    tokens = evidence_graph._vietnamese_legal_tokens(
        ("Không", "chấp_hành", "quyết_định", "xử_phạt"),
    )

    assert "w:không" in tokens
    assert "w:chấp_hành" in tokens
    assert "w:quyết_định" in tokens
    assert "w:xử_phạt" in tokens
    assert "b:chấp_hành|quyết_định" in tokens
    assert not any(token.startswith("s:") for token in tokens)


def test_vietnamese_legal_tokens_normalize_unicode_and_stop_bigrams_at_punctuation() -> None:
    """Equivalent Unicode forms match, and phrase features never cross sentence punctuation."""
    decomposed = normalize("NFD", "Không chấp hành. Ô tô")
    tokens = evidence_graph._vietnamese_legal_tokens(
        (decomposed.split()[0], "chấp_hành", ".", "Ô_tô"),
    )

    assert "w:không" in tokens
    assert "w:chấp_hành" in tokens
    assert "w:ô_tô" in tokens
    assert "b:chấp_hành|ô_tô" not in tokens


def test_fact_retrieval_uses_only_statement() -> None:
    """Dense and lexical retrieval ignore structured fields outside the complete statement."""
    fact = _fact(
        "fact-a",
        statement="Quyết định xử phạt phải được gửi cho người vi phạm.",
    )

    text = evidence_graph._fact_embedding_text(fact)
    corpus = evidence_graph._lexical_corpus([text], _FakeWordSegmenter())

    assert text == fact.statement
    assert "w:quyết" in corpus[0]
    assert "w:vượt" not in corpus[0]
    assert "w:điều" not in corpus[0]


def test_reciprocal_rank_fusion_gives_both_rankings_equal_weight() -> None:
    """Standard RRF sums reciprocal ranks without retrieval-leg weights."""
    fused = evidence_graph._reciprocal_rank_fusion(((1, 2), (2, 3)), 60)

    assert fused[1] == pytest.approx(1 / 61)
    assert fused[2] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[3] == pytest.approx(1 / 62)


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


def test_find_candidate_fact_pairs_respects_max_candidates_per_fact() -> None:
    """Only the top-K most similar partners are kept per fact."""
    anchor = _fact("anchor", statement="alpha")
    close = _fact("close", statement="beta")
    farther = _fact("farther", statement="gamma")
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
        _retrieval_settings(max_candidates_per_fact=1),
    )

    assert ("anchor", "close") in pairs
    assert ("anchor", "farther") not in pairs


class _FakeRelationClient:
    """Return pre-programmed relations keyed by sorted fact_id pair."""

    def __init__(self, relations_by_pair: dict[tuple[str, str], EvidenceRelation | None]) -> None:
        self._relations_by_pair = relations_by_pair

    def confirm_batch(self, pairs: Sequence[tuple[LegalFact, LegalFact]]) -> list[EvidenceRelation | None]:
        """Return the relation configured for each fact pair, in order."""
        return [self._relations_by_pair[tuple(sorted((fact_a.fact_id, fact_b.fact_id)))] for fact_a, fact_b in pairs]


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

    relations, failures = confirm_evidence_relations(
        [fact_a, fact_b, fact_c],
        [("fact-a", "fact-b"), ("fact-a", "fact-c")],
        client,
        batch_size=2,
        max_concurrency=1,
        max_retries=2,
        retry_delay_seconds=0,
    )

    assert relations == (_relation("fact-a", "fact-b"),)
    assert failures == ()


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
        batch_size=1,
        max_concurrency=1,
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
        def confirm_batch(self, pairs: Sequence[tuple[LegalFact, LegalFact]]) -> list[EvidenceRelation | None]:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                msg = "provider timed out"
                raise TimeoutError(msg)
            return [_relation("fact-a", "fact-b")]

    relations, failures = confirm_evidence_relations(
        [fact_a, fact_b],
        [("fact-a", "fact-b")],
        _FlakyClient(),
        batch_size=1,
        max_concurrency=1,
        max_retries=2,
        retry_delay_seconds=0,
    )

    assert attempts == 2
    assert relations == (_relation("fact-a", "fact-b"),)
    assert failures == ()


def test_confirm_evidence_relations_reports_a_pair_that_fails_every_attempt_as_a_failure() -> None:
    """A pair that keeps raising is reported as a failure, but other pairs are still confirmed and persisted."""
    fact_a, fact_b, fact_c = _fact("fact-a"), _fact("fact-b"), _fact("fact-c")

    class _FailsForOnePairClient:
        def confirm_batch(self, pairs: Sequence[tuple[LegalFact, LegalFact]]) -> list[EvidenceRelation | None]:
            fact_x, fact_y = pairs[0]
            if {fact_x.fact_id, fact_y.fact_id} == {"fact-a", "fact-b"}:
                msg = "provider timed out"
                raise TimeoutError(msg)
            return [_relation(fact_x.fact_id, fact_y.fact_id)]

    persisted: list[tuple[str, str]] = []
    relations, failures = confirm_evidence_relations(
        [fact_a, fact_b, fact_c],
        [("fact-a", "fact-b"), ("fact-a", "fact-c")],
        _FailsForOnePairClient(),
        batch_size=1,
        max_concurrency=1,
        max_retries=2,
        retry_delay_seconds=0,
        on_pair_confirmed=lambda pair, _relation: persisted.append(pair),
    )

    assert len(failures) == 1
    assert relations == (_relation("fact-a", "fact-c"),)
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
            '[{"pair_id":"fact-a::fact-b","relation":{"is_related":true,"relation_type":"AmendedBy",'
            '"from_fact_id":"fact-b","to_fact_id":"fact-a",'
            '"bridge":["Điều 6 khoản 9 điểm b"]}}]'
        ),
    )

    relations = client.confirm_batch([(_fact("fact-a"), _fact("fact-b"))])

    assert len(relations) == 1
    relation = relations[0]
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
            '[{"pair_id":"fact-a::fact-b","relation":'
            '{"is_related":false,"relation_type":null,"from_fact_id":null,"to_fact_id":null,"bridge":[]}}]'
        ),
    )

    relations = client.confirm_batch([(_fact("fact-a"), _fact("fact-b"))])

    assert relations == [None]


def test_baml_evidence_relation_client_sends_statements_and_evidence_chunk_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Relation confirmation sends complete statements and source chunk ids to BAML."""
    captured_items: list[object] = []
    request = object()

    def _capture_request(items: object, **kwargs: object) -> object:
        captured_items.extend(items)
        return request

    fact_a = replace(
        _fact("fact-a"),
        statement="Khi chuyển hướng tại nơi đường giao nhau, người điều khiển phải tuân thủ quy định.",
        evidence=(EvidenceReference(chunk_id="chunk-a"),),
    )
    fact_b = replace(
        _fact("fact-b"),
        statement="Quy định không áp dụng cho xe ưu tiên đang làm nhiệm vụ.",
        evidence=(EvidenceReference(chunk_id="chunk-b"),),
    )
    monkeypatch.setattr(evidence_graph.b.request, "ConfirmEvidenceRelationBatch", _capture_request)
    monkeypatch.setattr(
        evidence_graph,
        "execute_baml_request",
        lambda received_request, *args, **kwargs: (
            '[{"pair_id":"fact-a::fact-b","relation":'
            '{"is_related":false,"relation_type":null,"from_fact_id":null,"to_fact_id":null,"bridge":[]}}]'
            if received_request is request
            else pytest.fail("BAML request object was not forwarded to the transport")
        ),
    )

    relations = BamlEvidenceRelationClient(_settings(), _synthesis_settings(), run_id="run-1").confirm_batch(
        [(fact_a, fact_b)]
    )

    assert relations == [None]
    assert len(captured_items) == 1
    item = captured_items[0]
    assert item.fact_a.statement == fact_a.statement
    assert item.fact_a.evidence_chunk_ids == ["chunk-a"]
    assert item.fact_b.statement == fact_b.statement
    assert item.fact_b.evidence_chunk_ids == ["chunk-b"]


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
