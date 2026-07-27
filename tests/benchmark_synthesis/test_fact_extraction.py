"""Tests for chunk windowing and legal-fact extraction validation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from legal_rag.baml_client.types import ChunkWindowRole
from legal_rag.baml_client.types import EvidenceEntry as BamlEvidenceEntry
from legal_rag.baml_client.types import LegalFact as BamlLegalFact
from legal_rag.benchmark_synthesis import fact_extraction
from legal_rag.benchmark_synthesis.fact_extraction import (
    BamlFactExtractionClient,
    build_legal_fact,
    build_provision_index,
    build_reading_window,
    cited_articles,
    extract_facts_from_document,
    find_embedding_candidates,
)
from legal_rag.chunker.models import Chunk
from legal_rag.settings import EvaluationSettings, Settings


def _chunk(
    chunk_id: str,
    text: str,
    *,
    previous_chunk_id: str | None = None,
    next_chunk_id: str | None = None,
) -> Chunk:
    """Build one minimal chunk for fact-extraction tests."""
    return Chunk(
        chunk_id=chunk_id,
        content_sha256="content-hash",
        chunker_config_sha256="config-hash",
        scope_id="scope-id",
        chunk_index=0,
        kind="paragraph_group",
        source_name="168_2024_ND-CP.docx",
        source_sha256="source-hash",
        record_scope="root",
        parent_attachment_id=None,
        attachment_member_path=None,
        attachment_member_sha256=None,
        unit_ids=(f"unit-{chunk_id}",),
        source_refs=(f"word/document.xml#/body/{chunk_id}",),
        sequence_start=0,
        sequence_end=0,
        text=text,
        context_text="",
        embedding_text=text,
        text_token_count=5,
        embedding_token_count=5,
        previous_chunk_id=previous_chunk_id,
        next_chunk_id=next_chunk_id,
    )


def test_build_provision_index_only_indexes_chunks_opening_an_article() -> None:
    """Only chunks whose text starts with an article heading are indexed."""
    heading = _chunk("c1", "Điều 6. Xử phạt người điều khiển xe ô tô.")
    continuation = _chunk("c2", "9. Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng.")
    unrelated = _chunk("c3", "Chương II. Quy định chung.")

    index = build_provision_index([heading, continuation, unrelated])

    assert index.chunk_ids_by_article == {"6": ("c1",)}


def test_cited_articles_excludes_own_heading() -> None:
    """A chunk citing its own article number is not a cross-reference."""
    chunk = _chunk("c1", "Điều 6. Áp dụng theo Điều 6 và tham chiếu Điều 5.")

    assert cited_articles(chunk) == frozenset({"5"})


def test_cited_articles_returns_empty_when_no_citation_present() -> None:
    """A chunk without any article mention has no cited articles."""
    chunk = _chunk("c1", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng.")

    assert cited_articles(chunk) == frozenset()


def test_build_reading_window_includes_target_preceding_and_cited_chunks() -> None:
    """The window combines target, up to two preceding chunks, and citation-resolved chunks by role."""
    heading = _chunk("art5", "Điều 5. Thẩm quyền xử phạt.")
    preceding_two = _chunk("prev2", "Phần trước xa hơn.")
    preceding_one = _chunk("prev1", "Phần ngay trước.", previous_chunk_id="prev2")
    target = _chunk("target", "Áp dụng theo Điều 5.", previous_chunk_id="prev1", next_chunk_id="next")
    following = _chunk("next", "Phần sau.")
    chunks_by_id = {chunk.chunk_id: chunk for chunk in (heading, preceding_two, preceding_one, target, following)}
    provision_index = build_provision_index([heading, preceding_two, preceding_one, target, following])

    window = build_reading_window(target, chunks_by_id, provision_index)

    roles_by_id = {entry.chunk_id: entry.role for entry in window}
    assert roles_by_id == {
        "target": ChunkWindowRole.Target,
        "prev1": ChunkWindowRole.Adjacent,
        "prev2": ChunkWindowRole.Adjacent,
        "art5": ChunkWindowRole.CitedProvision,
    }
    assert "next" not in roles_by_id
    art5_entry = next(entry for entry in window if entry.chunk_id == "art5")
    assert art5_entry.provision_path == "Điều 5"


def test_build_reading_window_limits_preceding_chunk_depth() -> None:
    """Walking back through previous_chunk_id stops after max_preceding_chunks."""
    far = _chunk("far", "Xa nhất.")
    middle = _chunk("middle", "Giữa.", previous_chunk_id="far")
    near = _chunk("near", "Gần nhất.", previous_chunk_id="middle")
    target = _chunk("target", "Điều 5.", previous_chunk_id="near")
    chunks_by_id = {chunk.chunk_id: chunk for chunk in (far, middle, near, target)}
    provision_index = build_provision_index([far, middle, near, target])

    window = build_reading_window(target, chunks_by_id, provision_index, max_preceding_chunks=2)

    ids = {entry.chunk_id for entry in window}
    assert ids == {"target", "near", "middle"}


def test_build_reading_window_deduplicates_chunk_appearing_in_multiple_roles() -> None:
    """A chunk that is both adjacent and cited only appears once, with its first role."""
    heading = _chunk("prev", "Điều 5. Thẩm quyền xử phạt.")
    target = _chunk("target", "Áp dụng theo Điều 5.", previous_chunk_id="prev")
    chunks_by_id = {chunk.chunk_id: chunk for chunk in (heading, target)}
    provision_index = build_provision_index([heading, target])

    window = build_reading_window(target, chunks_by_id, provision_index)

    assert len(window) == 2
    prev_entry = next(entry for entry in window if entry.chunk_id == "prev")
    assert prev_entry.role == ChunkWindowRole.Adjacent


def test_build_reading_window_includes_every_embedding_candidate() -> None:
    """The window is not capped in total size — every embedding candidate given is included."""
    target = _chunk("target", "Điều 5.")
    candidates = tuple(_chunk(f"cand{i}", f"Nội dung {i}") for i in range(5))
    chunks_by_id = {chunk.chunk_id: chunk for chunk in (target, *candidates)}
    provision_index = build_provision_index([target, *candidates])

    window = build_reading_window(target, chunks_by_id, provision_index, candidates)

    assert len(window) == 1 + len(candidates)
    assert window[0].chunk_id == "target"


def test_find_embedding_candidates_returns_top_k_by_similarity() -> None:
    """Candidates are ranked by cosine similarity and truncated to max_candidates."""
    target = _chunk("target", "target text")
    close = _chunk("close", "close text")
    far = _chunk("far", "far text")
    vectors = {
        "target": np.array([1.0, 0.0]),
        "close": np.array([0.9, 0.1]),
        "far": np.array([0.0, 1.0]),
    }

    result = find_embedding_candidates(target, [target, close, far], vectors, max_candidates=1)

    assert result == (close,)


def _baml_fact(**overrides: object) -> BamlLegalFact:
    """Build one generated BAML legal fact for validation tests."""
    defaults: dict[str, object] = {
        "conduct": "Không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
        "subject": "Người điều khiển ô tô",
        "conditions": [],
        "monetary_penalty": "4.000.000 đồng đến 6.000.000 đồng",
        "license_points_deducted": "4 điểm",
        "additional_penalties": [],
        "remedial_measures": [],
        "valid_from": "2025-01-01",
        "valid_to": None,
        "document_number": "168/2024/NĐ-CP",
        "provision_path": "Điều 6 khoản 9 điểm b",
        "evidence": [BamlEvidenceEntry(chunk_id="chunk-1", quoted_span="phạt tiền từ 4.000.000 đồng")],
    }
    defaults.update(overrides)
    return BamlLegalFact(**defaults)


def test_build_legal_fact_cross_references_real_chunk_provenance() -> None:
    """unit_ids and source_refs come from the real chunk, not from the LLM."""
    chunk = _chunk("chunk-1", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng.")

    fact = build_legal_fact(_baml_fact(), {"chunk-1": chunk})

    assert fact.evidence[0].unit_ids == chunk.unit_ids
    assert fact.evidence[0].source_refs == chunk.source_refs
    assert fact.valid_from is not None
    assert fact.valid_from.isoformat() == "2025-01-01"
    assert fact.valid_to is None


def test_build_legal_fact_identity_is_stable_for_identical_identity_fields() -> None:
    """Two facts sharing conduct, subject, document_number, and provision_path share a fact_id."""
    chunk = _chunk("chunk-1", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng.")

    first = build_legal_fact(_baml_fact(), {"chunk-1": chunk})
    second = build_legal_fact(_baml_fact(monetary_penalty="5.000.000 đồng"), {"chunk-1": chunk})

    assert first.fact_id == second.fact_id
    assert first.fact_content_sha256 != second.fact_content_sha256


def test_build_legal_fact_trusts_llm_evidence_without_validation() -> None:
    """No defensive check runs; an empty evidence list simply produces empty evidence."""
    fact = build_legal_fact(_baml_fact(evidence=[]), {})

    assert fact.evidence == ()


def test_build_legal_fact_raises_naturally_when_evidence_chunk_is_missing() -> None:
    """A chunk_id absent from chunks_by_id raises a natural KeyError, with no defensive handling."""
    with pytest.raises(KeyError):
        build_legal_fact(_baml_fact(), {"other": _chunk("other", "text")})


class _FakeExtractionClient:
    """Return pre-programmed facts keyed by the window's target chunk_id."""

    def __init__(self, facts_by_target: dict[str, tuple[BamlLegalFact, ...]]) -> None:
        self._facts_by_target = facts_by_target

    def extract(self, window: Sequence[object]) -> tuple[BamlLegalFact, ...]:
        """Return the facts configured for this window's target chunk."""
        target_chunk_id = next(entry.chunk_id for entry in window if entry.role == ChunkWindowRole.Target)
        return self._facts_by_target.get(target_chunk_id, ())


class _FakeEncoderBackend:
    """Return one deterministic vector per input text, ignoring content."""

    def encode(self, sentences: list[str], *, convert_to_numpy: bool, normalize_embeddings: bool) -> np.ndarray:
        """Return a fixed unit vector for every input sentence."""
        return np.array([[1.0, 0.0] for _ in sentences])


def test_extract_facts_from_document_collects_every_fact_per_target() -> None:
    """Every fact the client returns for a target chunk is collected."""
    from legal_rag.embedding.encoder import EmbeddingEncoder
    from legal_rag.embedding.profiles import EmbeddingProfile

    target = _chunk("target", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng.")
    encoder = EmbeddingEncoder(
        EmbeddingProfile("p", "m", "r", "m", "r", 2, 512, None),
        _FakeEncoderBackend(),
    )
    fact_one = _baml_fact(evidence=[BamlEvidenceEntry(chunk_id="target", quoted_span="phạt tiền")])
    fact_two = _baml_fact(
        conduct="Hành vi khác", evidence=[BamlEvidenceEntry(chunk_id="target", quoted_span="phạt tiền")]
    )
    client = _FakeExtractionClient({"target": (fact_one, fact_two)})

    facts = extract_facts_from_document([target], client, encoder)

    assert len(facts) == 2
    assert {fact.conduct for fact in facts} == {fact_one.conduct, fact_two.conduct}


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
        dedup_cosine_threshold=0.9,
        max_quality_retries=2,
        max_simulator_turns=8,
    )


def test_baml_fact_extraction_client_parses_typed_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """The generated non-stream client returns typed candidate facts."""
    client = BamlFactExtractionClient(_settings(), _evaluation_settings())
    monkeypatch.setattr(
        fact_extraction,
        "execute_baml_request",
        lambda *args, **kwargs: (
            '[{"conduct":"Vượt đèn đỏ","subject":"Người điều khiển ô tô","conditions":[],'
            '"monetary_penalty":"4.000.000 đồng","license_points_deducted":null,'
            '"additional_penalties":[],"remedial_measures":[],"valid_from":null,"valid_to":null,'
            '"document_number":"168/2024/NĐ-CP","provision_path":"Điều 6",'
            '"evidence":[{"chunk_id":"c1","quoted_span":"4.000.000 đồng"}]}]'
        ),
    )
    target = _chunk("c1", "Vượt đèn đỏ phạt 4.000.000 đồng.")
    provision_index = build_provision_index([target])
    window = build_reading_window(target, {"c1": target}, provision_index)

    facts = client.extract(window)

    assert len(facts) == 1
    assert facts[0].conduct == "Vượt đèn đỏ"
    assert facts[0].evidence[0].chunk_id == "c1"
