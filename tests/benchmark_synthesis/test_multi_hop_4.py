"""Tests for four-fact multi-hop bundle synthesis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.benchmark_synthesis.config import SynthesisSettings
from legal_rag.benchmark_synthesis.evidence_graph import EvidenceRelation
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.multi_hop_3 import ConfirmedMultiHop3Bundle
from legal_rag.benchmark_synthesis.multi_hop_4 import (
    BamlMultiHop4Client,
    ConfirmedMultiHop4Bundle,
    MultiHop4Candidate,
    MultiHop4CandidateSet,
    MultiHop4ProgressRecord,
    confirm_multi_hop_4_bundles,
    find_multi_hop_4_candidates,
    multi_hop_4_candidate_id,
)
from legal_rag.benchmark_synthesis.writer import (
    read_multi_hop_4_bundles,
    read_multi_hop_4_candidate_set,
    read_multi_hop_4_progress,
    write_multi_hop_4_bundles,
    write_multi_hop_4_candidate_set,
    write_multi_hop_4_progress,
)
from legal_rag.multi_hop_4_cli import _validate_resume_state
from legal_rag.settings import Settings

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "data" / "config" / "groundtruth" / "synthesis_v1.json"
_PROMPT_PATH = _REPO_ROOT / "baml_src" / "multi_hop_4.baml"


def _fact(fact_id: str) -> LegalFact:
    """Build one minimal legal fact."""
    return LegalFact(
        fact_id=fact_id,
        statement=f"Quy định {fact_id}",
        valid_from=None,
        valid_to=None,
        document_number="168/2024/NĐ-CP",
        evidence=(EvidenceReference(chunk_id=f"chunk-{fact_id}"),),
    )


def _relation(fact_id_a: str, fact_id_b: str, *, kind: str = "multi_hop") -> EvidenceRelation:
    """Build one confirmed pair relation."""
    fact_ids = tuple(sorted((fact_id_a, fact_id_b)))
    return EvidenceRelation(
        relation_id=f"relation-{fact_ids[0]}-{fact_ids[1]}",
        fact_ids=fact_ids,
        bundle_kind=kind,
        bridge=f"Nối {fact_ids[0]} với {fact_ids[1]}",
        confirmed_by_model="provider/model",
        run_id="relation-run",
    )


def _source_bundle(
    bundle_id: str = "source-1",
    fact_ids: tuple[str, str, str] = ("a", "b", "c"),
) -> ConfirmedMultiHop3Bundle:
    """Build one confirmed three-fact source bundle."""
    return ConfirmedMultiHop3Bundle(
        bundle_id=bundle_id,
        fact_ids=fact_ids,
        bridge="Cần kết hợp đủ ba quy định",
        confirmed_by_model="provider/model",
        run_id="source-run",
    )


def _confirmed_bundle(fact_ids: tuple[str, str, str, str] = ("a", "b", "c", "d")) -> ConfirmedMultiHop4Bundle:
    """Build one confirmed four-fact bundle."""
    return ConfirmedMultiHop4Bundle(
        bundle_id="bundle-4",
        fact_ids=fact_ids,
        bridge="Cần kết hợp đủ bốn quy định",
        confirmed_by_model="provider/model",
        run_id="bundle-run",
    )


def test_multi_hop_4_prompt_requires_four_fact_ablation_and_subject_identity() -> None:
    """The prompt demands a dependent fourth step and exact subject identity."""
    prompt = _PROMPT_PATH.read_text(encoding="utf-8")

    assert "đủ cả bốn facts" in prompt
    assert "Bỏ bất kỳ một fact nào" in prompt
    assert "ít nhất ba bước nối" in prompt
    assert "danh tính và vai trò pháp lý" in prompt
    assert "nhánh điều kiện loại trừ nhau" in prompt
    assert "bridge về bản chất không đổi" in prompt
    assert "bản giấy/điện tử" in prompt
    assert "loại phương tiện hoặc vai trò" in prompt
    assert "văn bản hay bối cảnh khác" in prompt
    assert "không phải tiền đề cần thiết cho bước kế tiếp" in prompt
    assert "chuỗi A -> B -> C -> D" in prompt
    assert "hậu quả pháp lý song song" in prompt
    assert "mức phạt cá nhân, trách nhiệm của tổ chức" in prompt
    assert "không biến các nhánh này thành multi-hop" in prompt
    assert "checklist nghĩa vụ, hồ sơ, thời hạn" in prompt
    assert "không được là câu hỏi" in prompt
    assert "không kết thúc bằng dấu hỏi" in prompt
    assert "chỉ là gợi ý để kiểm tra facts" in prompt
    assert "Chỉ dùng tiếng Việt" in prompt


def test_find_multi_hop_4_candidates_extends_confirmed_sources_round_robin() -> None:
    """Candidate limiting covers separate source bundles before repeating one seed."""
    sources = (_source_bundle(), _source_bundle("source-2", ("x", "y", "z")))
    relations = (
        _relation("a", "b"),
        _relation("b", "c"),
        _relation("c", "d"),
        _relation("a", "e"),
        _relation("x", "y"),
        _relation("y", "z"),
        _relation("z", "w"),
    )

    candidates = find_multi_hop_4_candidates(sources, relations, limit=2)

    assert {candidate.source_bundle_id for candidate in candidates} == {"source-1", "source-2"}
    assert {candidate.fact_ids for candidate in candidates} == {("a", "b", "c", "d"), ("w", "x", "y", "z")}


def test_find_multi_hop_4_candidates_rejects_stale_source_and_deduplicates_fact_sets() -> None:
    """Disconnected source state is ignored and equivalent extensions appear once."""
    sources = (
        _source_bundle("source-stale", ("p", "q", "r")),
        _source_bundle("source-valid", ("b", "c", "d")),
        _source_bundle("source-duplicate", ("a", "b", "c")),
    )
    relations = (_relation("b", "c"), _relation("c", "d"), _relation("a", "b"), _relation("a", "d"))

    candidates = find_multi_hop_4_candidates(sources, relations, limit=10)

    assert [candidate.fact_ids for candidate in candidates].count(("a", "b", "c", "d")) == 1
    assert all(candidate.source_bundle_id != "source-stale" for candidate in candidates)


def test_multi_hop_4_candidate_id_requires_four_distinct_facts() -> None:
    """Persisted candidate identity cannot hide malformed fact sets."""
    with pytest.raises(ValueError, match="exactly four distinct"):
        multi_hop_4_candidate_id(("a", "b", "c"))
    with pytest.raises(ValueError, match="exactly four distinct"):
        multi_hop_4_candidate_id(("a", "b", "c", "c"))


class _FakeClient:
    """Confirm every four-fact input."""

    def confirm_batch(self, candidates):
        """Return one valid bundle per input."""
        return [
            _confirmed_bundle(tuple(sorted(fact.fact_id for fact in facts)))
            for facts, _source_bundle_value, _connections in candidates
        ]


def test_confirm_multi_hop_4_bundles_checkpoints_completed_candidates() -> None:
    """Concurrent confirmation exposes each completed candidate to persistence."""
    facts = [_fact(fact_id) for fact_id in ("a", "b", "c", "d")]
    relations = [_relation("a", "b"), _relation("b", "c"), _relation("c", "d")]
    candidate = MultiHop4Candidate(("a", "b", "c", "d"), "source-1")
    persisted: list[MultiHop4Candidate] = []

    bundles, failures = confirm_multi_hop_4_bundles(
        facts,
        relations,
        [_source_bundle()],
        [candidate],
        _FakeClient(),
        batch_size=1,
        max_concurrency=1,
        max_retries=1,
        retry_delay_seconds=0,
        on_batch_confirmed=lambda results: persisted.extend(item for item, _bundle in results),
    )

    assert bundles == (_confirmed_bundle(),)
    assert failures == ()
    assert persisted == [candidate]


def test_baml_multi_hop_4_client_translates_valid_bundle(monkeypatch) -> None:
    """A valid typed verdict becomes a provenance-carrying four-fact bundle."""
    settings = Settings(
        llm_base_url="http://localhost:20128/v1",
        llm_api_key="provider-test-key",
        llm_model="provider/model",
        llm_timeout_seconds=30,
    )
    client = BamlMultiHop4Client(settings, SynthesisSettings.from_path(_CONFIG_PATH), run_id="bundle-run")
    candidate_id = multi_hop_4_candidate_id(("a", "b", "c", "d"))
    monkeypatch.setattr(
        "legal_rag.benchmark_synthesis.multi_hop_4.execute_baml_request",
        lambda *args, **kwargs: json.dumps(
            [
                {
                    "candidate_id": candidate_id,
                    "bundle": {"is_valid": True, "bridge": "Cần đủ bốn quy định"},
                }
            ],
            ensure_ascii=False,
        ),
    )

    result = client.confirm_batch(
        [
            (
                (_fact("a"), _fact("b"), _fact("c"), _fact("d")),
                _source_bundle(),
                (_relation("a", "b"), _relation("b", "c"), _relation("c", "d")),
            )
        ]
    )

    assert result[0] is not None
    assert result[0].fact_ids == ("a", "b", "c", "d")
    assert result[0].bridge == "Cần đủ bốn quy định"
    assert result[0].run_id == "bundle-run"


def test_multi_hop_4_artifacts_round_trip(tmp_path: Path) -> None:
    """Candidate, bundle, and progress artifacts preserve typed provenance."""
    candidate_path = tmp_path / "candidates.json"
    bundle_path = tmp_path / "bundles.jsonl"
    progress_path = tmp_path / "progress.jsonl"
    candidate = MultiHop4Candidate(("a", "b", "c", "d"), "source-1")
    candidate_set = MultiHop4CandidateSet("candidate-run", (candidate,))
    progress = MultiHop4ProgressRecord(
        multi_hop_4_candidate_id(candidate.fact_ids),
        "bundle-run",
        confirmed_by_model="provider/model",
    )

    write_multi_hop_4_candidate_set(candidate_path, candidate_set)
    write_multi_hop_4_bundles(bundle_path, [_confirmed_bundle()])
    write_multi_hop_4_progress(progress_path, [progress])

    assert read_multi_hop_4_candidate_set(candidate_path) == candidate_set
    assert read_multi_hop_4_bundles(bundle_path) == (_confirmed_bundle(),)
    assert read_multi_hop_4_progress(progress_path) == (progress,)


def test_selective_resume_invalidates_only_requested_candidate() -> None:
    """A targeted refresh retains every unselected bundle and progress record."""
    selected = _confirmed_bundle()
    retained = ConfirmedMultiHop4Bundle(
        bundle_id="bundle-retained",
        fact_ids=("e", "f", "g", "h"),
        bridge="Retained",
        confirmed_by_model="provider/model",
        run_id="bundle-run",
    )
    selected_id = multi_hop_4_candidate_id(selected.fact_ids)
    retained_id = multi_hop_4_candidate_id(retained.fact_ids)
    progress = tuple(
        MultiHop4ProgressRecord(candidate_id, "bundle-run", confirmed_by_model="provider/model")
        for candidate_id in (selected_id, retained_id)
    )

    bundles_by_id, progress_by_id = _validate_resume_state(
        (selected, retained),
        progress,
        frozenset({selected_id, retained_id}),
        "current-run",
        frozenset({selected_id}),
    )

    assert bundles_by_id == {retained.bundle_id: retained}
    assert set(progress_by_id) == {retained_id}


def test_resume_rejects_stale_progress() -> None:
    """A normal resume cannot skip a candidate completed under different inputs."""
    candidate_id = multi_hop_4_candidate_id(("a", "b", "c", "d"))

    with pytest.raises(ValueError, match="different synthesis run"):
        _validate_resume_state(
            (),
            (MultiHop4ProgressRecord(candidate_id, "stale-run"),),
            frozenset({candidate_id}),
            "current-run",
            frozenset(),
        )
