"""Tests for three-fact multi-hop bundle synthesis."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from legal_rag.benchmark_synthesis.config import SynthesisSettings
from legal_rag.benchmark_synthesis.evidence_graph import EvidenceRelation
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.multi_hop_3 import (
    BamlMultiHop3Client,
    ConfirmedMultiHop3Bundle,
    MultiHop3CandidateSet,
    MultiHop3ProgressRecord,
    confirm_multi_hop_3_bundles,
    find_multi_hop_3_candidates,
    multi_hop_3_candidate_id,
)
from legal_rag.benchmark_synthesis.writer import (
    read_multi_hop_3_bundles,
    read_multi_hop_3_candidate_set,
    read_multi_hop_3_progress,
    write_multi_hop_3_bundles,
    write_multi_hop_3_candidate_set,
    write_multi_hop_3_progress,
)
from legal_rag.multi_hop_3_cli import _load_or_create_candidates, _validate_resume_state
from legal_rag.settings import Settings

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "baml_src" / "multi_hop_3.baml"


def test_multi_hop_3_prompt_requires_one_concrete_compositional_case() -> None:
    """The prompt rejects broad aggregation, exclusive branches, and mixed-language bridges."""
    prompt = _PROMPT_PATH.read_text(encoding="utf-8")

    assert "MỘT tình huống cụ thể" in prompt
    assert "Ít nhất hai bước nối phải có phụ thuộc nội dung" in prompt
    assert "những nhánh điều kiện loại trừ nhau" in prompt
    assert "chỉ xác nhận trạng thái nền hoặc tính hợp pháp chung" in prompt
    assert "Chỉ dùng tiếng Việt" in prompt


_CONFIG_PATH = Path(__file__).resolve().parents[2] / "data" / "config" / "groundtruth" / "synthesis_v1.json"


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


def _bundle(fact_ids: tuple[str, str, str] = ("a", "b", "c")) -> ConfirmedMultiHop3Bundle:
    """Build one confirmed three-fact bundle."""
    return ConfirmedMultiHop3Bundle(
        bundle_id="bundle-1",
        fact_ids=fact_ids,
        bridge="Cần kết hợp đủ ba quy định",
        confirmed_by_model="provider/model",
        run_id="bundle-run",
    )


def test_find_multi_hop_3_candidates_uses_connected_paths_and_round_robin_pivots() -> None:
    """Candidate limiting covers separate graph regions instead of exhausting one high-degree pivot."""
    relations = (
        _relation("a", "b"),
        _relation("b", "c"),
        _relation("b", "d"),
        _relation("x", "y"),
        _relation("y", "z"),
        _relation("m", "n", kind="temporal"),
        _relation("n", "o", kind="temporal"),
    )

    candidates = find_multi_hop_3_candidates(relations, limit=2)

    assert len(candidates) == 2
    assert ("a", "b", "c") in candidates
    assert ("x", "y", "z") in candidates
    assert all("n" not in candidate for candidate in candidates)


class _FakeClient:
    """Confirm only a configured candidate triple."""

    def confirm_batch(
        self,
        triples: Sequence[tuple[tuple[LegalFact, LegalFact, LegalFact], tuple[EvidenceRelation, ...]]],
    ) -> list[ConfirmedMultiHop3Bundle | None]:
        """Return one valid bundle per input for checkpoint testing."""
        return [_bundle(tuple(sorted(fact.fact_id for fact in facts))) for facts, _connections in triples]


def test_confirm_multi_hop_3_bundles_checkpoints_every_candidate() -> None:
    """Concurrent confirmation reports each completed candidate outcome to persistence."""
    facts = [_fact("a"), _fact("b"), _fact("c")]
    relations = [_relation("a", "b"), _relation("b", "c")]
    persisted: list[tuple[str, str, str]] = []

    bundles, failures = confirm_multi_hop_3_bundles(
        facts,
        relations,
        [("a", "b", "c")],
        _FakeClient(),
        batch_size=1,
        max_concurrency=1,
        max_retries=1,
        retry_delay_seconds=0,
        on_batch_confirmed=lambda results: persisted.extend(triple for triple, _bundle_result in results),
    )

    assert bundles == (_bundle(),)
    assert failures == ()
    assert persisted == [("a", "b", "c")]


def test_baml_multi_hop_3_client_translates_valid_bundle(monkeypatch) -> None:
    """A valid typed BAML verdict becomes a provenance-carrying three-fact bundle."""
    settings = Settings(
        llm_base_url="http://localhost:20128/v1",
        llm_api_key="provider-test-key",
        llm_model="provider/model",
        llm_timeout_seconds=30,
    )
    client = BamlMultiHop3Client(settings, SynthesisSettings.from_path(_CONFIG_PATH), run_id="bundle-run")
    monkeypatch.setattr(
        "legal_rag.benchmark_synthesis.multi_hop_3.execute_baml_request",
        lambda *args, **kwargs: (
            '[{"candidate_id":"'
            + multi_hop_3_candidate_id(("a", "b", "c"))
            + '","bundle":{"is_valid":true,"bridge":"Cần kết hợp đủ ba quy định"}}]'
        ),
    )

    result = client.confirm_batch([((_fact("a"), _fact("b"), _fact("c")), (_relation("a", "b"), _relation("b", "c")))])

    assert len(result) == 1
    bundle = result[0]
    assert bundle is not None
    assert bundle.fact_ids == ("a", "b", "c")
    assert bundle.bridge == "Cần kết hợp đủ ba quy định"
    assert bundle.confirmed_by_model == "provider/model"
    assert bundle.run_id == "bundle-run"


def test_multi_hop_3_artifacts_round_trip(tmp_path: Path) -> None:
    """Candidates, bundles, and progress preserve their typed provenance."""
    candidate_path = tmp_path / "candidates.json"
    bundles_path = tmp_path / "bundles.jsonl"
    progress_path = tmp_path / "progress.jsonl"
    candidate_set = MultiHop3CandidateSet(run_id="candidate-run", triples=(("a", "b", "c"),))
    progress = MultiHop3ProgressRecord(
        candidate_id=multi_hop_3_candidate_id(("a", "b", "c")),
        run_id="bundle-run",
        confirmed_by_model="provider/model",
    )

    write_multi_hop_3_candidate_set(candidate_path, candidate_set)
    write_multi_hop_3_bundles(bundles_path, [_bundle()])
    write_multi_hop_3_progress(progress_path, [progress])

    assert read_multi_hop_3_candidate_set(candidate_path) == candidate_set
    assert read_multi_hop_3_bundles(bundles_path) == (_bundle(),)
    assert read_multi_hop_3_progress(progress_path) == (progress,)


def test_selective_resume_invalidates_only_requested_candidate() -> None:
    """A targeted refresh retains every unselected bundle and progress record."""
    selected_bundle = _bundle(("a", "b", "c"))
    retained_bundle = ConfirmedMultiHop3Bundle(
        bundle_id="bundle-2",
        fact_ids=("d", "e", "f"),
        bridge="Retained",
        confirmed_by_model="provider/model",
        run_id="bundle-run",
    )
    selected_id = multi_hop_3_candidate_id(selected_bundle.fact_ids)
    retained_id = multi_hop_3_candidate_id(retained_bundle.fact_ids)
    progress = tuple(
        MultiHop3ProgressRecord(candidate_id=candidate_id, run_id="bundle-run", confirmed_by_model="provider/model")
        for candidate_id in (selected_id, retained_id)
    )

    bundles_by_id, progress_by_id = _validate_resume_state(
        (selected_bundle, retained_bundle),
        progress,
        frozenset({selected_id, retained_id}),
        "current-run",
        frozenset({selected_id}),
    )

    assert set(bundles_by_id) == {"bundle-2"}
    assert set(progress_by_id) == {retained_id}


def test_resume_rejects_progress_from_a_stale_run() -> None:
    """A normal resume cannot skip candidates completed under different inputs."""
    candidate_id = multi_hop_3_candidate_id(("a", "b", "c"))
    progress = MultiHop3ProgressRecord(candidate_id=candidate_id, run_id="stale-run")

    with pytest.raises(ValueError, match="different synthesis run"):
        _validate_resume_state(
            (),
            (progress,),
            frozenset({candidate_id}),
            "current-run",
            frozenset(),
        )


def test_resume_replays_bundle_written_before_progress() -> None:
    """A current-run bundle left by a crash is replaced when its candidate is replayed."""
    bundle = _bundle()
    candidate_id = multi_hop_3_candidate_id(bundle.fact_ids)

    bundles_by_id, progress_by_id = _validate_resume_state(
        (bundle,),
        (),
        frozenset({candidate_id}),
        bundle.run_id,
        frozenset(),
    )

    assert bundles_by_id == {bundle.bundle_id: bundle}
    assert progress_by_id == {}


def test_resume_accepts_provenance_rich_records_retained_by_selective_migration() -> None:
    """A later normal resume preserves unselected records from an older prompt run."""
    bundle = _bundle()
    candidate_id = multi_hop_3_candidate_id(bundle.fact_ids)
    progress = MultiHop3ProgressRecord(
        candidate_id=candidate_id,
        run_id=bundle.run_id,
        prompt_version=bundle.prompt_version,
        confirmed_by_model=bundle.confirmed_by_model,
    )

    bundles_by_id, progress_by_id = _validate_resume_state(
        (bundle,),
        (progress,),
        frozenset({candidate_id}),
        "new-run",
        frozenset(),
    )

    assert bundles_by_id == {bundle.bundle_id: bundle}
    assert progress_by_id == {candidate_id: progress}


def test_candidate_migration_rebuilds_cache_and_invalidates_changed_paths(tmp_path: Path) -> None:
    """A changed relation graph retains unaffected triples and selects changed paths for replay."""
    candidate_path = tmp_path / "candidates.json"
    previous = MultiHop3CandidateSet(
        run_id="previous-run",
        triples=(("a", "b", "c"), ("d", "e", "f"), ("x", "y", "z")),
    )
    write_multi_hop_3_candidate_set(candidate_path, previous)
    relations = (
        _relation("a", "b"),
        _relation("b", "c"),
        _relation("d", "e"),
        _relation("e", "f"),
    )

    run_id, candidates, invalidated_ids = _load_or_create_candidates(
        candidate_path,
        relations,
        SynthesisSettings.from_path(_CONFIG_PATH),
        frozenset({("a", "b")}),
    )

    assert set(candidates) == {("a", "b", "c"), ("d", "e", "f")}
    assert invalidated_ids == frozenset({multi_hop_3_candidate_id(("a", "b", "c"))})
    assert read_multi_hop_3_candidate_set(candidate_path) == MultiHop3CandidateSet(
        run_id=run_id,
        triples=candidates,
    )


def test_resume_migration_prunes_obsolete_and_invalidated_state() -> None:
    """Migration reuses only completed candidates whose relation inputs are unchanged."""
    retained_bundle = ConfirmedMultiHop3Bundle(
        bundle_id="retained-bundle",
        fact_ids=("a", "b", "c"),
        bridge="Retained",
        confirmed_by_model="provider/model",
        run_id="old-run",
    )
    invalidated_bundle = ConfirmedMultiHop3Bundle(
        bundle_id="invalidated-bundle",
        fact_ids=("d", "e", "f"),
        bridge="Invalidated",
        confirmed_by_model="provider/model",
        run_id="old-run",
    )
    obsolete_bundle = ConfirmedMultiHop3Bundle(
        bundle_id="obsolete-bundle",
        fact_ids=("x", "y", "z"),
        bridge="Obsolete",
        confirmed_by_model="provider/model",
        run_id="old-run",
    )
    retained_id = multi_hop_3_candidate_id(retained_bundle.fact_ids)
    invalidated_id = multi_hop_3_candidate_id(invalidated_bundle.fact_ids)
    new_id = multi_hop_3_candidate_id(("g", "h", "i"))
    progress = tuple(
        MultiHop3ProgressRecord(
            candidate_id=multi_hop_3_candidate_id(bundle.fact_ids),
            run_id=bundle.run_id,
            prompt_version=bundle.prompt_version,
            confirmed_by_model=bundle.confirmed_by_model,
        )
        for bundle in (retained_bundle, invalidated_bundle, obsolete_bundle)
    )

    bundles_by_id, progress_by_id = _validate_resume_state(
        (retained_bundle, invalidated_bundle, obsolete_bundle),
        progress,
        frozenset({retained_id, invalidated_id, new_id}),
        "new-run",
        frozenset(),
        frozenset({invalidated_id, new_id}),
    )

    assert bundles_by_id == {retained_bundle.bundle_id: retained_bundle}
    assert progress_by_id == {retained_id: progress[0]}
