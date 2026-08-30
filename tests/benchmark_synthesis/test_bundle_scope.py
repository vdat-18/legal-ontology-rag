"""Tests for evidence-bundle domain-scope confirmation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from legal_rag.baml_client.types import BundleScopeBatchResult, BundleScopeDecision
from legal_rag.benchmark_synthesis.bundle_scope import (
    BundleScopeClient,
    BundleScopeRecord,
    build_bundle_scope_run_id,
    confirm_bundle_scopes,
)
from legal_rag.benchmark_synthesis.config import SynthesisSettings
from legal_rag.benchmark_synthesis.evidence_graph import EvidenceBundle
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.writer import read_bundle_scope_records, write_bundle_scope_records

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPT_PATH = _REPO_ROOT / "baml_src" / "bundle_scope.baml"
_SYNTHESIS_CONFIG_PATH = _REPO_ROOT / "data" / "config" / "groundtruth" / "synthesis_v1.json"


def _fact(fact_id: str, statement: str) -> LegalFact:
    """Build one minimal legal fact."""
    return LegalFact(
        fact_id=fact_id,
        statement=statement,
        valid_from=None,
        valid_to=None,
        document_number="168/2024/NĐ-CP",
        evidence=(EvidenceReference(chunk_id=f"chunk-{fact_id}"),),
    )


def _bundle(bundle_id: str, fact_ids: tuple[str, ...]) -> EvidenceBundle:
    """Build one minimal evidence bundle."""
    return EvidenceBundle(
        bundle_id=bundle_id,
        kind="single_hop" if len(fact_ids) == 1 else "multi_hop",
        fact_ids=fact_ids,
        bridge=None if len(fact_ids) == 1 else "Kết hợp các căn cứ xử lý vi phạm",
    )


class _FakeScopeClient(BundleScopeClient):
    """Return configured scope decisions, deliberately in reverse order."""

    def __init__(self, decisions: Mapping[str, bool]) -> None:
        self._decisions = decisions

    def confirm_batch(
        self,
        bundles: Sequence[EvidenceBundle],
        facts_by_id: Mapping[str, LegalFact],
    ) -> Sequence[BundleScopeBatchResult]:
        """Return one typed result for every bundle."""
        assert facts_by_id
        return tuple(
            BundleScopeBatchResult(
                bundle_id=bundle.bundle_id,
                decision=BundleScopeDecision(
                    is_in_scope=self._decisions[bundle.bundle_id],
                    reason="Thuộc phạm vi" if self._decisions[bundle.bundle_id] else "Ngoài phạm vi",
                ),
            )
            for bundle in reversed(bundles)
        )


def test_bundle_scope_prompt_excludes_normal_administrative_topics() -> None:
    """The scope gate explicitly rejects ordinary domain-adjacent administration."""
    prompt = _PROMPT_PATH.read_text(encoding="utf-8")

    assert "xử phạt và xử lý" in prompt
    assert "cấp, đổi, cấp lại giấy phép" in prompt
    assert "khám sức khỏe" in prompt
    assert "quy tắc xử lý vi phạm hành chính chung" in prompt
    assert "Một fact ngoài scope không được cứu" in prompt


def test_confirm_bundle_scopes_restores_input_order_and_provenance() -> None:
    """Model result ordering cannot change deterministic bundle ordering."""
    facts = (
        _fact("fact-a", "Người điều khiển ô tô vượt đèn đỏ bị phạt tiền."),
        _fact("fact-b", "Người có giấy phép bị hỏng được đề nghị đổi giấy phép."),
    )
    bundles = (_bundle("bundle-a", ("fact-a",)), _bundle("bundle-b", ("fact-b",)))

    records, failures = confirm_bundle_scopes(
        bundles,
        facts,
        _FakeScopeClient({"bundle-a": True, "bundle-b": False}),
        run_id="run-1",
        model="test-model",
        batch_size=2,
        max_concurrency=1,
        max_retries=0,
        retry_delay_seconds=0,
    )

    assert failures == ()
    assert [record.bundle_id for record in records] == ["bundle-a", "bundle-b"]
    assert [record.is_in_scope for record in records] == [True, False]
    assert all(record.run_id == "run-1" for record in records)
    assert all(record.confirmed_by_model == "test-model" for record in records)


def test_bundle_scope_run_id_ignores_batch_and_concurrency() -> None:
    """Execution tuning does not invalidate reusable semantic decisions."""
    facts = (_fact("fact-a", "Người điều khiển ô tô vượt đèn đỏ bị phạt tiền."),)
    bundles = (_bundle("bundle-a", ("fact-a",)),)
    settings = SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH)
    changed = replace(settings, bundle_scope_batch_size=20, bundle_scope_max_concurrency=5)

    first = build_bundle_scope_run_id(
        bundles,
        facts,
        settings,
        llm_base_url="https://llm.test/v1",
        llm_model="test-model",
    )
    second = build_bundle_scope_run_id(
        bundles,
        facts,
        changed,
        llm_base_url="https://llm.test/v1",
        llm_model="test-model",
    )

    assert first == second


def test_bundle_scope_writer_round_trips_all_decisions(tmp_path: Path) -> None:
    """Rejected decisions remain persisted so resume never repeats them."""
    output_path = tmp_path / "scope.jsonl"
    records = (
        BundleScopeRecord("bundle-a", True, "Thuộc phạm vi", "run-1", "1.0.0", "test-model"),
        BundleScopeRecord("bundle-b", False, "Ngoài phạm vi", "run-1", "1.0.0", "test-model"),
    )

    write_bundle_scope_records(output_path, records)

    assert read_bundle_scope_records(output_path) == records
