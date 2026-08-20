"""Tests for the end-to-end evidence-graph and scenario-sampling command."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from legal_rag import evidence_graph_cli
from legal_rag.benchmark_synthesis import embedding as synthesis_embedding
from legal_rag.benchmark_synthesis import evidence_graph, vietnamese_lexical
from legal_rag.benchmark_synthesis.config import SynthesisSettings
from legal_rag.benchmark_synthesis.evidence_graph import EvidenceRelationProgressRecord
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.writer import (
    read_evidence_relation_progress,
    read_evidence_relations,
    read_scenarios,
    write_evidence_relation_progress,
    write_evidence_relations,
    write_legal_facts,
)
from legal_rag.embedding.profiles import load_configured_embedding_profile

_REPO_CONFIG_DIR = Path(__file__).resolve().parent.parent / "data" / "config" / "groundtruth"


def _fact(fact_id: str, conduct: str) -> LegalFact:
    """Build one legal fact for CLI tests."""
    return LegalFact(
        fact_id=fact_id,
        statement=f"Người điều khiển ô tô {conduct.lower()} bị phạt tiền.",
        valid_from=None,
        valid_to=None,
        document_number="168/2024/NĐ-CP",
        evidence=(EvidenceReference(chunk_id=f"chunk-{fact_id}"),),
    )


class _AlwaysRelatedClient:
    """Confirm every candidate pair as a fixed relation type, ignoring settings."""

    def __init__(self, settings: object, synthesis_settings: object, *, run_id: str) -> None:
        self._run_id = run_id

    def confirm_batch(self, pairs: Sequence[tuple[LegalFact, LegalFact]]) -> list[evidence_graph.EvidenceRelation]:
        """Return one confirmed relation for every pair in the batch."""
        return [
            evidence_graph.EvidenceRelation(
                relation_id=f"relation-{fact_a.fact_id}-{fact_b.fact_id}",
                relation_type="complementary_rule",
                from_fact_id=fact_a.fact_id,
                to_fact_id=fact_b.fact_id,
                bridge=("Điều 6",),
                confirmed_by_model="provider/test-model",
                run_id=self._run_id,
            )
            for fact_a, fact_b in pairs
        ]


class _FakeSynthesisEmbeddingEncoder:
    """Return one fixed unit vector per input text without provider access."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Accept production constructor arguments without using them."""

    def encode(self, sentences: list[str]) -> np.ndarray:
        """Return a fixed unit vector for every input sentence."""
        return np.array([[1.0, 0.0] for _ in sentences])


class _FakeVnCoreNlpWordSegmenter:
    """Return deterministic tokens without Java or model assets."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Accept production constructor arguments without using them."""

    def segment(self, texts: list[str]) -> tuple[tuple[str, ...], ...]:
        """Split every text into alphanumeric lexical units."""
        return tuple(tuple(evidence_graph._lexical_units(text)) for text in texts)


def _configure_cli(
    monkeypatch: pytest.MonkeyPatch,
    facts_path: Path,
    relations_output_path: Path,
    scenarios_output_path: Path,
    *,
    total_samples: int,
) -> Path:
    """Set deterministic CLI arguments, environment, and fake LLM/embedding clients; return the progress path."""
    from legal_rag.benchmark_synthesis import llm_execution

    relations_progress_path = relations_output_path.parent / "evidence_relation_progress.jsonl"
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:20128/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "provider/test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("SYNTHESIS_EMBEDDING_BASE_URL", "https://embedding.test/v1")
    monkeypatch.setenv("SYNTHESIS_EMBEDDING_API_KEY", "embedding-key")
    monkeypatch.setenv("SYNTHESIS_EMBEDDING_MODEL", "provider/bge-m3")
    monkeypatch.setenv("SYNTHESIS_EMBEDDING_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("SYNTHESIS_VNCORENLP_DIR", "C:/vncorenlp-1.2")
    monkeypatch.setattr(llm_execution.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(evidence_graph, "BamlEvidenceRelationClient", _AlwaysRelatedClient)
    monkeypatch.setattr(synthesis_embedding, "SynthesisEmbeddingEncoder", _FakeSynthesisEmbeddingEncoder)
    monkeypatch.setattr(vietnamese_lexical, "VnCoreNlpWordSegmenter", _FakeVnCoreNlpWordSegmenter)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "legal-build-evidence-graph",
            str(facts_path),
            str(relations_output_path),
            str(relations_progress_path),
            str(scenarios_output_path),
            "--query-taxonomy",
            str(_REPO_CONFIG_DIR / "query_taxonomy.json"),
            "--personas",
            str(_REPO_CONFIG_DIR / "personas.json"),
            "--scenario-distribution",
            str(_REPO_CONFIG_DIR / "scenario_distribution.json"),
            "--total-samples",
            str(total_samples),
            "--synthesis-config",
            str(_REPO_CONFIG_DIR / "synthesis_v1.json"),
        ],
    )
    return relations_progress_path


def _force_sequential_relation_confirmation(tmp_path: Path) -> None:
    """Point --synthesis-config at a copy with batch_size=1 and max_concurrency=1 for deterministic call order."""
    config = json.loads((_REPO_CONFIG_DIR / "synthesis_v1.json").read_text(encoding="utf-8"))
    config["stages"]["evidence_relation"]["batch_size"] = 1
    config["stages"]["evidence_relation"]["max_concurrency"] = 1
    sequential_config_path = tmp_path / "synthesis_v1.sequential.json"
    sequential_config_path.write_text(json.dumps(config), encoding="utf-8")
    sys.argv[sys.argv.index("--synthesis-config") + 1] = str(sequential_config_path)


def _relation_run_id(facts: list[LegalFact]) -> str:
    """Return the production fingerprint for the deterministic CLI test configuration."""
    synthesis_settings = SynthesisSettings.from_path(_REPO_CONFIG_DIR / "synthesis_v1.json")
    profile = load_configured_embedding_profile(synthesis_settings.embedding_profile)
    return evidence_graph.build_evidence_relation_run_id(
        facts,
        profile,
        synthesis_settings,
        embedding_base_url="https://embedding.test/v1",
        embedding_model="provider/bge-m3",
        llm_base_url="http://localhost:20128/v1",
        llm_model="provider/test-model",
    )


def test_evidence_graph_cli_confirms_relations_and_samples_scenarios(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The command confirms relations between facts and samples scenarios from them."""
    facts_path = tmp_path / "legal_facts.jsonl"
    relations_path = tmp_path / "evidence_relations.jsonl"
    scenarios_path = tmp_path / "scenarios.jsonl"
    fact_a = _fact("fact-a", "Vượt đèn đỏ")
    fact_b = _fact("fact-b", "Không chấp hành hiệu lệnh đèn tín hiệu")
    write_legal_facts(facts_path, [fact_a, fact_b])
    _configure_cli(monkeypatch, facts_path, relations_path, scenarios_path, total_samples=2)

    evidence_graph_cli.main()

    relations = read_evidence_relations(relations_path)
    scenarios = read_scenarios(scenarios_path)
    assert len(relations) == 1
    assert relations[0].from_fact_id == "fact-a"
    assert relations[0].to_fact_id == "fact-b"
    assert len(scenarios) > 0
    assert all(scenario.fact_ids for scenario in scenarios)


def test_evidence_graph_cli_raises_when_a_pair_fails_every_attempt_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without --allow-incomplete-relations, a permanently failing pair stops the run before scenario sampling."""
    facts_path = tmp_path / "legal_facts.jsonl"
    relations_path = tmp_path / "evidence_relations.jsonl"
    scenarios_path = tmp_path / "scenarios.jsonl"
    fact_a = _fact("fact-a", "Vượt đèn đỏ")
    fact_b = _fact("fact-b", "Không chấp hành hiệu lệnh đèn tín hiệu")
    write_legal_facts(facts_path, [fact_a, fact_b])
    _configure_cli(monkeypatch, facts_path, relations_path, scenarios_path, total_samples=1)
    _force_sequential_relation_confirmation(tmp_path)

    class _AlwaysFailsClient:
        """Fail every candidate pair, unconditionally, on every attempt."""

        def __init__(self, settings: object, synthesis_settings: object, *, run_id: str) -> None:
            pass

        def confirm_batch(self, pairs: Sequence[tuple[LegalFact, LegalFact]]) -> list[evidence_graph.EvidenceRelation]:
            """Always raise a retryable transport error."""
            msg = "provider timed out"
            raise TimeoutError(msg)

    monkeypatch.setattr(evidence_graph, "BamlEvidenceRelationClient", _AlwaysFailsClient)

    with pytest.raises(RuntimeError, match="evidence relation confirmation left 1 incomplete"):
        evidence_graph_cli.main()

    assert not scenarios_path.exists()


def test_evidence_graph_cli_allow_incomplete_relations_samples_from_confirmed_pairs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--allow-incomplete-relations samples scenarios from whatever pairs were confirmed, despite one failure."""
    facts_path = tmp_path / "legal_facts.jsonl"
    relations_path = tmp_path / "evidence_relations.jsonl"
    scenarios_path = tmp_path / "scenarios.jsonl"
    fact_a = _fact("fact-a", "Vượt đèn đỏ")
    fact_b = _fact("fact-b", "Không chấp hành hiệu lệnh đèn tín hiệu")
    fact_c = _fact("fact-c", "Không đội mũ bảo hiểm")
    write_legal_facts(facts_path, [fact_a, fact_b, fact_c])
    _configure_cli(monkeypatch, facts_path, relations_path, scenarios_path, total_samples=2)
    _force_sequential_relation_confirmation(tmp_path)
    sys.argv.append("--allow-incomplete-relations")

    class _FailsForOnePairClient:
        """Fail one specific candidate pair on every attempt; confirm every other pair."""

        def __init__(self, settings: object, synthesis_settings: object, *, run_id: str) -> None:
            self._run_id = run_id

        def confirm_batch(self, pairs: Sequence[tuple[LegalFact, LegalFact]]) -> list[evidence_graph.EvidenceRelation]:
            """Raise for the fact-a/fact-b pair, otherwise confirm the pair as related."""
            fact_x, fact_y = pairs[0]
            if {fact_x.fact_id, fact_y.fact_id} == {"fact-a", "fact-b"}:
                msg = "provider timed out"
                raise TimeoutError(msg)
            return [
                evidence_graph.EvidenceRelation(
                    relation_id=f"relation-{fact_x.fact_id}-{fact_y.fact_id}",
                    relation_type="complementary_rule",
                    from_fact_id=fact_x.fact_id,
                    to_fact_id=fact_y.fact_id,
                    bridge=("Điều 6",),
                    confirmed_by_model="provider/test-model",
                    run_id=self._run_id,
                )
            ]

    monkeypatch.setattr(evidence_graph, "BamlEvidenceRelationClient", _FailsForOnePairClient)

    evidence_graph_cli.main()

    assert len(read_evidence_relations(relations_path)) == 2
    assert scenarios_path.exists()
    assert len(read_scenarios(scenarios_path)) > 0


def test_evidence_graph_cli_does_not_recheck_a_pair_already_processed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A candidate pair recorded in the progress file is not sent to the LLM again."""
    facts_path = tmp_path / "legal_facts.jsonl"
    relations_path = tmp_path / "evidence_relations.jsonl"
    scenarios_path = tmp_path / "scenarios.jsonl"
    fact_a = _fact("fact-a", "Vượt đèn đỏ")
    fact_b = _fact("fact-b", "Không chấp hành hiệu lệnh đèn tín hiệu")
    fact_c = _fact("fact-c", "Không đội mũ bảo hiểm")
    write_legal_facts(facts_path, [fact_a, fact_b, fact_c])
    progress_path = _configure_cli(monkeypatch, facts_path, relations_path, scenarios_path, total_samples=2)
    write_evidence_relation_progress(
        progress_path,
        [
            EvidenceRelationProgressRecord(
                fact_id_a="fact-a",
                fact_id_b="fact-b",
                run_id=_relation_run_id([fact_a, fact_b, fact_c]),
            )
        ],
    )
    checked_pairs: list[tuple[str, str]] = []

    class _RecordingClient:
        """Confirm any pair as related, recording every pair it was asked about."""

        def __init__(self, settings: object, synthesis_settings: object, *, run_id: str) -> None:
            self._run_id = run_id

        def confirm_batch(self, pairs: Sequence[tuple[LegalFact, LegalFact]]) -> list[evidence_graph.EvidenceRelation]:
            """Record each pair and return one confirmed relation for it."""
            relations = []
            for fact_x, fact_y in pairs:
                checked_pairs.append(tuple(sorted((fact_x.fact_id, fact_y.fact_id))))
                relations.append(
                    evidence_graph.EvidenceRelation(
                        relation_id=f"relation-{fact_x.fact_id}-{fact_y.fact_id}",
                        relation_type="complementary_rule",
                        from_fact_id=fact_x.fact_id,
                        to_fact_id=fact_y.fact_id,
                        bridge=("Điều 6",),
                        confirmed_by_model="provider/test-model",
                        run_id=self._run_id,
                    )
                )
            return relations

    monkeypatch.setattr(evidence_graph, "BamlEvidenceRelationClient", _RecordingClient)

    evidence_graph_cli.main()

    assert ("fact-a", "fact-b") not in checked_pairs
    assert set(checked_pairs) == {("fact-a", "fact-c"), ("fact-b", "fact-c")}


def test_evidence_graph_cli_persists_progress_if_interrupted_mid_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A pair already confirmed is durably written even if the process dies before the run finishes."""
    facts_path = tmp_path / "legal_facts.jsonl"
    relations_path = tmp_path / "evidence_relations.jsonl"
    scenarios_path = tmp_path / "scenarios.jsonl"
    fact_a = _fact("fact-a", "Vượt đèn đỏ")
    fact_b = _fact("fact-b", "Không chấp hành hiệu lệnh đèn tín hiệu")
    fact_c = _fact("fact-c", "Không đội mũ bảo hiểm")
    write_legal_facts(facts_path, [fact_a, fact_b, fact_c])
    progress_path = _configure_cli(monkeypatch, facts_path, relations_path, scenarios_path, total_samples=2)
    _force_sequential_relation_confirmation(tmp_path)

    class _CrashesOnSecondPairClient:
        """Succeed on the first candidate pair, then simulate the whole process being killed."""

        def __init__(self, settings: object, synthesis_settings: object, *, run_id: str) -> None:
            self._calls = 0
            self._run_id = run_id

        def confirm_batch(self, pairs: Sequence[tuple[LegalFact, LegalFact]]) -> list[evidence_graph.EvidenceRelation]:
            """Return one confirmed relation on the first call, then raise an uncaught fatal error."""
            self._calls += 1
            if self._calls > 1:
                raise SystemExit
            fact_x, fact_y = pairs[0]
            return [
                evidence_graph.EvidenceRelation(
                    relation_id=f"relation-{fact_x.fact_id}-{fact_y.fact_id}",
                    relation_type="complementary_rule",
                    from_fact_id=fact_x.fact_id,
                    to_fact_id=fact_y.fact_id,
                    bridge=("Điều 6",),
                    confirmed_by_model="provider/test-model",
                    run_id=self._run_id,
                )
            ]

    monkeypatch.setattr(evidence_graph, "BamlEvidenceRelationClient", _CrashesOnSecondPairClient)

    with pytest.raises(SystemExit):
        evidence_graph_cli.main()

    assert len(read_evidence_relation_progress(progress_path)) == 1
    assert len(read_evidence_relations(relations_path)) == 1


def test_evidence_graph_cli_does_not_recheck_a_pair_confirmed_unrelated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A pair the model found unrelated produces no relation, but is still recorded as checked and never resent."""
    facts_path = tmp_path / "legal_facts.jsonl"
    relations_path = tmp_path / "evidence_relations.jsonl"
    scenarios_path = tmp_path / "scenarios.jsonl"
    fact_a = _fact("fact-a", "Vượt đèn đỏ")
    fact_b = _fact("fact-b", "Không chấp hành hiệu lệnh đèn tín hiệu")
    write_legal_facts(facts_path, [fact_a, fact_b])
    progress_path = _configure_cli(monkeypatch, facts_path, relations_path, scenarios_path, total_samples=1)

    class _UnrelatedClient:
        """Confirm that no candidate pair is related."""

        def __init__(self, settings: object, synthesis_settings: object, *, run_id: str) -> None:
            self._run_id = run_id

        def confirm_batch(self, pairs: Sequence[tuple[LegalFact, LegalFact]]) -> list[None]:
            """Return None for every pair."""
            return [None for _ in pairs]

    monkeypatch.setattr(evidence_graph, "BamlEvidenceRelationClient", _UnrelatedClient)

    evidence_graph_cli.main()

    assert read_evidence_relations(relations_path) == ()
    assert read_evidence_relation_progress(progress_path) == (
        EvidenceRelationProgressRecord(
            fact_id_a="fact-a",
            fact_id_b="fact-b",
            run_id=_relation_run_id([fact_a, fact_b]),
        ),
    )

    def _fail(*args: object, **kwargs: object) -> None:
        msg = "BamlEvidenceRelationClient must not be constructed when every pair is resumed"
        raise AssertionError(msg)

    monkeypatch.setattr(evidence_graph, "BamlEvidenceRelationClient", _fail)
    monkeypatch.setattr(
        synthesis_embedding,
        "SynthesisEmbeddingEncoder",
        lambda *args, **kwargs: pytest.fail("Cached candidate pairs must bypass embedding inference"),
    )

    evidence_graph_cli.main()


def test_evidence_graph_cli_rejects_progress_from_a_stale_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Progress created from different facts or settings cannot silently skip current work."""
    facts_path = tmp_path / "legal_facts.jsonl"
    relations_path = tmp_path / "evidence_relations.jsonl"
    scenarios_path = tmp_path / "scenarios.jsonl"
    facts = [_fact("fact-a", "Vượt đèn đỏ"), _fact("fact-b", "Không chấp hành hiệu lệnh đèn tín hiệu")]
    write_legal_facts(facts_path, facts)
    progress_path = _configure_cli(monkeypatch, facts_path, relations_path, scenarios_path, total_samples=1)
    write_evidence_relation_progress(
        progress_path,
        [EvidenceRelationProgressRecord(fact_id_a="fact-a", fact_id_b="fact-b", run_id="stale-run")],
    )

    with pytest.raises(ValueError, match="different synthesis run"):
        evidence_graph_cli.main()


def test_evidence_graph_cli_replaces_relation_written_before_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A retried pair replaces the relation left by a crash before its progress write."""
    facts_path = tmp_path / "legal_facts.jsonl"
    relations_path = tmp_path / "evidence_relations.jsonl"
    scenarios_path = tmp_path / "scenarios.jsonl"
    facts = [_fact("fact-a", "Vượt đèn đỏ"), _fact("fact-b", "Không chấp hành hiệu lệnh đèn tín hiệu")]
    write_legal_facts(facts_path, facts)
    _configure_cli(monkeypatch, facts_path, relations_path, scenarios_path, total_samples=1)
    run_id = _relation_run_id(facts)
    write_evidence_relations(
        relations_path,
        [
            evidence_graph.EvidenceRelation(
                relation_id="relation-before-crash",
                relation_type="complementary_rule",
                from_fact_id="fact-a",
                to_fact_id="fact-b",
                bridge=("Điều 6",),
                confirmed_by_model="provider/test-model",
                run_id=run_id,
            )
        ],
    )

    class _NowUnrelatedClient:
        """Return the updated unrelated result for the retried pair."""

        def __init__(self, settings: object, synthesis_settings: object, *, run_id: str) -> None:
            return None

        def confirm_batch(self, pairs: Sequence[tuple[LegalFact, LegalFact]]) -> list[None]:
            """Confirm that the pair is no longer considered related."""
            return [None for _ in pairs]

    monkeypatch.setattr(evidence_graph, "BamlEvidenceRelationClient", _NowUnrelatedClient)

    evidence_graph_cli.main()

    assert read_evidence_relations(relations_path) == ()


def test_evidence_graph_cli_writes_empty_progress_when_no_pairs_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A one-fact corpus still produces all declared output artifacts."""
    facts_path = tmp_path / "legal_facts.jsonl"
    relations_path = tmp_path / "evidence_relations.jsonl"
    scenarios_path = tmp_path / "scenarios.jsonl"
    write_legal_facts(facts_path, [_fact("fact-a", "Vượt đèn đỏ")])
    progress_path = _configure_cli(monkeypatch, facts_path, relations_path, scenarios_path, total_samples=1)

    evidence_graph_cli.main()

    assert relations_path.exists()
    assert progress_path.exists()
    assert read_evidence_relation_progress(progress_path) == ()


def test_evidence_graph_cli_can_prepare_candidates_without_llm_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Candidate-only mode writes the expensive retrieval result and stops before relation confirmation."""
    facts_path = tmp_path / "legal_facts.jsonl"
    relations_path = tmp_path / "evidence_relations.jsonl"
    scenarios_path = tmp_path / "scenarios.jsonl"
    write_legal_facts(
        facts_path,
        [_fact("fact-a", "Vượt đèn đỏ"), _fact("fact-b", "Không chấp hành hiệu lệnh đèn tín hiệu")],
    )
    _configure_cli(monkeypatch, facts_path, relations_path, scenarios_path, total_samples=1)
    sys.argv.append("--prepare-candidates-only")
    monkeypatch.setattr(
        evidence_graph,
        "BamlEvidenceRelationClient",
        lambda *args, **kwargs: pytest.fail("Candidate-only mode must not construct the relation client"),
    )

    evidence_graph_cli.main()

    assert (tmp_path / "evidence_relation_candidates.json").exists()
    assert not relations_path.exists()
    assert not scenarios_path.exists()
