"""Tests for independent gate-2 verification and relabeling."""

from __future__ import annotations

from pathlib import Path

import pytest

from legal_rag.baml_client.types import Answerability, FactContext
from legal_rag.baml_client.types import ClaimReference as BamlClaimReference
from legal_rag.baml_client.types import VerificationResult as BamlVerificationResult
from legal_rag.benchmark_synthesis import verification
from legal_rag.benchmark_synthesis.candidate_generation import CandidateDraft
from legal_rag.benchmark_synthesis.config import SynthesisSettings
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.verification import BamlVerificationClient, verify_candidate
from legal_rag.settings import Settings

_SYNTHESIS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "data" / "config" / "groundtruth" / "synthesis_v1.json"


def _fact(fact_id: str = "fact-1") -> LegalFact:
    """Build one minimal legal fact for verification tests."""
    return LegalFact(
        fact_id=fact_id,
        fact_content_sha256=f"content-{fact_id}",
        conduct="Vượt đèn đỏ",
        subject="Người điều khiển ô tô",
        conditions=(),
        monetary_penalty="4.000.000 đồng",
        license_points_deducted=None,
        additional_penalties=(),
        remedial_measures=(),
        valid_from=None,
        valid_to=None,
        document_number="168/2024/NĐ-CP",
        provision_path="Điều 6",
        evidence=(EvidenceReference(chunk_id="chunk-1", quoted_span="Phạt tiền."),),
    )


def _draft() -> CandidateDraft:
    """Build one minimal candidate draft for verification tests."""
    return CandidateDraft(
        sample_id="sample-1",
        scenario_id="scenario-1",
        query_vi="oto vuot den do bi phat bao nhieu",
        reference_answer_vi="Phạt tiền 4.000.000 đồng.",
        reasoning_vi="Theo Điều 6.",
        required_claims=(),
        answer_fields=(),
        evolutions=(),
        gold_chunk_ids=("chunk-1",),
        query_type="penalty_for_conduct",
        query_form="specific",
        persona="tai_xe_ca_nhan",
        query_style="standard",
        query_length="short",
        answerability="answerable",
        reasoning_hops=1,
        as_of_date=None,
        source_fact_ids=("fact-1",),
        source_fact_content_sha256=("content-fact-1",),
        generator_model="provider/generator-model",
    )


def _verification_result(**overrides: object) -> BamlVerificationResult:
    """Build one BAML verification result, all dimensions passing by default."""
    defaults: dict[str, object] = {
        "is_standalone": True,
        "is_specific": True,
        "is_question_grounded": True,
        "is_answer_grounded": True,
        "is_domain_relevant": True,
        "is_temporal_correct": True,
        "uses_required_bridge": True,
        "necessary_reasoning_hops": 1,
        "observed_query_type": "penalty_for_conduct",
        "observed_answerability": Answerability.Answerable,
        "reasoning": "Đầy đủ căn cứ.",
    }
    defaults.update(overrides)
    return BamlVerificationResult(**defaults)


class _FakeVerificationClient:
    """Return a pre-programmed verification result, ignoring call arguments."""

    def __init__(self, result: BamlVerificationResult) -> None:
        self._result = result

    def verify(self, *args: object, **kwargs: object) -> BamlVerificationResult:
        """Return the configured verification result."""
        return self._result


def test_verify_candidate_passes_when_all_six_dimensions_pass() -> None:
    """A candidate with every dimension true is marked passed."""
    client = _FakeVerificationClient(_verification_result())

    result = verify_candidate(
        _draft(), {"fact-1": _fact()}, ["penalty_for_conduct"], client, verifier_model="provider/verifier-model"
    )

    assert result.passed is True
    assert result.verifier_model == "provider/verifier-model"


@pytest.mark.parametrize(
    "failing_dimension",
    [
        "is_standalone",
        "is_specific",
        "is_question_grounded",
        "is_answer_grounded",
        "is_domain_relevant",
        "is_temporal_correct",
    ],
)
def test_verify_candidate_fails_when_any_single_dimension_fails(failing_dimension: str) -> None:
    """Passed is a deterministic AND: any one false dimension fails the whole candidate."""
    client = _FakeVerificationClient(_verification_result(**{failing_dimension: False}))

    result = verify_candidate(_draft(), {"fact-1": _fact()}, ["penalty_for_conduct"], client, verifier_model="m")

    assert result.passed is False
    assert getattr(result, failing_dimension) is False


def test_verify_candidate_translates_observed_answerability_to_snake_case() -> None:
    """The BAML PascalCase enum translates to the pipeline's snake_case label."""
    client = _FakeVerificationClient(_verification_result(observed_answerability=Answerability.PartiallyAnswerable))

    result = verify_candidate(_draft(), {"fact-1": _fact()}, ["penalty_for_conduct"], client, verifier_model="m")

    assert result.observed_answerability == "partially_answerable"


def test_verify_candidate_carries_necessary_reasoning_hops_unmodified() -> None:
    """The verifier's hop-necessity judgment passes through as observed, not the declared value."""
    client = _FakeVerificationClient(_verification_result(necessary_reasoning_hops=1))
    draft = _draft()
    assert draft.reasoning_hops == 1

    result = verify_candidate(draft, {"fact-1": _fact()}, ["penalty_for_conduct"], client, verifier_model="m")

    assert result.necessary_reasoning_hops == 1


def test_verify_candidate_fails_when_observed_hops_do_not_match_scenario() -> None:
    """A candidate fails when its declared multi-hop difficulty is not actually necessary."""
    client = _FakeVerificationClient(_verification_result(necessary_reasoning_hops=2))

    result = verify_candidate(_draft(), {"fact-1": _fact()}, ["penalty_for_conduct"], client, verifier_model="m")

    assert result.passed is False
    assert result.necessary_reasoning_hops == 2


def test_verify_candidate_carries_observed_query_type_from_catalog() -> None:
    """The relabeled query_type is the verifier's own choice, not the declared label."""
    client = _FakeVerificationClient(_verification_result(observed_query_type="out_of_scope"))

    result = verify_candidate(
        _draft(), {"fact-1": _fact()}, ["penalty_for_conduct", "out_of_scope"], client, verifier_model="m"
    )

    assert result.observed_query_type == "out_of_scope"


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


def test_baml_verification_client_parses_typed_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """The generated non-stream client returns a typed verification result."""
    client = BamlVerificationClient(_settings(), _synthesis_settings())
    monkeypatch.setattr(
        verification,
        "execute_baml_request",
        lambda *args, **kwargs: (
            '{"is_standalone":true,"is_specific":true,"is_question_grounded":true,'
            '"is_answer_grounded":true,"is_domain_relevant":true,"is_temporal_correct":true,'
            '"uses_required_bridge":true,'
            '"necessary_reasoning_hops":1,"observed_query_type":"penalty_for_conduct",'
            '"observed_answerability":"Answerable","reasoning":"ok"}'
        ),
    )
    fact_context = FactContext(
        fact_id="fact-1",
        conduct="Vượt đèn đỏ",
        subject="Người điều khiển ô tô",
        conditions=[],
        monetary_penalty="4.000.000 đồng",
        license_points_deducted=None,
        additional_penalties=[],
        remedial_measures=[],
        valid_from=None,
        valid_to=None,
        document_number="168/2024/NĐ-CP",
        provision_path="Điều 6",
        evidence_chunk_ids=["chunk-1"],
    )
    claim = BamlClaimReference(claim="c", fact_id="fact-1", evidence_chunk_ids=["chunk-1"])

    result = client.verify("q", "a", "r", [fact_context], [claim], None, 1, [], ["penalty_for_conduct"])

    assert result.is_standalone is True
    assert result.observed_answerability == Answerability.Answerable
