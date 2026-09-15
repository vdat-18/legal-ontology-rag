"""Tests for target-only multi-turn generation and resume."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

from legal_rag import multi_turn_generation_cli
from legal_rag.baml_client.types import ClaimReference as BamlClaimReference
from legal_rag.baml_client.types import (
    GeneratedMultiTurnTarget,
    GeneratedMultiTurnTargetBatchItem,
    MultiTurnTargetGenerationInput,
)
from legal_rag.benchmark_synthesis import multi_turn_candidate_generation
from legal_rag.benchmark_synthesis.conversation_scenario_sampling import (
    ConversationScenario,
    TurnSpec,
)
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.multi_turn_candidate_generation import (
    ReviewedHistoryCandidate,
)
from legal_rag.benchmark_synthesis.writer import (
    read_multi_turn_candidates,
    read_multi_turn_progress,
    write_conversation_scenarios,
    write_legal_facts,
    write_multi_turn_progress,
)
from legal_rag.io_utils import write_jsonl_atomic

_REPO_CONFIG_DIR = Path(__file__).resolve().parents[1] / "data" / "config" / "groundtruth"
_SEEN_BATCHES: list[list[MultiTurnTargetGenerationInput]] = []


class _FakeTargetClient:
    """Record target batches and return one valid result for every item."""

    def __init__(self, settings: object, synthesis_settings: object) -> None:
        return None

    def generate_target_batch(
        self,
        items: list[MultiTurnTargetGenerationInput],
    ) -> list[GeneratedMultiTurnTargetBatchItem]:
        _SEEN_BATCHES.append(items)
        return [_generated_item(item) for item in items]


class _MissingItemClient:
    """Omit one shared-batch item and return it when rerun alone."""

    def __init__(self) -> None:
        self.batch_task_ids: list[list[str]] = []

    def generate_target_batch(
        self,
        items: list[MultiTurnTargetGenerationInput],
    ) -> list[GeneratedMultiTurnTargetBatchItem]:
        self.batch_task_ids.append([item.task_id for item in items])
        selected = items[:1] if len(items) > 1 else items
        return [_generated_item(item) for item in selected]


class _InvalidItemClient:
    """Return one context-independent target, then repair it alone."""

    def __init__(self) -> None:
        self.batch_task_ids: list[list[str]] = []

    def generate_target_batch(
        self,
        items: list[MultiTurnTargetGenerationInput],
    ) -> list[GeneratedMultiTurnTargetBatchItem]:
        self.batch_task_ids.append([item.task_id for item in items])
        results = [_generated_item(item) for item in items]
        if len(items) > 1:
            results[1].target.standalone_query = results[1].target.query
        return results


def _generated_item(
    item: MultiTurnTargetGenerationInput,
) -> GeneratedMultiTurnTargetBatchItem:
    fact_id = item.target.facts[0].fact_id
    return GeneratedMultiTurnTargetBatchItem(
        task_id=item.task_id,
        target=GeneratedMultiTurnTarget(
            turn_number=item.target.turn_number,
            query=f"Vậy trường hợp {item.task_id} thì sao?",
            standalone_query=f"Câu hỏi đầy đủ cho {item.task_id} và đối tượng cụ thể",
            reference_answer=f"Đáp án mới cho {item.task_id}",
            reasoning="Căn cứ mới dẫn trực tiếp tới đáp án.",
            missing_information=(
                "mức xử lý bổ sung"
                if item.target.answerability == "partially_answerable"
                else None
            ),
            required_claims=[BamlClaimReference(claim="Ý mới", fact_id=fact_id)],
        ),
    )


def _fact(fact_id: str) -> LegalFact:
    """Build one fact with one evidence chunk."""
    return LegalFact(
        fact_id=fact_id,
        statement=f"Statement của {fact_id}.",
        valid_from=None,
        valid_to=None,
        document_number="168/2024/NĐ-CP",
        evidence=(EvidenceReference(chunk_id=f"chunk-{fact_id}"),),
    )


def _conversation(conversation_id: str, length: int) -> ConversationScenario:
    """Build one plan with disjoint reviewed history and target facts."""
    turns = [
        TurnSpec(
            turn_number=index,
            dialogue_act="inquiry" if index == 1 else "switch",
            context_dependency="standalone",
            answerability="answerable",
            anchor_scenario_id=f"scenario-{conversation_id}-{index}",
            reference_turn_number=None,
            fact_ids=(f"history-{conversation_id}-{index}",),
            gold_chunk_ids=(f"chunk-history-{conversation_id}-{index}",),
        )
        for index in range(1, length)
    ]
    turns.append(
        TurnSpec(
            turn_number=length,
            dialogue_act="continuity",
            context_dependency="anaphora",
            answerability="answerable",
            anchor_scenario_id=f"target-{conversation_id}",
            reference_turn_number=length - 1,
            fact_ids=(f"target-{conversation_id}",),
            gold_chunk_ids=(f"chunk-target-{conversation_id}",),
        )
    )
    return ConversationScenario(
        conversation_scenario_id=conversation_id,
        persona="tai_xe_ca_nhan",
        conversation_length=length,
        turns=tuple(turns),
        gold_chunk_ids=(f"chunk-target-{conversation_id}",),
    )


def _history_candidate(turn: TurnSpec) -> ReviewedHistoryCandidate:
    """Build the reviewed record referenced by one context turn."""
    return ReviewedHistoryCandidate(
        scenario_id=turn.anchor_scenario_id,
        persona="tai_xe_ca_nhan",
        answerability="answerable",
        query=f"Reviewed question {turn.anchor_scenario_id}",
        reference_answer=f"Reviewed answer {turn.anchor_scenario_id}",
        gold_chunk_ids=turn.gold_chunk_ids,
        source_fact_ids=turn.fact_ids,
    )


def _prepare_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write facts, conversation plans, and reviewed history records."""
    conversations = (_conversation("conv-a", 2), _conversation("conv-b", 3))
    fact_ids = {
        fact_id
        for conversation in conversations
        for turn in conversation.turns
        for fact_id in turn.fact_ids
    }
    facts_path = tmp_path / "facts.jsonl"
    write_legal_facts(facts_path, tuple(_fact(fact_id) for fact_id in sorted(fact_ids)))
    conversations_path = tmp_path / "conversations.jsonl"
    write_conversation_scenarios(conversations_path, conversations)
    history_path = tmp_path / "groundtruth.jsonl"
    history = [
        _history_candidate(turn)
        for conversation in conversations
        for turn in conversation.turns[:-1]
    ]
    write_jsonl_atomic(
        history_path,
        (
            {
                "interaction_type": "single_turn",
                "scenario_id": item.scenario_id,
                "persona": item.persona,
                "answerability": item.answerability,
                "question": item.query,
                "expected_answer": item.reference_answer,
                "expected_chunk_ids": list(item.gold_chunk_ids),
                "source_fact_ids": list(item.source_fact_ids),
            }
            for item in history
        ),
    )
    return facts_path, conversations_path, history_path


def _configure_cli(
    monkeypatch: pytest.MonkeyPatch,
    facts_path: Path,
    conversations_path: Path,
    history_path: Path,
    output_path: Path,
) -> None:
    """Configure deterministic CLI arguments and a fake provider."""
    from legal_rag.benchmark_synthesis import llm_execution

    _SEEN_BATCHES.clear()
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:20128/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "provider/test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setattr(llm_execution.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        multi_turn_candidate_generation,
        "BamlMultiTurnGenerationClient",
        _FakeTargetClient,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "legal-generate-multi-turn-candidates",
            str(facts_path),
            str(conversations_path),
            str(output_path),
            "--history-groundtruth",
            str(history_path),
            "--personas",
            str(_REPO_CONFIG_DIR / "personas.json"),
            "--synthesis-config",
            str(_REPO_CONFIG_DIR / "synthesis_v1.json"),
        ],
    )


def _context(
    conversations: tuple[ConversationScenario, ...],
    client: object,
) -> multi_turn_candidate_generation.MultiTurnPipelineContext:
    history = {
        turn.anchor_scenario_id: _history_candidate(turn)
        for conversation in conversations
        for turn in conversation.turns[:-1]
    }
    facts = {
        fact_id: _fact(fact_id)
        for conversation in conversations
        for turn in conversation.turns
        for fact_id in turn.fact_ids
    }
    return multi_turn_candidate_generation.MultiTurnPipelineContext(
        facts_by_id=facts,
        history_candidates_by_scenario_id=history,
        persona_descriptions={"tai_xe_ca_nhan": "persona"},
        generation_client=client,
        generator_model="model",
        llm_max_retries=0,
        llm_retry_delay_seconds=0,
    )


def test_cli_reuses_history_and_only_requests_targets(tmp_path: Path) -> None:
    """Reviewed history is copied verbatim and omitted from model output."""
    facts_path, conversations_path, history_path = _prepare_inputs(tmp_path)
    output_path = tmp_path / "multi_turn_v3.jsonl"
    monkeypatch = pytest.MonkeyPatch()
    _configure_cli(monkeypatch, facts_path, conversations_path, history_path, output_path)
    with monkeypatch.context():
        multi_turn_generation_cli.main()

    conversations = read_multi_turn_candidates(output_path)
    assert len(_SEEN_BATCHES) == 1
    assert len(_SEEN_BATCHES[0]) == 2
    assert len(_SEEN_BATCHES[0][1].history) == 2
    assert _SEEN_BATCHES[0][1].target.facts[0].fact_id == "T3F1"
    assert _SEEN_BATCHES[0][1].target.reference_turn_number == 2
    assert _SEEN_BATCHES[0][1].target.reference_question.endswith("scenario-conv-b-2")
    assert _SEEN_BATCHES[0][1].target.reference_facts[0].fact_id == "H2F1"
    assert conversations[1].turns[0].query.startswith("Reviewed question")
    assert conversations[1].turns[-1].standalone_query.startswith("Câu hỏi đầy đủ")
    assert conversations[1].gold_chunk_ids == conversations[1].turns[-1].gold_chunk_ids


def test_cli_resumes_only_missing_conversations(tmp_path: Path) -> None:
    """Completed conversation checkpoints survive a later resume."""
    facts_path, conversations_path, history_path = _prepare_inputs(tmp_path)
    output_path = tmp_path / "multi_turn_v3.jsonl"
    monkeypatch = pytest.MonkeyPatch()
    _configure_cli(monkeypatch, facts_path, conversations_path, history_path, output_path)
    with monkeypatch.context():
        multi_turn_generation_cli.main()
    progress_path = output_path.with_suffix(".progress.jsonl")
    retained = tuple(
        record for record in read_multi_turn_progress(progress_path) if record.task_id == "conv-a"
    )
    write_multi_turn_progress(progress_path, retained)

    _configure_cli(monkeypatch, facts_path, conversations_path, history_path, output_path)
    with monkeypatch.context():
        multi_turn_generation_cli.main()

    assert [[item.task_id for item in batch] for batch in _SEEN_BATCHES] == [["conv-b"]]


@pytest.mark.parametrize("client_type", [_MissingItemClient, _InvalidItemClient])
def test_shared_batch_salvages_valid_items_and_reruns_only_bad_item(client_type: type) -> None:
    """Missing and semantically invalid results do not discard valid siblings."""
    conversations = (_conversation("conv-a", 2), _conversation("conv-b", 2))
    client = client_type()
    context = _context(conversations, client)
    progress: dict[str, multi_turn_candidate_generation.MultiTurnProgressRecord] = {}

    failures = multi_turn_candidate_generation.generate_multi_turn_candidates(
        conversations,
        progress,
        context,
        lambda records: progress.update({record.task_id: record for record in records}),
        batch_size=2,
        max_concurrency=1,
    )

    assert failures == ()
    assert client.batch_task_ids == [["conv-a", "conv-b"], ["conv-b"]]
    assert set(progress) == {"conv-a", "conv-b"}


def test_generation_rejects_target_fact_present_in_reviewed_history() -> None:
    """Runtime validation protects against stale or manually edited scenario plans."""
    conversation = _conversation("conv", 2)
    context = _context((conversation,), _FakeTargetClient(None, None))
    target_fact_id = conversation.turns[-1].fact_ids[0]
    history_id = conversation.turns[0].anchor_scenario_id
    original = context.history_candidates_by_scenario_id[history_id]
    context = multi_turn_candidate_generation.MultiTurnPipelineContext(
        facts_by_id=context.facts_by_id,
        history_candidates_by_scenario_id={
            history_id: ReviewedHistoryCandidate(
                scenario_id=original.scenario_id,
                persona=original.persona,
                answerability=original.answerability,
                query=original.query,
                reference_answer=original.reference_answer,
                gold_chunk_ids=original.gold_chunk_ids,
                source_fact_ids=(target_fact_id,),
            )
        },
        persona_descriptions=context.persona_descriptions,
        generation_client=context.generation_client,
        generator_model=context.generator_model,
        llm_max_retries=0,
        llm_retry_delay_seconds=0,
    )

    with pytest.raises(ValueError, match="Target facts must not appear"):
        multi_turn_candidate_generation.generate_conversation_batch((conversation,), context)


def test_target_requires_a_context_dependent_query() -> None:
    """A standalone final question cannot pass as a multi-turn target."""
    turn_spec = _conversation("conv", 2).turns[-1]
    raw = GeneratedMultiTurnTarget(
        turn_number=2,
        query="Câu hỏi giống nhau",
        standalone_query="Câu hỏi giống nhau",
        reference_answer="Đáp án",
        reasoning="Căn cứ",
        missing_information=None,
        required_claims=[BamlClaimReference(claim="Ý", fact_id=turn_spec.fact_ids[0])],
    )

    with pytest.raises(ValueError, match="depend on history"):
        multi_turn_candidate_generation._target_turn_from_generated(
            turn_spec,
            raw,
            {turn_spec.fact_ids[0]: _fact(turn_spec.fact_ids[0])},
        )


def test_generated_target_text_is_preserved_before_persistence() -> None:
    """Typed model text is persisted without rule-based rewriting."""
    conversation = _conversation("conv", 2)
    context = _context((conversation,), _FakeTargetClient(None, None))
    generated = GeneratedMultiTurnTarget(
        turn_number=2,
        query="Vậy T2F1 áp dụng thế nào?",
        standalone_query="T2F1 áp dụng cho trường hợp cụ thể thế nào?",
        reference_answer="Target fact xác định kết quả.",
        reasoning="History và T2F1 dẫn tới kết luận.",
        missing_information=None,
        required_claims=[BamlClaimReference(claim="Claim theo T2F1.", fact_id="T2F1")],
    )

    draft = multi_turn_candidate_generation._conversation_from_generated(
        conversation, generated, context
    )
    target = draft.turns[-1]

    assert target.query == generated.query
    assert target.standalone_query == generated.standalone_query
    assert target.reference_answer == generated.reference_answer
    assert target.reasoning == generated.reasoning
    assert target.required_claims[0].claim == generated.required_claims[0].claim
    assert target.required_claims[0].fact_id == conversation.turns[-1].fact_ids[0]


def test_return_target_receives_the_first_history_turn_as_its_reference() -> None:
    """Return generation highlights the older topic instead of the latest turn."""
    original = _conversation("conv", 3)
    target = replace(original.turns[-1], dialogue_act="return", reference_turn_number=1)
    conversation = replace(original, turns=(*original.turns[:-1], target))
    context = _context((conversation,), _FakeTargetClient(None, None))

    item = multi_turn_candidate_generation._generation_input(conversation, context)

    assert item.target.reference_turn_number == 1
    assert item.target.reference_question.endswith("scenario-conv-1")
    assert item.target.reference_answer.endswith("scenario-conv-1")
    assert item.target.reference_facts[0].fact_id == "H1F1"


def test_partially_answerable_target_requires_missing_information() -> None:
    """A partial target must state the exact detail unavailable from evidence."""
    turn_spec = TurnSpec(
        **{
            **_conversation("conv", 2).turns[-1].to_dict(),
            "answerability": "partially_answerable",
        }
    )
    raw = GeneratedMultiTurnTarget(
        turn_number=2,
        query="Vậy trường hợp đó thì sao?",
        standalone_query="Trường hợp cụ thể đó được xử lý như thế nào?",
        reference_answer="Đáp án một phần",
        reasoning="Căn cứ chỉ xác định được một phần.",
        missing_information=None,
        required_claims=[BamlClaimReference(claim="Ý bắt buộc", fact_id=turn_spec.fact_ids[0])],
    )

    with pytest.raises(ValueError, match="must declare missing information"):
        multi_turn_candidate_generation._target_turn_from_generated(
            turn_spec,
            raw,
            {turn_spec.fact_ids[0]: _fact(turn_spec.fact_ids[0])},
        )


def test_answerable_target_rejects_missing_information() -> None:
    """A complete target cannot be mislabeled by declaring an evidence gap."""
    turn_spec = _conversation("conv", 2).turns[-1]
    raw = GeneratedMultiTurnTarget(
        turn_number=2,
        query="Vậy trường hợp đó thì sao?",
        standalone_query="Trường hợp cụ thể đó được xử lý như thế nào?",
        reference_answer="Đáp án đầy đủ",
        reasoning="Căn cứ xác định đầy đủ kết luận.",
        missing_information="mức xử lý bổ sung",
        required_claims=[BamlClaimReference(claim="Ý bắt buộc", fact_id=turn_spec.fact_ids[0])],
    )

    with pytest.raises(ValueError, match="must not declare missing information"):
        multi_turn_candidate_generation._target_turn_from_generated(
            turn_spec,
            raw,
            {turn_spec.fact_ids[0]: _fact(turn_spec.fact_ids[0])},
        )


def test_target_rejects_multiple_claims_for_one_fact() -> None:
    """One source fact must map to one merged required claim."""
    turn_spec = _conversation("conv", 2).turns[-1]
    raw = GeneratedMultiTurnTarget(
        turn_number=2,
        query="Vậy trường hợp đó thì sao?",
        standalone_query="Trường hợp cụ thể đó được xử lý như thế nào?",
        reference_answer="Đáp án",
        reasoning="Căn cứ",
        missing_information=None,
        required_claims=[
            BamlClaimReference(claim="Ý thứ nhất", fact_id=turn_spec.fact_ids[0]),
            BamlClaimReference(claim="Ý thứ hai", fact_id=turn_spec.fact_ids[0]),
        ],
    )

    with pytest.raises(ValueError, match="only one required claim"):
        multi_turn_candidate_generation._target_turn_from_generated(
            turn_spec,
            raw,
            {turn_spec.fact_ids[0]: _fact(turn_spec.fact_ids[0])},
        )


def test_valid_partial_target_persists_missing_information() -> None:
    """The explicit evidence gap survives candidate construction."""
    turn_spec = TurnSpec(
        **{
            **_conversation("conv", 2).turns[-1].to_dict(),
            "answerability": "partially_answerable",
        }
    )
    raw = GeneratedMultiTurnTarget(
        turn_number=2,
        query="Vậy mức bổ sung là bao nhiêu?",
        standalone_query="Mức xử lý bổ sung trong trường hợp cụ thể là bao nhiêu?",
        reference_answer="Căn cứ hiện có chỉ xác định hình thức xử lý chính.",
        reasoning="Nguồn hiện có chưa xác định mức bổ sung.",
        missing_information="mức xử lý bổ sung",
        required_claims=[BamlClaimReference(claim="Hình thức xử lý chính", fact_id=turn_spec.fact_ids[0])],
    )

    target = multi_turn_candidate_generation._target_turn_from_generated(
        turn_spec,
        raw,
        {turn_spec.fact_ids[0]: _fact(turn_spec.fact_ids[0])},
    )

    assert target.missing_information == "mức xử lý bổ sung"
