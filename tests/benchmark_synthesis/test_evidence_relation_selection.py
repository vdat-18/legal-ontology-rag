"""Tests for selective evidence-relation confirmation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.benchmark_synthesis.evidence_graph import (
    EvidenceRelation,
    EvidenceRelationProgressRecord,
)
from legal_rag.benchmark_synthesis.evidence_relation_selection import (
    load_reconfirm_pairs,
    validate_selective_confirmation,
)
from legal_rag.evidence_graph_cli import _validate_resume_state


def test_load_reconfirm_pairs_combines_inline_and_file_selections(tmp_path: Path) -> None:
    """Pair selections are canonicalized and deduplicated across both CLI sources."""
    selections_path = tmp_path / "pairs.json"
    selections_path.write_text(json.dumps(["fact-d::fact-c"]), encoding="utf-8")

    pairs = load_reconfirm_pairs(["fact-b::fact-a"], selections_path)

    assert pairs == frozenset({("fact-a", "fact-b"), ("fact-c", "fact-d")})


def test_validate_selective_confirmation_requires_complete_unselected_progress() -> None:
    """A selective run cannot silently omit a candidate pair outside its selection."""
    candidates = frozenset({("fact-a", "fact-b"), ("fact-a", "fact-c")})

    with pytest.raises(ValueError, match="prior progress"):
        validate_selective_confirmation(candidates, frozenset(), frozenset({("fact-a", "fact-b")}))


def test_validate_resume_state_invalidates_only_selected_pair() -> None:
    """Selective confirmation keeps old per-pair provenance while removing selected outcomes."""
    selected = ("fact-a", "fact-b")
    retained = ("fact-a", "fact-c")
    relations = (
        EvidenceRelation(
            relation_id="relation-selected",
            fact_ids=selected,
            bundle_kind="multi_hop",
            bridge="Selected",
            confirmed_by_model="old-model",
            run_id="old-run",
            prompt_version="5.0.0",
        ),
        EvidenceRelation(
            relation_id="relation-retained",
            fact_ids=retained,
            bundle_kind="multi_hop",
            bridge="Retained",
            confirmed_by_model="old-model",
            run_id="old-run",
            prompt_version="5.0.0",
        ),
    )
    progress = tuple(
        EvidenceRelationProgressRecord(
            fact_id_a=pair[0],
            fact_id_b=pair[1],
            run_id="old-run",
            prompt_version="5.0.0",
            confirmed_by_model="old-model",
        )
        for pair in (selected, retained)
    )

    relations_by_id, progress_by_pair = _validate_resume_state(
        relations,
        progress,
        (selected, retained),
        "current-run",
        frozenset({selected}),
    )

    assert set(relations_by_id) == {"relation-retained"}
    assert set(progress_by_pair) == {retained}
    assert progress_by_pair[retained].prompt_version == "5.0.0"
