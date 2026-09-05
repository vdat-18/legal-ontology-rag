"""Tests for deterministic ontology population gold evaluation."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from legal_rag import ontology_population_evaluation_cli
from legal_rag.ontology_population.evaluation import (
    ConsequenceExpectation,
    PopulationGoldStandard,
    evaluate_population_records,
)
from legal_rag.ontology_population.models import OntologyPopulationRecord, PopulationProgressRecord
from legal_rag.ontology_population.writer import write_population_progress, write_population_records


def _write_gold(path: Path, chunk_id: str) -> None:
    path.write_text(
        "\n".join(
            (
                'schema_version: "0.2.0"',
                "document_number: 168/2024/NĐ-CP",
                "cases:",
                f"  - source_chunk_id: {chunk_id}",
                "    expected_record_count: 1",
                "    required_records:",
                "      - id: REQUIRED-MONEY",
                "        provision_path: Điều 13 khoản 9 điểm b",
                "        conduct_label_vi: Điều khiển xe không gắn biển số",
                "        consequences:",
                "          - kind: MonetaryPenalty",
                '            minimum_amount: "4000000"',
                '            maximum_amount: "6000000"',
                "    forbidden_records:",
                "      - id: FORBIDDEN-MONEY",
                "        provision_path: Điều 13 khoản 9 điểm b",
                "        consequences:",
                "          - kind: MonetaryPenalty",
                '            minimum_amount: "100"',
                '            maximum_amount: "200"',
            )
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("kind", ["Warning", "ReferencedPenalty"])
def test_population_gold_accepts_extended_consequence_kinds(kind: str) -> None:
    """Gold assertions can cover every consequence kind emitted by population."""
    assert ConsequenceExpectation.from_dict({"kind": kind}).kind == kind


def test_population_gold_evaluation_passes_exact_structured_assertions(
    tmp_path: Path, population_record: OntologyPopulationRecord
) -> None:
    """Required and forbidden checks compare typed fields without text matching heuristics."""
    gold_path = tmp_path / "gold.yaml"
    _write_gold(gold_path, population_record.source_chunk_id)

    report = evaluate_population_records(
        (population_record,),
        (PopulationProgressRecord(population_record.source_chunk_id, "test-run"),),
        PopulationGoldStandard.from_path(gold_path),
    )

    assert report.overall_status == "passed"
    assert report.expected_record_count == 1
    assert report.issues == ()


def test_population_gold_evaluation_reports_missing_and_forbidden_records(
    tmp_path: Path, population_record: OntologyPopulationRecord
) -> None:
    """Incomplete progress and present forbidden signatures fail with explicit issue IDs."""
    gold_path = tmp_path / "gold.yaml"
    _write_gold(gold_path, population_record.source_chunk_id)
    gold_text = (
        gold_path.read_text(encoding="utf-8")
        .replace('minimum_amount: "100"', 'minimum_amount: "4000000"')
        .replace('maximum_amount: "200"', 'maximum_amount: "6000000"')
    )
    gold_path.write_text(gold_text, encoding="utf-8")

    report = evaluate_population_records((population_record,), (), PopulationGoldStandard.from_path(gold_path))

    assert report.overall_status == "failed"
    assert {issue.check_id for issue in report.issues} == {"EVAL-TARGET-001", "EVAL-FORBIDDEN-001"}


def test_population_gold_evaluation_reports_over_generated_records(
    tmp_path: Path, population_record: OntologyPopulationRecord
) -> None:
    """Exact reviewed inventory rejects extra records even when required assertions remain present."""
    gold_path = tmp_path / "gold.yaml"
    _write_gold(gold_path, population_record.source_chunk_id)
    extra_record = replace(population_record, population_id="unexpected-record")

    report = evaluate_population_records(
        (population_record, extra_record),
        (PopulationProgressRecord(population_record.source_chunk_id, "test-run"),),
        PopulationGoldStandard.from_path(gold_path),
    )

    assert report.overall_status == "failed"
    assert {issue.check_id for issue in report.issues} == {"EVAL-COVERAGE-002"}


def test_population_gold_requires_distinct_records_for_repeated_signatures(
    tmp_path: Path, population_record: OntologyPopulationRecord
) -> None:
    """Two expected variants cannot both be satisfied by one actual record."""
    gold_path = tmp_path / "gold.yaml"
    _write_gold(gold_path, population_record.source_chunk_id)
    gold_text = gold_path.read_text(encoding="utf-8")
    repeated_assertion = "\n".join(
        (
            "      - id: REQUIRED-MONEY-VARIANT",
            "        provision_path: Điều 13 khoản 9 điểm b",
            "        conduct_label_vi: Điều khiển xe không gắn biển số",
            "        consequences:",
            "          - kind: MonetaryPenalty",
            '            minimum_amount: "4000000"',
            '            maximum_amount: "6000000"',
        )
    )
    gold_path.write_text(
        gold_text.replace("    expected_record_count: 1", "    expected_record_count: 2").replace(
            "    forbidden_records:", f"{repeated_assertion}\n    forbidden_records:"
        ),
        encoding="utf-8",
    )

    report = evaluate_population_records(
        (population_record,),
        (PopulationProgressRecord(population_record.source_chunk_id, "test-run"),),
        PopulationGoldStandard.from_path(gold_path),
    )

    assert {issue.check_id for issue in report.issues} == {"EVAL-COVERAGE-001", "EVAL-REQUIRED-001"}
    assert [issue.assertion_id for issue in report.issues if issue.check_id == "EVAL-REQUIRED-001"] == [
        "REQUIRED-MONEY-VARIANT"
    ]


def test_population_evaluation_cli_writes_report_before_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, population_record: OntologyPopulationRecord
) -> None:
    """The CLI always persists an actionable report before failing a gold gate."""
    records_path = tmp_path / "records.jsonl"
    progress_path = tmp_path / "progress.jsonl"
    gold_path = tmp_path / "gold.yaml"
    report_path = tmp_path / "report.json"
    write_population_records(records_path, (population_record,))
    write_population_progress(progress_path, ())
    _write_gold(gold_path, population_record.source_chunk_id)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "legal-evaluate-ontology-population",
            str(records_path),
            str(progress_path),
            str(gold_path),
            str(report_path),
        ],
    )

    with pytest.raises(ValueError, match="evaluation failed"):
        ontology_population_evaluation_cli.main()

    assert json.loads(report_path.read_text(encoding="utf-8"))["overall_status"] == "failed"


def test_population_gold_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    """A target chunk can appear only once in one gold artifact."""
    gold_path = tmp_path / "gold.yaml"
    _write_gold(gold_path, "chunk-1")
    content = gold_path.read_text(encoding="utf-8")
    duplicate_case = "\n".join(content.splitlines()[3:])
    gold_path.write_text(f"{content}\n{duplicate_case}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate source chunk IDs"):
        PopulationGoldStandard.from_path(gold_path)


def test_population_gold_rejects_unused_schema_fields(tmp_path: Path) -> None:
    """Every persisted gold field must have explicit downstream semantics."""
    gold_path = tmp_path / "gold.yaml"
    _write_gold(gold_path, "chunk-1")
    gold_path.write_text(f"{gold_path.read_text(encoding='utf-8')}\nunused: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown gold standard fields: unused"):
        PopulationGoldStandard.from_path(gold_path)


def test_population_gold_rejects_more_required_assertions_than_records(tmp_path: Path) -> None:
    """A distinct required inventory cannot be larger than the reviewed total."""
    gold_path = tmp_path / "gold.yaml"
    _write_gold(gold_path, "chunk-1")
    content = gold_path.read_text(encoding="utf-8")
    duplicate = "\n".join(
        (
            "      - id: REQUIRED-SECOND",
            "        provision_path: Điều 13 khoản 9 điểm c",
        )
    )
    gold_path.write_text(
        content.replace("    forbidden_records:", f"{duplicate}\n    forbidden_records:"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="required assertions cannot exceed"):
        PopulationGoldStandard.from_path(gold_path)
