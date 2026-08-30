"""Tests for ontology semantic evolution diffing."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from legal_rag.ontology_validation.evolution import compare_ontology_projects
from legal_rag.ontology_validation.evolution_config import OntologyEvolutionSettings
from tests.ontology_evolution.conftest import write_ontology_project


def _settings() -> OntologyEvolutionSettings:
    project_root = Path(__file__).resolve().parents[2]
    return OntologyEvolutionSettings.from_path(project_root / "ontology" / "evolution_policy.yaml")


def test_additions_and_retained_deprecation_are_non_breaking(tmp_path: Path) -> None:
    """The gate reports additive changes while allowing a declared old IRI to be deprecated."""
    before_root = tmp_path / "before"
    after_root = tmp_path / "after"
    write_ontology_project(
        before_root,
        terms={"Rule": "class", "oldProperty": "object_property"},
        shapes=("RuleShape",),
        competency_questions=("CQ-RULE-001",),
    )
    write_ontology_project(
        after_root,
        terms={"Rule": "class", "oldProperty": "object_property", "newProperty": "object_property"},
        deprecated_terms=("oldProperty",),
        replacements={"oldProperty": "newProperty"},
        shapes=("RuleShape", "NewShape"),
        competency_questions=("CQ-RULE-001", "CQ-NEW-001"),
    )

    report = compare_ontology_projects(before_root, after_root, _settings(), datetime(2026, 8, 26, tzinfo=UTC))

    assert report.overall_status == "passed"
    assert [(item.iri, item.kinds) for item in report.added_terms] == [
        ("https://example.test/ontology#newProperty", ("object_property",))
    ]
    assert report.added_node_shape_iris == ("https://example.test/shapes#NewShape",)
    assert report.added_mandatory_cq_ids == ("CQ-NEW-001",)
    assert report.newly_deprecated_term_iris == ("https://example.test/ontology#oldProperty",)
    assert [item.to_dict() for item in report.replacements] == [
        {
            "source_iri": "https://example.test/ontology#oldProperty",
            "replacement_iri": "https://example.test/ontology#newProperty",
        }
    ]
    assert report.issues == ()


def test_breaking_changes_fail_with_exact_subjects(tmp_path: Path) -> None:
    """Active term removal, kind changes, shape removal, and CQ removal are independently visible."""
    before_root = tmp_path / "before"
    after_root = tmp_path / "after"
    write_ontology_project(
        before_root,
        terms={"Rule": "class", "hasPenalty": "object_property"},
        shapes=("RuleShape", "PenaltyShape"),
        competency_questions=("CQ-RULE-001", "CQ-PENALTY-001"),
    )
    write_ontology_project(
        after_root,
        terms={"Rule": "object_property"},
        shapes=("RuleShape",),
        competency_questions=("CQ-PENALTY-001",),
    )

    report = compare_ontology_projects(before_root, after_root, _settings(), datetime(2026, 8, 26, tzinfo=UTC))

    assert report.overall_status == "failed"
    assert {(item.check_id, item.subject) for item in report.issues} == {
        ("changed_term_kind", "https://example.test/ontology#Rule"),
        ("removed_active_term", "https://example.test/ontology#hasPenalty"),
        ("removed_node_shape", "https://example.test/shapes#PenaltyShape"),
        ("removed_mandatory_cq", "CQ-RULE-001"),
    }


def test_deprecated_term_must_remain_declared(tmp_path: Path) -> None:
    """Stable deprecated IRIs remain resolvable instead of disappearing in a later version."""
    before_root = tmp_path / "before"
    after_root = tmp_path / "after"
    write_ontology_project(
        before_root,
        terms={"OldRule": "class"},
        deprecated_terms=("OldRule",),
    )
    write_ontology_project(after_root, terms={})

    report = compare_ontology_projects(before_root, after_root, _settings(), datetime(2026, 8, 26, tzinfo=UTC))

    assert report.overall_status == "failed"
    assert [item.iri for item in report.removed_terms] == ["https://example.test/ontology#OldRule"]
    assert [(item.check_id, item.subject) for item in report.issues] == [
        ("removed_deprecated_term", "https://example.test/ontology#OldRule")
    ]


def test_changed_owl_axioms_and_shacl_constraints_are_breaking(tmp_path: Path) -> None:
    """Retained names do not hide changed domains, ranges, restrictions, or SHACL cardinalities."""
    before_root = tmp_path / "before"
    after_root = tmp_path / "after"
    common_terms = {"Rule": "class", "Penalty": "class", "hasPenalty": "object_property"}
    write_ontology_project(
        before_root,
        terms=common_terms,
        shapes=("RuleShape",),
        ontology_statements=(
            "test:hasPenalty rdfs:domain test:Rule ; rdfs:range test:Penalty .",
            "test:Rule rdfs:subClassOf [ a owl:Restriction ; owl:onProperty test:hasPenalty ; owl:minCardinality 1 ] .",
        ),
        shape_statements=("shape:RuleShape sh:property [ sh:path test:hasPenalty ; sh:minCount 1 ] .",),
    )
    write_ontology_project(
        after_root,
        terms=common_terms,
        shapes=("RuleShape",),
        ontology_statements=(
            "test:hasPenalty rdfs:domain test:Penalty ; rdfs:range test:Rule .",
            "test:Rule rdfs:subClassOf [ a owl:Restriction ; owl:onProperty test:hasPenalty ; owl:minCardinality 2 ] .",
        ),
        shape_statements=("shape:RuleShape sh:property [ sh:path test:hasPenalty ; sh:minCount 2 ] .",),
    )

    report = compare_ontology_projects(before_root, after_root, _settings(), datetime(2026, 8, 26, tzinfo=UTC))

    assert report.overall_status == "failed"
    assert {item.iri for item in report.changed_term_semantics} == {
        "https://example.test/ontology#Rule",
        "https://example.test/ontology#hasPenalty",
    }
    assert [item.iri for item in report.changed_node_shape_constraints] == ["https://example.test/shapes#RuleShape"]
    assert {(item.check_id, item.subject) for item in report.issues} == {
        ("changed_active_term_semantics", "https://example.test/ontology#Rule"),
        ("changed_active_term_semantics", "https://example.test/ontology#hasPenalty"),
        ("changed_node_shape_constraints", "https://example.test/shapes#RuleShape"),
    }


def test_labels_comments_and_traceability_do_not_change_semantic_hashes(tmp_path: Path) -> None:
    """Documentation edits remain visible in Git without being classified as semantic breaking changes."""
    before_root = tmp_path / "before"
    after_root = tmp_path / "after"
    write_ontology_project(
        before_root,
        terms={"Rule": "class"},
        shapes=("RuleShape",),
        ontology_statements=('test:Rule rdfs:label "Old label" ; rdfs:comment "Old comment" .',),
        shape_statements=('shape:RuleShape rdfs:label "Old shape label" .',),
    )
    write_ontology_project(
        after_root,
        terms={"Rule": "class"},
        shapes=("RuleShape",),
        ontology_statements=('test:Rule rdfs:label "New label" ; rdfs:comment "New comment" .',),
        shape_statements=('shape:RuleShape rdfs:label "New shape label" .',),
    )

    report = compare_ontology_projects(before_root, after_root, _settings(), datetime(2026, 8, 26, tzinfo=UTC))

    assert report.overall_status == "passed"
    assert report.changed_term_semantics == ()
    assert report.changed_node_shape_constraints == ()
    assert report.issues == ()
