"""Shared fixtures for ontology evolution tests."""

from __future__ import annotations

from pathlib import Path

import yaml

_NAMESPACE = "https://example.test/ontology#"


def write_ontology_project(
    root: Path,
    *,
    terms: dict[str, str],
    deprecated_terms: tuple[str, ...] = (),
    replacements: dict[str, str] | None = None,
    shapes: tuple[str, ...] = (),
    competency_questions: dict[str, str] | None = None,
    ontology_statements: tuple[str, ...] = (),
    shape_statements: tuple[str, ...] = (),
) -> None:
    """Write the minimal formal artifacts consumed by the semantic diff."""
    ontology_directory = root / "ontology"
    shapes_directory = ontology_directory / "shapes"
    requirements_directory = ontology_directory / "requirements"
    shapes_directory.mkdir(parents=True)
    requirements_directory.mkdir(parents=True)
    policy = {
        "module_policy": {
            "root_ontology": {
                "id": "MOD-ROOT",
                "file": "ontology/legal_ontology.ttl",
                "ontology_iri": "https://example.test/ontology",
                "version_iri": "https://example.test/ontology/0.1.0",
                "namespace": _NAMESPACE,
                "active_imports": [],
            },
            "modules": [],
        }
    }
    (ontology_directory / "formalization_policy.yaml").write_text(
        yaml.safe_dump(policy, sort_keys=False),
        encoding="utf-8",
    )
    declarations = {
        "class": "owl:Class",
        "datatype_property": "owl:DatatypeProperty",
        "object_property": "owl:ObjectProperty",
    }
    lines = [
        "@prefix dcterms: <http://purl.org/dc/terms/> .",
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix test: <https://example.test/ontology#> .",
        "<https://example.test/ontology> a owl:Ontology .",
    ]
    for local_name, kind in terms.items():
        lines.append(f"test:{local_name} a {declarations[kind]} .")
    lines.extend(f"test:{local_name} owl:deprecated true ." for local_name in deprecated_terms)
    lines.extend(
        f"test:{source} dcterms:isReplacedBy test:{replacement} ."
        for source, replacement in (replacements or {}).items()
    )
    lines.extend(ontology_statements)
    (ontology_directory / "legal_ontology.ttl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shape_lines = [
        "@prefix sh: <http://www.w3.org/ns/shacl#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix shape: <https://example.test/shapes#> .",
        "@prefix test: <https://example.test/ontology#> .",
        *(f"shape:{local_name} a sh:NodeShape ." for local_name in shapes),
        *shape_statements,
    ]
    (shapes_directory / "legal_ontology.shacl.ttl").write_text(
        "\n".join(shape_lines) + "\n",
        encoding="utf-8",
    )
    cq_document = {
        "competency_questions": [
            {"id": cq_id, "priority": priority} for cq_id, priority in (competency_questions or {}).items()
        ]
    }
    (requirements_directory / "competency_questions.yaml").write_text(
        yaml.safe_dump(cq_document, sort_keys=False),
        encoding="utf-8",
    )
