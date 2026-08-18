"""Load ontology policy, manifests, and RDF graphs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from rdflib import Graph

from legal_rag.ontology_validation.models import ModulePolicy, QueryCase, ValidationCase


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a trusted versioned YAML artifact."""
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_graph(path: Path) -> Graph:
    """Parse one Turtle graph."""
    return Graph().parse(path, format="turtle")


def merge_graphs(paths: tuple[Path, ...]) -> Graph:
    """Merge parsed Turtle artifacts into one graph."""
    graph = Graph()
    for path in paths:
        graph += load_graph(path)
    return graph


def load_module_policy(project_root: Path, policy_path: Path) -> tuple[ModulePolicy, ...]:
    """Load the root ontology and active module declarations."""
    policy = load_yaml(policy_path)["module_policy"]
    root = policy["root_ontology"]
    modules = [
        ModulePolicy(
            module_id=root["id"],
            path=project_root / root["file"],
            ontology_iri=root["ontology_iri"],
            version_iri=root["version_iri"],
            namespace=root["namespace"],
            imports=tuple(root["active_imports"]),
        )
    ]
    modules.extend(
        ModulePolicy(
            module_id=module["id"],
            path=project_root / module["file"],
            ontology_iri=module["ontology_iri"],
            version_iri=module["version_iri"],
            namespace=module["namespace"],
            imports=tuple(module["imports"]),
        )
        for module in policy["modules"]
        if module["formalization_status"] == "active"
    )
    return tuple(modules)


def load_query_cases(project_root: Path, manifest_path: Path) -> tuple[QueryCase, ...]:
    """Load executable CQ cases from the query manifest."""
    manifest = load_yaml(manifest_path)
    return tuple(
        QueryCase(
            cq_id=item["cq_id"],
            query_path=project_root / item["file"],
            fixture_path=project_root / item["fixture"],
            expected_columns=tuple(item["expected_columns"]),
            expected_rows=item.get("expected_rows"),
            expected_minimum_rows=item.get("expected_minimum_rows"),
            deferred_output_fields=tuple(item.get("deferred_output_fields", [])),
        )
        for item in manifest["queries"]
    )


def load_validation_cases(project_root: Path, manifest_path: Path) -> tuple[ValidationCase, ...]:
    """Load SHACL conformance cases from the query manifest."""
    manifest = load_yaml(manifest_path)
    return tuple(
        ValidationCase(
            fixture_path=project_root / item["fixture"],
            expected_conforms=item["expected_conforms"],
        )
        for item in manifest["validation_cases"]
    )
