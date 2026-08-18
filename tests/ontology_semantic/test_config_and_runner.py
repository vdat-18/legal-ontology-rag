"""Tests for semantic-QA configuration, packaging, and safe tool execution."""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

from rdflib import OWL, RDF, URIRef

from legal_rag.ontology_validation.loader import load_graph
from legal_rag.ontology_validation.semantic_config import SemanticQaSettings
from legal_rag.ontology_validation.semantic_runner import DockerRobotRunner


def test_semantic_contract_and_cli_are_versioned_and_packaged(
    project_root: Path,
    semantic_settings: SemanticQaSettings,
) -> None:
    """The pinned contract exposes bounded OWL profile, HermiT, and quality checks."""
    scripts = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]

    assert semantic_settings.image == "obolibrary/robot:v1.9.10"
    assert semantic_settings.profile == "DL"
    assert semantic_settings.reasoner == "HermiT"
    assert semantic_settings.timeout_seconds == 300
    assert scripts["legal-validate-ontology-semantics"] == "legal_rag.ontology_semantic_qa_cli:main"


def test_external_reference_terms_are_declared_in_the_foundational_import(project_root: Path) -> None:
    """Referenced external entities remain declared without importing external ontology bundles."""
    graph = load_graph(project_root / "ontology" / "modules" / "source_provenance.ttl")

    assert (URIRef("http://www.w3.org/ns/prov#Entity"), RDF.type, OWL.Class) in graph
    assert (URIRef("http://www.w3.org/2004/02/skos/core#Concept"), RDF.type, OWL.Class) in graph
    assert (URIRef("http://www.w3.org/2006/time#Instant"), RDF.type, OWL.Class) in graph
    assert (
        URIRef("http://purl.org/dc/terms/title"),
        RDF.type,
        OWL.AnnotationProperty,
    ) in graph
    assert (
        URIRef("http://purl.org/dc/terms/license"),
        RDF.type,
        OWL.AnnotationProperty,
    ) in graph


def test_docker_runner_uses_argument_vector_read_only_workspace_and_no_network(
    monkeypatch,
    tmp_path: Path,
    project_root: Path,
    semantic_settings: SemanticQaSettings,
) -> None:
    """External execution is bounded and never constructs a shell command string."""
    captured = {}

    def fake_run(arguments, **kwargs):
        captured["arguments"] = arguments
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(arguments, 0, "ok", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = DockerRobotRunner(project_root, semantic_settings).run(("--version",), tmp_path)

    arguments = captured["arguments"]
    assert isinstance(arguments, tuple)
    assert arguments[:5] == ("docker", "run", "--rm", "--network", "none")
    assert any(value.endswith("target=/work,readonly") for value in arguments)
    assert semantic_settings.image in arguments
    assert captured["kwargs"]["check"] is False
    assert captured["kwargs"]["timeout"] == 300
    assert result.exit_code == 0
