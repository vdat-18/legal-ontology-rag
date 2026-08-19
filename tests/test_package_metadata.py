"""Tests for package extras and console scripts."""

from __future__ import annotations

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_corrective_rag_extra_declares_runtime_dependencies() -> None:
    """The CRAG extra installs the direct dependencies used by its API and UI."""
    metadata = _project_metadata()
    dependencies = set(metadata["optional-dependencies"]["corrective-rag"])

    assert {
        "httpx>=0.27",
        "pymilvus>=3.0",
        "sentence-transformers>=3.0",
        "streamlit>=1.37",
        "torch>=2.1",
        "uvicorn>=0.30",
    } <= dependencies


def test_crag_console_script_points_to_fastapi_entrypoint() -> None:
    """The CRAG API console script is published by the package metadata."""
    metadata = _project_metadata()

    assert metadata["scripts"]["legal-crag-api"] == "legal_rag.corrective_rag.api.server:main"


def test_groundtruth_export_console_script_is_published() -> None:
    """The synthesis exporter is available as a package console script."""
    metadata = _project_metadata()

    assert metadata["scripts"]["legal-export-groundtruth"] == "legal_rag.groundtruth_export_cli:main"


def test_groundtruth_document_preparation_console_script_is_published() -> None:
    """Concurrent synthesis document preparation is available as a console script."""
    metadata = _project_metadata()

    assert metadata["scripts"]["legal-prepare-groundtruth-document"] == "legal_rag.document_preparation_cli:main"


def test_evaluation_extra_excludes_unused_dependencies() -> None:
    """The evaluation extra only declares dependencies used by project code."""
    metadata = _project_metadata()
    dependencies = set(metadata["optional-dependencies"]["evaluation"])

    assert {"rapidfuzz>=3.0", "sentence-transformers>=3.0"} <= dependencies
    assert not {
        "deepeval",
        "langchain-community",
        "langchain-openai",
        "ragas",
    } & {dependency.split("==", maxsplit=1)[0].split(">=", maxsplit=1)[0] for dependency in dependencies}


def _project_metadata() -> dict[str, object]:
    """Load project metadata from pyproject.toml."""
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)["project"]
