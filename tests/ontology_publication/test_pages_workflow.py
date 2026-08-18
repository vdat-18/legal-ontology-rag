"""Tests for the manually triggered ontology Pages deployment."""

from __future__ import annotations

from pathlib import Path


def test_pages_workflow_is_manual_least_privilege_and_commit_pinned(project_root: Path) -> None:
    """Deployment cannot run on push and uses pinned official actions with only required permissions."""
    workflow = (project_root / ".github" / "workflows" / "ontology-pages.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "push:" not in workflow
    assert "contents: read" in workflow
    assert "pages: write" in workflow
    assert "id-token: write" in workflow
    assert "actions/checkout@11d5960a326750d5838078e36cf38b85af677262" in workflow
    assert "astral-sh/setup-uv@d0d8abe699bfb85fec6de9f7adb5ae17292296ff" in workflow
    assert "actions/deploy-pages@d6db90164ac5ed86f2b6aed7e0febac5b3c0c03e" in workflow
    assert "legal-build-ontology-publication" in workflow
