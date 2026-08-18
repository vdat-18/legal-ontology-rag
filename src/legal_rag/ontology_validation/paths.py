"""Repository paths for ontology validation artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OntologyPaths:
    """Resolved paths needed by the ontology QA suite."""

    project_root: Path

    @property
    def ontology_dir(self) -> Path:
        """Return the ontology artifact directory."""
        return self.project_root / "ontology"

    @property
    def policy(self) -> Path:
        """Return the formalization policy path."""
        return self.ontology_dir / "formalization_policy.yaml"

    @property
    def query_manifest(self) -> Path:
        """Return the CQ query manifest path."""
        return self.ontology_dir / "tests" / "cq_queries" / "query_manifest.yaml"

    @property
    def shapes(self) -> Path:
        """Return the SHACL shapes path."""
        return self.ontology_dir / "shapes" / "legal_ontology.shacl.ttl"

    @property
    def catalog(self) -> Path:
        """Return the XML catalog path."""
        return self.ontology_dir / "catalog-v001.xml"

    @property
    def competency_questions(self) -> Path:
        """Return the competency-question registry path."""
        return self.ontology_dir / "requirements" / "competency_questions.yaml"

    @property
    def requirements(self) -> Path:
        """Return the ontology-requirement registry path."""
        return self.ontology_dir / "requirements" / "ontology_requirements.yaml"

    @property
    def glossary(self) -> Path:
        """Return the glossary registry path."""
        return self.ontology_dir / "glossary" / "terms.yaml"

    @property
    def modeling_decisions(self) -> Path:
        """Return the modeling-decision registry path."""
        return self.ontology_dir / "decisions" / "modeling_decisions.yaml"

    @property
    def semantic_qa_contract(self) -> Path:
        """Return the ontology semantic-QA contract path."""
        return self.ontology_dir / "semantic_qa_contract.yaml"

    @property
    def release_contract(self) -> Path:
        """Return the ontology release contract path."""
        return self.ontology_dir / "release_contract.yaml"

    @property
    def namespace_publication_contract(self) -> Path:
        """Return the ontology namespace-publication contract path."""
        return self.ontology_dir / "namespace_publication.yaml"

    @property
    def external_reuse_registry(self) -> Path:
        """Return the ontology external-reuse registry path."""
        return self.ontology_dir / "external_reuse_registry.yaml"

    @property
    def legal_review_contract(self) -> Path:
        """Return the internal legal-review contract path."""
        return self.ontology_dir / "legal_review_contract.yaml"

    @property
    def legal_review_decisions(self) -> Path:
        """Return the internal legal-review decisions path."""
        return self.ontology_dir / "legal_review_decisions.yaml"
