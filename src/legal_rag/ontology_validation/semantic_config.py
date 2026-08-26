"""Typed configuration for ontology semantic QA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from legal_rag.ontology_validation.loader import load_yaml
from legal_rag.ontology_validation.semantic_models import SEMANTIC_QA_REPORT_SCHEMA_VERSION

SEMANTIC_QA_CONTRACT_SCHEMA_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class SemanticQaSettings:
    """External tool, semantic audit, and report settings."""

    container_engine: str
    image: str
    command: str
    network: str
    memory: str
    cpus: str
    timeout_seconds: float
    ontology_path: Path
    catalog_path: Path
    profile: str
    reasoner: str
    equivalent_classes_allowed: str
    axiom_generators: str
    fail_on_unsatisfiable_classes: bool
    fail_on_unexpected_equivalences: bool
    quality_format: str
    quality_fail_on: str
    require_resolved_references: bool
    require_local_term_traceability: bool
    require_named_shape_traceability: bool
    require_mandatory_cq_coverage: bool
    require_ontology_pattern_coverage: bool
    profile_report_file: str
    reasoned_ontology_file: str
    quality_report_file: str

    @classmethod
    def from_path(cls, path: Path, project_root: Path) -> SemanticQaSettings:
        """Load the versioned semantic-QA contract."""
        document = load_yaml(path)
        if document["schema_version"] != SEMANTIC_QA_CONTRACT_SCHEMA_VERSION:
            msg = f"Unsupported semantic-QA contract: {document['schema_version']}"
            raise ValueError(msg)
        tool = document["tool"]
        inputs = document["input"]
        profile = document["profile_check"]
        reasoning = document["reasoning"]
        quality = document["quality_report"]
        traceability = document["traceability"]
        output = document["output"]
        if output["report_schema_version"] != SEMANTIC_QA_REPORT_SCHEMA_VERSION:
            msg = f"Unsupported semantic-QA report schema: {output['report_schema_version']}"
            raise ValueError(msg)
        return cls(
            container_engine=tool["container_engine"],
            image=tool["image"],
            command=tool["command"],
            network=tool["network"],
            memory=tool["memory"],
            cpus=tool["cpus"],
            timeout_seconds=tool["timeout_seconds"],
            ontology_path=project_root / inputs["ontology"],
            catalog_path=project_root / inputs["catalog"],
            profile=profile["profile"],
            reasoner=reasoning["reasoner"],
            equivalent_classes_allowed=reasoning["equivalent_classes_allowed"],
            axiom_generators=reasoning["axiom_generators"],
            fail_on_unsatisfiable_classes=reasoning["fail_on_unsatisfiable_classes"],
            fail_on_unexpected_equivalences=reasoning["fail_on_unexpected_equivalences"],
            quality_format=quality["format"],
            quality_fail_on=quality["fail_on"],
            require_resolved_references=traceability["require_resolved_references"],
            require_local_term_traceability=traceability["require_local_term_traceability"],
            require_named_shape_traceability=traceability["require_named_shape_traceability"],
            require_mandatory_cq_coverage=traceability["require_mandatory_cq_coverage"],
            require_ontology_pattern_coverage=traceability["require_ontology_pattern_coverage"],
            profile_report_file=output["profile_report_file"],
            reasoned_ontology_file=output["reasoned_ontology_file"],
            quality_report_file=output["quality_report_file"],
        )
