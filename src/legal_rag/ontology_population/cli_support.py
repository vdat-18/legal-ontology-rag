"""Shared input preparation for independently runnable population steps."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from legal_rag.chunker.models import Chunk
from legal_rag.io_utils import write_json_atomic
from legal_rag.ontology_population.config import PopulationSettings
from legal_rag.ontology_population.preflight import (
    PopulationInputBundle,
    load_population_inputs,
    load_population_target_selection,
    population_ontology_semantic_sha256,
    run_population_preflight,
    select_population_targets,
)


@dataclass(frozen=True, slots=True)
class PreparedPopulationStep:
    """Validated source inputs shared by the two population LLM steps."""

    bundle: PopulationInputBundle
    settings: PopulationSettings
    targets: tuple[Chunk, ...]
    pilot_run_id: str
    ontology_semantic_sha256: str


def prepare_population_step(
    *,
    chunks_path: Path,
    screening_path: Path,
    metadata_path: Path,
    contract_path: Path,
    target_limit: int | None,
    target_ids_path: Path | None,
    preflight_report_path: Path | None,
) -> PreparedPopulationStep:
    """Run the shared structural preflight and select exact target chunks."""
    project_root = Path(__file__).resolve().parents[3]
    ontology_semantic_sha256 = population_ontology_semantic_sha256(project_root)
    settings = PopulationSettings.from_path(contract_path)
    bundle = load_population_inputs(chunks_path, screening_path, metadata_path)
    target_selection = load_population_target_selection(target_ids_path) if target_ids_path is not None else None
    artifact_paths = [
        ("chunks", chunks_path),
        ("screening", screening_path),
        ("metadata", metadata_path),
        ("population_contract", contract_path),
    ]
    if target_ids_path is not None:
        artifact_paths.append(("target_selection", target_ids_path))
    preflight = run_population_preflight(
        bundle,
        settings,
        tuple(artifact_paths),
        datetime.now(UTC),
        ontology_semantic_sha256=ontology_semantic_sha256,
        target_limit=target_limit,
        target_selection=target_selection,
    )
    if preflight_report_path is not None:
        write_json_atomic(preflight_report_path, preflight.to_dict())
    if preflight.overall_status != "ready":
        report_hint = f"; inspect {preflight_report_path}" if preflight_report_path else ""
        msg = f"Ontology population inputs are blocked{report_hint}"
        raise ValueError(msg)
    return PreparedPopulationStep(
        bundle=bundle,
        settings=settings,
        targets=select_population_targets(bundle, target_limit, target_selection),
        pilot_run_id=preflight.pilot_run_id,
        ontology_semantic_sha256=ontology_semantic_sha256,
    )
