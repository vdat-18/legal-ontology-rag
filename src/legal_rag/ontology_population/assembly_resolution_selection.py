"""Parse explicit component selections for assembly-resolution reruns."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from legal_rag.ontology_population.assembly_models import AssemblyConflictComponent, AssemblyResolutionProgress

ASSEMBLY_RESOLUTION_SELECTION_SCHEMA_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class AssemblyResolutionSelection:
    """Reviewed component IDs selected for one resolution replay."""

    component_ids: tuple[str, ...]


def load_assembly_resolution_selection(path: Path) -> AssemblyResolutionSelection:
    """Load one versioned component-rerun selection."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["schema_version"] != ASSEMBLY_RESOLUTION_SELECTION_SCHEMA_VERSION:
        msg = f"Unsupported assembly resolution selection schema: {payload['schema_version']}"
        raise ValueError(msg)
    component_ids = payload["component_ids"]
    if not isinstance(component_ids, list) or not all(isinstance(value, str) for value in component_ids):
        msg = "Assembly resolution selection component_ids must be a JSON string array"
        raise ValueError(msg)
    return AssemblyResolutionSelection(component_ids=tuple(component_ids))


def select_assembly_resolution_rerun_components(
    components: Sequence[AssemblyConflictComponent],
    progress: Sequence[AssemblyResolutionProgress],
    selection: AssemblyResolutionSelection,
) -> tuple[AssemblyConflictComponent, ...]:
    """Require a complete prior review and return explicitly selected components."""
    if not selection.component_ids:
        msg = "Assembly resolution rerun selection must contain at least one component ID"
        raise ValueError(msg)
    if len(set(selection.component_ids)) != len(selection.component_ids):
        msg = "Assembly resolution rerun component IDs must be unique"
        raise ValueError(msg)

    components_by_id = {component.component_id: component for component in components}
    unknown_ids = set(selection.component_ids) - set(components_by_id)
    if unknown_ids:
        msg = f"Assembly resolution rerun selection contains an unknown component: {sorted(unknown_ids)[0]}"
        raise ValueError(msg)

    progress_ids = [item.component_id for item in progress]
    if len(set(progress_ids)) != len(progress_ids) or set(progress_ids) != set(components_by_id):
        msg = "Selective assembly resolution requires prior progress for every component"
        raise ValueError(msg)
    selected_ids = set(selection.component_ids)
    return tuple(component for component in components if component.component_id in selected_ids)
