"""Build bounded structural candidates for downstream rule-assembly confirmation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from legal_rag.chunker.models import Chunk
from legal_rag.ontology_population.assembly_config import AssemblySettings
from legal_rag.ontology_population.assembly_models import AssemblyCandidateSet, AssemblyCandidateWindow
from legal_rag.ontology_population.models import OntologyPopulationRecord, population_identity


def build_assembly_candidates(
    records: Sequence[OntologyPopulationRecord],
    chunks: Sequence[Chunk],
    settings: AssemblySettings,
) -> AssemblyCandidateSet:
    """Build proximity-only candidate windows without making a merge decision."""
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    records_by_document_scope: dict[tuple[str, str], list[OntologyPopulationRecord]] = defaultdict(list)
    for record in records:
        chunk = chunks_by_id[record.source_chunk_id]
        records_by_document_scope[(record.document_number, chunk.scope_id)].append(record)

    unique_windows: dict[tuple[str, ...], AssemblyCandidateWindow] = {}
    for document_records in records_by_document_scope.values():
        ordered = sorted(
            document_records,
            key=lambda record: (chunks_by_id[record.source_chunk_id].chunk_index, record.population_id),
        )
        for start, anchor in enumerate(ordered):
            anchor_index = chunks_by_id[anchor.source_chunk_id].chunk_index
            selected = tuple(
                record
                for record in ordered[start:]
                if chunks_by_id[record.source_chunk_id].chunk_index - anchor_index <= settings.max_chunk_span
            )[: settings.max_records_per_window]
            if len(selected) < 2:
                continue
            population_ids = tuple(record.population_id for record in selected)
            identity_members = tuple(sorted(population_ids))
            unique_windows[identity_members] = AssemblyCandidateWindow(
                candidate_id=population_identity(
                    {"candidate_version": settings.candidate_version, "population_ids": identity_members}
                ),
                population_ids=population_ids,
                source_chunk_ids=tuple(record.source_chunk_id for record in selected),
                provision_paths=tuple(record.provision_path for record in selected),
            )

    windows = tuple(sorted(unique_windows.values(), key=lambda window: window.candidate_id))
    run_id = population_identity(
        {
            "candidate_version": settings.candidate_version,
            "max_chunk_span": settings.max_chunk_span,
            "max_records_per_window": settings.max_records_per_window,
            "records": sorted(record.population_id for record in records),
            "chunk_positions": sorted((chunk.chunk_id, chunk.scope_id, chunk.chunk_index) for chunk in chunks),
        }
    )
    return AssemblyCandidateSet(run_id=run_id, windows=windows)
