"""Tests for corpus loading and structural legal-change candidates."""

from __future__ import annotations

import json
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from legal_rag.ontology_change.candidates import build_change_candidates
from legal_rag.ontology_change.config import ChangeResolutionSettings
from legal_rag.ontology_change.corpus import ChangeCorpus, ChangeCorpusDocument, read_change_corpus_manifest
from legal_rag.ontology_change.writer import read_change_candidates, write_change_candidates
from legal_rag.ontology_population.models import PopulationEvidence
from legal_rag.ontology_validation.loader import load_yaml

from .conftest import write_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_change_contract_and_console_scripts_are_packaged() -> None:
    """The checked-in contract and operator commands expose bounded workflow settings."""
    settings = ChangeResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "change_resolution_contract.yaml")
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert settings.max_source_chunks_per_window == 6
    assert settings.max_earlier_records_per_window == 32
    assert settings.max_later_records_per_window == 32
    assert settings.max_total_windows == 10_000
    assert settings.batch_size == 1
    assert set(load_yaml(PROJECT_ROOT / "ontology" / "change_resolution_contract.yaml")) == {
        "schema_version",
        "candidates",
        "llm",
    }
    assert project["scripts"]["legal-build-change-candidates"] == "legal_rag.ontology_change_candidates_cli:main"
    assert project["scripts"]["legal-propose-legal-changes"] == "legal_rag.ontology_change_proposals_cli:main"
    assert project["scripts"]["legal-finalize-legal-changes"] == "legal_rag.ontology_change_finalize_cli:main"


def test_manifest_and_candidates_are_deterministic_and_round_trip(
    tmp_path: Path,
    change_documents: tuple[ChangeCorpusDocument, ChangeCorpusDocument],
) -> None:
    """Ordered manifests produce stable provenance-aligned windows without semantic matching."""
    manifest_path = tmp_path / "corpus.json"
    output_path = tmp_path / "candidates.json"
    write_manifest(manifest_path, change_documents)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["documents"][1]["comparison_document_ids"] = [change_documents[0].document_id]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    corpus = read_change_corpus_manifest(manifest_path)
    settings = ChangeResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "change_resolution_contract.yaml")
    first = build_change_candidates(corpus, settings)
    second = build_change_candidates(corpus, settings)
    write_change_candidates(output_path, first)

    assert first == second
    assert corpus.documents[1].comparison_document_ids == (change_documents[0].document_id,)
    assert len(first.windows) == 1
    assert first.windows[0].earlier_population_ids == ("earlier-population",)
    assert first.windows[0].later_population_ids == ("later-population",)
    assert first.windows[0].change_source_chunk_ids == ("later-chunk",)
    assert read_change_candidates(output_path) == first


def test_later_records_only_enter_windows_for_their_source_chunk_batch(
    change_documents: tuple[ChangeCorpusDocument, ChangeCorpusDocument],
) -> None:
    """A later rule is not repeated across unrelated change-source batches."""
    earlier_document, later_document = change_documents
    first_chunk = later_document.chunks[0]
    second_chunk = replace(
        first_chunk,
        chunk_id="later-chunk-2",
        content_sha256="content-later-chunk-2",
        chunk_index=1,
        source_refs=("document.xml#/later-chunk-2",),
        unit_ids=("unit-later-chunk-2",),
        text="Bãi bỏ một quy định xử phạt khác.",
        context_text="Bãi bỏ một quy định xử phạt khác.",
        embedding_text="Bãi bỏ một quy định xử phạt khác.",
    )
    second_evidence = (PopulationEvidence(second_chunk.chunk_id, second_chunk.text),)
    second_record = replace(
        later_document.records[0],
        population_id="later-population-2",
        source_chunk_id=second_chunk.chunk_id,
        evidence=second_evidence,
        consequences=(replace(later_document.records[0].consequences[0], evidence=second_evidence),),
    )
    corpus = ChangeCorpus(
        documents=(
            earlier_document,
            replace(
                later_document,
                chunks=(*later_document.chunks, second_chunk),
                records=(*later_document.records, second_record),
            ),
        )
    )
    settings = replace(
        ChangeResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "change_resolution_contract.yaml"),
        max_source_chunks_per_window=1,
    )

    candidates = build_change_candidates(corpus, settings)
    later_ids_by_source = {window.change_source_chunk_ids: window.later_population_ids for window in candidates.windows}

    assert later_ids_by_source == {
        ("later-chunk",): ("later-population",),
        ("later-chunk-2",): ("later-population-2",),
    }


@pytest.mark.parametrize(
    ("comparison_ids", "message"),
    [
        (["missing-document"], "must exist"),
        (["nd-238"], "must reference an earlier document"),
    ],
)
def test_manifest_rejects_invalid_comparison_scope(
    tmp_path: Path,
    change_documents: tuple[ChangeCorpusDocument, ChangeCorpusDocument],
    comparison_ids: list[str],
    message: str,
) -> None:
    """Comparison scopes are validated at the untrusted manifest boundary."""
    manifest_path = tmp_path / "corpus.json"
    write_manifest(manifest_path, change_documents)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["documents"][1]["comparison_document_ids"] = comparison_ids
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_change_corpus_manifest(manifest_path)


def test_candidate_generation_uses_only_forward_document_order(
    change_documents: tuple[ChangeCorpusDocument, ChangeCorpusDocument],
) -> None:
    """Only records from an earlier manifest document enter the earlier side of a window."""
    settings = ChangeResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "change_resolution_contract.yaml")
    corpus = ChangeCorpus(documents=tuple(reversed(change_documents)))
    candidates = build_change_candidates(corpus, settings)

    assert candidates.windows[0].earlier_document_id == "nd-238"
    assert candidates.windows[0].later_document_id == "nd-168"


def test_explicit_comparison_scope_skips_unrelated_document_pairs(
    change_documents: tuple[ChangeCorpusDocument, ChangeCorpusDocument],
) -> None:
    """A manifest can retain an independent document without scheduling model work."""
    earlier_document, later_document = change_documents
    settings = ChangeResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "change_resolution_contract.yaml")
    independent = ChangeCorpus(documents=(earlier_document, replace(later_document, comparison_document_ids=())))
    related = ChangeCorpus(
        documents=(
            earlier_document,
            replace(later_document, comparison_document_ids=(earlier_document.document_id,)),
        )
    )

    assert build_change_candidates(independent, settings).windows == ()
    assert len(build_change_candidates(related, settings).windows) == 1


def test_candidate_generation_enforces_total_window_bound(
    change_corpus: ChangeCorpus,
) -> None:
    """Provenance-aligned retrieval fails before creating an unbounded workload."""
    settings = replace(
        ChangeResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "change_resolution_contract.yaml"),
        max_total_windows=0,
    )

    with pytest.raises(ValueError, match="exceeds the configured bound"):
        build_change_candidates(change_corpus, settings)


def test_one_document_corpus_has_no_cross_document_candidates(
    change_documents: tuple[ChangeCorpusDocument, ChangeCorpusDocument],
) -> None:
    """A cross-document phase cannot propose a change with only one document."""
    settings = ChangeResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "change_resolution_contract.yaml")

    assert build_change_candidates(ChangeCorpus(documents=change_documents[:1]), settings).windows == ()
