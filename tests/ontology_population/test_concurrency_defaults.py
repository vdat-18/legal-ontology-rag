"""Tests for ontology command-line concurrency defaults."""

from __future__ import annotations

import argparse

import pytest

from legal_rag import (
    ontology_assembly_proposals_cli,
    ontology_change_proposals_cli,
    ontology_concept_proposals_cli,
    ontology_conduct_extraction_cli,
    ontology_exception_proposals_cli,
    ontology_population_binding_cli,
    ontology_population_cli,
    ontology_semantic_step_cli,
)
from legal_rag.ontology_proposal_execution import DEFAULT_ONTOLOGY_MAX_CONCURRENCY


@pytest.mark.parametrize(
    "parser",
    [
        ontology_conduct_extraction_cli._parser(),
        ontology_population_binding_cli._parser(),
        ontology_semantic_step_cli._assembly_parser(),
        ontology_semantic_step_cli._change_parser(),
        ontology_semantic_step_cli._exception_parser(),
        ontology_semantic_step_cli._concept_parser(),
        ontology_assembly_proposals_cli._parser(),
        ontology_change_proposals_cli._parser(),
        ontology_exception_proposals_cli._parser(),
        ontology_concept_proposals_cli._parser(),
        ontology_population_cli._parser(),
    ],
)
def test_ontology_cli_defaults_to_bounded_concurrency(parser: argparse.ArgumentParser) -> None:
    """Every ontology LLM command uses the shared concurrency default."""
    assert parser.get_default("max_concurrency") == DEFAULT_ONTOLOGY_MAX_CONCURRENCY == 30
