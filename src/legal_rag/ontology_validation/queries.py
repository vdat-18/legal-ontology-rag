"""Execute competency-question SPARQL projections."""

from __future__ import annotations

from pathlib import Path

from rdflib import Graph

from legal_rag.ontology_validation.models import QueryResult


def run_query(graph: Graph, query_path: Path) -> QueryResult:
    """Execute one versioned SPARQL query and materialize its result."""
    query = query_path.read_text(encoding="utf-8")
    result = graph.query(query)
    return QueryResult(
        columns=tuple(str(variable) for variable in result.vars),
        rows=tuple(tuple(row) for row in result),
    )
