"""Execute closed-world SHACL fixture validation."""

from __future__ import annotations

from pyshacl import validate
from rdflib import Graph

from legal_rag.ontology_validation.models import ShaclResult


def run_shacl(data_graph: Graph, shapes_graph: Graph, ontology_graph: Graph) -> ShaclResult:
    """Validate one data graph with RDFS inference and advanced SHACL."""
    conforms, _, report_text = validate(
        data_graph=data_graph,
        shacl_graph=shapes_graph,
        ont_graph=ontology_graph,
        inference="rdfs",
        advanced=True,
    )
    return ShaclResult(conforms=conforms, report_text=report_text)
