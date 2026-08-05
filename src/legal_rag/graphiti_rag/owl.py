"""Serialize the traffic-penalty ontology to OWL 2 in Turtle syntax.

The OWL file is generated from ``ontology.py`` rather than maintained by hand, so the schema the
extractor enforces and the schema published for inspection cannot drift apart. Generating it also
keeps the Python definitions as the single source of truth: Graphiti reads those directly, and a
hand-written OWL copy would be documentation that silently goes stale.

What is deliberately not asserted:

``owl:disjointWith`` between the classes. The built graph contains entities carrying two ontology
labels at once (``Subject``+``Violation``, ``Authority``+``Subject``, ``RemedialMeasure``+
``Violation``), because a legal entity really can be both — a driver is a subject who commits a
violation. Declaring the classes pairwise disjoint would make the published ontology inconsistent
with the data it describes, and a reasoner would report those individuals as errors. Disjointness
belongs here only once the domain genuinely forbids the overlap.
"""

from __future__ import annotations

from legal_rag.graphiti_rag.ontology import EDGE_TYPE_MAP, EDGE_TYPES, ENTITY_TYPES

ONTOLOGY_IRI = "https://github.com/vdat-18/legal-ontology-rag/ontology/traffic-penalty"

_PREAMBLE = f"""@prefix : <{ONTOLOGY_IRI}#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

<{ONTOLOGY_IRI}> a owl:Ontology ;
    rdfs:comment "Vietnamese road-traffic-penalty ontology, generated from legal_rag.graphiti_rag.ontology." .
"""


def _docstring(model: type) -> str:
    """Return a type's one-line description, escaped for a Turtle string literal."""
    text = " ".join((model.__doc__ or "").split())
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _class_expression(class_names: list[str]) -> str:
    """Render one or more classes as a Turtle term, using owl:unionOf beyond a single class.

    Listing several classes on one ``rdfs:domain`` would mean their *intersection*, so a property
    allowed from either a violation or a provision would end up requiring an individual to be both.
    A union class expression states the intended "either" instead.
    """
    if len(class_names) == 1:
        return f":{class_names[0]}"
    members = " ".join(f":{name}" for name in class_names)
    return f"[ a owl:Class ; owl:unionOf ( {members} ) ]"


def _sorted_unique(names: list[str]) -> list[str]:
    """Return the names deduplicated in a stable, readable order."""
    return sorted(set(names))


def build_owl() -> str:
    """Render the ontology as an OWL 2 document in Turtle syntax.

    Object property domains and ranges come from ``EDGE_TYPE_MAP``, which is the same constraint
    the indexer applies, so the published axioms describe what the graph is actually built to hold.
    """
    lines = [_PREAMBLE, "#################  Classes  #################\n"]
    for name, model in ENTITY_TYPES.items():
        lines.append(f':{name} a owl:Class ;\n    rdfs:label "{name}" ;\n    rdfs:comment "{_docstring(model)}" .\n')

    lines.append("#################  Object properties  #################\n")
    for name, model in EDGE_TYPES.items():
        domains = _sorted_unique([source for (source, _), edges in EDGE_TYPE_MAP.items() if name in edges])
        ranges = _sorted_unique([target for (_, target), edges in EDGE_TYPE_MAP.items() if name in edges])
        lines.append(
            f":{name} a owl:ObjectProperty ;\n"
            f'    rdfs:label "{name}" ;\n'
            f'    rdfs:comment "{_docstring(model)}" ;\n'
            f"    rdfs:domain {_class_expression(domains)} ;\n"
            f"    rdfs:range {_class_expression(ranges)} .\n"
        )
    return "\n".join(lines)
