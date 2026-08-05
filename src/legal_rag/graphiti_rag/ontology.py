"""Custom traffic-penalty entity and edge types for Graphiti extraction.

Adapted from the domain vocabulary already established for ground-truth synthesis
(``baml_src/legal_fact_extraction.baml``'s ``LegalFact`` and
``baml_src/evidence_relations.baml``'s ``RelationType``), which models fact-to-fact
bridging for QA generation. Here the same vocabulary is reshaped into entity-to-entity
graph types for Graphiti: each class's docstring is shown to the LLM as the type
description used for node/edge classification.

Types are deliberately kept field-less (docstring only). Graphiti skips its extra
per-type "attribute extraction" LLM call entirely when a type has zero fields
(``graphiti_core/utils/maintenance/node_operations.py::_extract_entity_attributes``,
``if entity_type is None or len(entity_type.model_fields) == 0: return {}``). Adding
fields here was tried and reliably broke under this project's configured LLM
(DeepSeek, ``json_object`` structured-output fallback): the model echoed the JSON
*schema* it was shown back as the "value" (e.g. a ``document_number`` property
literally containing ``{"type": "string", "default": "...", "description": "...",
"title": "..."}``), which Neo4j then rejected as a non-primitive property value. Node
labels/classification and edge names still work correctly with field-less types —
only the extra structured attributes are sacrificed. Revisit adding fields back if a
more schema-reliable model becomes available.
"""

from __future__ import annotations

from pydantic import BaseModel


class Violation(BaseModel):
    """A specific traffic violation or prohibited conduct described in a legal provision."""


class Penalty(BaseModel):
    """A monetary fine or administrative sanction imposed for a violation."""


class RemedialMeasure(BaseModel):
    """A corrective or remedial action required in addition to a penalty."""


class LegalDocument(BaseModel):
    """A legal document: decree (Nghị định), circular (Thông tư), law (Luật), or decision (Quyết định)."""


class Provision(BaseModel):
    """A specific article, clause, or point within a legal document."""


class Subject(BaseModel):
    """A category of person or vehicle that a rule applies to."""


class Authority(BaseModel):
    """A government body responsible for enforcement, sanctioning, or issuance of a rule."""


class Penalizes(BaseModel):
    """This violation is penalized by this fine or sanction."""


class AppliesTo(BaseModel):
    """This violation or rule applies to this subject or vehicle category."""


class RequiresRemedy(BaseModel):
    """This violation requires this remedial measure in addition to any penalty."""


class DefinedIn(BaseModel):
    """This violation or rule is defined in this legal provision."""


class PartOf(BaseModel):
    """This provision is part of this legal document."""


class EnforcedBy(BaseModel):
    """This rule or violation is enforced or sanctioned by this authority."""


class Amends(BaseModel):
    """This legal document amends, supersedes, or modifies that legal document."""


ENTITY_TYPES: dict[str, type[BaseModel]] = {
    "Violation": Violation,
    "Penalty": Penalty,
    "RemedialMeasure": RemedialMeasure,
    "LegalDocument": LegalDocument,
    "Provision": Provision,
    "Subject": Subject,
    "Authority": Authority,
}

EDGE_TYPES: dict[str, type[BaseModel]] = {
    "PENALIZES": Penalizes,
    "APPLIES_TO": AppliesTo,
    "REQUIRES_REMEDY": RequiresRemedy,
    "DEFINED_IN": DefinedIn,
    "PART_OF": PartOf,
    "ENFORCED_BY": EnforcedBy,
    "AMENDS": Amends,
}

EDGE_TYPE_MAP: dict[tuple[str, str], list[str]] = {
    ("Violation", "Penalty"): ["PENALIZES"],
    ("Violation", "Subject"): ["APPLIES_TO"],
    ("Provision", "Subject"): ["APPLIES_TO"],
    ("Violation", "RemedialMeasure"): ["REQUIRES_REMEDY"],
    ("Violation", "Provision"): ["DEFINED_IN"],
    ("Penalty", "Provision"): ["DEFINED_IN"],
    ("RemedialMeasure", "Provision"): ["DEFINED_IN"],
    ("Provision", "LegalDocument"): ["PART_OF"],
    ("Violation", "Authority"): ["ENFORCED_BY"],
    ("Provision", "Authority"): ["ENFORCED_BY"],
    ("LegalDocument", "LegalDocument"): ["AMENDS"],
}
"""Which edge types may connect which entity types, as (source, target) -> allowed names.

Without this map Graphiti defaults to ``{('Entity', 'Entity'): every edge type}``, so the model
may attach any relation between any two entities. That default produced clearly wrong edges in
the built graph: only ~62% of ``PENALIZES`` ran ``Violation -> Penalty``, the rest including
``Penalty -> Violation`` (reversed) and ``Subject -> Subject``.

Graphiti matches a candidate pair against the cartesian product of both nodes' labels, each
extended with ``Entity`` (``edge_operations.py``), and offers the model only the types found
here. ``('Entity', 'Entity')`` is deliberately absent: a pair the extractor could not classify
falls back to a generic relation name rather than being forced into a typed edge it does not
satisfy, which keeps the typed edges trustworthy for traversal.
"""
