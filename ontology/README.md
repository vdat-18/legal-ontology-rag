# Road Traffic Legal Ontology

This directory contains versioned ontology-engineering artifacts derived from
`docs/ontology_construction_plan.md`.

Current status: draft formalization. The project owner accepted pilot legal-review items `LR-001` through `LR-007`
on 2026-08-17. Independent legal-professional certification is not asserted.

The current increment contains machine-readable requirements, competency questions, glossary terms, modeling decisions,
formalization policy, the source/provenance OWL module, its SHACL shapes, executable source/evidence CQ projections,
the occurrence-context OWL module, the temporal-change OWL module, and pilot fixtures. Remaining OWL modules,
retrieval, GraphRAG, Graphiti, and ontology population remain outside this increment.

`formalization_policy.yaml` selects OWL 2 DL and defines proposed canonical namespaces and an acyclic local import graph.
Publishing under the proposed `w3id.org` namespace remains blocked until redirect registration and licensing are complete.

`legal_ontology.ttl` currently imports only modules already formalized in the repository. `catalog-v001.xml` resolves
their canonical and version IRIs to local files without network access.

The SHACL graph validates approved source/provenance record completeness without closing arbitrary properties or adding
OWL cardinality axioms. `CQ-ASSEMBLY-001` remains deferred until the normative-rule module defines rule components.

Occurrence-context shapes validate controlled concept schemes, contextual role assignments, vehicle classification,
normalized conduct, and occurrence time. Their queries expose complete input context and missing dimensions without
selecting or scoring a penalty rule.

The temporal-change module models verified validity intervals and evidenced legal-change events. It keeps missing
interval ends unknown, allows repeal without a later rule, and does not infer change links from textual similarity.
Temporal SHACL constraints validate interval structure and ordering, controlled change types, optional replacement,
and mandatory change evidence. Temporal fixtures and CQ projections preserve applicability and changed-field decisions
for the later normative-rule increment.
