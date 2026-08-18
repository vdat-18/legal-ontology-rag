# Road Traffic Legal Ontology

This directory contains versioned ontology-engineering artifacts derived from
`docs/ontology_construction_plan.md`.

Current status: draft formalization. The project owner accepted pilot legal-review items `LR-001` through `LR-007`
on 2026-08-17. Independent legal-professional certification is not asserted.

The current increment contains machine-readable requirements, competency questions, glossary terms, modeling decisions,
formalization policy, the source/provenance OWL module, its SHACL shapes, executable source/evidence CQ projections,
the occurrence-context, temporal-change, normative-rule, and legal-consequence OWL modules, plus pilot fixtures.
Retrieval, GraphRAG, Graphiti, and ontology population remain outside this increment.

`formalization_policy.yaml` selects OWL 2 DL and defines proposed canonical namespaces and an acyclic local import graph.
Publishing under the proposed `w3id.org` namespace remains blocked until redirect registration and licensing are complete.

`legal_ontology.ttl` currently imports only modules already formalized in the repository. `catalog-v001.xml` resolves
their canonical and version IRIs to local files without network access.

The SHACL graph validates approved source/provenance record completeness without closing arbitrary properties or adding
OWL cardinality axioms. `CQ-ASSEMBLY-001` now projects each independently evidenced normative component.

Occurrence-context shapes validate controlled concept schemes, contextual role assignments, vehicle classification,
normalized conduct, and occurrence time. Their queries expose complete input context and missing dimensions without
selecting or scoring a penalty rule.

The temporal-change module models verified validity intervals and evidenced legal-change events. It keeps missing
interval ends unknown, allows repeal without a later rule, and does not infer change links from textual similarity.
Temporal SHACL constraints validate interval structure and ordering, controlled change types, optional replacement,
and mandatory change evidence. Temporal fixtures and CQ projections defer applicability and changed-field decisions
to the normative validation increment.

The normative-rule module formalizes version-specific penalty rules, independently evidenced components, contextual
applicability profiles, non-executable condition trees, and evidenced exceptions with optional versioned redirects.
Normative SHACL constraints validate required rule structure, component evidence, applicability values, condition-tree
cardinalities and cycles, and exception targets. Its fixtures and CQ projections expose candidates and modeled source
structure without evaluating legal conditions or making a final applicability decision.

The legal-consequence module separates monetary penalties, point deductions, additional sanctions, and remedial
measures. Qualified monetary ranges retain decimal bounds and controlled currency, while targets and responsible roles
remain explicit. All 16 mandatory CQs now have executable structural projections; legal applicability, condition
satisfaction, and change interpretation still require semantic confirmation outside these projections.

Automated ontology QA lives in `tests/ontology/`, backed by reusable loaders and validators in
`src/legal_rag/ontology_validation/`. It checks Turtle parsing, policy and import consistency, catalog resolution,
modeling guardrails, SHACL fixtures, all mandatory CQ projections, and traceability. See
`ontology/reports/v0.1_qa_report.md` for the validated scope, semantic limitations, and open public-release gates.
