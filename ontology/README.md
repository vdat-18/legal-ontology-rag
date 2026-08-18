# Road Traffic Legal Ontology

This directory contains versioned ontology-engineering artifacts derived from
`docs/ontology_construction_plan.md`.

Current status: draft formalization. The project owner accepted pilot legal-review items `LR-001` through `LR-007`
on 2026-08-17. Independent legal-professional certification is not asserted.

The current increment contains machine-readable requirements, competency questions, glossary terms, modeling decisions,
formalization policy, the source/provenance OWL module, its SHACL shapes, executable source/evidence CQ projections,
the occurrence-context, temporal-change, normative-rule, and legal-consequence OWL modules, plus pilot fixtures.
Retrieval, GraphRAG, and Graphiti remain outside this increment. Source-grounded pilot population is implemented as a
separate downstream pipeline so construction artifacts remain independent from extracted instance data.

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

`population_contract.yaml` defines the versioned chunk-to-instance boundary. The corresponding pipeline creates
reviewable typed records, deterministic RDF instances, exact evidence links, and writes Turtle only after SHACL
conformance. It does not populate concrete occurrences or infer cross-document legal changes and concept equivalence.
See `docs/ontology_population_pipeline.md` for the workflow and command.

`assembly_contract.yaml` defines the next within-document canonicalization boundary. It uses bounded structural windows
only to prepare typed LLM inputs, requires a legal reviewer to accept or reject every proposed cross-chunk assembly, and
rejects overlapping accepted members. Candidate generation and finalization are offline; only proposal generation calls
the model. Canonical population JSONL and Turtle are written after review completion and full-graph SHACL conformance.
Cross-document legal changes are handled only by the subsequent reviewed change-resolution boundary; exception redirects
and concept equivalence remain deferred.

`change_resolution_contract.yaml` implements the reviewed cross-document `LegalChange` boundary after per-document
assembly. An explicit ordered corpus manifest drives exhaustive bounded candidate windows without lexical matching or
semantic scoring. A typed LLM may only propose changes supported by exact later-document evidence; a human reviewer owns
acceptance. Accepted changes update canonical validity identities and materialize amendment, replacement, or
repeal/supersession events before full multi-document SHACL validation. Exception redirects are handled only by the
subsequent reviewed exception-resolution boundary; transitional applicability and concept equivalence remain deferred.

`exception_resolution_contract.yaml` implements the next reviewed boundary. It uses exhaustive bounded source/general/
target windows without keyword classification or similarity scoring. Typed proposals preserve exact exception evidence
and grouped citation wording, while humans approve the general rule, cited provisions, and version-specific redirects.
The formal model adds `hasProvisionCitation` so one exception can retain locator grouping separately from direct
`citesExcludedProvision` and `redirectsTo` projections. Finalization validates the complete graph against SHACL and
`CQ-EXCEPTION-001`. Concept equivalence is handled only by the subsequent reviewed concept-resolution boundary; legal
priority evaluation, unpopulated provision targets, and occurrence applicability remain deferred.

`concept_resolution_contract.yaml` implements reviewed consolidation for rule-scoped `RegulatedConduct`,
`LegalSubjectRole`, and `VehicleCategory` occurrences. Candidate construction exhaustively enumerates same-kind pairs
without lexical filtering or semantic scoring. Typed LLM proposals must bind exact evidence from both occurrences, and
every proposed merge or taxonomy relation requires attributable human acceptance. Finalization computes canonical
identities only from accepted `SameConcept` decisions, remaps rule references, and materializes reviewed SKOS
`broader`/`narrower`/`related` relations. It does not assert OWL equivalence or subclass semantics. Subject kinds remain
the pre-approved controlled concepts; consequence-target consolidation, legal priority, and occurrence applicability
remain deferred.
