# Road Traffic Legal Ontology

This directory contains versioned ontology-engineering artifacts derived from
`docs/ontology_construction_plan.md`.

Current status: V0.1 is preserved as an immutable internal release under `dist/ontology/v0.1.0/`; the editable ontology
workspace is now the V0.2 candidate. Prior V0.1 approvals remain historical evidence and are not reused as approval of
V0.2. Independent legal-professional certification is not asserted.

The current increment contains machine-readable requirements, competency questions, glossary terms, modeling decisions,
formalization policy, the source/provenance OWL module, its SHACL shapes, executable source/evidence CQ projections,
the occurrence-context, temporal-change, normative-rule, and legal-consequence OWL modules, plus pilot fixtures.
Retrieval, GraphRAG, and Graphiti remain outside this increment. Source-grounded pilot population is implemented as a
separate downstream pipeline so construction artifacts remain independent from extracted instance data.

`formalization_policy.yaml` selects OWL 2 DL and defines proposed canonical namespaces and an acyclic local import graph.
Publishing under the proposed `w3id.org` namespace remains optional and is pending V0.2 semantic QA and review.

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
`ontology/reports/v0.1_qa_report.md` for the historical V0.1 validated scope. It is not QA evidence for V0.2.

`semantic_qa_contract.yaml` adds the semantic release-hardening boundary. It pins ROBOT `v1.9.10` and HermiT in a
network-isolated bounded Docker invocation, validates the OWL 2 DL import closure through the local XML catalog,
materializes reasoning output, audits unsatisfiable classes and unintended inferred equivalences, runs ROBOT quality
rules, and repeats traceability plus mandatory-CQ coverage in one machine-readable report. Minimal declarations for
referenced PROV-O, SKOS, OWL-Time, and Dublin Core entities live in the foundational source module so the import closure
is OWL 2 DL without importing external ontology bundles. The ontology and its construction artifacts are licensed under
CC BY 4.0 through `LICENSE` and root `dcterms:license` metadata. The archived V0.1 semantic checks passed; no passing V0.2
report is asserted yet. Each report contains a digest of every semantic input so changed ontology, shape, CQ, fixture,
registry, catalog, or policy files invalidate stale QA evidence. V0.2 semantic QA must be rerun before release.

`external_reuse_registry.yaml` records the completed ontology-engineering review of ten external resources and the
exact 23 external term IRIs used by OWL or SHACL. Dublin Core, PROV-O, SKOS, and OWL-Time are reused by term reference
without `owl:imports`; ELI remains candidate-only; TLKB, LKIF-Core, Akoma Ntoso, LegalRuleML, and the n-ary relation
pattern remain references or local pattern adaptations. The local review command passes all six structural gates, and
the project owner accepted `REUSE-DCTERMS-001` through `REUSE-ODP-001` on 2026-08-18.

`legal_review_contract.yaml` and `legal_review_decisions.yaml` define the AI-assisted internal legal-review boundary
before the V0.1 freeze. Eleven review items cover all open modeling-decision legal topics and every glossary term, with
explicit questions, evidence files, affected IDs, acceptance criteria, and Codex rationale. The generated packet and
readiness report live under `reviews/v0.1/`. All 11 internal recommendations and project-owner acceptance are recorded;
no licensed or independent legal-professional certification is asserted.

`release_contract.yaml` now defines the blocked V0.2 candidate release boundary. The release command checks the bound semantic
report, ontology identity, license, the live external-reuse registry, legal-review packet completeness, explicit
project-owner approvals, and the declared file
inventory. A successful run creates a metadata-normalized ZIP with an internal `MANIFEST.json` plus a separate SHA-256
checksum. V0.2 packaging remains fail-closed until fresh semantic QA and explicit V0.2 approvals exist. The immutable
V0.1 archive and checksum remain unchanged under `dist/ontology/v0.1.0/`.

`evolution_policy.yaml` defines how a frozen ontology may change after new source data arrives. New chunks populate
instances and may produce reviewed gap proposals, but neither the data nor an LLM mutates the TBox directly. The
`legal-diff-ontology` command compares two project roots by declared local classes and properties, their OWL axiom
closures, named SHACL node shapes and constraint closures, mandatory CQs, deprecations, and replacement links. It blocks
active or deprecated term removal, term-kind or active-term semantic changes, named-shape removal or constraint changes,
and mandatory-CQ removal while allowing additive changes. Labels, comments, and traceability annotations do not count as
semantic changes. A changed meaning must receive a new IRI, and the old IRI remains declared with `owl:deprecated true`.

`namespace_publication.yaml` defines the canonical namespace deployment and W3ID registration boundary. The publication
command validates all ontology/version IRIs against Turtle and the XML catalog, creates HTML plus byte-identical Turtle
representations for 14 canonical/versioned endpoints, and emits a SHA-256 manifest. The copy-ready W3ID pull-request
bundle lives under `publication/w3id/legal-ontology-rag/`. Namespace approval remains pending until the proposed hosting
target is deployed, externally verified, and the W3ID pull request is merged. See
`docs/ontology_namespace_publication.md`.

The repository includes a manually triggered, least-privilege GitHub Pages workflow for the generated site and a live
verification command. Verification performs bounded sequential requests and requires exact HTML/Turtle bytes, media
types, redirect status, and `Location` targets; it does not retry or silently accept stale publication content.

`population_contract.yaml` defines the versioned chunk-to-instance boundary. The corresponding pipeline creates
reviewable typed records, deterministic RDF instances, exact evidence links, and writes Turtle only after SHACL
conformance. It does not populate concrete occurrences or infer cross-document legal changes and concept equivalence.
Population records and checkpoints are bound to the ontology version, ontology semantic hash, input, prompt, provider
configuration, model, and run ID so incompatible ontology/run states cannot be mixed. A source-reviewed gold evaluator
gates the pilot on exact per-chunk inventory plus required and forbidden
rule signatures before scaling.
See `docs/ontology_population_pipeline.md` for the workflow and command.

`assembly_contract.yaml` defines the next within-document canonicalization boundary. It uses bounded structural windows
only to prepare typed LLM inputs, requires attributable project-owner review for every proposed cross-chunk assembly, and
rejects overlapping accepted members. Candidate generation and finalization are offline; only proposal generation calls
the model. Canonical population JSONL and Turtle are written after review completion and full-graph SHACL conformance.
Cross-document legal changes are handled only by the subsequent reviewed change-resolution boundary; exception redirects
and concept equivalence remain deferred.

`change_resolution_contract.yaml` implements the reviewed cross-document `LegalChange` boundary after per-document
assembly. An explicit ordered corpus manifest drives exhaustive bounded candidate windows without lexical matching or
semantic scoring. A typed LLM may only propose changes supported by exact later-document evidence; the project owner owns
acceptance. Accepted changes update canonical validity identities and materialize amendment, replacement, or
repeal/supersession events before full multi-document SHACL validation. Exception redirects are handled only by the
subsequent reviewed exception-resolution boundary; transitional applicability and concept equivalence remain deferred.

`exception_resolution_contract.yaml` implements the next reviewed boundary. It uses exhaustive bounded source/general/
target windows without keyword classification or similarity scoring. Typed proposals preserve exact exception evidence
and grouped citation wording, while project-owner review approves the general rule, cited provisions, and redirects.
The formal model adds `hasProvisionCitation` so one exception can retain locator grouping separately from direct
`citesExcludedProvision` and `redirectsTo` projections. Finalization validates the complete graph against SHACL and
`CQ-EXCEPTION-001`. Concept equivalence is handled only by the subsequent reviewed concept-resolution boundary; legal
priority evaluation, unpopulated provision targets, and occurrence applicability remain deferred.

`concept_resolution_contract.yaml` implements reviewed consolidation for rule-scoped `RegulatedConduct`,
`LegalSubjectRole`, and `VehicleCategory` occurrences. Candidate construction exhaustively enumerates same-kind pairs
without lexical filtering or semantic scoring. Typed LLM proposals must bind exact evidence from both occurrences, and
every proposed merge or taxonomy relation requires attributable project-owner acceptance. Finalization computes canonical
identities only from accepted `SameConcept` decisions, remaps rule references, and materializes reviewed SKOS
`broader`/`narrower`/`related` relations. It does not assert OWL equivalence or subclass semantics. Subject kinds remain
the pre-approved controlled concepts; consequence-target consolidation, legal priority, and occurrence applicability
remain deferred.
