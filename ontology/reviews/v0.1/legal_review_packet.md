# AI-Assisted Internal Legal Review for Road Traffic Legal Ontology V0.1

Artifact: `legal-ontology-rag-internal-legal-review` version `0.1.0`

This packet records AI-assisted internal legal-domain review for a solo project. It is not licensed legal
advice or independent professional certification; project-owner acceptance remains explicit.

## Reviewer Record

- Name: Codex
- Qualifications: AI-assisted internal Vietnamese legal-domain and ontology analysis; not licensed legal-professional certification
- Organization: Solo-project coding agent
- Reviewed on: 2026-08-18
- Project-owner acceptance: approved
- Project-owner accepted on: 2026-08-18

## Review Items

### ILR-001 - Rule and legal-expression identity

Question: Are version-specific penalty rules, legal resources, and legal expressions distinguished correctly?

Current decision: `accept`

Affected IDs: `MD-RULE-001`, `MD-DOC-001`, `TERM-RULE-001`, `TERM-RULE-002`

Evidence files: `ontology/modules/normative_rule.ttl`, `ontology/modules/source_provenance.ttl`

Acceptance criteria:

- Similar wording alone never establishes identity across legal instruments or versions.
- Each accepted rule identity is grounded in an authoritative expression and provision.

Reviewer rationale: Version-specific rule and expression identities are correctly separated; similarity cannot create cross-version identity without an explicit legal-change source.

### ILR-002 - Conduct normalization boundary

Question: Does the model correctly separate regulated conduct concepts, rule specifications, and concrete occurrences?

Current decision: `accept`

Affected IDs: `MD-CONDUCT-001`, `TERM-CONDUCT-001`, `TERM-CONDUCT-002`, `TERM-APP-001`

Evidence files: `ontology/modules/occurrence_context.ttl`, `ontology/modules/normative_rule.ttl`

Acceptance criteria:

- Normalized conduct does not erase legally material actor, vehicle, condition, or source differences.
- Conduct similarity does not establish rule identity or applicability.

Reviewer rationale: Shared conduct concepts are safe because exact wording, applicability, consequences, and rule identity remain separate and independently evidenced.

### ILR-003 - Legal subject and contextual role taxonomy

Question: Are subject kinds, legal subjects, contextual roles, and role assignments legally distinguished?

Current decision: `accept`

Affected IDs: `MD-SUBJECT-001`, `TERM-SUBJECT-002`, `TERM-SUBJECT-005`, `TERM-SUBJECT-006`

Evidence files: `ontology/modules/occurrence_context.ttl`, `ontology/modules/normative_rule.ttl`

Acceptance criteria:

- Driver, owner, custodian, and operator roles are not inferred from one another.
- Individual and organization values are not confused with contextual legal roles.

Reviewer rationale: Subject kind and contextual role separation preserves the legal difference between a person, driver, owner, custodian, and vehicle operator.

### ILR-004 - Vehicle legal taxonomy

Question: Are vehicle categories represented at the distinctions required by the reviewed penalty provisions?

Current decision: `accept`

Affected IDs: `MD-VEHICLE-001`, `TERM-VEHICLE-001`

Evidence files: `ontology/modules/occurrence_context.ttl`, `ontology/shapes/legal_ontology.shacl.ttl`

Acceptance criteria:

- Vehicle categories preserve every distinction that changes applicability or consequence.
- SKOS hierarchy does not silently become an OWL subclass or legal subsumption conclusion.

Reviewer rationale: Vehicle categories are controlled legal-context values and SKOS relations do not assert OWL or automatic legal subsumption.

### ILR-005 - Condition expression grouping

Question: Do condition trees preserve conjunction, alternatives, negation, and legal scope without evaluating them?

Current decision: `accept`

Affected IDs: `MD-COND-001`, `TERM-COND-001`, `TERM-COND-002`

Evidence files: `ontology/modules/normative_rule.ttl`, `ontology/shapes/legal_ontology.shacl.ttl`

Acceptance criteria:

- Grouping matches the authoritative syntax and provision hierarchy.
- The ontology records conditions but does not claim their truth for an occurrence.

Reviewer rationale: Condition trees preserve conjunction, alternatives, negation, and source scope without evaluating condition truth or executing legal rules.

### ILR-006 - Exception scope, citation grouping, and legal priority

Question: Are exceptions, grouped citations, excluded provisions, and redirects modeled without deciding legal priority automatically?

Current decision: `accept`

Affected IDs: `MD-EXCEPTION-001`, `MD-REF-001`, `TERM-EXCEPTION-001`, `TERM-CITATION-001`, `TERM-PROVISION-001`

Evidence files: `ontology/modules/normative_rule.ttl`, `ontology/tests/cq_queries/cq_exception_001.rq`

Acceptance criteria:

- Citation grouping and source provenance are preserved.
- Redirects target reviewed version-specific rules and do not automate legal priority.

Reviewer rationale: Explicit exception entities preserve the Article 6 exclusion wording and grouped citations while leaving legal priority and target applicability for reviewed interpretation.

### ILR-007 - Legal consequence classification

Question: Are monetary penalties, point deductions, additional sanctions, and remedial measures classified correctly?

Current decision: `accept`

Affected IDs: `MD-CONSEQUENCE-001`, `MD-CONSEQUENCE-002`, `TERM-CONSEQUENCE-001`, `TERM-MONEY-001`, `TERM-POINTS-001`, `TERM-ADDITIONAL-001`, `TERM-REMEDY-001`

Evidence files: `ontology/modules/legal_consequence.ttl`, `ontology/tests/cq_queries/cq_consequence_001.rq`

Acceptance criteria:

- Every subtype is supported by normative source text, not a form or attachment alone.
- Missing consequence data remains unknown and is not interpreted as zero or legal absence.

Reviewer rationale: Consequence subtypes remain distinct and require normative classification evidence; attachment or form content alone cannot establish a remedial measure.

### ILR-008 - Normative document and attachment legal status

Question: Does the model avoid treating files, attachments, or forms as normative provisions without legal authority?

Current decision: `accept`

Affected IDs: `MD-DOC-002`

Evidence files: `ontology/modules/source_provenance.ttl`, `ontology/tests/cq_queries/cq_source_001.rq`

Acceptance criteria:

- File containment does not establish normative status.
- Provision status and governing expression remain independently evidenced.

Reviewer rationale: Legal resource, expression, provision, file, attachment, and chunk identities are correctly separated, preventing file containment from creating normative authority.

### ILR-009 - Effective dates and legal change types

Question: Are validity intervals and amendment, replacement, repeal, or supersession events legally supported?

Current decision: `accept`

Affected IDs: `MD-TIME-001`, `MD-CHANGE-001`, `TERM-TIME-001`, `TERM-CHANGE-004`

Evidence files: `ontology/modules/temporal_change.ttl`, `ontology/tests/cq_queries/cq_temporal_002_change_projection.rq`

Acceptance criteria:

- Effective dates are supported at rule level by authoritative evidence.
- Missing interval ends remain unknown and change types are never inferred from textual similarity.

Reviewer rationale: Validity and change events require authoritative dates and explicit change evidence; missing ends remain unknown and textual similarity cannot prove succession.

### ILR-010 - Source provenance boundary

Question: Is every material legal assertion traceable to an authoritative provision and source chunk?

Current decision: `accept`

Affected IDs: `MD-PROV-001`, `MD-PROV-002`, `TERM-PROVISION-001`, `TERM-CITATION-001`

Evidence files: `ontology/modules/source_provenance.ttl`, `ontology/tests/cq_queries/cq_source_001.rq`

Acceptance criteria:

- Provenance for one component does not silently support unrelated components.
- Heading-derived scope retains both ancestor and leaf provision provenance.

Reviewer rationale: Component-level provision and source-chunk links keep unrelated legal assertions independently traceable.

### ILR-011 - V0.1 scope, competency questions, and legal limitations

Question: Are the V0.1 scope, exclusions, CQs, and stated limitations legally appropriate and non-misleading?

Current decision: `accept`

Affected IDs: `PURP-001`, `USE-001`, `SCOPE-001`, `GRAN-001`, `CQ-RULE-001`, `CQ-TEMPORAL-001`, `CQ-EXCEPTION-001`

Evidence files: `ontology/requirements/ontology_requirements.yaml`, `ontology/requirements/competency_questions.yaml`, `ontology/reports/v0.1_qa_report.md`

Acceptance criteria:

- Structural CQ projections are not represented as final legal conclusions.
- Out-of-scope procedure, appeals, civil/criminal liability, and sanction aggregation remain explicit.

Reviewer rationale: V0.1 scope and CQs clearly separate structural ontology answers from final applicability, procedure, adjudication, liability, and sanction aggregation.
