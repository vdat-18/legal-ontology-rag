# Independent Legal Review Packet for Road Traffic Legal Ontology V0.1

Artifact: `legal-ontology-rag-independent-legal-review` version `0.1.0`

This packet requests independent legal review. It does not authorize an LLM or the project owner to substitute
for the named legal-domain reviewer.

## Reviewer Record

- Name: [pending]
- Qualifications: [pending]
- Organization: [pending]
- Reviewed on: [pending]

## Review Items

### ILR-001 - Rule and legal-expression identity

Question: Are version-specific penalty rules, legal resources, and legal expressions distinguished correctly?

Current decision: `pending`

Affected IDs: `MD-RULE-001`, `MD-DOC-001`, `TERM-RULE-001`, `TERM-RULE-002`, `TERM-RULE-003`, `TERM-DOC-001`, `TERM-EXPR-001`

Evidence files: `ontology/modules/normative_rule.ttl`, `ontology/modules/source_provenance.ttl`

Acceptance criteria:

- Similar wording alone never establishes identity across legal instruments or versions.
- Each accepted rule identity is grounded in an authoritative expression and provision.

Reviewer rationale: [pending]

### ILR-002 - Conduct normalization boundary

Question: Does the model correctly separate regulated conduct concepts, rule specifications, and concrete occurrences?

Current decision: `pending`

Affected IDs: `MD-CONDUCT-001`, `TERM-CONDUCT-001`, `TERM-CONDUCT-002`, `TERM-CONDUCT-003`, `TERM-APP-001`

Evidence files: `ontology/modules/occurrence_context.ttl`, `ontology/modules/normative_rule.ttl`

Acceptance criteria:

- Normalized conduct does not erase legally material actor, vehicle, condition, or source differences.
- Conduct similarity does not establish rule identity or applicability.

Reviewer rationale: [pending]

### ILR-003 - Legal subject and contextual role taxonomy

Question: Are subject kinds, legal subjects, contextual roles, and role assignments legally distinguished?

Current decision: `pending`

Affected IDs: `MD-SUBJECT-001`, `TERM-SUBJECT-001`, `TERM-SUBJECT-002`, `TERM-SUBJECT-003`, `TERM-SUBJECT-004`, `TERM-SUBJECT-005`, `TERM-SUBJECT-006`

Evidence files: `ontology/modules/occurrence_context.ttl`, `ontology/modules/normative_rule.ttl`

Acceptance criteria:

- Driver, owner, custodian, and operator roles are not inferred from one another.
- Individual and organization values are not confused with contextual legal roles.

Reviewer rationale: [pending]

### ILR-004 - Vehicle legal taxonomy

Question: Are vehicle categories represented at the distinctions required by the reviewed penalty provisions?

Current decision: `pending`

Affected IDs: `MD-VEHICLE-001`, `TERM-VEHICLE-001`, `TERM-VEHICLE-002`

Evidence files: `ontology/modules/occurrence_context.ttl`, `ontology/shapes/legal_ontology.shacl.ttl`

Acceptance criteria:

- Vehicle categories preserve every distinction that changes applicability or consequence.
- SKOS hierarchy does not silently become an OWL subclass or legal subsumption conclusion.

Reviewer rationale: [pending]

### ILR-005 - Condition expression grouping

Question: Do condition trees preserve conjunction, alternatives, negation, and legal scope without evaluating them?

Current decision: `pending`

Affected IDs: `MD-COND-001`, `TERM-COND-001`, `TERM-COND-002`

Evidence files: `ontology/modules/normative_rule.ttl`, `ontology/shapes/legal_ontology.shacl.ttl`

Acceptance criteria:

- Grouping matches the authoritative syntax and provision hierarchy.
- The ontology records conditions but does not claim their truth for an occurrence.

Reviewer rationale: [pending]

### ILR-006 - Exception scope, citation grouping, and legal priority

Question: Are exceptions, grouped citations, excluded provisions, and redirects modeled without deciding legal priority automatically?

Current decision: `pending`

Affected IDs: `MD-EXCEPTION-001`, `MD-REF-001`, `TERM-EXCEPTION-001`, `TERM-CITATION-001`, `TERM-PROVISION-001`

Evidence files: `ontology/modules/normative_rule.ttl`, `ontology/tests/cq_queries/cq_exception_001.rq`

Acceptance criteria:

- Citation grouping and exact exception evidence are preserved.
- Redirects target reviewed version-specific rules and do not automate legal priority.

Reviewer rationale: [pending]

### ILR-007 - Legal consequence classification

Question: Are monetary penalties, point deductions, additional sanctions, and remedial measures classified correctly?

Current decision: `pending`

Affected IDs: `MD-CONSEQUENCE-001`, `MD-CONSEQUENCE-002`, `TERM-CONSEQUENCE-001`, `TERM-MONEY-001`, `TERM-MONEY-002`, `TERM-MONEY-003`, `TERM-POINTS-001`, `TERM-ADDITIONAL-001`, `TERM-REMEDY-001`, `TERM-TARGET-001`

Evidence files: `ontology/modules/legal_consequence.ttl`, `ontology/tests/cq_queries/cq_consequence_001.rq`

Acceptance criteria:

- Every subtype is supported by normative source text, not a form or attachment alone.
- Missing consequence data remains unknown and is not interpreted as zero or legal absence.

Reviewer rationale: [pending]

### ILR-008 - Normative document and attachment legal status

Question: Does the model avoid treating files, attachments, or forms as normative provisions without legal authority?

Current decision: `pending`

Affected IDs: `MD-DOC-002`, `TERM-FILE-001`, `TERM-CHUNK-001`

Evidence files: `ontology/modules/source_provenance.ttl`, `ontology/tests/cq_queries/cq_source_001.rq`

Acceptance criteria:

- File containment does not establish normative status.
- Provision status and governing expression remain independently evidenced.

Reviewer rationale: [pending]

### ILR-009 - Effective dates and legal change types

Question: Are validity intervals and amendment, replacement, repeal, or supersession events legally supported?

Current decision: `pending`

Affected IDs: `MD-TIME-001`, `MD-CHANGE-001`, `TERM-TIME-001`, `TERM-CHANGE-001`, `TERM-CHANGE-002`, `TERM-CHANGE-003`, `TERM-CHANGE-004`

Evidence files: `ontology/modules/temporal_change.ttl`, `ontology/tests/cq_queries/cq_temporal_002_change_projection.rq`

Acceptance criteria:

- Effective dates are supported at rule level by authoritative evidence.
- Missing interval ends remain unknown and change types are never inferred from textual similarity.

Reviewer rationale: [pending]

### ILR-010 - Evidence and supported-assertion boundary

Question: Is every material legal assertion reviewable through exact evidence and authoritative provenance?

Current decision: `pending`

Affected IDs: `MD-PROV-001`, `MD-PROV-002`, `TERM-EVIDENCE-001`, `TERM-ASSERTION-001`, `TERM-ISSUER-001`

Evidence files: `ontology/modules/source_provenance.ttl`, `ontology/tests/cq_queries/cq_evidence_001.rq`

Acceptance criteria:

- Evidence for one component does not silently support unrelated components.
- Heading-derived scope retains both ancestor and leaf provision provenance.

Reviewer rationale: [pending]

### ILR-011 - V0.1 scope, competency questions, and legal limitations

Question: Are the V0.1 scope, exclusions, CQs, and stated limitations legally appropriate and non-misleading?

Current decision: `pending`

Affected IDs: `PURP-001`, `USE-001`, `SCOPE-001`, `EXCL-001`, `GRAN-001`, `CQ-RULE-001`, `CQ-TEMPORAL-001`, `CQ-EXCEPTION-001`

Evidence files: `ontology/requirements/ontology_requirements.yaml`, `ontology/requirements/competency_questions.yaml`, `ontology/reports/v0.1_qa_report.md`

Acceptance criteria:

- Structural CQ projections are not represented as final legal conclusions.
- Out-of-scope procedure, appeals, civil/criminal liability, and sanction aggregation remain explicit.

Reviewer rationale: [pending]
