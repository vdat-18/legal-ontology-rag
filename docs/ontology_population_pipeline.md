# Ontology Population Pipeline

## Scope

This pipeline converts reviewed legal-document chunks into a SHACL-validated RDF instance graph for ontology V0.1. It
runs after chunking and chunk screening, but before retrieval, GraphRAG, or Graphiti indexing.

The versioned execution and semantic contract is `ontology/population_contract.yaml`. The typed LLM contract is
`baml_src/ontology_population.baml`.

```text
chunk JSONL + screening + document metadata
  -> bounded reading windows
  -> typed BAML extraction
  -> exact evidence binding
  -> reviewable population JSONL
  -> deterministic RDF materialization
  -> SHACL validation
  -> validated Turtle graph
```

## Populated Modules

The source-document pipeline creates instances for:

- source/provenance: legal resource, expression, source file, chunk, provision, and exact evidence span;
- normative rule: penalty rule, conduct specification, applicability profile, and condition-expression tree;
- temporal validity: one reviewed validity interval per populated rule version;
- legal consequence: monetary penalty, point deduction, additional sanction, remedial measure, targets, and roles.

It intentionally does not create `ConductOccurrence` instances. Occurrences describe a concrete case supplied later as
query or application input, not a normative statement extracted from a decree.

It also does not create amendment, replacement, repeal, cross-document rule identity, or concept-equivalence links.
Those relations require reviewed multi-document assembly. Similar wording and chronological order are insufficient.

## Semantic Responsibilities

The LLM decides source-grounded semantic structure:

- regulated conduct candidate;
- explicit subject kind, roles, and vehicle categories;
- AND/OR/NOT condition structure;
- consequence subtype and structured values;
- exact supporting spans and provision path.

Deterministic code owns:

- exact-span verification against chunk text;
- stable record hashes and instance IRIs;
- RDF serialization and provenance edges;
- rule-scoped candidate concepts, avoiding unsupported cross-rule merging;
- SHACL execution and rejection of non-conformant graphs;
- checkpoint and resume behavior.

An LLM result is a candidate population record, not a final legal-applicability decision. Independent legal review
still owns effective dates, legal classification, rule identity, change type, and disputed interpretations.

## Inputs And Outputs

Inputs:

| Artifact | Purpose |
| --- | --- |
| Chunk JSONL | Stable chunk IDs, source hashes, text, and adjacency |
| Screening JSONL | Limits LLM calls to chunks already confirmed as penalty-content candidates |
| Document metadata JSONL | Reviewed document number and fallback validity dates |
| Population contract YAML | Versioned execution limits, namespace, scope, and semantic boundaries |

Outputs:

| Artifact | Purpose |
| --- | --- |
| Population-record JSONL | Reviewable typed intermediate data before RDF |
| Population-progress JSONL | Completed target chunks, including chunks that yielded zero rules |
| Turtle graph | Written only after the complete instance graph conforms to SHACL |

`valid_from` falls back to reviewed document metadata only when the extracted rule has no source-specific date.
`valid_to` remains absent when unknown. If no reviewed beginning is available, SHACL rejects the graph instead of code
inventing a date.

## Command

```powershell
uv run --extra dev legal-populate-ontology `
  data/chunks/document.jsonl `
  data/benchmark_synthesis/screening/document.jsonl `
  data/benchmark_synthesis/metadata/document.jsonl `
  ontology/population_contract.yaml `
  data/ontology/population/document.jsonl `
  data/ontology/progress/document.jsonl `
  data/ontology/graphs/document.ttl
```

Required provider environment variables are `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, and `LLM_TIMEOUT_SECONDS`.

The command writes each successful batch before marking its target chunks complete. A rerun skips completed targets,
including valid zero-rule results, then rematerializes and revalidates the graph from all persisted records.

## Failure Boundaries

- A missing or paraphrased evidence span fails exact evidence binding.
- Missing condition references raise during RDF materialization.
- Missing required consequence values, targets, provenance, or validity fail SHACL.
- The Turtle output is not written when SHACL validation fails.
- Provider failures use the configured bounded retry count and explicit timeout.

No regex, keyword classifier, handcrafted semantic score, or rule engine makes ontology population decisions. Chunk
screening only proposes which chunks reach the downstream typed LLM extraction boundary.

## Tests

`tests/ontology_population/` covers contract loading, typed extraction, exact evidence rejection, deterministic identity,
all four consequence types, condition trees, JSONL round trips, SHACL rejection, and the complete CLI pipeline with an
offline fake typed client.

## Cross-Chunk Rule Assembly

Population is intentionally chunk-local. A monetary penalty in one chunk and a point deduction in the next may belong
to one legal rule, but proximity alone cannot decide that. The versioned `ontology/assembly_contract.yaml` adds a
separate reviewed workflow after population:

```text
population records
  -> bounded same-document, same-scope structural windows
  -> typed LLM assembly proposals
  -> mandatory human decisions
  -> accepted-member replacement
  -> complete SHACL validation
  -> canonical population JSONL + Turtle graph
```

This phase only assembles rule components within one legal document. Cross-document amendment, replacement, repeal,
exception redirection, and concept equivalence remain outside its scope.

### 1. Build candidates offline

```powershell
uv run --extra dev legal-build-assembly-candidates `
  data/chunks/document.jsonl `
  data/ontology/population/document.jsonl `
  ontology/assembly_contract.yaml `
  data/ontology/assembly/document.candidates.json
```

Candidate generation uses only bounded chunk position within the same document and chunk scope. It proposes model input
windows; it never asserts that two population records are the same rule.

### 2. Run the LLM proposal step

This is the only assembly command that calls an LLM. Configure `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, and
`LLM_TIMEOUT_SECONDS`, regenerate the BAML client after pulling prompt changes, then run:

```powershell
.venv\Scripts\baml-cli.exe generate

uv run --extra dev legal-propose-rule-assemblies `
  data/chunks/document.jsonl `
  data/benchmark_synthesis/metadata/document.jsonl `
  data/ontology/population/document.jsonl `
  data/ontology/assembly/document.candidates.json `
  ontology/assembly_contract.yaml `
  data/ontology/assembly/document.proposals.jsonl `
  data/ontology/assembly/document.progress.jsonl `
  data/ontology/assembly/document.reviews.jsonl
```

The command persists each completed candidate window. Rerunning it with the same candidate, prompt, and model identity
skips completed windows, including windows for which the model correctly returned no proposal.

### 3. Review every proposal manually

The proposal command creates pending rows in `document.reviews.jsonl`. A legal reviewer must inspect the member records,
assembled rule, provision path, condition tree, consequence values, validity, and every exact evidence span. Edit each
row to either `accepted` or `rejected` and provide reviewer identity and review date:

```json
{
  "schema_version": "0.1.0",
  "proposal_id": "...",
  "decision": "accepted",
  "reviewer": "reviewer-name",
  "reviewed_on": "2026-08-18",
  "note": "Confirmed against the cited provisions and exact spans."
}
```

Do not accept a proposal merely because records are adjacent or use similar labels. The LLM output remains a proposal;
the human decision is the authority for canonical assembly.

### 4. Finalize offline

```powershell
uv run --extra dev legal-finalize-rule-assemblies `
  data/chunks/document.jsonl `
  data/benchmark_synthesis/metadata/document.jsonl `
  data/ontology/population/document.jsonl `
  data/ontology/assembly/document.proposals.jsonl `
  data/ontology/assembly/document.reviews.jsonl `
  ontology/population_contract.yaml `
  ontology/assembly_contract.yaml `
  data/ontology/canonical/document.jsonl `
  data/ontology/canonical/document.ttl
```

Finalization rejects missing, pending, unattributed, unknown, duplicate, or overlapping accepted reviews. Rejected
proposals preserve their original records. Accepted proposals replace exactly their reviewed member records. Neither
canonical output is written until the complete materialized graph passes SHACL.

`tests/ontology_assembly/` covers structural bounds, typed proposal translation, exact evidence, deterministic identity,
resume behavior, review gates, accepted/rejected outcomes, persistence, SHACL, and the complete three-command workflow
without making a real LLM call.

## Cross-Document Legal Change Resolution

Run this phase only after every document has a reviewed canonical population file. It resolves explicit amendment,
replacement, repeal, or supersession events across rule versions. It does not use lexical similarity or document order
as proof of legal succession.

```text
ordered corpus manifest + per-document canonical rules
  -> exhaustive bounded cross-document windows
  -> typed LLM legal-change proposals with exact later-document evidence
  -> mandatory human review
  -> reviewed earlier/later validity updates
  -> canonical LegalChange events
  -> complete multi-document SHACL validation
```

The versioned execution and semantic boundary is `ontology/change_resolution_contract.yaml`. The typed LLM contract is
`baml_src/ontology_change_resolution.baml`.

### 1. Create the ordered corpus manifest

The manifest order is reviewed candidate-generation context. It must follow the authoritative document/expression order
chosen for the corpus, but order alone never establishes a legal change. Paths are resolved relative to the manifest:

```json
{
  "schema_version": "0.1.0",
  "documents": [
    {
      "document_id": "nd-168-original",
      "order_index": 0,
      "chunks_path": "nd-168/chunks.jsonl",
      "metadata_path": "nd-168/metadata.jsonl",
      "canonical_records_path": "nd-168/canonical_population.jsonl"
    },
    {
      "document_id": "nd-238-amending",
      "order_index": 1,
      "chunks_path": "nd-238/chunks.jsonl",
      "metadata_path": "nd-238/metadata.jsonl",
      "canonical_records_path": "nd-238/canonical_population.jsonl"
    }
  ]
}
```

Each document ID, order index, and document number must be unique. Every canonical record must reference metadata and a
chunk belonging to its own manifest document.

### 2. Build exhaustive bounded candidates offline

```powershell
uv run --extra dev legal-build-change-candidates `
  data/ontology/change/corpus_manifest.json `
  ontology/change_resolution_contract.yaml `
  data/ontology/change/candidates.json
```

The command pairs earlier and later documents according to manifest order, then partitions earlier rules, later rules,
and later-document source chunks into bounded windows. It performs no keyword matching, semantic scoring, rule identity,
or change classification. `max_total_windows` fails the run before an unbounded workload is produced.

### 3. Run the LLM proposal step

This is the only legal-change command that calls an LLM. Configure `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, and
`LLM_TIMEOUT_SECONDS`, regenerate the BAML client after prompt changes, then run:

```powershell
.venv\Scripts\baml-cli.exe generate

uv run --extra dev legal-propose-legal-changes `
  data/ontology/change/corpus_manifest.json `
  data/ontology/change/candidates.json `
  ontology/change_resolution_contract.yaml `
  data/ontology/change/proposals.jsonl `
  data/ontology/change/progress.jsonl `
  data/ontology/change/reviews.jsonl
```

The LLM may propose a change only when a later-document chunk directly supports its earlier target, optional later rule,
controlled change type, effective date, authoritative provision path, and exact evidence. Similar conduct, provisions,
consequences, publication order, or missing `valid_to` are explicitly insufficient. Completed windows, including valid
zero-proposal results, are checkpointed and skipped on a compatible rerun.

### 4. Review every proposal manually

Inspect both rule versions and the exact change-source quotations. Set every generated row to `accepted` or `rejected`
and record reviewer identity, date, and notes:

```json
{
  "schema_version": "0.1.0",
  "proposal_id": "...",
  "decision": "accepted",
  "reviewer": "reviewer-name",
  "reviewed_on": "2026-08-18",
  "note": "Confirmed change target, type, effective date, and exact authoritative evidence."
}
```

Human acceptance owns the legal conclusion. Do not accept based on similarity, chronology, or the model rationale alone.

### 5. Finalize offline

```powershell
uv run --extra dev legal-finalize-legal-changes `
  data/ontology/change/corpus_manifest.json `
  data/ontology/change/proposals.jsonl `
  data/ontology/change/reviews.jsonl `
  ontology/population_contract.yaml `
  ontology/change_resolution_contract.yaml `
  data/ontology/change/canonical_population.jsonl `
  data/ontology/change/legal_change_events.jsonl `
  data/ontology/change/canonical_graph.ttl
```

An accepted event closes the earlier rule at the reviewed effective date and, when a later rule exists, opens that later
version at the same date. Because validity is part of population identity, affected rule versions receive new canonical
population IDs and the `LegalChange` event links those new IDs. A reviewed repeal may omit the later rule. Rejected and
unmatched rules remain unchanged.

Finalization rejects missing, pending, unattributed, unknown, duplicate, or overlapping accepted changes. Canonical
population JSONL, accepted-event JSONL, and Turtle are written only after the complete multi-document graph conforms to
SHACL. Transitional applicability and exception redirection remain deferred.

`tests/ontology_change/` covers manifest boundaries, exhaustive candidate bounds, typed proposals, exact-evidence
rejection, zero-result checkpoints, review gates, validity reidentification, amendment/repeal structures, persistence,
temporal CQ projection, SHACL, resume behavior, and the complete workflow without a real LLM call.

## Legal Exception And Provision Citation Resolution

Run this phase after cross-document canonical validity has been finalized. It identifies source-evidenced exceptions to
general rule versions, preserves grouped citation wording, resolves cited provisions through canonical target rules, and
creates version-specific redirects.

```text
canonical multi-document rules + source chunks
  -> exhaustive bounded source/general/target windows
  -> typed LLM exception and grouped-citation proposals
  -> mandatory human review
  -> merged canonical exception records
  -> LegalException + ProvisionCitation + redirectsTo RDF
  -> complete SHACL and CQ-EXCEPTION-001 validation
```

The versioned boundary is `ontology/exception_resolution_contract.yaml`; the typed model contract is
`baml_src/ontology_exception_resolution.baml`. The ontology relation `lornr:hasProvisionCitation` preserves the link
between an exception and each grouped `lorsp:ProvisionCitation`, while `lornr:citesExcludedProvision` remains the direct
CQ projection to resolved provisions.

### 1. Build candidates offline

Use the same ordered corpus manifest as legal-change resolution and the flattened canonical population produced after
reviewed validity changes:

```powershell
uv run --extra dev legal-build-exception-candidates `
  data/ontology/change/corpus_manifest.json `
  data/ontology/change/canonical_population.jsonl `
  ontology/exception_resolution_contract.yaml `
  data/ontology/exception/candidates.json
```

Candidate generation partitions source chunks, same-document possible general rules, and all corpus target rule versions
into bounded exhaustive windows. It performs no regex, exception-keyword classification, lexical matching, target
scoring, or rule selection. The configured total-window limit bounds the workload.

### 2. Run the LLM proposal step

This is the only exception command that calls an LLM. Configure the provider environment, regenerate BAML after prompt
changes, and run:

```powershell
.venv\Scripts\baml-cli.exe generate

uv run --extra dev legal-propose-legal-exceptions `
  data/ontology/change/corpus_manifest.json `
  data/ontology/change/canonical_population.jsonl `
  data/ontology/exception/candidates.json `
  ontology/exception_resolution_contract.yaml `
  data/ontology/exception/proposals.jsonl `
  data/ontology/exception/progress.jsonl `
  data/ontology/exception/reviews.jsonl
```

Every proposal must identify one general canonical rule, the exception provision path, exact exception evidence, and at
least one grouped exact citation. Each citation resolves only to version-specific target population IDs supplied in the
window. Similar wording, locator shape, conduct labels, chronology, or a token such as `trừ` are insufficient.

Completed windows, including valid zero-proposal results, are checkpointed. Proposal fragments for the same exception
may come from different target batches and are merged only after each fragment is independently accepted.

### 3. Review every proposal manually

For each row, verify the general rule, complete exception scope, exact evidence, citation grouping, every cited target
provision, and each redirected rule version. Then set `accepted` or `rejected` with reviewer attribution:

```json
{
  "schema_version": "0.1.0",
  "proposal_id": "...",
  "decision": "accepted",
  "reviewer": "reviewer-name",
  "reviewed_on": "2026-08-18",
  "note": "Confirmed general rule, exception scope, grouped locator, provisions, and target versions."
}
```

Do not accept a partial fragment unless it is legally correct as part of the final merged exception. Human review owns
exception classification, locator grouping, provision resolution, and redirect selection.

### 4. Finalize offline

```powershell
uv run --extra dev legal-finalize-legal-exceptions `
  data/ontology/change/corpus_manifest.json `
  data/ontology/change/canonical_population.jsonl `
  data/ontology/change/legal_change_events.jsonl `
  data/ontology/exception/proposals.jsonl `
  data/ontology/exception/reviews.jsonl `
  ontology/population_contract.yaml `
  ontology/exception_resolution_contract.yaml `
  data/ontology/exception/legal_exceptions.jsonl `
  data/ontology/exception/canonical_graph.ttl
```

Finalization merges accepted fragments with the same general rule, source provision, and exact exception evidence. It
deduplicates target versions without collapsing different citation wordings. The RDF graph contains bidirectional
`hasException`/`exceptionTo`, grouped `ProvisionCitation` objects, direct cited-provision projections, version-specific
`redirectsTo`, source provision, and exact evidence.

Outputs are written only after full SHACL conformance. Missing, pending, unattributed, unknown, or duplicate reviews
block finalization. Rejected proposals produce no exception semantics.

V0.1 resolves citations only through provisions belonging to canonical populated target rules. A citation whose target
provision has not yet been populated must remain unresolved; the LLM must not invent a target path. Legal priority and
occurrence-level exception applicability remain deferred.

`tests/ontology_exception/` covers contract and command packaging, exhaustive bounds, typed source/target membership,
exact exception and citation binding, zero-result resume, fragment merging, review gates, grouped citation RDF, SHACL,
CQ projection, persistence, and the complete three-command workflow without a real LLM call.

## Reviewed Concept Normalization And Taxonomy Resolution

Run this phase after canonical rules, legal changes, and exceptions are available. Population intentionally creates one
rule-scoped conduct, role, or vehicle concept per canonical rule. This phase is the only boundary allowed to merge those
occurrences into reusable identities or add taxonomy relations.

```text
canonical population
  -> extract rule-scoped conduct, role, and vehicle occurrences
  -> exhaustive same-kind candidate pairs
  -> typed LLM semantic proposals with exact evidence from both sides
  -> mandatory human review
  -> accepted-equivalence components and reviewed taxonomy relations
  -> canonical concept registry + remapped RDF
  -> complete SHACL and CQ validation
```

The versioned boundary is `ontology/concept_resolution_contract.yaml`; the typed model contract is
`baml_src/ontology_concept_resolution.baml`. Subject kinds are not re-extracted because `NaturalPerson` and
`Organization` are already controlled ontology concepts. Consequence targets remain deferred because one target label
may denote a person, role, license, vehicle, object, or state rather than one homogeneous taxonomy.

### 1. Build candidates offline

```powershell
uv run --extra dev legal-build-concept-candidates `
  data/ontology/change/canonical_population.jsonl `
  ontology/concept_resolution_contract.yaml `
  data/ontology/concept/candidates.json
```

Candidate generation extracts `RegulatedConduct`, `LegalSubjectRole`, and `VehicleCategory` occurrences and enumerates
every pair within each kind. It performs no regex, keyword filtering, lexical similarity, embeddings, handcrafted score,
or automatic semantic merge. `max_total_pairs` bounds the exhaustive workload.

### 2. Run the LLM proposal step

This is the only concept-resolution command that calls an LLM:

```powershell
.venv\Scripts\baml-cli.exe generate

uv run --extra dev legal-propose-concept-relations `
  data/ontology/change/corpus_manifest.json `
  data/ontology/concept/candidates.json `
  ontology/concept_resolution_contract.yaml `
  data/ontology/concept/proposals.jsonl `
  data/ontology/concept/progress.jsonl `
  data/ontology/concept/reviews.jsonl
```

For each pair the model may propose `SameConcept`, `LeftBroaderThanRight`, `LeftNarrowerThanRight`, `Related`, or
`Distinct`, or return no proposal when evidence is insufficient. Every proposal must copy exact evidence owned by both
occurrences. `SameConcept` must select one supplied Vietnamese label; other relations cannot silently choose a canonical
label. Completed zero-proposal pairs are checkpointed so retries do not repeat finished model work.

### 3. Review every proposal manually

Review legal meaning, both exact evidence spans, direction, and preferred label. Do not accept equality from shared words,
provision location, consequence values, or similar rule scope. Resolve every generated review row:

```json
{
  "schema_version": "0.1.0",
  "proposal_id": "...",
  "decision": "accepted",
  "reviewer": "reviewer-name",
  "reviewed_on": "2026-08-18",
  "note": "Confirmed legal identity and both evidence sources."
}
```

### 4. Finalize offline

```powershell
uv run --extra dev legal-finalize-concept-resolution `
  data/ontology/change/corpus_manifest.json `
  data/ontology/change/canonical_population.jsonl `
  data/ontology/change/legal_change_events.jsonl `
  data/ontology/exception/legal_exceptions.jsonl `
  data/ontology/concept/candidates.json `
  data/ontology/concept/proposals.jsonl `
  data/ontology/concept/reviews.jsonl `
  ontology/population_contract.yaml `
  ontology/concept_resolution_contract.yaml `
  data/ontology/concept/canonical_concepts.jsonl `
  data/ontology/concept/concept_relations.jsonl `
  data/ontology/concept/canonical_graph.ttl
```

Finalization verifies that candidates still match the canonical population, requires complete attributable reviews,
forms equivalence components only from accepted `SameConcept` proposals, and rejects conflicting labels, contradictory
relations, self-relations after merging, or broader cycles. Rejected and zero-proposal pairs remain separate concepts.

The RDF graph replaces rule-scoped concept references with canonical concept IRIs, preserves all Vietnamese aliases and
source evidence, and emits SKOS `broader`, `narrower`, or `related` only for reviewed relations. `Distinct` remains an
auditable JSON decision and is not converted into an OWL disjointness axiom. SKOS hierarchy does not imply
`rdfs:subClassOf`, rule equivalence, legal priority, condition satisfaction, or occurrence-level applicability.

`tests/ontology_concept/` covers exhaustive same-kind pairing, configured bounds, persistence, typed fake-model input,
exact evidence binding, invalid labels and paraphrases, zero-result resume, mandatory review, conservative rejection,
canonical RDF remapping, reviewed SKOS hierarchy, SHACL, and the complete three-command workflow without a real LLM.
