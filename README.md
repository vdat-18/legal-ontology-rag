# Legal Ontology RAG

This repository provides multiple retrieval backends over the same versioned legal-document chunks:

- Graphiti uses Neo4j for temporal entities, relationships, and source provenance.
- Naive RAG uses Milvus with semantic dense-vector search (default), BM25 keyword search, hybrid RRF, Cross-Encoder reranking, or bounded article-level hierarchical expansion selected by `NAIVE_RAG_SEARCH_MODE`.
- Adaptive RAG uses one LLM planning call to select direct or one-shot multi-hop retrieval, reuses a configured Naive search mode for every leg, and fuses multi-hop results with RRF before one answer-generation call.
- Corrective RAG uses Milvus retrieval plus CRAG grading, refinement, optional web search, and answer generation.
- Ontology RAG combines ontology rule retrieval with Naive semantic/BM25 fallback and preserves source-chunk provenance for shared scoring.
- Attu is an optional Milvus administration UI; it is not part of the application data path.

## Local Infrastructure

Copy `.env.example` to `.env`, replace every credential placeholder, and keep the file uncommitted. Services are grouped by Compose profile so each pipeline starts only its own dependencies:

```powershell
docker compose --profile neo4j up -d neo4j
docker compose --profile milvus up -d milvus
docker compose --profile attu up -d attu
```

Compose waits for etcd and MinIO before starting Milvus, and waits for Milvus before starting Attu. Published ports bind to `127.0.0.1` by default. Override `INFRA_HOST_BIND` only when remote access is intentionally protected by a firewall or private network.
The MinIO service uses a digest-pinned Chainguard image and retains the previous container's root UID so existing named volumes remain readable.

The `app` image is a batch CLI image, not a daemon. Start the required backend first, then run a command explicitly:

```powershell
docker compose --profile milvus run --rm app legal-naive-rag-index data/processed/groundtruth
docker compose --profile neo4j run --rm app legal-graphiti-index data/processed/groundtruth `
  --corpus-config data/config/groundtruth_corpus_v1.json
```

The image contains LibreOffice for legacy DOC/XLS attachment conversion. Pass `--renderer-executable /usr/bin/soffice` to parsing commands that need it. Source code is built into the image so the generated BAML client is not hidden by a host bind mount; rebuild the image after code changes.

## Milvus Data Compatibility

Milvus uses one shared database configured by `MILVUS_DATABASE`. Each retrieval method owns a separate collection inside that database, such as `naive_rag_chunks` and `legal_chunks`, so methods share infrastructure without mixing incompatible schemas or embeddings.

When upgrading an existing environment, rename `NAIVE_RAG_MILVUS_URI` to `MILVUS_URI`, `NAIVE_RAG_MILVUS_TOKEN` to `MILVUS_TOKEN`, and `NAIVE_RAG_MILVUS_DATABASE` to `MILVUS_DATABASE`. No automatic data migration runs: keep the previous database value to reuse existing data, or select `legal_rag` and rebuild the method-specific collections there.

The naive index stores one source-text field (`text`), its vector, and chunker/embedding-model identity metadata with each chunk. The vector is generated from the chunk's `embedding_text` in memory, but that duplicate input is not persisted in Milvus. Re-running the indexer resumes missing chunks after a partial run. It fails instead of mixing data when the collection schema, vector dimension, index metric, corpus IDs, chunk content, chunker configuration, or embedding revision differs.

The `hierarchical` Naive mode keeps the same Milvus rows. It uses hybrid RRF seeds, validates them against `NAIVE_RAG_CHUNKS_DIR`, and expands only chunks with one unambiguous parser-owned article context. Each article unit is bounded by `NAIVE_RAG_HIERARCHICAL_MAX_CHUNKS_PER_UNIT` and `NAIVE_RAG_HIERARCHICAL_MAX_TOKENS_PER_UNIT`; chunks without safe structure remain leaf results.

Adaptive RAG does not create another Milvus collection. It defaults to `hybrid_rerank`, always retrieves the original question, and adds two or more standalone subquery legs only when the planner selects `multi_hop`. The planner route and subqueries are stored in evaluation prediction metadata for audit. Run it through the shared evaluation command:

```powershell
uv run --extra naive-rag legal-evaluate run --pipeline adaptive `
  --max-samples 20 --limit 10 --reference-date 2026-10-01 `
  --run-dir data/evaluation/runs/adaptive_smoke
```

Existing naive collections that still contain the former `embedding_text`, `record_scope`, or `chunk_index` fields must be dropped and rebuilt once with the current schema.

Collections created before this metadata was introduced are intentionally incompatible. Back them up, then configure the shared `MILVUS_DATABASE` and a method-specific collection name, or explicitly drop and rebuild the old collection.

Server-backed indexing flushes inserts before final verification. Milvus Lite on Windows currently has a broken flush RPC; set `NAIVE_RAG_MILVUS_FLUSH_AFTER_INSERT=false` only for that local backend.

## Corrective RAG

Install the CRAG API, Milvus, embedding, and Streamlit runtime dependencies:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[corrective-rag]"
```

Corrective RAG uses the shared Docker Milvus instance, not Milvus Lite. Start Milvus before indexing, retrieval, or evaluation:

```powershell
docker compose --profile milvus up -d milvus
```

Use these settings in `.env`:

```env
EMBEDDING_LOCAL_FILES_ONLY=true
MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=root:<MILVUS_ROOT_PASSWORD>
MILVUS_DATABASE=legal_rag
CRAG_MILVUS_COLLECTION_NAME=legal_chunks
CRAG_MILVUS_INDEX_TYPE=IVF_FLAT
CRAG_MILVUS_METRIC_TYPE=COSINE
CRAG_MILVUS_NLIST=128
CRAG_MILVUS_NPROBE=16
CRAG_MILVUS_FLUSH_AFTER_INSERT=true
MILVUS_INDEX_BATCH_SIZE=32
CRAG_WEB_SEARCH_ALLOWED_DOMAINS=chinhphu.vn,vanban.chinhphu.vn,moj.gov.vn
```

Corrective RAG validates its Milvus collection schema, vector dimension, vector
metric, chunk content hashes, chunker config hash, and embedding
model/revision metadata before reusing indexed rows. Collections created by an
older CRAG indexer without these metadata fields are intentionally rejected;
drop and rebuild `CRAG_MILVUS_COLLECTION_NAME` rather than mixing stale chunks.

When `CRAG_WEB_SEARCH_ENABLED=true`, web search requires a non-empty domain
allowlist. Only HTTP(S) result URLs are fetched, redirects are checked again at
their final URL, and loopback, private, link-local, reserved, or multicast
addresses are blocked.

Index the processed chunks once after starting a fresh Milvus database:

```powershell
python -m legal_rag.corrective_rag.api.cli index --data-dir data/processed/groundtruth
```

Start the FastAPI server after Milvus is running and chunks are indexed:

```powershell
python -m legal_rag.corrective_rag.api --host 127.0.0.1 --port 8000
```

Open the interactive API docs at <http://127.0.0.1:8000/docs>.

Run the full CRAG pipeline:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/crag/query `
  -ContentType "application/json" `
  -Body '{"query":"Vuot den do xe may bi phat bao nhieu?","limit":5}'
```

Run vector-only retrieval from Milvus:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/crag/search `
  -ContentType "application/json" `
  -Body '{"query":"muc phat vuot den do xe may","limit":5}'
```

Test direct web search without loading Milvus or the embedding model:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/crag/web-search `
  -ContentType "application/json" `
  -Body '{"query":"muc phat vuot den do xe may nghi dinh 168","limit":3}'
```

Inside the full CRAG pipeline, web search is used only for `INCORRECT` and `AMBIGUOUS` retrieval actions. A `CORRECT` retrieval intentionally keeps `web_search_count` at `0`.

## Comparing RAG variants

Use `legal-evaluate run` for the common end-to-end contract. `--max-samples` limits the number of ground-truth rows; `--limit` is the maximum number of pipeline-native retrieval units supplied per question. Keep `--limit`, `--k-values`, `--concurrency`, `--reference-date`, the answer model, embedding profile, corpus snapshot, and optional judge settings equal across compared runs.

Naive, Adaptive, and Ontology need the `naive-rag` extra. Corrective needs `corrective-rag`; Graphiti dependencies are part of the base install. Query rewriting for multi-turn rows is enabled by default. `--no-rewrite-query` is an explicit ablation for Naive, Adaptive, and Ontology. Graphiti and Corrective always use their intrinsic history handling.

```powershell
$common = @(
  "--ground-truth", "data/benchmark_synthesis/groundtruth.jsonl",
  "--max-samples", "20",
  "--limit", "10",
  "--k-values", "1,3,5,10",
  "--concurrency", "1",
  "--reference-date", "2026-10-01"
)

uv run --extra naive-rag legal-evaluate run --pipeline naive --naive-search-mode hybrid @common `
  --run-dir data/evaluation/runs/naive_hybrid_smoke
uv run --extra naive-rag legal-evaluate run --pipeline adaptive @common `
  --run-dir data/evaluation/runs/adaptive_smoke
uv run --extra corrective-rag legal-evaluate run --pipeline corrective @common `
  --run-dir data/evaluation/runs/corrective_closed_smoke
uv run legal-evaluate run --pipeline graphiti @common `
  --group-id traffic_penalty_groundtruth_v1 `
  --corpus-config data/config/groundtruth_corpus_v1.json `
  --run-dir data/evaluation/runs/graphiti_smoke
uv run --extra naive-rag legal-evaluate run --pipeline ontology @common `
  --ontology-paths data/ontology/exception/graphs/traffic_penalty_v0.5.ttl `
  --run-dir data/evaluation/runs/ontology_smoke
```

Add `--generation` to every run when comparing the six judge metrics: faithfulness, correctness, answer relevancy, temporal correctness, answerability correctness, and required-claim coverage. Judge calls are scored after prediction and are not included in prediction-stage latency or token usage. Graphiti evaluation also omits product-only citation diagnostics and follow-up suggestions from those operational measurements.

Run Corrective twice when web search is part of the study, once with `CRAG_WEB_SEARCH_ENABLED=false` and once with `true`. The comparison command deliberately treats live-web and closed-corpus runs as incompatible because web content is not a frozen evidence snapshot.

```powershell
uv run legal-evaluate compare `
  data/evaluation/runs/naive_hybrid_smoke/metrics.json `
  data/evaluation/runs/adaptive_smoke/metrics.json `
  data/evaluation/runs/corrective_closed_smoke/metrics.json `
  data/evaluation/runs/graphiti_smoke/metrics.json `
  data/evaluation/runs/ontology_smoke/metrics.json `
  --output data/evaluation/rag_variant_comparison.json
```

Comparison validates the evaluated sample set and shared experiment controls before reporting native-unit retrieval metrics, source-evidence coverage, generation metrics, context budget, latency, and provider-reported token usage. History resolution must be enabled consistently across variants. Each pipeline's more specific rewrite policy is recorded and compared within that pipeline because Naive, Adaptive, Corrective, Graphiti, and Ontology implement history resolution differently by design. The report does not create a composite score. `--allow-incompatible` only writes diagnostics; it does not make incompatible results fair to rank.

The current committed ontology runtime graph is partial. The CLI records coverage of gold chunk provenance and warns when coverage is incomplete. Treat its result as an ontology-assisted hybrid result with fallback context, not as evidence that a full-corpus ontology has been completed. The Graphiti corpus config is required so the indexed group and source-document boundary can be validated.

For multi-turn evaluation, export one of the versioned conversation artifacts first:

```powershell
uv run legal-export-groundtruth --multi-turn `
  data/benchmark_synthesis/multi_turn_candidates_1k_luna_high_v1_2.jsonl `
  data/evaluation/groundtruth_multi_turn.jsonl `
  --scenarios data/benchmark_synthesis/scenarios.jsonl `
  --legal-facts data/benchmark_synthesis/legal_facts.jsonl
```

The scenarios artifact supplies the target's reviewed `bundle_kind`; the exporter never infers a temporal comparison merely because required facts have different effective dates. For a temporal target, `query_dates` comes only from the final target's required-claim facts plus explicit dates in its standalone question. Dates from history and unused scenario facts are excluded. Legacy schema `1.0.0` exports preserve gold prior answers as oracle history; they cannot reconstruct `standalone_question` or `missing_information`, so those fields remain null. Keep single-turn and oracle-history multi-turn reports separate.

## Ontology Population

Populate the ontology from screened legal-document chunks before any GraphRAG indexing:

```powershell
uv run --extra dev legal-check-ontology-population `
  <chunks.jsonl> <screening.jsonl> <metadata.jsonl> ontology/population_contract.yaml `
  <preflight-report.json> --target-ids-path <target-selection.json>
```

This offline preflight fingerprints inputs, validates structural consistency, and records the selected pilot targets.

The recommended runtime has six user-facing steps. Each command wraps deterministic candidate preparation, resumable
semantic LLM work, and finalization. Assembly adds a second LLM phase to review proposal components globally. All six
steps default to at most 30 concurrent LLM batches; pass `--max-concurrency` only when a different limit is required:

```powershell
uv run --extra dev legal-extract-ontology-conducts `
  <chunks.jsonl> <screening.jsonl> <metadata.jsonl> ontology/population_contract.yaml `
  <conducts.jsonl> --target-ids-path <selection.json> --max-concurrency 30

uv run --extra dev legal-bind-ontology-population `
  <chunks.jsonl> <screening.jsonl> <metadata.jsonl> ontology/population_contract.yaml `
  <conducts.jsonl> <population.jsonl> <population-progress.jsonl> <population.ttl> `
  --target-ids-path <selection.json> --max-concurrency 30

uv run --extra dev legal-assemble-ontology-rules `
  <chunks.jsonl> <metadata.jsonl> <population.jsonl> `
  ontology/population_contract.yaml ontology/assembly_contract.yaml `
  <assembly-candidates.json> <assembly-proposals.jsonl> <assembly-progress.jsonl> `
  <canonical-population.jsonl> <canonical.ttl> --max-concurrency 30

uv run --extra dev legal-resolve-ontology-changes `
  <corpus-manifest.json> ontology/change_resolution_contract.yaml ontology/population_contract.yaml `
  <change-candidates.json> <change-proposals.jsonl> <change-progress.jsonl> `
  <canonical-population.jsonl> <change-events.jsonl> <canonical.ttl> --max-concurrency 30

uv run --extra dev legal-resolve-ontology-exceptions `
  <corpus-manifest.json> <canonical-population.jsonl> <change-events.jsonl> `
  ontology/exception_resolution_contract.yaml ontology/population_contract.yaml `
  <exception-candidates.json> <exception-proposals.jsonl> <exception-progress.jsonl> `
  <legal-exceptions.jsonl> <canonical.ttl> --max-concurrency 30

uv run --extra dev legal-resolve-ontology-concepts `
  <corpus-manifest.json> <canonical-population.jsonl> <change-events.jsonl> <legal-exceptions.jsonl> `
  ontology/population_contract.yaml ontology/concept_resolution_contract.yaml `
  <concept-candidates.json> <concept-proposals.jsonl> <concept-progress.jsonl> `
  <canonical-concepts.jsonl> <concept-relations.jsonl> <canonical.ttl> `
  --max-concurrency 30
```

Step 6 builds one global inventory for each reusable controlled-vocabulary kind. After a complete run, add
`--rerun-candidate-ids-path <rerun-selection.json>` to replace only a reviewed inventory while preserving the other
concept proposals.

Omit pilot-selection options for the full document. Re-running a step resumes from compatible conduct/proposal progress
instead of repeating completed LLM work. The lower-level candidate, proposal, and finalize commands remain available for
debugging and backward compatibility. See `docs/ontology_population_pipeline.md` for dependencies and output contracts.

After reviewing Step 1, keep the existing `conducts.jsonl` and add
`--rerun-target-ids-path <rerun-selection.json>` to replace only incorrect targets. The rerun selection uses the pilot
selection schema, must be a subset of the original targets, and resumes completed rerun batches when invoked again.

Review the ontology's external vocabulary boundary locally with:

```powershell
uv run --extra dev legal-review-ontology-reuse `
  ontology/external_reuse_registry.yaml `
  data/ontology/qa/external_reuse_review.json
```

The registry records source, version, license, attribution, compatibility, boundaries, and exact term IRIs for Dublin
Core, PROV-O, SKOS, and OWL-Time. It also records ELI as candidate-only and the remaining legal resources as reviewed
references or patterns. The command rejects unregistered semantic IRIs, candidate/reference namespace use, missing
foundational declarations, and external OWL imports. Technical checks pass, and project-owner acceptance of all ten
reviewed decisions was recorded on 2026-08-18.

Run the AI-assisted internal legal review locally with:

```powershell
uv run --extra dev legal-review-ontology-internally `
  ontology/legal_review_contract.yaml `
  ontology/legal_review_decisions.yaml `
  ontology/reviews/v0.1/legal_review_packet.md `
  ontology/reviews/v0.1/legal_review_readiness.json
```

The packet contains 11 bounded legal questions covering every open modeling topic and every V0.1 glossary term. Codex
records evidence-backed internal recommendations, while the command verifies traceability, evidence files, coverage,
and attributable review records. This is AI-assisted project analysis, not licensed legal advice or independent
professional certification. Project-owner acceptance remains the final V0.1 freeze gate.

Run the ontology semantic release gates with pinned ROBOT and HermiT tooling:

```powershell
uv run --extra dev legal-validate-ontology-semantics `
  ontology/semantic_qa_contract.yaml `
  data/ontology/qa/artifacts `
  data/ontology/qa/semantic_qa_report.json
```

This command uses the pinned `obolibrary/robot:v1.9.10` Docker image with networking disabled. It validates the complete
OWL 2 DL import closure, runs HermiT, audits unsatisfiable classes and unintended inferred equivalences, executes the
ROBOT quality report, and records traceability/CQ coverage. The report is written before the command fails an open gate.
The report also binds its result to a SHA-256 digest of the complete semantic input closure, so a report becomes stale
when an ontology module, SHACL graph, CQ query, fixture, registry, catalog, policy, or semantic contract changes.
The command produces the semantic report required by the packaging step. No semantic QA report or packaged release is
currently committed under `data/ontology/`; do not describe a local ontology build as released until these gates have
been run for that exact input closure.

After every semantic gate passes, build the immutable V0.1 distribution with:

```powershell
uv run --extra dev legal-package-ontology-release `
  ontology/release_contract.yaml `
  data/ontology/qa/semantic_qa_report.json `
  data/ontology/releases/v0.1.0 `
  data/ontology/releases/v0.1.0/release_report.json
```

The release command verifies the semantic-input digest, ontology/version IRIs, license metadata and file, the complete
external-reuse registry, internal legal-review packet structure, explicit project-owner approvals, and the complete
source inventory. It always writes a release report. Once all gates pass, it emits a reproducible ZIP containing a
machine-readable manifest and
a separate `.sha256` checksum. Runtime data, populated instances, retrieval indexes, and GraphRAG artifacts are excluded.

Optionally build the proposed canonical namespace publication before requesting W3ID registration:

```powershell
uv run --extra dev legal-build-ontology-publication `
  ontology/namespace_publication.yaml `
  data/ontology/publication/site `
  data/ontology/publication/report.json
```

This creates 14 canonical/versioned HTML and byte-identical Turtle endpoints plus a SHA-256 publication manifest. It
also validates the copy-ready W3ID `.htaccess` and `README.md` bundle. The proposed GitHub Pages target must be deployed
and externally verified before the W3ID pull request is submitted. See `docs/ontology_namespace_publication.md`.

The manual `Deploy ontology publication` GitHub Actions workflow publishes the generated site with least-privilege Pages
permissions and commit-pinned actions. After deployment, run the bounded live verifier in `deployment` mode; after the
W3ID pull request is merged, run it again in `w3id` mode to verify all 28 content-negotiated requests and redirects.

## Operations

- Health: Neo4j exposes its browser health endpoint on port 7474; Milvus exposes health and Prometheus metrics on port 9091. Container logs use bounded rotation.
- Secrets: Neo4j, MinIO, and Milvus have no source-controlled fallback credentials. Milvus authentication is enabled and clients use `MILVUS_TOKEN`.
- Backup: use Neo4j's supported dump/backup tooling for `neo4j_data`. Use a Milvus Backup release compatible with Milvus 2.4 for collections, or stop Milvus and snapshot `etcd_data`, `minio_data`, and `milvus_data` together. A partial snapshot of only one Milvus volume is not a valid backup.
- Restore: restore into an isolated environment first, start the matching database version, run the integration smoke tests, and only then replace the active deployment. Keep the original volumes until the restored corpus row IDs and representative queries are verified.

Authentication environment variables initialize fresh datastores; they do not reliably rotate credentials already persisted in old volumes. Before reusing an existing Neo4j or Milvus volume, rotate its credential through the database's supported administration API and make `.env` match it, or restore the data into freshly initialized secured volumes.

Run opt-in datastore smoke tests against running services with:

```powershell
$env:RUN_INFRA_INTEGRATION="1"
.venv\Scripts\python.exe -m pytest -q tests/integration/test_datastores.py
```

## Checks

Regenerate the local BAML client after changing or pulling files under `baml_src/`:

```powershell
.venv\Scripts\baml-cli.exe generate
```

```powershell
.venv\Scripts\ruff.exe format --check .
.venv\Scripts\ruff.exe check .
.venv\Scripts\python.exe -m pytest -q
uv run --extra dev legal-validate-ontology-semantics ontology/semantic_qa_contract.yaml `
  data/ontology/qa/artifacts data/ontology/qa/semantic_qa_report.json
uv run --extra dev legal-package-ontology-release ontology/release_contract.yaml `
  data/ontology/qa/semantic_qa_report.json data/ontology/releases/v0.1.0 `
  data/ontology/releases/v0.1.0/release_report.json
uv run --extra dev legal-build-ontology-publication ontology/namespace_publication.yaml `
  data/ontology/publication/site data/ontology/publication/report.json
uv run --extra dev legal-verify-ontology-publication ontology/namespace_publication.yaml `
  deployment data/ontology/publication/deployment_verification.json
docker compose config --quiet
```
