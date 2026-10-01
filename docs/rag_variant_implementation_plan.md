# Kế hoạch triển khai và đánh giá các biến thể RAG

> Trạng thái cập nhật: phần implementation cho các biến thể và harness so sánh đã có trong codebase. Công việc còn lại là chuẩn bị hạ tầng/index khớp corpus, chạy smoke trên dịch vụ thật, chạy đủ 1.000 mẫu và viết báo cáo kết quả. Tài liệu này vừa lưu các quyết định thiết kế đã triển khai, vừa là runbook cho phần benchmark còn lại.

## 1. Mục tiêu

Đánh giá công bằng các biến thể RAG trên cùng bộ 1.000 câu hỏi pháp luật giao thông Việt Nam, cùng corpus và cùng các control của thí nghiệm. Kết quả phải cho thấy riêng chất lượng retrieval, chất lượng câu trả lời và chi phí vận hành; không gộp các chiều này thành một composite score tự đặt trọng số.

Mục tiêu hiện tại không còn là bổ sung một pipeline còn thiếu. Code đã có:

- năm mode Naive RAG;
- Adaptive RAG one-shot multi-hop;
- Corrective RAG với hai chế độ closed-corpus và live-web;
- Graphiti RAG;
- Ontology RAG;
- contract evaluation chung, LLM judge sáu metric và module so sánh nhiều run.

## 2. Phạm vi và các quyết định đã chốt

- Giữ ba baseline Naive cũ (`keyword`, `semantic`, `hybrid`) để kết quả trước và sau có thể đối chiếu.
- Milvus lưu một row cho mỗi `chunk_id`. Keyword, semantic, hierarchy và reranker dùng lại cùng corpus; không tạo collection riêng theo mode.
- Không thêm MRR hoặc NDCG. Retrieval được báo cáo bằng Hit@k, Precision@k, Recall@k, Context Precision, Context Recall và claim evidence coverage.
- Multi-hop v1 dùng one-shot decomposition: plan một lần, retrieve query gốc và các subquery, fuse một lần rồi generate một lần.
- `--limit` luôn là budget retrieval cho từng câu hỏi. `--max-samples` mới là số dòng ground truth được chạy.
- Multi-turn history được truyền dưới dạng có cấu trúc. Naive, Adaptive và Ontology mặc định bật resolution cho câu hỏi phụ thuộc lịch sử; Graphiti và Corrective dùng cơ chế history nội tại của pipeline.
- `--reference-date` phải được truyền rõ trong campaign để câu hỏi thời gian tương đối và identity không thay đổi theo ngày chạy.
- Cùng một giá trị `--concurrency` phải được dùng khi so latency. Judge concurrency và batch size cũng phải giống nhau nếu so generation metrics.
- Corrective RAG bật web là một experiment khác với closed-corpus. Kết quả live-web không được trộn vào bảng xếp hạng closed-corpus như thể chúng có cùng nguồn bằng chứng.
- Ontology runtime hiện dùng graph có structured coverage một phần. Không mô tả nó là ontology canonical bao phủ toàn corpus.
- Không tự động chạy đủ 1.000 câu trong lúc sửa code. Chỉ chạy full benchmark sau khi live preflight và smoke pass.

## 3. Hiện trạng codebase

### 3.1 Dataset và corpus

Các kiểm tra trực tiếp trên artifact hiện tại cho thấy:

- `data/benchmark_synthesis/groundtruth.jsonl`: 1.000 evaluation samples;
- `single_hop`: 334, `multi_hop`: 333, `temporal`: 333;
- bảy `query_style`: sáu nhóm có 143 mẫu và `keyword_style` có 142 mẫu;
- `data/processed/groundtruth/*.chunks.jsonl`: 35 file, tổng 8.351 chunks.

Mỗi chunk có stable `chunk_id`, text dùng cho retrieval, structural context, neighbor links và provenance. Evaluation chấm gold theo evidence `chunk_id`, kể cả khi pipeline trả về một graph fact, ontology rule hoặc hierarchical unit làm native retrieval unit.

Kết quả Naive hybrid cũ trên 1.000 mẫu (`Hit@10 = 0.679`, `Recall@10 = 0.465`, `Precision@10 = 0.1097`) chỉ là historical baseline. Phải chạy lại bằng contract và manifest hiện tại trước khi dùng trong bảng cuối.

### 3.2 Storage và active-index preflight

Naive, Adaptive, Ontology và Corrective dùng Milvus. Preflight đọc active collection, đối chiếu chính xác tập `chunk_id` và các field không phải vector với corpus local, rồi tạo fingerprint gồm row count, corpus digest và embedding identity. Collection thiếu, thừa, stale hoặc dùng embedding khác phải fail trước khi benchmark tốn LLM calls.

Graphiti dùng một group Neo4j. Preflight snapshot đọc có phân trang các episode, entity, mention và fact của đúng `group_id`; kiểm tra group boundary, fact provenance và tập episode so với chunks được chọn bởi corpus config; sau đó tạo digest ổn định, bỏ qua UUID ngẫu nhiên và ingestion timestamp. Graph rỗng, thiếu provenance, vượt group hoặc không khớp corpus phải fail.

Corpus config chuẩn cho campaign này là:

- path: `data/config/groundtruth_corpus_v1.json`;
- group: `traffic_penalty_groundtruth_v1`.

`--corpus-config` và `--group-id` là bắt buộc cho Graphiti evaluation. Corpus config có thể chỉ chứa selection metadata; document number được đọc từ metadata artifact khi có và dùng fallback ổn định khi cần.

### 3.3 Data flow chung

```text
groundtruth row
  -> EvaluationRecord
       question + structured history + temporal metadata + gold evidence
  -> selected pipeline
       optional/intrinsic standalone-query resolution
       pipeline-native retrieval
       answer generation
  -> normalized PipelinePrediction
       retrieved_units
       evidence chunk IDs
       generation contexts
       latency + observed provider usage
  -> retrieval metrics + slices
  -> optional six-metric LLM judge
  -> metrics.json + predictions.jsonl + manifest.json
```

`retrieved_units`, evidence chunks và generation contexts là ba lớp riêng:

- native unit phản ánh thứ pipeline thực sự xếp hạng;
- evidence chunk dùng để đối chiếu gold provenance;
- generation context phản ánh lượng nội dung thực sự đưa cho answer model.

Module comparison không đánh đồng số graph fact, ontology rule hoặc hierarchical unit với số chunk.

### 3.4 Ma trận biến thể hiện tại

| Nhóm | Experiment | Implementation | Trạng thái code | Trạng thái benchmark ngoài |
|---|---|---|---|---|
| Naive | `naive_keyword` | BM25L | Đã implement và có test | Chưa chạy full campaign mới |
| Naive | `naive_semantic` | Milvus dense retrieval | Đã implement và có test | Chưa chạy full campaign mới |
| Naive | `naive_hybrid` | BM25L + dense + deterministic RRF | Đã implement và có test | Chưa chạy full campaign mới |
| Naive | `naive_hybrid_rerank` | Hybrid candidate pool + pinned Cross-Encoder | Đã implement và có test | Chưa chạy full campaign mới |
| Naive | `naive_hierarchical` | Hybrid seeds + bounded structural units | Đã implement và có test | Chưa chạy full campaign mới |
| Adaptive | `adaptive_multihop` | Direct/multi-hop planner + shared RRF | Đã implement và có test | Chưa chạy full campaign mới |
| Corrective | `corrective_closed` | CRAG, web search disabled | Đã implement và có test | Cần live Milvus/LLM smoke |
| Corrective | `corrective_live_web` | CRAG, allowlisted web search enabled | Đã implement và có test | Cần live web smoke riêng |
| Graph | `graphiti` | Graph search, expansion và rank fusion | Đã implement và có test | Cần live Neo4j snapshot preflight |
| Ontology | `ontology_partial` | Ontology-assisted hybrid retrieval | Đã implement và có test | Cần benchmark với coverage caveat |

`retrieve --retriever vector` là retrieval-only baseline của Corrective RAG. Nó không thay thế end-to-end `corrective` vì CRAG còn grading, refinement, reformulation và optional web search.

## 4. Lịch sử triển khai và quyết định theo phase

### Phase 0 - Khóa dataset và contract

Trạng thái: hoàn thành ở mức code/data.

- [x] Xác nhận đúng 1.000 samples và phân phối 334/333/333.
- [x] Xác nhận 35 file chunk, 8.351 chunks.
- [x] Dùng stable evaluation record ID cho single-turn và multi-turn.
- [x] Tách output theo deterministic run identity và run directory.
- [x] Không ghi đè historical report mặc định.

Live collection/graph là state ngoài repository nên vẫn phải preflight ở đầu mỗi campaign.

### Phase 1 - Accent-aware BM25 cho query không dấu

Trạng thái: deferred có chủ đích.

Shared tokenizer chưa được đổi để tạo thêm folded-token namespace. Query rewrite của Naive và Adaptive chỉ chạy khi có history, nên không thể coi nó là lời giải cho single-turn `query_style=no_diacritics`. Báo cáo cuối phải giữ slice này riêng. Chỉ mở lại phase này nếu benchmark mới xác nhận regression còn đáng kể.

Thiết kế đã lưu nếu cần tiếp tục:

- tạo token giữ dấu và token ASCII-folded trong namespace riêng;
- normalize Unicode, đổi `đ/Đ`, giữ số và không nối bigram qua punctuation;
- dùng cùng semantics ở VnCoreNLP path và fallback path;
- không thêm field hoặc row Milvus, không re-chunk corpus.

### Phase 2 - Hybrid RRF + Cross-Encoder rerank

Trạng thái: hoàn thành.

`hybrid_rerank` lấy semantic và BM25 candidate pools, fuse/deduplicate bằng RRF, sau đó rerank cặp `(query, chunk.text)` và mới cắt final top-k. Model mặc định được pin là `BAAI/bge-reranker-v2-m3`, revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. Candidate pool mặc định là `top_k * 5` và có typed settings.

Lỗi load/inference hoặc score count không hợp lệ được propagate; mode này không âm thầm fallback về hybrid.

### Phase 3 - Structure-aware hierarchical retrieval

Trạng thái: hoàn thành.

`hierarchical` dùng hybrid seed retrieval, local chunk catalog và legal structure để tạo bounded units. Mỗi unit có stable unit ID và danh sách evidence chunk IDs thật. Budget v1 là tối đa 5 chunks và 2.500 source tokens. Chunk thiếu hoặc ambiguous structure giữ leaf semantics; context không được trộn qua `scope_id`.

Mode này không tạo summary node, không persist parent node và không tăng số row Milvus.

### Phase 4 - Shared deterministic RRF

Trạng thái: hoàn thành.

Naive hybrid và Adaptive dùng helper RRF chung, deduplicate theo stable key và giữ deterministic first-seen tie-break. Ontology và Graphiti giữ implementation riêng vì chúng fuse các loại native unit và có semantics khác.

### Phase 5 - Adaptive one-shot multi-hop

Trạng thái: hoàn thành.

Adaptive planner trả `direct` hoặc `multi_hop`, không trả answer. Route multi-hop luôn giữ original standalone query làm retrieval leg đầu tiên, thêm tối đa ba subqueries, dùng per-leg limit tối thiểu 10 và fuse bằng shared RRF. Base mode mặc định là `hybrid_rerank`.

Plan, route và subqueries được lưu trong prediction metadata. Planner output sai typed contract fail rõ; không dùng gold `bundle_kind` để route và không âm thầm đổi về direct.

Các phần vẫn deferred: IRCoT nhiều vòng, answer-conditioned retrieval, ontology-aware temporal router và self-reflection retry.

### Phase 6 - Shared evaluation và comparison

Trạng thái: hoàn thành ở mức implementation và test.

Đã có:

- `run --max-samples` và `retrieve --max-samples`;
- năm mode Naive và năm end-to-end pipeline trong cùng CLI;
- structured multi-turn history và standalone retrieval query;
- bounded sample concurrency;
- explicit reference date;
- resume chỉ khi manifest identity khớp;
- metrics theo aggregate và slice;
- ontology coverage report;
- generation metric coverage;
- per-sample latency và observed token usage, không điền token giả bằng 0 khi provider không trả usage;
- `legal-evaluate compare` để validate compatibility và xuất JSON report;
- tách native-unit, evidence-coverage, context-budget và operational results;
- không tạo composite score.

### Phase 7 - Documentation, infrastructure và benchmark

Trạng thái: documentation/runbook đang được đồng bộ; external benchmark chưa chạy.

Các bước còn lại:

1. chuẩn bị active Milvus và Neo4j indexes đúng corpus;
2. chạy preflight và smoke cho từng variant;
3. khóa experiment protocol, gồm regression threshold;
4. chạy đủ 1.000 samples;
5. so sánh closed-corpus runs;
6. phân tích live-web CRAG như experiment riêng;
7. viết report cuối theo aggregate và các slice quan trọng.

## 5. Contract đánh giá hiện tại

### 5.1 Retrieval metrics

Các metric chính gồm:

- `hit_rate_at_k`/Hit@k trên ranked native units;
- `precision_at_k` trên native units;
- `recall_at_k` trên union evidence chunk IDs;
- Context Precision và Context Recall;
- `claim_evidence_any_at_k`;
- `claim_evidence_complete_at_k`.

Grouped evidence được giữ nguyên: một hierarchical unit, graph fact hoặc ontology rule có thể trỏ tới nhiều chunks. Precision xét native unit có chứa evidence liên quan; recall và claim coverage xét union provenance chunks.

Các slice tối thiểu cần báo cáo:

- `bundle_kind`: `single_hop`, `multi_hop`, `temporal`;
- `query_style`, đặc biệt `no_diacritics`;
- `answerability`;
- `reasoning_hops`.

### 5.2 Sáu generation metrics

Khi có `--generation`, LLM judge trả sáu metric:

1. `faithfulness` (0.0-1.0);
2. `correctness` (1-5);
3. `answer_relevancy` (1-5);
4. `temporal_correctness` (boolean/null);
5. `answerability_correctness` (boolean/null);
6. `required_claim_coverage` (0.0-1.0/null).

Aggregate luôn đi cùng metric coverage. Giá trị null đúng contract không được biến thành 0; ví dụ temporal correctness có thể null khi answer abstains, và required claim coverage null khi sample không có required claims.

### 5.3 Experiment identity và compatibility

Manifest/metrics fingerprint các control ảnh hưởng ý nghĩa hoặc tính tái lập của run, gồm:

- pipeline, mode, ground-truth digest và exact evaluated sample set;
- `max_samples`, retrieval limit, k-values và concurrency;
- reference date và history-resolution policy;
- answer model, endpoint fingerprint, timeout và provider extra-body fingerprint;
- embedding profile/model/revision;
- Milvus database, collection, verified corpus fingerprint;
- BM25, RRF, reranker, hierarchy và adaptive settings;
- Graphiti group, corpus-config digest, graph expansion/ranker settings và active Neo4j snapshot;
- Corrective thresholds, refinement settings và web-search policy;
- ontology file digests và measured gold-provenance coverage;
- runtime Python source hashes và BAML prompt hashes;
- judge model, prompt, batch size và concurrency khi generation judge được bật.

Secrets và raw provider configuration không được ghi vào manifest.

`legal-evaluate compare` fail khi các run không tương thích, gồm khác sample set, retrieval budget, concurrency, reference date, history-resolution capability, answer/judge model, embedding/corpus identity hoặc external-evidence policy. Rewrite policy chi tiết được so trong cùng pipeline; giữa các pipeline nó là metadata mô tả vì năm architecture dùng cơ chế history khác nhau. `--allow-incompatible` chỉ xuất chẩn đoán; nó không biến các run đó thành một so sánh công bằng.

### 5.4 Multi-turn policy

- Evaluation reader giữ `conversation_id`, `turn_number`, history và gold của turn cuối từ cả schema hiện tại lẫn legacy multi-turn export.
- Multi-turn exporter lấy `bundle_kind` từ reviewed target scenario. Chỉ target `temporal` nhận `query_dates` từ required-claim facts của turn cuối; history và unused scenario facts không đóng góp ngày.
- Naive và Adaptive rewrite câu follow-up thành standalone query khi history có mặt; query đó được dùng cho retrieval và generation.
- Graphiti và Corrective luôn dùng history handling nội tại của pipeline.
- Ontology planner nhận structured history cùng frozen `reference_date`; planned search query được dùng xuyên retrieval/generation khi resolution được bật.
- Dùng `--no-rewrite-query` chỉ như một ablation cho Naive, Adaptive hoặc Ontology, và phải đặt run directory riêng.

## 6. Ontology coverage caveat

Artifact runtime đang được dùng trong ví dụ là:

`data/ontology/exception/graphs/traffic_penalty_v0.5.ttl`

Đây là graph structured coverage một phần, không phải full-corpus canonical ontology. Đo trực tiếp trên 1.000 samples hiện tại:

- 168 provenance chunk IDs có trong ontology;
- 589/2.565 gold chunk references được phủ (22,96%);
- 122/473 unique gold chunks được phủ (25,79%);
- 278/1.000 samples có ít nhất một gold chunk trong ontology;
- 155/1.000 samples được ontology phủ đủ gold chunks;
- riêng multi-hop, chỉ 8/333 samples được phủ đủ.

Ontology pipeline vẫn có semantic/BM25 candidates và context fallback, nên tên chính xác của experiment là ontology-assisted hybrid với partial structured coverage. `metrics.json` ghi `ontology_coverage` toàn bộ và theo `bundle_kind`; report cuối phải trình bày số này cạnh kết quả chất lượng.

Không dùng kết quả của experiment này để tuyên bố hiệu quả của một full-corpus ontology. Muốn trả lời câu hỏi đó phải tạo và version một graph bao phủ corpus rộng hơn rồi chạy campaign mới với identity mới.

## 7. Chiến lược verification

Trước live benchmark:

```powershell
uv run --extra dev --extra naive-rag --extra corrective-rag pytest
uv run --extra dev ruff check src tests
uv run --extra dev ruff format --check src tests
```

Focused suites cần xanh ít nhất:

```powershell
uv run --extra dev --extra naive-rag --extra corrective-rag pytest `
  tests/naive_rag `
  tests/adaptive_rag `
  tests/corrective_rag `
  tests/graphiti_rag `
  tests/ontology_rag `
  tests/evaluation
```

Unit tests hiện bao phủ dispatch của năm Naive modes, grouped evidence, adaptive planning, structured history, CRAG correction/web boundaries, Graphiti snapshot, Milvus preflight, ontology temporal planning/coverage, deterministic identity và comparison compatibility. Live service smoke vẫn cần vì unit tests không chứng minh active Milvus/Neo4j/provider state đúng.

## 8. Runbook benchmark

Các lệnh dưới đây dùng PowerShell. Ngày `2026-10-01` là frozen campaign date minh họa cho campaign hiện tại; nếu chọn ngày khác, dùng đúng một ngày cho mọi end-to-end run và không thay khi resume.

### 8.1 Chuẩn bị hạ tầng và index

Khởi động service cần dùng:

```powershell
docker compose --profile milvus --profile neo4j up -d
docker compose ps
```

Index Milvus nếu active collection chưa khớp corpus:

```powershell
uv run --extra naive-rag legal-naive-rag-index data/processed/groundtruth
```

Index Graphiti vào đúng group/corpus config nếu snapshot chưa khớp:

```powershell
uv run --extra naive-rag legal-graphiti-index data/processed/groundtruth `
  --corpus-config data/config/groundtruth_corpus_v1.json
```

Corrective RAG phải trỏ tới cùng Milvus corpus/embedding identity. Không tạo collection riêng chỉ để đặt tên experiment khác. Chạy một smoke evaluation sẽ thực hiện active preflight; nếu preflight báo stale/missing/unexpected rows hoặc episodes, rebuild đúng index rồi chạy lại.

### 8.2 Retrieval-only smoke, 20 samples

Ở retrieval-only CLI, `--limit 10` là tối đa 10 native units mỗi câu và `--max-samples 20` mới giới hạn smoke ở 20 câu.

```powershell
$retrievalCommon = @(
  '--max-samples', '20',
  '--limit', '10',
  '--k-values', '1,3,5,10'
)

uv run --extra naive-rag legal-evaluate retrieve --retriever naive --naive-search-mode keyword @retrievalCommon `
  --predictions-output data/evaluation/smoke/naive_keyword.jsonl `
  --metrics-output data/evaluation/smoke/naive_keyword_metrics.json

uv run --extra naive-rag legal-evaluate retrieve --retriever naive --naive-search-mode semantic @retrievalCommon `
  --predictions-output data/evaluation/smoke/naive_semantic.jsonl `
  --metrics-output data/evaluation/smoke/naive_semantic_metrics.json

uv run --extra naive-rag legal-evaluate retrieve --retriever naive --naive-search-mode hybrid @retrievalCommon `
  --predictions-output data/evaluation/smoke/naive_hybrid.jsonl `
  --metrics-output data/evaluation/smoke/naive_hybrid_metrics.json

uv run --extra naive-rag legal-evaluate retrieve --retriever naive --naive-search-mode hybrid_rerank @retrievalCommon `
  --predictions-output data/evaluation/smoke/naive_hybrid_rerank.jsonl `
  --metrics-output data/evaluation/smoke/naive_hybrid_rerank_metrics.json

uv run --extra naive-rag legal-evaluate retrieve --retriever naive --naive-search-mode hierarchical @retrievalCommon `
  --predictions-output data/evaluation/smoke/naive_hierarchical.jsonl `
  --metrics-output data/evaluation/smoke/naive_hierarchical_metrics.json

uv run --extra corrective-rag legal-evaluate retrieve --retriever vector @retrievalCommon `
  --predictions-output data/evaluation/smoke/corrective_vector.jsonl `
  --metrics-output data/evaluation/smoke/corrective_vector_metrics.json

uv run --extra naive-rag legal-evaluate retrieve --retriever graph @retrievalCommon `
  --group-id traffic_penalty_groundtruth_v1 `
  --corpus-config data/config/groundtruth_corpus_v1.json `
  --predictions-output data/evaluation/smoke/graphiti.jsonl `
  --metrics-output data/evaluation/smoke/graphiti_metrics.json
```

Graphiti retrieval lấy configured candidate pool rồi mới cắt về `--limit`, nên `--limit` có cùng ý nghĩa native-unit budget như end-to-end run.

### 8.3 End-to-end smoke, 20 samples

```powershell
$smokeCommon = @(
  '--max-samples', '20',
  '--limit', '10',
  '--k-values', '1,3,5,10',
  '--concurrency', '1',
  '--reference-date', '2026-10-01',
  '--judge-batch-size', '5',
  '--judge-concurrency', '1',
  '--generation'
)

uv run --extra naive-rag legal-evaluate run --pipeline naive --naive-search-mode hybrid --rewrite-query @smokeCommon `
  --run-dir data/evaluation/runs/smoke_naive_hybrid

uv run --extra naive-rag legal-evaluate run --pipeline adaptive --rewrite-query @smokeCommon `
  --run-dir data/evaluation/runs/smoke_adaptive

$env:CRAG_WEB_SEARCH_ENABLED = 'false'
uv run --extra corrective-rag legal-evaluate run --pipeline corrective @smokeCommon `
  --run-dir data/evaluation/runs/smoke_corrective_closed

uv run --extra naive-rag legal-evaluate run --pipeline graphiti @smokeCommon `
  --group-id traffic_penalty_groundtruth_v1 `
  --corpus-config data/config/groundtruth_corpus_v1.json `
  --run-dir data/evaluation/runs/smoke_graphiti

uv run --extra naive-rag legal-evaluate run --pipeline ontology --rewrite-query @smokeCommon `
  --ontology-paths data/ontology/exception/graphs/traffic_penalty_v0.5.ttl `
  --run-dir data/evaluation/runs/smoke_ontology_partial
```

Chỉ bật Corrective live web sau khi đã cấu hình non-empty `CRAG_WEB_SEARCH_ALLOWED_DOMAINS` và chốt allowlist cho campaign:

```powershell
$env:CRAG_WEB_SEARCH_ENABLED = 'true'
uv run --extra corrective-rag legal-evaluate run --pipeline corrective @smokeCommon `
  --run-dir data/evaluation/runs/smoke_corrective_live_web
```

### 8.4 Full retrieval benchmark, 1.000 samples

Không truyền `--max-samples`. Dùng lại các lệnh retrieval smoke sau khi bỏ hai phần tử `--max-samples`, `20` khỏi `$retrievalCommon`, và ghi vào `data/evaluation/full/`. Full retrieval gate phải gồm cả năm Naive modes, Corrective vector baseline và Graphiti.

Ví dụ khởi tạo common arguments:

```powershell
$retrievalFull = @(
  '--limit', '10',
  '--k-values', '1,3,5,10'
)
```

Không reuse file smoke làm output full run.

### 8.5 Full end-to-end benchmark, 1.000 samples

```powershell
$fullCommon = @(
  '--limit', '10',
  '--k-values', '1,3,5,10',
  '--concurrency', '1',
  '--reference-date', '2026-10-01',
  '--judge-batch-size', '5',
  '--judge-concurrency', '1',
  '--generation'
)

uv run --extra naive-rag legal-evaluate run --pipeline naive --naive-search-mode keyword --rewrite-query @fullCommon `
  --run-dir data/evaluation/runs/naive_keyword_full

uv run --extra naive-rag legal-evaluate run --pipeline naive --naive-search-mode semantic --rewrite-query @fullCommon `
  --run-dir data/evaluation/runs/naive_semantic_full

uv run --extra naive-rag legal-evaluate run --pipeline naive --naive-search-mode hybrid --rewrite-query @fullCommon `
  --run-dir data/evaluation/runs/naive_hybrid_full

uv run --extra naive-rag legal-evaluate run --pipeline naive --naive-search-mode hybrid_rerank --rewrite-query @fullCommon `
  --run-dir data/evaluation/runs/naive_hybrid_rerank_full

uv run --extra naive-rag legal-evaluate run --pipeline naive --naive-search-mode hierarchical --rewrite-query @fullCommon `
  --run-dir data/evaluation/runs/naive_hierarchical_full

uv run --extra naive-rag legal-evaluate run --pipeline adaptive --rewrite-query @fullCommon `
  --run-dir data/evaluation/runs/adaptive_full

$env:CRAG_WEB_SEARCH_ENABLED = 'false'
uv run --extra corrective-rag legal-evaluate run --pipeline corrective @fullCommon `
  --run-dir data/evaluation/runs/corrective_closed_full

uv run --extra naive-rag legal-evaluate run --pipeline graphiti @fullCommon `
  --group-id traffic_penalty_groundtruth_v1 `
  --corpus-config data/config/groundtruth_corpus_v1.json `
  --run-dir data/evaluation/runs/graphiti_full

uv run --extra naive-rag legal-evaluate run --pipeline ontology --rewrite-query @fullCommon `
  --ontology-paths data/ontology/exception/graphs/traffic_penalty_v0.5.ttl `
  --run-dir data/evaluation/runs/ontology_partial_full

$env:CRAG_WEB_SEARCH_ENABLED = 'true'
uv run --extra corrective-rag legal-evaluate run --pipeline corrective @fullCommon `
  --run-dir data/evaluation/runs/corrective_live_web_full
```

Mỗi full run phải bắt đầu từ active snapshot đã preflight. Nếu gián đoạn, chạy lại chính xác cùng lệnh và thêm `--resume`; không đổi model, date, concurrency, corpus, web policy hoặc output directory khi resume.

### 8.6 So sánh kết quả bằng explicit PowerShell paths

Primary closed-corpus comparison dùng explicit metrics paths, không dùng wildcard:

```powershell
uv run legal-evaluate compare `
  data/evaluation/runs/naive_keyword_full/metrics.json `
  data/evaluation/runs/naive_semantic_full/metrics.json `
  data/evaluation/runs/naive_hybrid_full/metrics.json `
  data/evaluation/runs/naive_hybrid_rerank_full/metrics.json `
  data/evaluation/runs/naive_hierarchical_full/metrics.json `
  data/evaluation/runs/adaptive_full/metrics.json `
  data/evaluation/runs/corrective_closed_full/metrics.json `
  data/evaluation/runs/graphiti_full/metrics.json `
  data/evaluation/runs/ontology_partial_full/metrics.json `
  --output data/evaluation/rag_variant_closed_corpus_comparison.json
```

Corrective closed-corpus và live-web có external evidence policy khác nhau. Lệnh sau chỉ tạo incompatibility diagnostic và vẫn phải báo cáo hai experiment riêng:

```powershell
uv run legal-evaluate compare `
  data/evaluation/runs/corrective_closed_full/metrics.json `
  data/evaluation/runs/corrective_live_web_full/metrics.json `
  --allow-incompatible `
  --output data/evaluation/corrective_web_policy_diagnostic.json
```

## 9. Cách ra quyết định sau benchmark

Không chọn “best RAG” từ một con số. Report cuối cần đánh giá theo thứ tự:

1. Recall@10 và Hit@10 toàn dataset;
2. claim evidence complete trên `multi_hop` và `temporal`;
3. Precision@5/10 và generation-context budget;
4. sáu generation metrics cùng coverage;
5. latency và token usage, chỉ trên samples/provider calls có observation;
6. regressions theo `no_diacritics`, answerability và reasoning hops.

Trước full run phải chốt ngưỡng regression có ý nghĩa, ví dụ absolute delta tối đa cho metric chính. Không đặt ngưỡng sau khi đã nhìn kết quả.

Ontology partial phải được diễn giải theo measured structured coverage. Corrective live-web phải được diễn giải theo external evidence policy, allowlist và thời điểm truy cập. Latency chỉ so trực tiếp khi concurrency, provider và các auxiliary-call policy giống nhau.

## 10. Definition of done

### Implementation trong repository

- [x] Dataset và corpus counts được xác nhận từ artifact hiện tại.
- [x] Ba baseline Naive giữ nguyên và tổng cộng năm Naive modes có contract/test.
- [x] `hybrid_rerank` tách candidate fusion khỏi final reranking và pin model revision.
- [x] `hierarchical` dùng local structure, giữ evidence chunks và không thêm row Milvus.
- [x] Shared RRF có deterministic tie-break.
- [x] Adaptive route direct/multi-hop và lưu plan để audit.
- [x] Naive, Adaptive, Corrective, Graphiti và Ontology hỗ trợ structured multi-turn evaluation.
- [x] Tất cả variant normalize native units, evidence chunks và generation contexts.
- [x] Evaluation dùng retrieval/claim metrics hiện tại, không có MRR/NDCG.
- [x] LLM judge có sáu metric và báo metric coverage.
- [x] Run identity và comparison kiểm tra dataset, model, corpus, temporal/history, concurrency và judge controls.
- [x] Active Milvus fingerprint và Neo4j group snapshot preflight đã được implement và có unit tests.
- [x] Ontology partial coverage được đo và ghi trong metrics.
- [x] Focused unit tests và Ruff cho các phần triển khai đã pass trong quá trình phát triển.
- [x] Quyết định defer accent-aware BM25 được ghi rõ; slice `no_diacritics` vẫn là gate.

### Campaign ngoài repository

- [ ] Active Milvus collection đã được index và live preflight pass với corpus hiện tại.
- [ ] Active Neo4j group đã được index và live snapshot preflight pass với corpus config/group hiện tại.
- [ ] Provider/embedding/reranker models và CRAG web allowlist đã được khóa cho campaign.
- [ ] Smoke end-to-end pass cho năm Naive modes, Adaptive, Corrective closed/live, Graphiti và Ontology partial.
- [ ] Full 1.000 samples hoàn tất cho tất cả experiment đã vượt smoke gate.
- [ ] Closed-corpus comparison report được tạo từ các metrics paths explicit.
- [ ] Report cuối phân tích aggregate, `single_hop`, `multi_hop`, `temporal`, `no_diacritics`, answerability, latency và token usage.
- [ ] Kết luận về Ontology được giới hạn đúng theo structured coverage của artifact đã chạy.
