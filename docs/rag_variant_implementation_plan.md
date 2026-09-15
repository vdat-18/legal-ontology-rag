# Kế hoạch triển khai và đánh giá các biến thể RAG

## 1. Mục tiêu

Triển khai các biến thể RAG còn thiếu trên kiến trúc hiện tại, sau đó đánh giá công bằng trên cùng bộ 1.000 câu hỏi để xác định biến thể phù hợp nhất cho bài toán tra cứu xử phạt giao thông theo pháp luật Việt Nam.

Kế hoạch này chỉ tập trung vào retrieval, orchestration và evaluation. Không thay đổi ontology, pipeline xây dựng ontology hoặc dữ liệu ontology đang được phát triển ở task khác.

## 2. Phạm vi và quyết định đã chốt

- Giữ nguyên các baseline hiện có để kết quả cũ vẫn tái lập được.
- Milvus tiếp tục lưu đúng một row cho mỗi chunk, không tạo bản sao riêng cho BM25, semantic, hierarchy hoặc reranker.
- Không thêm lại MRR hoặc NDCG. Các metric chính là Hit@k, Precision@k, Recall@k, Context Precision, Context Recall và claim evidence coverage.
- Không vendor hoặc sao chép nguyên repo RAPTOR, Adaptive-RAG hay IRCoT. Chỉ triển khai phần tối thiểu phù hợp với codebase hiện tại.
- Multi-hop phiên bản đầu dùng one-shot decomposition: lập danh sách subquery một lần, retrieve từng subquery, fuse kết quả rồi generate một lần.
- Chưa triển khai temporal router phụ thuộc ontology. Phần này chỉ làm sau khi ontology ổn định và contract truy vấn đã được chốt.
- Không thay schema chunk trên disk và không migrate collection Milvus trong các phase đầu.
- Không tự động chạy đủ 1.000 câu khi đang implement. Chạy smoke test và sample nhỏ trước; người dùng chạy full benchmark bằng các lệnh ở cuối tài liệu sau khi metric/output đã được xác nhận đúng.

## 3. Hiện trạng codebase

### 3.1 Dữ liệu

- Corpus retrieval nằm tại `data/processed/groundtruth/*.chunks.jsonl`.
- Hiện có khoảng 8.351 chunks thuộc 35 file chunk.
- Ground truth nằm tại `data/benchmark_synthesis/groundtruth.jsonl` với 1.000 câu:
  - 334 `single_hop`;
  - 333 `multi_hop`;
  - 333 `temporal`.
- Mỗi chunk trên disk có `chunk_id`, `text`, `context_text`, `embedding_text`, `scope_id`, `chunk_index`, `previous_chunk_id`, `next_chunk_id` và provenance.
- `embedding_text` hiện bằng `text`; đây là contract của file chunk, không phải hai chunk khác nhau.

### 3.2 Milvus

`src/legal_rag/naive_rag/indexer.py` tạo một row cho mỗi `chunk_id` với các field:

- `chunk_id`;
- `vector`;
- `text`;
- `source_name`;
- `kind`;
- `content_sha256`;
- `chunker_config_sha256`;
- `embedding_profile_id`;
- `embedding_model_id`;
- `embedding_model_revision`.

BM25 đọc `text` từ chính các row này để dựng index trong memory. Semantic search dùng `vector` của cùng row. Vì vậy keyword và semantic không tạo hai bản sao dữ liệu.

### 3.3 Luồng Naive RAG hiện tại

```text
question
  -> optional query rewrite
  -> Retriever.retrieve()
       -> keyword: BM25L trên text
       -> semantic: Milvus dense search
       -> hybrid: semantic + BM25L -> RRF
  -> top-k RetrievedChunk
  -> BAML answer generation
  -> answer + sources
```

Các component chính:

- `src/legal_rag/naive_rag/pipeline.py`: orchestration retrieve-then-generate.
- `src/legal_rag/naive_rag/retriever.py`: keyword, semantic và hybrid retrieval.
- `src/legal_rag/naive_rag/indexer.py`: đọc chunk, embed và upsert vào Milvus.
- `src/legal_rag/naive_rag/models.py`: `SearchMode` và `RetrievedChunk`.
- `src/legal_rag/retrieval/vietnamese_bm25.py`: tokenization BM25 tiếng Việt.
- `src/legal_rag/settings.py`: toàn bộ cấu hình qua environment.

Ba mode baseline phải tiếp tục tồn tại với semantics cũ:

- `keyword`: BM25L only;
- `semantic`: dense embedding only;
- `hybrid`: BM25L + dense, fuse bằng RRF.

### 3.4 Evaluation hiện tại

`src/legal_rag/evaluation/` đã có contract chung cho bốn pipeline:

- `naive`;
- `corrective`;
- `graphiti`;
- `ontology`.

Luồng evaluation:

```text
groundtruth.jsonl
  -> EvaluationSample/EvaluationRecord
  -> backend.predict(question, top_k)
  -> normalized retrieved_units + evidence chunk IDs
  -> predictions.jsonl
  -> aggregate metrics + metrics theo slice
```

Các slice quan trọng đã có gồm `bundle_kind`, `answerability`, `reasoning_hops` và `query_style`.

Baseline Naive hybrid đã chạy trên 1.000 câu:

- Hit@10: `0.679`;
- Recall@10: `0.465`;
- Precision@10: `0.1097`;
- lỗi lớn nhất là nhóm query `no_diacritics`.

## 4. Ma trận biến thể cần đánh giá

| Nhóm | Tên experiment | Cách triển khai | Trạng thái ban đầu |
|---|---|---|---|
| Naive | `naive_keyword` | `pipeline=naive`, mode `keyword` | Đã có |
| Naive | `naive_semantic` | `pipeline=naive`, mode `semantic` | Đã có |
| Hybrid | `naive_hybrid_rrf` | `pipeline=naive`, mode `hybrid` | Đã có |
| Hybrid rerank | `naive_hybrid_rerank` | mode mới `hybrid_rerank` | Đã implement, smoke 20 câu đạt |
| Structure-aware | `naive_hierarchical` | mode mới `hierarchical` | Đã implement, chờ smoke benchmark |
| Adaptive multi-hop | `adaptive_multihop` | pipeline mới `adaptive` | Cần làm |
| Corrective | `corrective` | backend hiện có | Đã có, cần kiểm tra hạ tầng/index |
| Graph RAG | `graphiti` | backend hiện có | Đã có, cần kiểm tra Neo4j/index |
| Ontology RAG | `ontology` | backend hiện có | Chờ task ontology ổn định |

Không gộp `hybrid_rerank`, `hierarchical` hoặc `adaptive` vào mode `hybrid` hiện tại. Baseline `hybrid` phải giữ nguyên để so sánh trước/sau có ý nghĩa.

## 5. Thứ tự triển khai

## Phase 0 - Khóa baseline và contract đánh giá

Mục tiêu: bảo đảm mọi thay đổi sau đó được so với cùng một baseline và không làm sai cách tính metric.

### Công việc

- [ ] Xác nhận `data/benchmark_synthesis/groundtruth.jsonl` có đúng 1.000 evaluation samples sau khi expand row.
- [ ] Xác nhận phân phối `single_hop`, `multi_hop`, `temporal` và `query_style` khớp số liệu ở trên.
- [ ] Chạy lại unit tests nhỏ nhất cho `tests/evaluation/` và `tests/naive_rag/` trước khi sửa.
- [ ] Lưu kết quả baseline hiện có; không sửa hoặc ghi đè các file kết quả trong `data/evaluation/`.
- [ ] Dùng run directory riêng cho từng experiment.

### Files liên quan

- Không sửa source ở phase này.
- Chỉ đọc `data/benchmark_synthesis/groundtruth.jsonl` và các report hiện có trong `data/evaluation/`.

### Validation

```powershell
uv run --extra dev pytest tests/evaluation tests/naive_rag
uv run --extra dev ruff check src/legal_rag/evaluation src/legal_rag/naive_rag tests/evaluation tests/naive_rag
```

### Điều kiện hoàn thành

- Test baseline pass.
- Có bản ghi rõ commit/worktree, embedding profile, Milvus collection, top-k và RRF config dùng cho baseline.

## Phase 1 - Sửa BM25 cho query không dấu (deferred)

Trạng thái: chưa triển khai trong đợt hiện tại theo quyết định của người dùng. Ưu tiên đo tác động của Cross-Encoder rerank trước; chỉ quay lại phase này nếu benchmark `query_style=no_diacritics` vẫn là regression quan trọng.

Lưu ý kiến trúc: retrieval-only evaluation hiện gọi `Retriever` trực tiếp và không query rewrite. `NaiveRAGPipeline` trong worktree hiện cũng chỉ rewrite khi có `history`. Vì vậy query rewriting không bảo đảm xử lý query không dấu trong benchmark single-turn; đây là rủi ro đã biết, không được âm thầm coi là đã giải quyết.

### Thiết kế

Mở rộng tokenizer BM25 để mỗi word/bigram có hai namespace:

- token giữ dấu, ví dụ `w:tiền` và `b:phạt|tiền`;
- token ASCII-folded, ví dụ `f:tien` và `fb:phat|tien`.

ASCII folding phải:

- normalize Unicode ổn định;
- bỏ combining marks;
- đổi `đ/Đ` thành `d/D` trước khi casefold;
- giữ chữ số;
- không nối bigram qua punctuation boundary;
- tạo cùng token cho query không dấu và document có dấu.

Không dùng LLM diacritic restoration ở phase này vì làm retrieval chậm, tốn chi phí và khó tái lập.

### Files cần sửa

- `src/legal_rag/retrieval/vietnamese_bm25.py`
  - thêm helper nội bộ để ASCII-fold một lexical unit;
  - mở rộng `vietnamese_legal_tokens()` để sinh token giữ dấu và folded;
  - giữ signature public hiện tại nếu không thật sự cần thay đổi.
- `tests/benchmark_synthesis/test_evidence_graph.py`
  - cập nhật expectation của shared tokenizer;
  - thêm test NFC/NFD, `đ`, punctuation và số điều/khoản.
- `tests/naive_rag/test_retriever.py`
  - thêm case corpus có dấu, query không dấu vẫn retrieve đúng;
  - thêm case query có dấu không bị mất exact lexical match.

### Không sửa

- Không thêm field Milvus.
- Không thêm `unaccented_text` vào chunk hoặc collection.
- Không đổi `chunk_id`.
- Không re-chunk corpus.

### Edge cases và error handling

- Query rỗng hoặc chỉ có punctuation trả về không có BM25 hit như hiện tại.
- Token chỉ gồm số vẫn phải ổn định.
- Không sinh token trùng lặp trong cùng namespace cho một segmented token.
- Nếu VnCoreNLP không được inject, fallback whitespace hiện tại cũng phải đi qua cùng lexical normalization; không để hai code path có semantics khác nhau.

### Validation

```powershell
uv run --extra dev pytest tests/benchmark_synthesis/test_evidence_graph.py tests/naive_rag/test_retriever.py
uv run --extra dev ruff check src/legal_rag/retrieval/vietnamese_bm25.py tests/benchmark_synthesis/test_evidence_graph.py tests/naive_rag/test_retriever.py
```

Sau unit test, chạy retrieval smoke trên cả query có dấu và không dấu. So sánh ít nhất các slice:

- `query_style=no_diacritics`;
- `query_style=standard`;
- tổng thể.

### Điều kiện chấp nhận

- Hit@10/Recall@10 của `no_diacritics` tăng rõ ràng.
- Metric của query có dấu không giảm đáng kể. Nếu giảm, kiểm tra token weighting trước khi sang phase 2.
- Không đổi schema/index Milvus.

## Phase 2 - Thêm Hybrid RRF + Cross-Encoder rerank

Mục tiêu: tách candidate recall và final ranking precision thành hai bước đo được độc lập.

### Thiết kế

Thêm mode `hybrid_rerank`:

```text
query
  -> semantic candidate pool
  -> BM25 candidate pool
  -> RRF deduplicate/fuse
  -> Cross-Encoder score (query, chunk.text)
  -> sort theo reranker score
  -> top-k
```

RRF vẫn là candidate fusion. Cross-Encoder là final reranker; không gọi Cross-Encoder riêng trên hai nhánh trước khi fuse.

Model mặc định đề xuất: `BAAI/bge-reranker-v2-m3`, nhưng model ID và revision phải là typed settings để benchmark ghi lại chính xác. Không tự fallback sang model khác khi load/inference lỗi; fail rõ để tránh một run mang tên rerank nhưng thực tế chỉ chạy RRF.

### Files cần tạo

- `src/legal_rag/naive_rag/reranker.py`
  - protocol/interface nhỏ cho reranker để unit test inject fake;
  - implementation dùng `sentence_transformers.CrossEncoder`;
  - nhận query và bounded candidate list;
  - trả candidates theo score giảm dần;
  - validate số score bằng số candidate.
- `tests/naive_rag/test_reranker.py`
  - ranking order;
  - empty candidates;
  - duplicate/tie behavior ổn định;
  - invalid score count fail rõ.

### Files cần sửa

- `src/legal_rag/naive_rag/models.py`
  - thêm `SearchMode.HYBRID_RERANK = "hybrid_rerank"`;
  - nếu cần trace score, thêm optional `retrieval_score` và `rerank_score` với default để không phá caller hiện tại; `score` luôn là score quyết định thứ hạng cuối.
- `src/legal_rag/naive_rag/retriever.py`
  - tách việc lấy hybrid candidate pool khỏi bước slice top-k;
  - thêm `_hybrid_rerank_search()`;
  - chỉ khởi tạo/load reranker khi mode này được dùng;
  - deduplicate theo `chunk_id` trước rerank;
  - tie-break ổn định theo thứ tự RRF hiện tại để không đổi semantics baseline.
- `src/legal_rag/settings.py`
  - thêm typed settings tối thiểu: reranker model ID, pinned revision, batch size và candidate pool multiplier;
  - validate string không rỗng, số dương và pool không nhỏ hơn final `top_k` theo logic runtime.
- `.env.example`
  - document biến `NAIVE_RAG_RERANKER_*` mới, không chứa token/secret.
- `src/legal_rag/evaluation/cli.py`
  - thêm `hybrid_rerank` vào choices của cả `run --naive-search-mode` và `retrieve --naive-search-mode`.
- `tests/naive_rag/test_search_modes.py`
  - xác nhận mode mới dispatch đúng và rerank sau RRF.
- `tests/naive_rag/test_retriever.py`
  - candidate pool đủ lớn;
  - dedup;
  - deterministic tie-break;
  - reranker không load ở các mode baseline.
- `tests/test_settings.py` hoặc file test settings hiện có
  - defaults, environment overrides và invalid values.
- `tests/evaluation/test_cli.py`
  - parser và adapter chấp nhận mode mới;
  - output/run identity phân biệt `hybrid` và `hybrid_rerank`.

### Dependencies

- Dùng dependency `sentence-transformers` đã có; không thêm package mới.
- Phase 1 không còn là dependency bắt buộc. Phase 2 dùng nguyên candidate pool baseline hiện tại để đo riêng tác động của reranker.

### Edge cases và error handling

- Candidate list rỗng trả về rỗng, không gọi model.
- Một candidate trả về chính candidate đó.
- Model không tải được hoặc output không hợp lệ: propagate exception với model ID/revision trong message, không fallback âm thầm.
- Cross-Encoder có giới hạn input: dùng batching và truncation của model; không tự cắt `text` bằng số ký tự tùy ý.
- Không để candidate pool tăng theo cấp số nhân do gọi lồng `_hybrid_search()` với multiplier lần hai.

### Validation

```powershell
uv run --extra dev pytest tests/naive_rag/test_reranker.py tests/naive_rag/test_retriever.py tests/naive_rag/test_search_modes.py tests/evaluation/test_cli.py
uv run --extra dev ruff check src/legal_rag/naive_rag src/legal_rag/evaluation/cli.py tests/naive_rag tests/evaluation/test_cli.py
```

Chạy sample nhỏ với cùng query order cho `hybrid` và `hybrid_rerank`. Không kết luận từ file `data/evaluation/reranker_comparison.json` cũ vì report đó dùng metric/schema cũ có MRR/NDCG và chỉ 50 samples.

### Điều kiện chấp nhận

- Không giảm Hit@10/Recall@10 ngoài sai số nhỏ của sample.
- Ưu tiên cải thiện Hit@1, Precision@3/5 và claim coverage.
- Chỉ giữ variant này trong full benchmark nếu sample đủ lớn cho thấy không tệ hơn RRF thuần ở các slice chính.

## Phase 3 - Thêm structure-aware hierarchical retrieval

Mục tiêu: tận dụng cấu trúc chương/mục/điều và neighbor links đã có trong file chunk mà không tạo collection hoặc row Milvus mới.

### Thiết kế v1

Mode mới `hierarchical` dùng hybrid retrieval làm seed retrieval, sau đó mở rộng context theo cấu trúc:

```text
query
  -> hybrid RRF seed candidates
  -> lookup seed trong local chunk catalog
  -> xác định article context trong cùng scope
  -> gom các chunk cùng article, có giới hạn
  -> tạo ranked structural units
  -> top-k units -> generation
```

`context_text` là structural context, không phải một chunk độc lập. Catalog được load từ `data/processed/groundtruth/*.chunks.jsonl` và index trong memory theo:

- `chunk_id`;
- `scope_id`;
- `source_name`;
- article structural label;
- `previous_chunk_id`/`next_chunk_id`.

Một structural unit phải có:

- stable synthetic `unit_id` từ `scope_id` + normalized article label;
- text gồm heading/context một lần và các chunk text theo `chunk_index`;
- danh sách evidence `chunk_id` thật để evaluation tính recall đúng;
- source name/provenance;
- score kế thừa từ seed tốt nhất; tie-break theo seed rank.

Nếu một chunk có nhiều structural variants hoặc không xác định được đúng một article:

- không đoán article;
- giữ chunk đó như leaf unit;
- có thể thêm neighbor trực tiếp chỉ khi cùng `scope_id` và trong budget cấu hình.

`top_k` ở mode này là số ranked units. Recall được tính trên union evidence chunk IDs của mỗi unit thông qua contract `retrieved_units`, không so synthetic unit ID với gold chunk ID.

### Files cần tạo

- `src/legal_rag/naive_rag/structure.py`
  - `ChunkCatalog` load typed chunks bằng reader hiện có;
  - parse `context_text` theo format `[LEGAL_STRUCTURE ...]` và `[LEGAL_STRUCTURE_VARIANTS ...]`;
  - map chunk -> article group;
  - build bounded structural unit từ seed.
- `tests/naive_rag/test_structure.py`
  - same-article grouping;
  - không trộn `scope_id` hoặc attachment;
  - giữ source order;
  - ambiguous/missing context fallback;
  - neighbor cycle/missing ID không gây loop;
  - token/chunk budget được enforce.

### Files cần sửa

- `src/legal_rag/naive_rag/models.py`
  - thêm `SearchMode.HIERARCHICAL = "hierarchical"`;
  - mở rộng result model để mang `unit_id` và evidence chunk IDs mà vẫn tương thích ba mode cũ.
- `src/legal_rag/naive_rag/retriever.py`
  - inject optional `ChunkCatalog`;
  - thêm `_hierarchical_search()`;
  - lazy-load catalog chỉ khi mode này được gọi;
  - deduplicate structural units khi nhiều seed thuộc cùng article.
- `src/legal_rag/naive_rag/pipeline.py`
  - truyền chunk directory/catalog theo settings;
  - generation nhận text đã expand nhưng sources vẫn giữ evidence IDs.
- `src/legal_rag/settings.py`
  - thêm `chunks_dir`, maximum chunks per structural unit và maximum context token/chars theo cách đo đã có trong repo;
  - default `chunks_dir` là `data/processed/groundtruth`.
- `.env.example`
  - document `NAIVE_RAG_CHUNKS_DIR` và budget liên quan.
- `src/legal_rag/evaluation/runner.py`
  - `NaiveEvaluationBackend` map một structural result thành một `RetrievedUnit` có nhiều `PredictionEvidence`;
  - baseline result không có evidence list vẫn map thành một evidence như hiện tại.
- `src/legal_rag/evaluation/cli.py`
  - thêm `hierarchical` vào naive mode choices và run identity.
- `tests/naive_rag/test_naive_rag_pipeline.py`
  - generation nhận đúng grouped context.
- `tests/evaluation/test_runner.py`
  - structural unit với nhiều evidence chunk IDs được normalize đúng.
- `tests/evaluation/test_metrics.py`
  - Recall@k dùng union evidence;
  - Precision@k đếm ranked unit relevant nếu unit chứa ít nhất một gold evidence;
  - claim complete hoạt động khi một unit chứa nhiều evidence.

### Không làm trong v1

- Không sinh summary node bằng LLM như RAPTOR.
- Không persist parent nodes hoặc duplicate vectors vào Milvus.
- Không sửa chunker schema.
- Không expand toàn bộ document.

### Edge cases và error handling

- Chunk directory không tồn tại, rỗng hoặc thiếu seed ID: fail rõ khi mode `hierarchical` khởi tạo/chạy; các mode khác không bị ảnh hưởng.
- Duplicate `chunk_id` trong catalog: fail thay vì lấy row cuối.
- Chunk Milvus và local catalog khác `content_sha256`: cần validation nếu metadata có sẵn; không được ghép context từ corpus khác.
- Group vượt budget: ưu tiên seed, sau đó neighbor gần seed nhất; output cuối vẫn theo thứ tự tài liệu.
- Không lặp heading nhiều lần trong generated context.

### Validation

```powershell
uv run --extra dev pytest tests/naive_rag/test_structure.py tests/naive_rag/test_retriever.py tests/naive_rag/test_naive_rag_pipeline.py tests/evaluation/test_runner.py tests/evaluation/test_metrics.py
uv run --extra dev ruff check src/legal_rag/naive_rag src/legal_rag/evaluation tests/naive_rag tests/evaluation
```

### Điều kiện chấp nhận

- Recall@10 và `claim_evidence_complete@10` tăng ở `multi_hop` hoặc câu cần nhiều khoản trong cùng điều.
- Precision@k không bị tính sai do synthetic unit ID.
- Không tăng số row Milvus và không cần re-index.

## Phase 4 - Chuẩn hóa helper RRF dùng chung

Mục tiêu: tránh mỗi variant mới tự viết một bản RRF khác nhau, nhưng không refactor rộng vào Ontology/Graphiti đang ổn định.

Trạng thái: đã hoàn thành. Naive hybrid dùng helper generic; characterization test xác nhận thứ tự, điểm số và semantic representative không đổi.

### Thiết kế

Tạo helper generic nhận nhiều ranked lists, một key function và `rrf_k`, trả về item, fused score và deterministic tie information. Công thức giữ nguyên:

```text
score(item) = sum(1 / (rrf_k + rank_in_list))
```

Rank bắt đầu từ 1. Một item chỉ đóng góp một lần trong mỗi leg; duplicate trong cùng leg phải được bỏ trước khi cộng điểm.

### Files cần tạo

- `src/legal_rag/retrieval/rrf.py`
- `tests/retrieval/test_rrf.py`

### Files cần sửa

- `src/legal_rag/naive_rag/retriever.py`
  - thay block RRF inline bằng helper sau khi characterization tests đã khóa behavior.
- Các module mới ở phase 2/3 dùng cùng helper nếu cần fuse rankings.

### Không sửa

- Không đổi `src/legal_rag/ontology_rag/fusion/rrf_fusion.py` trong task này.
- Không đổi Graphiti RRF vì Graphiti có semantics expansion/reranking riêng.
- Không đổi private RRF của benchmark synthesis vì dataset generation là pipeline khác và đã được version hóa.

### Tests bắt buộc

- hai rankings overlap;
- item chỉ có ở một leg;
- duplicate trong một leg;
- empty rankings;
- invalid `rrf_k`;
- tie-break deterministic;
- Naive hybrid output giống characterization test trước khi extract helper.

### Validation

```powershell
uv run --extra dev pytest tests/retrieval/test_rrf.py tests/naive_rag/test_search_modes.py tests/naive_rag/test_retriever.py
uv run --extra dev ruff check src/legal_rag/retrieval src/legal_rag/naive_rag tests/retrieval tests/naive_rag
```

## Phase 5 - Thêm Adaptive Multi-hop one-shot

Mục tiêu: cải thiện nhóm `multi_hop` bằng query planning tối thiểu, không triển khai vòng lặp agent/IRCoT đầy đủ.

Trạng thái: đã implement với base mode mặc định `hybrid_rerank`, tối đa 3 subqueries, per-leg limit tối thiểu 10 và shared deterministic RRF. Chờ smoke evaluation thực tế trước khi đánh dấu hoàn tất.

### Thiết kế v1

Pipeline mới `adaptive`:

```text
question
  -> planner(route=direct|multi_hop, subqueries)
  -> direct: retrieve original question một lần
  -> multi_hop:
       retrieve original question
       retrieve từng subquery
       RRF fuse tất cả ranked lists
  -> final top-k evidence chunks/units
  -> generate answer một lần bằng original question
```

Planner chỉ được trả:

- `route`: `direct` hoặc `multi_hop`;
- `subqueries`: danh sách câu hỏi retrieval độc lập;
- không trả answer hoặc căn cứ pháp luật tự suy đoán.

Với `multi_hop`, original query luôn là một retrieval leg để tránh decomposition làm mất ý định chính. V1 không retrieve -> reason -> retrieve lặp nhiều vòng.

### Files cần tạo

- `baml_src/adaptive_query_planning.baml`
  - typed output cho route và subqueries;
  - prompt tiếng Việt cho domain pháp luật;
  - cấm trả lời câu hỏi trong planning step.
- `src/legal_rag/adaptive_rag/__init__.py`
- `src/legal_rag/adaptive_rag/models.py`
  - typed plan/result models.
- `src/legal_rag/adaptive_rag/planner.py`
  - sync BAML client wrapper theo pattern query rewriting hiện tại;
  - validate và normalize planner output.
- `src/legal_rag/adaptive_rag/pipeline.py`
  - orchestration direct/multi-hop;
  - reuse `Retriever`, shared RRF helper và Naive generation client;
  - close resource đúng một lần.
- `tests/adaptive_rag/test_planner.py`
- `tests/adaptive_rag/test_pipeline.py`

### Files generated

- Regenerate `src/legal_rag/baml_client/` bằng workflow BAML của repo.
- Không edit generated files thủ công.

### Files cần sửa

- `src/legal_rag/settings.py`
  - thêm `AdaptiveRagSettings` hoặc nhóm setting tối thiểu theo pattern hiện tại;
  - gồm base search mode, max subqueries, per-leg pool size và RRF k;
  - validate base mode không được tự tham chiếu `adaptive`.
- `.env.example`
  - document settings mới.
- `src/legal_rag/evaluation/runner.py`
  - thêm `AdaptiveEvaluationBackend` trả đúng normalized units/evidence và metadata gồm route/subqueries.
- `src/legal_rag/evaluation/cli.py`
  - thêm `adaptive` vào `run --pipeline`;
  - thêm builder/backend wiring;
  - thêm adaptive settings vào deterministic run identity.
- `pyproject.toml`
  - chỉ thêm CLI entry point riêng nếu cần query thủ công; evaluation có thể gọi pipeline trực tiếp nên không bắt buộc.
- `tests/evaluation/test_runner.py`
  - adapter direct và multi-hop.
- `tests/evaluation/test_cli.py`
  - parser, builder, run identity và resource close.

### Planner validation

- `direct`: bỏ qua subqueries thừa hoặc yêu cầu output rỗng theo schema; chọn một rule và test cố định.
- `multi_hop`: phải có từ 2 đến `max_subqueries` subqueries không rỗng.
- Deduplicate subqueries bằng normalized text, giữ thứ tự đầu tiên.
- Nếu planner output sai schema, để typed BAML parsing fail rõ; không chuyển âm thầm sang direct.
- Không dùng ground-truth metadata `bundle_kind` để route trong runtime.

### Retrieval behavior

- Dùng cùng embedding profile, Milvus collection và lexical fixes với Naive để khác biệt chỉ nằm ở planning/orchestration.
- Dùng bounded pool cho từng leg để số candidates tối đa xác định được.
- Deduplicate theo evidence `chunk_id` trước generation.
- Final score là RRF score; tie-break theo first-seen order của shared helper, trong đó original query là leg đầu tiên.
- Chưa dùng answer từ hop trước để tạo hop sau.

### Error handling

- Planner timeout/authentication/network error: propagate; manifest evaluation phải ghi `failed` và cho phép `--resume`.
- Một subquery không có hit không làm fail toàn question; fuse các leg còn lại.
- Tất cả leg rỗng: generation nhận empty context theo behavior hiện tại hoặc abstain theo generator contract.
- `max_subqueries`, pool size hoặc RRF k không hợp lệ: fail ngay khi load settings.

### Validation

```powershell
uv run baml-cli generate
uv run --extra dev pytest tests/adaptive_rag tests/evaluation/test_runner.py tests/evaluation/test_cli.py
uv run --extra dev ruff check src/legal_rag/adaptive_rag src/legal_rag/evaluation tests/adaptive_rag tests/evaluation
```

### Điều kiện chấp nhận

- `multi_hop` slice cải thiện Recall@10 và `claim_evidence_complete@10` so với Naive hybrid.
- `single_hop` không giảm mạnh do router lạm dụng decomposition.
- Planner route/subqueries được lưu trong prediction metadata để audit.

### Deferred

- Full IRCoT iterative reasoning.
- Query planning dựa trên graph/ontology temporal state.
- Temporal route chuyên biệt.
- Self-reflection hoặc retry planner.

Chỉ mở các hạng mục này nếu full benchmark chứng minh one-shot decomposition chưa đủ.

## Phase 6 - Hoàn thiện evaluation để so sánh công bằng

Mục tiêu: mọi variant chạy trên cùng contract, output không ghi đè nhau và có report đủ để quyết định.

### Files cần sửa

- `src/legal_rag/evaluation/cli.py`
  - thêm `--max-samples` cho `run`, giống retrieval smoke test;
  - đưa `max_samples` vào run identity;
  - slice records trước khi tạo manifest để resume không đổi dataset giữa các run;
  - choices mới như các phase trên;
  - update help: `--limit` là retrieval top-k, không phải số câu đánh giá.
- `src/legal_rag/evaluation/runner.py`
  - giữ shared prediction contract;
  - bảo toàn ranked units và evidence chunk IDs cho hierarchy/adaptive.
- `src/legal_rag/evaluation/metrics.py`
  - không thêm MRR/NDCG;
  - chỉ sửa nếu test chứng minh grouped evidence hiện bị chấm sai.
- `tests/evaluation/test_cli.py`
  - sample limit, deterministic identity, resume và output isolation.
- `tests/evaluation/test_runner.py`
  - tất cả backend mới map về cùng contract.
- `tests/evaluation/test_metrics.py`
  - grouped evidence và slice aggregation.

### Files cần tạo

- `src/legal_rag/evaluation/comparison.py`
  - đọc nhiều `metrics.json`;
  - validate cùng ground-truth digest, sample count, k-values và retrieval limit;
  - xuất bảng tổng và các slice quan trọng;
  - không tự tạo weighted composite score không có căn cứ.
- `tests/evaluation/test_comparison.py`
  - reject runs không cùng dataset/config;
  - handle generation metrics optional/null;
  - deterministic ordering.

Có thể expose comparison bằng subcommand `legal-evaluate compare <metrics...>` trong `src/legal_rag/evaluation/cli.py`. Output tối thiểu gồm JSON; Markdown là tùy chọn nếu không làm phức tạp implementation.

### Metric dùng để quyết định

Ưu tiên theo thứ tự:

1. Recall@10 và Hit@10 toàn bộ dataset.
2. `claim_evidence_complete@10` trên `multi_hop` và `temporal`.
3. Precision@5/10 để kiểm soát context noise.
4. Nếu chạy generation judge: correctness, faithfulness, answer relevancy và temporal correctness.
5. Báo cáo riêng `no_diacritics`, không để aggregate che mất regression.

Không gọi một variant là tốt nhất chỉ dựa trên một metric. Variant được chọn phải thắng hoặc không thua đáng kể trên metric chính và không có regression lớn ở slice quan trọng.

### Experiment identity bắt buộc

Manifest phải phân biệt ít nhất:

- pipeline/mode;
- ground-truth SHA-256;
- sample limit;
- retrieval top-k và k-values;
- embedding profile;
- collection/database;
- BM25 config;
- RRF config;
- reranker model/revision/pool;
- hierarchy budgets/chunk directory digest hoặc corpus identity;
- adaptive planner model và planning limits;
- query rewrite enabled/disabled;
- generation/judge model nếu có.

Không ghi API key/token vào manifest.

### Validation

```powershell
uv run --extra dev pytest tests/evaluation
uv run --extra dev ruff check src/legal_rag/evaluation tests/evaluation
```

## Phase 7 - Documentation, Docker và benchmark

### Files cần sửa

- `README.md`
  - cập nhật danh sách mode/pipeline;
  - giải thích rõ Milvus vẫn một row/chunk;
  - thêm lệnh smoke/full benchmark đúng semantics.
- `.env.example`
  - thêm settings mới, không thêm secret thật.
- `docs/naive_rag_evaluation_readiness.md`
  - sửa các lệnh cũ đang dùng `--limit` như sample count;
  - cập nhật metric hiện tại và variant mới.
- `docker-compose.yml`
  - chỉ sửa nếu app service cần nhận thêm environment variables; không tạo thêm Milvus service/collection cho từng variant.

### Không cần xóa file/source nào

Không có file nào cần xóa trong kế hoạch hiện tại. Chỉ cân nhắc đánh dấu report cũ là historical nếu schema metric của report không còn khớp code hiện tại; không xóa dữ liệu đánh giá cũ.

## 6. Chiến lược test tổng thể

Mỗi phase thực hiện theo thứ tự:

1. Characterization/unit test cho behavior hiện tại.
2. Implement thay đổi nhỏ nhất.
3. Chạy unit tests trực tiếp liên quan.
4. Chạy toàn bộ `tests/naive_rag`, `tests/evaluation` và test package mới.
5. Chạy Ruff.
6. Chạy retrieval smoke trên sample nhỏ.
7. Chỉ khi output/metric đúng mới chạy full 1.000 câu.

Full checks trước khi benchmark:

```powershell
uv run --extra dev pytest tests/naive_rag tests/retrieval tests/adaptive_rag tests/evaluation
uv run --extra dev ruff check src tests
```

Nếu `tests/retrieval` hoặc `tests/adaptive_rag` chưa tồn tại ở phase đang làm, bỏ path đó khỏi lệnh thay vì tạo test rỗng.

## 7. Lệnh chạy sau khi implement

Các lệnh dưới đây là target CLI sau khi hoàn thành các phase tương ứng.

### 7.1 Khởi động Milvus

```powershell
docker compose --profile milvus up -d
docker compose ps
```

### 7.2 Index corpus nếu collection chưa đúng/đủ

```powershell
uv run legal-naive-rag-index --data-dir data/processed/groundtruth
```

Phase 1, 2 và 3 không đổi schema Milvus. Tuy nhiên BM25 được rebuild trong memory mỗi process; không cần re-index chỉ vì đổi tokenizer. Chỉ re-index khi collection thiếu, metadata không khớp embedding profile/corpus hoặc indexer báo cần rebuild.

### 7.3 Smoke test retrieval, 20 câu

```powershell
uv run legal-evaluate retrieve --retriever naive --naive-search-mode keyword --max-samples 20 --limit 10 --k-values 1,3,5,10 --predictions-output data/evaluation/smoke/naive_keyword.jsonl --metrics-output data/evaluation/smoke/naive_keyword_metrics.json

uv run legal-evaluate retrieve --retriever naive --naive-search-mode semantic --max-samples 20 --limit 10 --k-values 1,3,5,10 --predictions-output data/evaluation/smoke/naive_semantic.jsonl --metrics-output data/evaluation/smoke/naive_semantic_metrics.json

uv run legal-evaluate retrieve --retriever naive --naive-search-mode hybrid --max-samples 20 --limit 10 --k-values 1,3,5,10 --predictions-output data/evaluation/smoke/naive_hybrid.jsonl --metrics-output data/evaluation/smoke/naive_hybrid_metrics.json

uv run legal-evaluate retrieve --retriever naive --naive-search-mode hybrid_rerank --max-samples 20 --limit 10 --k-values 1,3,5,10 --predictions-output data/evaluation/smoke/naive_hybrid_rerank.jsonl --metrics-output data/evaluation/smoke/naive_hybrid_rerank_metrics.json

uv run legal-evaluate retrieve --retriever naive --naive-search-mode hierarchical --max-samples 20 --limit 10 --k-values 1,3,5,10 --predictions-output data/evaluation/smoke/naive_hierarchical.jsonl --metrics-output data/evaluation/smoke/naive_hierarchical_metrics.json
```

Adaptive smoke sau phase 5/6:

```powershell
uv run legal-evaluate run --pipeline adaptive --max-samples 20 --limit 10 --k-values 1,3,5,10 --run-dir data/evaluation/runs/adaptive_smoke
```

### 7.4 Full retrieval benchmark, 1.000 câu

Không truyền `--max-samples` để chạy toàn bộ ground truth.

```powershell
uv run legal-evaluate retrieve --retriever naive --naive-search-mode keyword --limit 10 --k-values 1,3,5,10 --predictions-output data/evaluation/full/naive_keyword.jsonl --metrics-output data/evaluation/full/naive_keyword_metrics.json

uv run legal-evaluate retrieve --retriever naive --naive-search-mode semantic --limit 10 --k-values 1,3,5,10 --predictions-output data/evaluation/full/naive_semantic.jsonl --metrics-output data/evaluation/full/naive_semantic_metrics.json

uv run legal-evaluate retrieve --retriever naive --naive-search-mode hybrid --limit 10 --k-values 1,3,5,10 --predictions-output data/evaluation/full/naive_hybrid.jsonl --metrics-output data/evaluation/full/naive_hybrid_metrics.json

uv run legal-evaluate retrieve --retriever naive --naive-search-mode hybrid_rerank --limit 10 --k-values 1,3,5,10 --predictions-output data/evaluation/full/naive_hybrid_rerank.jsonl --metrics-output data/evaluation/full/naive_hybrid_rerank_metrics.json

uv run legal-evaluate retrieve --retriever naive --naive-search-mode hierarchical --limit 10 --k-values 1,3,5,10 --predictions-output data/evaluation/full/naive_hierarchical.jsonl --metrics-output data/evaluation/full/naive_hierarchical_metrics.json
```

### 7.5 Full end-to-end benchmark

Chạy một run directory riêng cho mỗi pipeline/mode. Bật `--generation` chỉ khi muốn gọi judge trong cùng workflow.

```powershell
uv run legal-evaluate run --pipeline naive --naive-search-mode hybrid --limit 10 --k-values 1,3,5,10 --run-dir data/evaluation/runs/naive_hybrid_full --generation

uv run legal-evaluate run --pipeline naive --naive-search-mode hybrid_rerank --limit 10 --k-values 1,3,5,10 --run-dir data/evaluation/runs/naive_hybrid_rerank_full --generation

uv run legal-evaluate run --pipeline naive --naive-search-mode hierarchical --limit 10 --k-values 1,3,5,10 --run-dir data/evaluation/runs/naive_hierarchical_full --generation

uv run legal-evaluate run --pipeline adaptive --limit 10 --k-values 1,3,5,10 --run-dir data/evaluation/runs/adaptive_full --generation

uv run legal-evaluate run --pipeline corrective --limit 10 --k-values 1,3,5,10 --run-dir data/evaluation/runs/corrective_full --generation

uv run legal-evaluate run --pipeline graphiti --limit 10 --k-values 1,3,5,10 --run-dir data/evaluation/runs/graphiti_full --generation
```

Ontology chỉ chạy sau khi task ontology xác nhận artifacts và query contract ổn định:

```powershell
uv run legal-evaluate run --pipeline ontology --ontology-paths <ontology-file-1.ttl> <ontology-file-2.ttl> --limit 10 --k-values 1,3,5,10 --run-dir data/evaluation/runs/ontology_full --generation
```

Nếu run bị gián đoạn, chạy lại đúng lệnh và thêm `--resume`. Không đổi config hoặc model trong khi resume.

### 7.6 So sánh kết quả

Sau phase 6, target command:

```powershell
uv run legal-evaluate compare data/evaluation/runs/*/metrics.json --output data/evaluation/rag_variant_comparison.json
```

Nếu PowerShell không expand wildcard theo cách CLI mong đợi, truyền từng path metrics rõ ràng.

## 8. Dependency giữa các phase

```text
Phase 0 baseline lock
  -> Phase 2 hybrid reranker
      -> Phase 3 hierarchy
          -> Phase 4 shared RRF cleanup
              -> Phase 5 adaptive multi-hop
                  -> Phase 6 evaluation comparison
                      -> Phase 7 docs + full benchmark

Phase 1 accent-aware BM25: deferred, chỉ mở lại nếu số liệu yêu cầu.
```

- Phase 2 chỉ phụ thuộc Phase 0; giữ nguyên BM25 để cô lập tác động của reranker.
- Phase 3 reuse hybrid seed retrieval từ Phase 1/2 nhưng không bắt buộc dùng Cross-Encoder.
- Phase 5 phụ thuộc shared RRF helper ở Phase 4.
- Phase 6 cần result/evidence contracts của Phase 3 và 5 đã ổn định.
- Ontology full benchmark độc lập với việc implement các phase trên nhưng bị chặn bởi task ontology hiện tại.

## 9. Các điểm cần quyết định trước khi code từng phần

Các quyết định dưới đây không nên tự đoán trong lúc implement:

1. [Đã chốt cho v1] Cross-Encoder dùng `BAAI/bge-reranker-v2-m3`, revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`.
2. [Đã chốt cho v1] Candidate pool dùng `top_k * 5`, cấu hình bằng `NAIVE_RAG_RERANKER_POOL_MULTIPLIER`.
3. [Đã chốt cho v1] Hierarchical unit budget là tối đa 5 chunks và 2.500 source tokens.
4. [Đã chốt cho v1] Hierarchical dùng `hybrid` làm seed để đo riêng đóng góp của structure.
5. [Đã chốt cho v1] Adaptive base retriever mặc định dùng `hybrid_rerank` và có thể đổi bằng `ADAPTIVE_RAG_BASE_SEARCH_MODE`.
6. [Đã chốt cho v1] Planner dùng shared `LLM_MODEL`; model và toàn bộ adaptive planner config được ghi trong run identity.
7. Ngưỡng "không giảm đáng kể" cho regression. Nên chốt trước full benchmark, ví dụ absolute delta tối đa trên metric chính, thay vì chọn sau khi thấy kết quả.

## 10. Definition of done

- [ ] Tất cả baseline cũ vẫn chạy và cho output cùng semantics.
- [ ] Quyết định về query không dấu được ghi nhận; accent-aware BM25 đang deferred và phải được đánh giá lại theo slice.
- [x] `hybrid_rerank` có candidate fusion và reranking tách biệt, model được pin.
- [x] `hierarchical` dùng local chunk structure, không thêm row Milvus.
- [x] Naive hybrid dùng shared RRF helper với deterministic tie-break.
- [ ] `adaptive` route direct/multi-hop và lưu plan để audit.
- [ ] Tất cả variant trả về ranked units có evidence chunk IDs đúng.
- [ ] Evaluation không có MRR/NDCG và không tính sai grouped evidence.
- [ ] Run identity ngăn resume/so sánh các config không tương thích.
- [ ] Unit tests, Ruff và smoke evaluation pass.
- [ ] Full 1.000 câu được chạy cho các variant đã vượt smoke gate.
- [ ] Report cuối phân tích aggregate và các slice `single_hop`, `multi_hop`, `temporal`, `no_diacritics`.
- [ ] Không có thay đổi ngoài phạm vi, đặc biệt không sửa ontology internals.
