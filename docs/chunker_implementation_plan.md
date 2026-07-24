# Kế hoạch triển khai Chunker Pipeline

## 1. Kết luận thiết kế

Chunker sẽ đọc trực tiếp các file `*.source.jsonl` do parser tạo và ghi ra
`*.chunks.jsonl`. Phiên bản đầu tiên dùng chunking theo cấu trúc nguồn, giới hạn
bằng tokenizer của embedding model và không dùng LLM để quyết định điểm cắt.

Luồng mục tiêu:

```text
Parser source JSONL
    -> đọc và tách document scope
    -> chuẩn hóa paragraph/table/image/spreadsheet row
    -> chia source unit quá lớn
    -> gộp source unit liền kề theo token budget
    -> gắn provenance và quan hệ chunk trước/sau
    -> ghi atomically thành chunk JSONL
```

Các nâng cấp contextual retrieval, hybrid retrieval và reranking được đánh giá
sau khi baseline deterministic đã hoàn thành. Chúng không được làm thay đổi
ranh giới hoặc nội dung nguồn của chunk nếu chưa có benchmark chứng minh lợi ích.

## 2. Cơ sở lựa chọn

- [Beyond Chunk-Then-Embed](https://arxiv.org/abs/2602.16974) cho thấy
  structure-based chunking hiệu quả hơn LLM-guided chunking trong in-corpus
  retrieval, là dạng retrieval chính của dự án này.
- [Chunking German Legal Code](https://arxiv.org/abs/2605.19806) cho thấy việc
  giữ cấu trúc pháp lý nguồn hiệu quả hơn các phương pháp phức tạp như Lumber,
  RAPTOR và semantic clustering.
- [A Systematic Investigation of Document Chunking Strategies](https://arxiv.org/abs/2603.06976)
  cho thấy paragraph grouping đứng đầu tổng thể và đặc biệt phù hợp với legal.
- [Summary-Augmented Chunking](https://arxiv.org/abs/2510.06999) cho thấy context
  cấp document có thể giảm việc lấy đúng nội dung từ nhầm văn bản pháp luật.
- [dsRAG](https://github.com/D-Star-AI/dsRAG) cho thấy contextual headers và mở
  rộng segment tại query time có thể hữu ích mà không cần LLM semantic sectioning.

## 3. Trạng thái đầu vào hiện tại

Parser schema `2.0.0` đã cung cấp:

- Thứ tự ổn định bằng `sequence_index` hoặc `child_sequence_index`.
- Paragraph với `text`, `unit_id`, `source_ref` và annotations.
- Table với row, cell, nested table và source reference ở từng cấp.
- Image caption với `description_vi` và provenance của ảnh.
- Spreadsheet row với sheet, cell, value và formula.
- Root document và attachment-child records với đầy đủ quan hệ cha-con.

Chunker không suy luận heading vì parser đã xác định Word style không đủ tin cậy.
Mỗi root document và mỗi attachment child phải được chunk thành một stream độc
lập; chunk không bao giờ đi qua ranh giới giữa hai stream.

## 4. Mục tiêu

- Tạo chunk phù hợp cho semantic retrieval và lexical retrieval.
- Tính token bằng đúng tokenizer dùng cho embedding model.
- Giữ paragraph, table row, image caption và spreadsheet row làm đơn vị cấu trúc.
- Không làm mất thứ tự, source reference hoặc attachment lineage.
- Kết quả deterministic: cùng input và config phải tạo cùng output byte-for-byte.
- Cho phép mở rộng chunk liền kề ở retrieval time mà không lưu overlap dư thừa.
- Ghi đè output cũ atomically; lỗi phải giữ nguyên exception và cause đầy đủ.
- Không thay đổi parser output hoặc schema nguồn.

## 5. Ngoài phạm vi chunker v1

- Không gọi embedding API và không ghi vector database.
- Không triển khai BM25, reranker hoặc answer generation trong package chunker.
- Không dùng regex, keyword, font, numbering hoặc handcrafted score để đoán heading.
- Không dùng LLM để tìm semantic boundary.
- Không tạo overlap cố định giữa các chunk bình thường.
- Không lặp lại hàng đầu bảng như header nếu parser chưa có metadata header đáng tin.
- Không biến attachment metadata hoặc icon thành nội dung pháp lý có thể tìm kiếm.
- Không triển khai RAPTOR, LumberChunker hoặc adaptive rule generation ở baseline.

## 6. Cấu trúc code đề xuất

```text
src/legal_rag/
├── embedding/
│   ├── __init__.py
│   ├── profiles.py         # Đọc model/tokenizer profile từ config versioned
│   └── tokenizer.py        # Adapter tokenizer dùng chung cho chunking và indexing
├── chunker/
│   ├── __init__.py
│   ├── models.py          # Typed source units, fragments và output chunks
│   ├── reader.py          # JSONL boundary, schema và document scopes
│   ├── rendering.py       # Text representation cho từng loại source unit
│   ├── splitting.py       # Chia paragraph/table/cell quá token budget
│   ├── pipeline.py        # Packing, IDs, neighbor links và orchestration
│   └── writer.py          # Atomic JSONL writer
└── chunk_cli.py           # CLI cho một source JSONL

tests/chunker/
├── test_reader.py
├── test_rendering.py
├── test_splitting.py
├── test_pipeline.py
├── test_writer.py
└── test_cli.py

tests/embedding/
├── test_profiles.py
└── test_tokenizer.py
```

Chỉ tách module khi trách nhiệm thực sự khác nhau. Không tạo `utils.py`, service
wrapper hoặc helper chỉ được gọi một lần.

## 7. Cấu hình

### 7.1. Quyết định bắt buộc trước khi viết tokenizer adapter

- Chọn embedding model đầu tiên.
- Xác nhận tokenizer chính thức của model.
- Xác nhận token input tối đa của model hoặc embedding endpoint.
- Xác nhận tokenizer hỗ trợ offset mapping để cắt trên chuỗi gốc.

### 7.2. Typed settings

Thêm `ChunkerSettings` trong `src/legal_rag/settings.py`, tách khỏi các biến LLM
để chạy chunker không yêu cầu image-caption credentials. Model, tokenizer, revision,
dimensions và input limit nằm trong `data/config/embedding_models.json`; environment
chỉ chọn profile và các budget của chunker:

```text
EMBEDDING_PROFILE=<profile-id-trong-embedding_models.json>
CHUNK_MAX_TOKENS=<giới-hạn-body-chunk>
CHUNK_SPLIT_OVERLAP_TOKENS=<overlap-chỉ-cho-source-unit-bị-cắt>
```

Không thêm `target_tokens`, `min_tokens`, threshold semantic hoặc các knobs chưa
có nhu cầu thực tế. Các giá trị thay đổi được không được hardcode trong source.

Nếu tokenizer cần package mới, thêm nó thành direct dependency với version được
pin và cập nhật `uv.lock`. Không dựa vào dependency gián tiếp của package khác.

## 8. Schema output đề xuất

Mỗi dòng `*.chunks.jsonl` là một chunk:

```json
{
  "schema_version": "1.0.0",
  "chunk_id": "<sha256>",
  "content_sha256": "<sha256>",
  "chunker_config_sha256": "<sha256>",
  "scope_id": "<sha256>",
  "chunk_index": 12,
  "kind": "paragraph_group",
  "source_name": "168_2024_ND-CP_619502.docx",
  "source_sha256": "<sha256>",
  "record_scope": "root",
  "parent_attachment_id": null,
  "attachment_member_path": null,
  "attachment_member_sha256": null,
  "unit_ids": ["<unit-id-1>", "<unit-id-2>"],
  "source_refs": [
    "word/document.xml#/body/41",
    "word/document.xml#/body/42"
  ],
  "sequence_start": 34,
  "sequence_end": 35,
  "text": "Nội dung nguyên gốc đã được ghép...",
  "context_text": "",
  "embedding_text": "Nội dung nguyên gốc đã được ghép...",
  "text_token_count": 481,
  "embedding_token_count": 481,
  "previous_chunk_id": "<sha256-or-null>",
  "next_chunk_id": "<sha256-or-null>"
}
```

### 8.1. Ý nghĩa field

- `chunk_id`: ID deterministic từ scope, ranh giới nguồn, nội dung và phiên bản
  thuật toán; không phụ thuộc đường dẫn output hoặc thời gian chạy.
- `content_sha256`: hash của `text` để audit và deduplicate kỹ thuật.
- `chunker_config_sha256`: fingerprint của tokenizer và các token settings ảnh
  hưởng đến boundary.
- `scope_id`: root source hoặc một attachment-child stream cụ thể.
- `kind`: một trong `paragraph_group`, `table`, `image`, `spreadsheet_group`.
- `unit_ids`, `source_refs`: provenance quay lại parser output.
- `text`: nội dung source-derived đưa cho reranker hoặc LLM.
- `context_text`: metadata contextualization; rỗng trong baseline.
- `embedding_text`: baseline bằng `text`; phase sau có thể là context cộng text.
- `text_token_count`: số token của source-derived `text`.
- `embedding_token_count`: số token cuối cùng gửi tới embedding model.
- `previous_chunk_id`, `next_chunk_id`: chỉ liên kết trong cùng `scope_id`.

Không lưu toàn bộ parser record lặp lại trong chunk. Source JSONL vẫn là nguồn
chuẩn; chunk chỉ giữ text cần retrieval và provenance để truy ngược.

## 9. Quy tắc normalize và chunk

### 9.1. Document scope

- Root records không có `record_scope` được chuẩn hóa thành `root` theo contract
  parser hiện tại.
- Attachment-child records được nhóm theo root source, parent attachment và
  attachment member SHA.
- Sắp xếp root bằng `sequence_index`; child bằng `child_sequence_index`.
- Flush chunk đang xây dựng khi scope thay đổi.

### 9.2. Paragraph

- Paragraph có text được giữ nguyên và trở thành một atomic text unit.
- Các paragraph liền kề được nối bằng hai newline và greedily pack tới token limit.
- Table, image-only record hoặc spreadsheet boundary sẽ kết thúc paragraph group.
- Paragraph rỗng không tạo chunk; image hoặc attachment tại cùng record vẫn được
  xử lý bằng pipeline tương ứng.
- Paragraph vượt giới hạn được chia bằng token offsets trên chuỗi gốc.

### 9.3. Table

Table được render theo thứ tự row và cell bằng một format ổn định, có version.
Không dùng Markdown-table heuristic vì bảng có merge cell và nested table.

Thứ tự chia:

1. Giữ toàn bộ table nếu vừa token limit.
2. Nếu quá lớn, greedily pack các row liên tiếp.
3. Nếu một row quá lớn, pack các cell liên tiếp.
4. Nếu một cell quá lớn, đi theo các block paragraph/nested-table gốc.
5. Nếu một block vẫn quá lớn, cắt bằng tokenizer offset mapping.

Mỗi table chunk giữ source ref của table và các row/cell/block thực sự xuất hiện.
Không đoán hoặc lặp header row.

### 9.4. Image

- Mỗi `description_vi` tạo một image chunk độc lập.
- `text` chỉ chứa caption, không sao chép paragraph trước/sau vào source content.
- Source refs gồm image occurrence và parent source position.
- Context cục bộ được bổ sung ở retrieval time bằng neighbor expansion thay vì
  nhân bản text vào nhiều chunk.
- Caption vượt giới hạn được chia bằng tokenizer như paragraph.

### 9.5. Spreadsheet

- Các row liên tiếp trong cùng sheet được greedily pack theo token limit.
- Đổi sheet sẽ flush chunk hiện tại.
- Representation giữ cell reference, value và formula theo thứ tự nguồn.
- Một row hoặc cell quá lớn được chia bằng cùng cơ chế token offsets.

### 9.6. Attachment

- Attachment parent record không tự tạo semantic chunk.
- Nội dung DOC/DOCX/XLS đã parse được chunk trong attachment-child scope riêng.
- Mọi child chunk giữ `parent_attachment_id`, member path và member hash.
- Unsupported attachment chỉ tồn tại ở source layer; chunker không giả lập nội dung.

## 10. Token counting và overlap

- Encode bằng tokenizer thật của embedding model; không dùng số ký tự hoặc
  `cl100k_base` làm runtime approximation.
- `text_token_count` không được vượt `CHUNK_MAX_TOKENS`.
- `embedding_token_count` không được vượt `EMBEDDING_MAX_INPUT_TOKENS`; baseline
  có hai token count bằng nhau.
- Chunk bình thường không overlap.
- `CHUNK_SPLIT_OVERLAP_TOKENS` chỉ áp dụng khi một atomic unit tự nó quá lớn.
- Dùng offset mapping để lấy substring từ text nguồn thay vì decode token rồi làm
  thay đổi whitespace hoặc Unicode.
- Overlap phải nhỏ hơn max tokens; đây là boundary validation của untrusted config.

## 11. Determinism và provenance

- Dùng `sha256_json` hiện có cho config fingerprint, scope ID và chunk ID.
- Hash input không chứa timestamp, absolute path, temporary path hoặc object order
  không ổn định.
- Tạo toàn bộ chunk trong một scope trước, sau đó gắn previous/next links.
- Neighbor links có thể đi qua paragraph/table/image boundary nhưng không được đi
  qua document hoặc attachment-child scope.
- Chạy hai lần cùng input/config phải tạo file có hash giống nhau.

## 12. Error handling và persistence

- Đọc JSON bằng `json.loads` từng physical line.
- JSON lỗi giữ nguyên `JSONDecodeError`; reader không catch hoặc đổi exception thành
  thông báo chung.
- Schema version hoặc `kind` không hỗ trợ phải fail fast; không skip record.
- Không catch exception trong core pipeline nếu không thể bổ sung context hữu ích.
- CLI không đổi exception thành thông báo chung như `error` hoặc fake success.
- Writer dùng temporary file trong output directory rồi `replace()` như parser
  writer hiện tại.
- Nếu chunking thất bại, output cũ phải còn nguyên và temporary file được dọn.

## 13. CLI dự kiến

```powershell
legal-chunk `
  data/processed/groundtruth/168_2024_ND-CP_619502.source.jsonl `
  data/chunked/groundtruth/168_2024_ND-CP_619502.chunks.jsonl
```

CLI chỉ xử lý một source file để khớp kiến trúc `legal-parse`. Batch toàn corpus
dùng PowerShell orchestration; không thêm batch service vào core package ở v1.

Thêm console script:

```toml
[project.scripts]
legal-parse = "legal_rag.cli:main"
legal-chunk = "legal_rag.chunk_cli:main"
```

Log cuối chỉ ghi số scope, số chunk, token statistics và đường dẫn; không log full
document content hoặc credentials.

## 14. Kế hoạch triển khai theo phase

### Phase 0 — Chốt embedding boundary

- [x] ✅ DONE — Chọn `Qwen/Qwen3-Embedding-0.6B` làm embedding profile mặc định.
- [x] ✅ DONE — Dùng tokenizer chính thức tại revision đã pin; cả ba profile đều
  là fast tokenizer và đã xác nhận offset mapping trên tiếng Việt/Unicode.
- [x] ✅ DONE — Chốt body budget `512`, overlap forced-split `64` trong environment;
  embedding input limit nằm trong profile versioned, không hardcode trong Python.
- [x] ✅ DONE — Ghi model, tokenizer, immutable revision, dimensions và input limit
  của ba profile vào `data/config/embedding_models.json`.

**Hoàn thành khi:** có thể encode/decode tiếng Việt và lấy offset mapping ổn định
trên paragraph, ký hiệu pháp luật và Unicode trong corpus.

### Phase 1 — Models và input contract

- [x] ✅ DONE — Tạo package `legal_rag.chunker`.
- [x] ✅ DONE — Tạo dataclass cho source record/scope, normalized unit, fragment và output chunk.
- [x] ✅ DONE — Định nghĩa chunk schema version `1.0.0` và `Chunk.to_dict()`.
- [x] ✅ DONE — Đọc JSONL, kiểm tra schema/kind ở boundary và nhóm đúng root/child scopes.
- [x] ✅ DONE — Test ordering, native malformed-JSON error, unsupported contract values
  và attachment provenance.

**Hoàn thành khi:** một fixture root + attachment child được đọc thành các stream
độc lập, đúng thứ tự và không mất provenance.

### Phase 2 — Tokenizer adapter

- [x] ✅ DONE — Thêm direct dependency `transformers==5.14.1` và kiểm tra security audit.
- [x] ✅ DONE — Implement token count có special tokens và token-window character offsets.
- [x] ✅ DONE — Test tiếng Việt, xuống dòng, ký hiệu `§`, superscript text và chuỗi dài.
- [x] ✅ DONE — Regenerate `uv.lock` cục bộ; file tiếp tục được gitignore theo policy hiện tại.

**Hoàn thành khi:** mọi substring sau split đều là slice của chuỗi đầu vào và số
token đo lại không vượt giới hạn.

### Phase 3 — Rendering source units

- [x] ✅ DONE — Render paragraph nguyên văn.
- [x] ✅ DONE — Render table row/cell/nested blocks theo format versioned, ổn định.
- [x] ✅ DONE — Render mỗi image occurrence từ `description_vi` thành unit độc lập.
- [x] ✅ DONE — Render spreadsheet row với sheet, cell reference, value và formula.
- [x] ✅ DONE — Bỏ qua base record không có searchable content mà không bỏ image/child document.
- [x] ✅ DONE — Test ordering và round-trip provenance cho từng kind.

**Hoàn thành khi:** mỗi loại record trong corpus tạo representation ổn định, dễ đọc
và source refs khớp parser output.

### Phase 4 — Oversized-unit splitting

- [x] ✅ DONE — Chia paragraph/image bằng tokenizer character offsets.
- [x] ✅ DONE — Chia table theo row → cell → block → token window.
- [x] ✅ DONE — Chia spreadsheet theo cell → token window khi một row quá lớn.
- [x] ✅ DONE — Áp dụng overlap chỉ tại forced token split; unit/group bình thường không overlap.
- [x] ✅ DONE — Test nested table, oversized row, oversized cell/block và spreadsheet cell.

**Hoàn thành khi:** không fragment nào vượt max tokens và không thay đổi thứ tự text.

### Phase 5 — Packing và output chunks

- [x] ✅ DONE — Greedily pack paragraph liền kề.
- [x] ✅ DONE — Greedily pack table rows và spreadsheet rows tương thích.
- [x] ✅ DONE — Giữ image chunk độc lập.
- [x] ✅ DONE — Sinh scope/config/content/chunk hashes.
- [x] ✅ DONE — Gắn chunk index và previous/next links trong cùng scope.
- [x] ✅ DONE — Test deterministic IDs, scope isolation và boundary flushing.

**Hoàn thành khi:** chạy hai lần cho cùng fixture tạo output byte-for-byte giống nhau.

### Phase 6 — Writer và CLI

- [ ] Implement atomic JSONL writer.
- [ ] Thêm `legal-chunk` CLI và `.env.example` settings.
- [ ] Ghi đè output cũ khi thành công.
- [ ] Giữ output cũ nếu pipeline lỗi giữa chừng.
- [ ] Không che stack trace hoặc original exception.
- [ ] Test CLI success, overwrite và failure behavior.

**Hoàn thành khi:** command một file chạy end-to-end và lỗi có đầy đủ nguyên nhân.

### Phase 7 — Full-corpus verification

- [ ] Chạy chunker trên toàn bộ `data/processed/groundtruth/*.source.jsonl`.
- [ ] Xác nhận mọi chunk có text, unit ID, source ref và scope.
- [ ] Xác nhận body và final embedding text đều nằm trong giới hạn tương ứng.
- [ ] Xác nhận chunk IDs unique và neighbor links đối xứng.
- [ ] Xác nhận root/attachment streams không bị trộn.
- [ ] Chạy lại và so sánh output hashes.
- [ ] Báo cáo count, token mean/median/p95/max theo kind.
- [ ] Review thủ công các file có bảng và ảnh nhiều nhất.

**Hoàn thành khi:** toàn corpus chạy không lỗi, deterministic và các invariants đều
được kiểm tra bằng script/test thay vì nhận xét cảm tính.

### Phase 8 — Retrieval benchmark baseline

- [ ] Dùng ground-truth corpus hiện có để tạo query/relevant-source labels đã review.
- [ ] Index `embedding_text` và một lexical index từ cùng chunk output.
- [ ] Đo Recall@5/10, MRR hoặc nDCG, citation accuracy, latency và index size.
- [ ] Lưu config fingerprint cùng từng benchmark run.
- [ ] Không thay chunker theo một vài ví dụ riêng lẻ; chỉ đổi khi metric tổng thể tốt hơn.

**Hoàn thành khi:** có baseline tái lập được để so sánh các nâng cấp SOTA.

### Phase 9 — Contextual retrieval experiments

Giữ nguyên `text` và source boundaries, chỉ thay `context_text` và
`embedding_text` trong các experiment riêng:

- [ ] Baseline: `embedding_text = text`.
- [ ] Structured context: source name, attachment path và metadata tin cậy từ
  versioned corpus config.
- [ ] Document-summary context: summary do BAML sinh một lần cho document, chỉ
  triển khai nếu structured context chưa đủ và có validation/provenance riêng.
- [ ] Late chunking nếu embedding model thực sự hỗ trợ long-context token pooling.
- [ ] So sánh với baseline bằng cùng queries, retriever và reranker.

Context phải được rút ngắn hoặc fail experiment nếu làm `embedding_text` vượt
embedding input limit; không truncate âm thầm và không thay source `text`.

Chỉ chọn phương án cải thiện retrieval/citation đủ rõ so với chi phí indexing và
LLM. Không dùng LLM summary để thay đổi source chunk boundary.

### Phase 10 — Retrieval-time neighbor expansion

Phần này thuộc retrieval pipeline, nhưng chunk schema phải hỗ trợ từ đầu:

- [ ] Retrieve top-k bằng dense + lexical retrieval.
- [ ] Rerank candidates.
- [ ] Với candidate tốt, lấy chunk trước/sau trong cùng `scope_id` theo token budget
  của answer model.
- [ ] Deduplicate các segment chồng nhau.
- [ ] Đưa source refs và unit IDs sang citation layer.

Không quay lại lưu overlap cố định nếu query-time expansion giải quyết được context.

## 15. Test matrix tối thiểu

| Case | Kỳ vọng |
| --- | --- |
| Hai paragraph nhỏ liền nhau | Một `paragraph_group` |
| Paragraph -> table -> paragraph | Ba boundary độc lập |
| Paragraph vượt max token | Nhiều fragment, chỉ case này có overlap |
| Table nhỏ | Một table chunk |
| Table nhiều row | Chia theo nhóm row liên tiếp |
| Một row quá lớn | Chia theo cell/block |
| Nested table | Giữ đúng block order và source refs |
| Nhiều image cùng record | Mỗi caption là một image chunk |
| Spreadsheet nhiều sheet | Không gộp row qua sheet boundary |
| Root + attachment child | Scope và neighbor links tách biệt |
| Invalid JSON line | Fail với path, line number và original cause |
| Writer lỗi | Output cũ không bị ghi đè |
| Chạy hai lần | File output có cùng SHA-256 |

## 16. Checks trước mỗi handoff

```powershell
uv run ruff format --check .
uv run ruff check .
uv run pytest tests/embedding
uv run pytest tests/chunker
uv run pytest
uv run pre-commit run --all-files
```

Sau Phase 7, chạy thêm full-corpus invariant check và lưu report dưới `data/`
hoặc `docs/` tùy report có phải artifact versioned hay không.

## 17. Definition of Done

Chunker pipeline được xem là sẵn sàng khi:

- [ ] API, CLI và output schema có test.
- [ ] Toàn bộ parser record có searchable content được xử lý hoặc có lý do loại bỏ rõ.
- [ ] Không chunk body hoặc final embedding text vượt giới hạn cấu hình tương ứng.
- [ ] Không có chunk đi qua document/attachment scope.
- [ ] Table, image và spreadsheet provenance truy ngược được tới source JSONL.
- [ ] Output deterministic và atomically overwrite.
- [ ] Exception không bị swallow hoặc thay bằng thông báo lỗi chung.
- [ ] Ruff, pytest và pre-commit pass.
- [ ] Có retrieval baseline và benchmark trước khi thêm contextual/LLM complexity.

## 18. Thứ tự bắt đầu khuyến nghị

1. Hoàn thành Phase 0: chọn embedding model và tokenizer.
2. Implement Phase 1-3 để có typed input và normalized representations.
3. Implement Phase 4-6 để tạo được `*.chunks.jsonl` end-to-end.
4. Chạy Phase 7 trên toàn corpus và sửa correctness issues.
5. Xây baseline retrieval ở Phase 8.
6. Chỉ sau đó thử contextual retrieval và neighbor expansion.
