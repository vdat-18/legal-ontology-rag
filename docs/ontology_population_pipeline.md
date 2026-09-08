# Ontology Population Pipeline

## Mục tiêu

Luồng này chuyển các chunk văn bản pháp luật đã screening thành ontology instance trước khi làm retrieval hoặc
GraphRAG. Thiết kế tối giản chỉ giữ dữ liệu cần cho câu hỏi pháp lý và khả năng truy nguồn.

```text
chunks đã screening
  -> inventory hành vi trong target
  -> population theo từng văn bản
  -> assembly các phần cùng một rule
  -> legal change giữa các văn bản
  -> legal exception và citation
  -> chuẩn hóa concept
  -> RDF graph
```

Runtime có sáu bước người dùng chạy. Mỗi bước chứa đúng một loại quyết định LLM; chuẩn bị candidate, kiểm tra
ID/reference và materialization chạy nội bộ trong cùng lệnh:

```text
1. extract conducts       -> LLM inventory hành vi
2. bind population       -> LLM hoàn thiện rule
3. assemble rules        -> LLM ghép fragment
4. resolve changes       -> LLM xác định thay đổi
5. resolve exceptions    -> LLM xác định ngoại lệ
6. resolve concepts      -> LLM chuẩn hóa concept
```

Một bước có thể gửi nhiều batch đồng thời, nhưng mọi request trong bước đó dùng cùng một prompt và cùng một nhiệm vụ
semantic. Candidate, proposal và progress vẫn được ghi ra file; chạy lại lệnh dùng checkpoint tương thích thay vì gọi
lại LLM đã hoàn tất.

Không có file review, reviewer, ngày review, vòng LLM thứ hai hoặc semantic validator trong runtime. Code không đánh giá
nội dung pháp lý của proposal; nó chỉ parse typed output, ràng buộc ID vào input hiện tại và bảo đảm transformation có
thể persistence/materialize xác định. SHACL, CQ và reasoner là QA tách rời, không chặn runtime.

## Nguyên tắc dữ liệu

- LLM quyết định ngữ nghĩa từ toàn bộ context được cung cấp; code không dùng regex, keyword hay similarity để phân loại
  hành vi, hậu quả, thay đổi, ngoại lệ hoặc concept.
- Candidate generation chỉ giới hạn không gian làm việc bằng cấu trúc, thứ tự văn bản và giới hạn cấu hình.
- Canonical RDF giữ provenance tối thiểu: `groundedIn -> LegalProvision` và `sourcedFromChunk -> SourceChunk`.
- Text nguồn trong JSON trung gian chỉ phục vụ grounding và resume của LLM; RDF không tạo `EvidenceSpan`, `exactText`,
  `containsSpan` hoặc `supportedBy`.
- Dữ liệu chưa biết giữ `null`; không biến thành `0`, `false`, vô hạn hay “không áp dụng”.
- Không bước nào tự sửa TBox. Thay đổi class/property/constraint phải đi qua ontology evolution riêng.

## 1. Inventory hành vi

Sáu bước LLM mặc định chạy tối đa 30 batch đồng thời; `--max-concurrency` chỉ cần truyền khi muốn đổi giới hạn này.

LLM chỉ đọc target chunk và lập closed inventory các hành vi độc lập được target trực tiếp mô tả. Kết quả được ghi cho
cả target không có hành vi, vì vậy `conducts.jsonl` đồng thời là checkpoint resume của bước này.

```powershell
uv run --extra dev legal-extract-ontology-conducts `
  <chunks.jsonl> <screening.jsonl> <document_metadata.jsonl> `
  ontology/population_contract.yaml <conducts.jsonl> `
  --preflight-report-path <preflight.json> --max-concurrency 30
```

Thêm `--target-ids-path <selection.json>` khi chạy pilot; dùng cùng option ở bước 2.

Sau khi review output, có thể selective rerun mà không xóa các inventory đúng:

```powershell
uv run --extra dev legal-extract-ontology-conducts `
  <chunks.jsonl> <screening.jsonl> <document_metadata.jsonl> `
  ontology/population_contract.yaml <conducts.jsonl> `
  --rerun-target-ids-path <rerun-selection.json> `
  --preflight-report-path <preflight.json> --max-concurrency 30
```

`rerun-selection.json` dùng cùng schema với target selection và chỉ chứa các `chunk_ids` đã review là sai. CLI giữ
nguyên nội dung cùng provenance của các target khác, thay atomically từng batch rerun và resume khi chạy lại cùng
selection. Bước 2 fingerprint toàn bộ inventory đã review nên chấp nhận provenance theo từng record nhưng vẫn phát
hiện mọi thay đổi nội dung đầu vào.

## 2. Population binding

Population đọc target chunk cùng hai chunk trước, một chunk sau và context cùng Điều trong giới hạn cấu hình. LLM thực
hiện đúng một tác vụ: gắn applicability, condition tree và consequences từ reading window vào inventory hành vi đã
khóa ở bước 1.

Code kiểm tra typed schema cùng ID/reference cần để persistence và materialization. Nó không áp rule về số consequence,
condition arity, completeness, mức giá trị hay phân loại pháp lý. Bước này chưa tạo legal change, exception redirect,
concept equivalence hoặc occurrence của một vụ việc cụ thể.

```powershell
uv run --extra dev legal-bind-ontology-population `
  <chunks.jsonl> `
  <screening.jsonl> `
  <document_metadata.jsonl> `
  ontology/population_contract.yaml `
  <conducts.jsonl> `
  <population.jsonl> `
  <progress.jsonl> `
  <graph.ttl> `
  --preflight-report-path <preflight.json> `
  --max-concurrency 30
```

Sau khi review, có thể giữ nguyên các chunk đúng và rerun riêng các chunk sai. Với chunk có nhiều conduct, dùng
`--batch-size 1` để mỗi response có đủ không gian trả toàn bộ rule:

```powershell
uv run --extra dev legal-bind-ontology-population `
  <chunks.jsonl> <screening.jsonl> <document_metadata.jsonl> `
  ontology/population_contract.yaml <conducts.jsonl> `
  <population.jsonl> <progress.jsonl> <graph.ttl> `
  --rerun-target-ids-path <rerun-selection.json> `
  --batch-size 1 --preflight-report-path <preflight.json> --max-concurrency 30
```

Selective rerun thay atomically records của target được chọn, giữ nguyên records của target khác và dùng run ID riêng
cho selection để lần chạy lại cùng lệnh chỉ tiếp tục các target chưa hoàn thành.

CLI `legal-populate-ontology` cũ vẫn được giữ để tương thích, nhưng nó gộp cả hai prompt và không thuộc flow sáu bước
được khuyến nghị.

## 3. Assembly trong một văn bản

Một rule có thể bị chia qua nhiều chunk. Candidate builder tạo các window gần nhau theo cấu trúc; LLM quyết định record
nào cùng thuộc một rule. Resolver gom proposal có liên hệ thành component rồi review cả component đơn lẫn component
chồng lấn để loại over-merge và tạo quyết định toàn cục. Finalizer sau đó ghi canonical output.

```powershell
uv run --extra dev legal-assemble-ontology-rules `
  <chunks.jsonl> <metadata.jsonl> <population.jsonl> `
  ontology/population_contract.yaml ontology/assembly_contract.yaml `
  <candidates.json> <proposals.jsonl> <progress.jsonl> `
  <canonical_population.jsonl> <canonical_graph.ttl> --max-concurrency 30
```

Resolver tự tạo `<proposals>.resolved.jsonl` và `<progress>.resolution.jsonl`; chạy lại cùng lệnh sẽ resume cả proposal
cục bộ lẫn conflict component đã hoàn thành.

## 4. Legal change giữa các văn bản

Corpus manifest sắp xếp các văn bản theo thứ tự cần so sánh. Candidate builder tạo bounded cross-document windows mà
không chấm điểm similarity. Later records chỉ đi cùng nhóm source chunks đã sinh ra chúng, thay vì bị nhân với mọi
source chunk của văn bản sau. Rule summaries giữ đủ semantics nhưng không lặp evidence vì authoritative change evidence
đã được cung cấp riêng. Mỗi document có thể khai báo `comparison_document_ids`; danh sách rỗng giữ văn bản độc lập trong
canonical output nhưng không tạo candidate, còn khi bỏ field này thì vẫn so với mọi văn bản đứng trước để tương thích.
Ngày hiệu lực mặc định từ document metadata được đưa vào mọi window; ngày riêng trong source của provision được ưu tiên.
LLM đề xuất `Amendment`, `Replacement` hoặc `RepealOrSupersession` chỉ khi nguồn nêu trực tiếp.
Locator được đối chiếu trước semantics để không ghép các rule giống nội dung nhưng khác điều khoản. Repeal phải khớp đúng
Điều/Khoản/Điểm. Việc chuyển leaf giữa hai Điểm chỉ được suy ra trong cùng Điều/Khoản khi source trực tiếp viết lại cả hai
Điểm. Records khác Điều/Khoản không được liên kết vì quy định cùng hành vi tại locator khác có thể tồn tại đồng thời. Một
rule version có thể tham gia nhiều link khi source thực sự tách hoặc nhập các leaf. Finalizer ràng buộc rule IDs vào corpus
và cập nhật validity của các rule version mà không đánh giá lại nội dung thay đổi.

```powershell
uv run --extra dev legal-resolve-ontology-changes `
  <corpus_manifest.json> ontology/change_resolution_contract.yaml `
  ontology/population_contract.yaml <candidates.json> <proposals.jsonl> <progress.jsonl> `
  <canonical_population.jsonl> <legal_change_events.jsonl> <canonical_graph.ttl> --max-concurrency 30
```

## 5. Legal exception và citation

Candidate builder chỉ ghép source chunks với các general rules phát sinh từ chính các chunks đó. LLM thứ nhất phát hiện
ngoại lệ và chuẩn hóa citation thành số văn bản cùng locator tuyệt đối. Code tra cứu cấu trúc đúng document/locator,
sau đó LLM thứ hai chọn rule versions thực sự thuộc phạm vi citation trong tập nhỏ này; không dùng từ khóa, regex hay
similarity để quyết định ngữ nghĩa. Finalizer gộp các fragment cùng một ngoại lệ mà không đánh giá lại nội dung pháp lý.

Candidate schema `0.2.0` không còn nhúng toàn bộ target IDs. Artifact candidate cũ phải được tạo lại bằng candidate
builder hoặc command tích hợp trước khi resume Step 5.

Khi review phát hiện candidate sai, truyền selection JSON dạng
`{"schema_version":"0.1.0","candidate_ids":[...]}` qua `--rerun-candidate-ids-path`. Selective rerun yêu cầu progress
cũ đã đủ toàn bộ candidate, chỉ thay proposals của tập được chọn và giữ nguyên các kết quả còn lại.

```powershell
uv run --extra dev legal-resolve-ontology-exceptions `
  <corpus_manifest.json> <canonical_population.jsonl> <legal_change_events.jsonl> `
  ontology/exception_resolution_contract.yaml ontology/population_contract.yaml `
  <candidates.json> <proposals.jsonl> <progress.jsonl> `
  <legal_exceptions.jsonl> <canonical_graph.ttl> --max-concurrency 30
```

## 6. Chuẩn hóa concept

Population ban đầu giữ concept theo từng rule để tránh merge sai. Step 6 chỉ chuẩn hóa hai controlled vocabulary tái sử
dụng là `LegalSubjectRole` và `VehicleCategory`; `RegulatedConduct` vẫn gắn với canonical rule vì đó là mô tả hành vi đầy
đủ, không phải thuật ngữ nguyên tử. Candidate builder tạo một inventory toàn cục cho mỗi loại, dùng một occurrence có
bằng chứng làm đại diện cho mỗi label nguyên văn. Nhờ vậy batch size 1 tương ứng một inventory LLM task, thay vì tạo mọi
cặp occurrence. LLM chỉ đề xuất `SameConcept`, broader/narrower hoặc `Related` khi có căn cứ; cặp không liên quan bị bỏ.
Finalizer hợp nhất các occurrence có cùng loại và cùng label nguyên văn, áp dụng quyết định semantic, đồng thời chặn quan
hệ xung đột và chu trình broader trước khi tạo canonical concepts và SKOS relations.

Contract schema `0.3.0` và candidate schema `0.2.0` dùng `inventories` thay cho exhaustive `pairs`; artifact Step 6 cũ
phải được tạo lại. Sau một
lần chạy đầy đủ, có thể selective rerun một inventory bằng selection JSON
`{"schema_version":"0.1.0","candidate_ids":[...]}` và `--rerun-candidate-ids-path`.

```powershell
uv run --extra dev legal-resolve-ontology-concepts `
  <corpus_manifest.json> <canonical_population.jsonl> <legal_change_events.jsonl> `
  <legal_exceptions.jsonl> ontology/population_contract.yaml ontology/concept_resolution_contract.yaml `
  <candidates.json> <proposals.jsonl> <progress.jsonl> `
  <canonical_concepts.jsonl> <concept_relations.jsonl> <canonical_graph.ttl> --max-concurrency 30
```

## Khi có dữ liệu mới

1. Chunk và screening văn bản mới.
2. Chạy conduct inventory rồi population binding cho văn bản đó.
3. Chạy assembly trong văn bản.
4. Thêm văn bản vào corpus manifest và chạy lại change, exception, concept resolution.
5. Nếu dữ liệu không biểu diễn được bằng ontology hiện tại, tạo ontology gap proposal; không tự thêm class/property từ
   population output.

Ontology instance tăng theo dữ liệu. TBox chỉ thay đổi khi có nhu cầu mới được chứng minh bằng CQ và vượt qua ontology
evolution checks.

## Kiểm tra

```powershell
uv run --extra dev pytest `
  tests/ontology `
  tests/ontology_population `
  tests/ontology_assembly `
  tests/ontology_change `
  tests/ontology_exception `
  tests/ontology_concept

uv run ruff check src tests
```

Semantic QA bằng ROBOT/HermiT là release check bổ sung. Publication/W3ID là tùy chọn và không thuộc core ontology flow.
