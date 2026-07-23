# Parser Audit: Các hạng mục cần sửa

## Kết luận ngắn

Parser hiện tại **đủ tốt cho RAG dựa trên văn bản, bảng và caption ảnh**. Đối chiếu 35 DOCX với 22.307 record và 171 ảnh không phát hiện nội dung nào bị sai trong phạm vi schema đang hỗ trợ.

Nội dung DOC, DOCX và XLS trong attachment đã được đưa vào cùng JSONL với provenance cha–con. Field, hyperlink và bookmark cũng đã giữ được target cùng vị trí text gốc. Metadata heading không đủ tin cậy để dựng hierarchy, còn OCR/QR chuyên dụng không cần cho phạm vi semantic RAG hiện tại, nên hai hạng mục này được chủ động không triển khai.

## Thứ tự ưu tiên

| Ưu tiên | Hạng mục | Trạng thái | Ảnh hưởng |
| --- | --- | --- | --- |
| P0 | Trích xuất file đính kèm OLE | Đã hoàn thành | Payload và provenance được bảo toàn |
| P0 | Chuẩn hóa ký tự điều khiển khi ghi JSONL | Đã hoàn thành | Một số consumer có thể đọc sai số dòng/record |
| P1 | Parse tài liệu con trong attachment | Đã hoàn thành | DOC, DOCX và XLS nhúng đã có record tìm kiếm được |
| P1 | Bảo toàn field, hyperlink và bookmark | Đã hoàn thành | Ký hiệu trích dẫn truy được đúng bookmark hoặc URL đích |
| P1 | Bảo toàn cấp heading gốc | Không triển khai | Style Word được gán thiếu nhất quán và không đủ tin cậy cho hierarchy |
| P1 | Tách caption, OCR và QR decoding | Không triển khai | `description_vi` từ LLM đã đủ cho semantic retrieval hiện tại |
| P2 | Bổ sung metadata tài liệu và section | Chưa làm | Thiếu thông tin nguồn; metadata kỹ thuật có thể không đáng tin |
| P2 | Bổ sung layout bảng khi có nhu cầu | Chưa làm | Nội dung bảng đúng nhưng chưa đủ để tái dựng hình thức Word |

## 1. Trích xuất file đính kèm OLE — P0

**Trạng thái:** Đã hoàn thành phần trích xuất và bảo toàn payload. Parser đọc `o:OLEObject` và relationship gốc, lấy file từ stream `Ole10Native`, lưu dưới `assets/` theo SHA-256, ghi provenance vào JSONL và không gửi icon attachment cho LLM.

### Hiện trạng trước khi sửa

Parser chỉ thấy ảnh biểu tượng của file đính kèm và tạo caption cho biểu tượng. Payload thật bên trong DOCX không xuất hiện trong JSONL.

Bằng chứng:

- `118_2021_ND-CP_477969.docx`: `Bieu mau.zip` chứa `Bieu mau.doc`.
- `12_2025_TT-BCA_645271.docx`: `Phu luc.zip` chứa `Phu luc.doc`.
- `13_2025_TT-BCA_m_645302.docx`: `Bieu mau.zip` chứa 28 file DOC, DOCX và XLS.
- `158...` và `161...`: có `Phu luc.zip` chưa được trích xuất.
- `51_2024...`: có `Quy chuan.zip` khoảng 26,4 MB chưa được trích xuất.

### Cần sửa

- Đọc relationships và embedded package bằng OOXML thay vì suy luận từ icon hoặc tên hiển thị.
- Sinh record `kind: "attachment"` gồm tối thiểu: tên file, MIME type, SHA-256, `source_ref`, quan hệ với phần tử cha và đường dẫn asset.
- Lưu payload vào thư mục sidecar theo content hash để JSONL không phình lớn.
- Chỉ parse đệ quy DOCX hoặc định dạng đã được parser hỗ trợ; không tự động thực thi hay giải nén không giới hạn.

### Tiêu chí hoàn thành

- [x] Sáu nhóm attachment trên đều xuất hiện trong output và hash khớp payload gốc.
- [x] Attachment giữ đúng vị trí/quan hệ với icon hoặc đoạn văn chứa nó.
- [x] Archive chỉ được đọc trong bộ nhớ với giới hạn member/byte và không ghi đường dẫn nguồn ra filesystem.
- [x] Test bao phủ OLE Package, payload ZIP, relationship ngoài, stream lỗi và icon không được caption.

### Ingestion tài liệu con

**Trạng thái:** Đã hoàn thành. Parser nhận diện container từ cấu trúc ZIP/OLE, đọc ZIP trong bộ nhớ với giới hạn tài nguyên, chuyển DOC→DOCX và XLS→XLSX bằng LibreOffice theo batch, rồi đưa record con vào ngay sau record cha chứa attachment.

- DOCX dùng lại toàn bộ parser hiện có, gồm paragraph, table, image caption và attachment metadata.
- XLSX được đọc trực tiếp từ OOXML, giữ sheet, row, cell reference, giá trị và công thức; không biến bảng tính thành text phẳng.
- Mỗi record con giữ `parent_attachment_id`, `attachment_member_index`, `attachment_member_path`, hash member, định dạng nguồn và thông tin chuyển đổi.
- ZIP member không được giải nén theo đường dẫn nguồn; parser chặn path traversal, member mã hóa, quá số lượng và quá giới hạn byte.
- Parser chỉ đi sâu một cấp attachment. Attachment lồng trong tài liệu con vẫn được bảo toàn nhưng không tiếp tục parse đệ quy.

### Tiêu chí hoàn thành tài liệu con

- [x] Sáu DOCX có attachment đều parse thành công trong smoke test thực tế.
- [x] `13_2025_TT-BCA_m_645302.docx` tạo đủ 28 tài liệu con: 9 DOCX, 18 DOC chuyển sang DOCX và 1 XLS chuyển sang XLSX.
- [x] `51_2024_TT-BGTVT_m_633562.docx` tạo 3.925 record từ `Quy chuan.doc`.
- [x] Batch conversion giảm case 28 file từ khoảng 235 giây xuống khoảng 19 giây trên máy kiểm thử.
- [x] Test bao phủ provenance, XLSX row/cell/formula, ZIP traversal, giới hạn member và nhận diện OLE trước ZIP signature nhúng.

## 2. Ghi JSONL an toàn — P0

**Trạng thái:** Đã hoàn thành. Writer escape `U+0085`, `U+2028` và `U+2029`; regression test xác nhận mỗi record vẫn nằm trên một physical line và nội dung được khôi phục đúng khi đọc JSON.

### Hiện trạng

`67_2023_ND-CP_m_578283.parsed.jsonl` chứa ký tự điều khiển thô `U+0085` tại các body ref 793–795. JSON vẫn hợp lệ nếu đọc từng physical line, nhưng consumer dùng `splitlines()` có thể tách sai một record thành nhiều dòng.

### Cần sửa

- Khi serialize JSONL, escape các ký tự điều khiển có thể được Unicode xem là line separator, trong khi vẫn giữ tiếng Việt dễ đọc.
- Không thay đổi schema hoặc nội dung logic của record.

### Tiêu chí hoàn thành

- Mỗi record luôn chiếm đúng một physical line.
- Parse lại toàn bộ JSONL cho kết quả đúng 22.307 record.
- Regression test có chuỗi chứa `U+0085`.

## 3. Bảo toàn field, hyperlink và bookmark — P1

**Trạng thái:** Đã hoàn thành. Parser output schema `2.0.0` giữ field instruction, field result, hyperlink relationship, bookmark ID/name, `source_ref` và character offset ngay trên paragraph tương ứng, kể cả paragraph trong bảng và bookmark không có text.

### Hiện trạng

Text hiển thị vẫn còn nhưng quan hệ liên kết bị mất.

Bằng chứng:

- `90_VBHN-VPQH_699548.docx`: 556 field và 878 bookmark.
- Field nguồn `HYPERLINK \\l "_ftn1"` được output thành text `[1]`, nhưng không còn biết `[1]` trỏ tới bookmark nào.
- `17_2026_TT-BXD_704696.docx`: có hai URL dạng field; URL hiển thị còn nhưng ngữ nghĩa hyperlink không còn.

### Cần sửa

- Bảo toàn field instruction, result text, target và bookmark ID/name dưới dạng dữ liệu có cấu trúc.
- Giữ quan hệ bằng `source_ref`/ID gốc; không dùng regex hoặc keyword để đoán trích dẫn.
- Không thay text hiện có; thêm metadata quan hệ vào record tương ứng.

### Tiêu chí hoàn thành

- [x] Từ record `[1]` có thể truy ngược đúng bookmark `_ftn1` và nội dung đích.
- [x] URL field giữ được target độc lập với text hiển thị.
- [x] Test bao phủ internal hyperlink, external hyperlink và bookmark không có text.
- [x] `90_VBHN-VPQH_699548.docx` giữ đủ 556 field và 878 bookmark; `_ftn1` trỏ đúng paragraph chú thích.
- [x] `17_2026_TT-BXD_704696.docx` giữ đủ hai field URL và target gốc dù Word chia visible URL thành hai field.

## 4. Bảo toàn hierarchy của heading — P1

**Trạng thái:** Không triển khai. Phần lớn paragraph không có Heading Style; các style còn lại bị gán cho cả paragraph rỗng, tên biểu mẫu và nội dung như `Kính gửi`, nên không được coi là hierarchy pháp lý đáng tin cậy.

### Hiện trạng

Nội dung heading được giữ như paragraph thường, nhưng cấp heading gốc bị mất.

Bằng chứng:

- `14_2025_TT-BXD_m_664151.docx`: 60 heading có cấu trúc.
- `53_2024_TT-BGTVT_m_624420.docx`: 15 heading có cấu trúc.
- Ví dụ `word/document.xml#/body/409` có style `heading 3`, outline level `3`, text `II. Môn học kỹ thuật lái xe`.

### Hướng đã cân nhắc

- Bổ sung `style_id` và `outline_level` lấy trực tiếp từ OOXML/style definitions.
- Dùng hierarchy gốc để hỗ trợ chunking hoặc dựng cây tài liệu ở bước sau.
- Không suy luận heading từ chữ in hoa, số thứ tự, font hoặc regex.

### Lý do không triển khai

- Metadata style chỉ phản ánh cách file Word được soạn, không xác nhận cấu trúc pháp lý.
- Gắn hierarchy theo style gốc có thể nâng paragraph thường hoặc paragraph rỗng thành heading sai.
- Text hiện tại đã được bảo toàn và downstream chưa cần metadata layout này.

## 5. Tách caption, OCR và QR decoding — P1

**Trạng thái:** Không triển khai. Image caption chỉ giữ `description_vi` do LLM sinh; không chép OCR riêng, không decode QR/barcode và không coi caption là bản chép ký tự chính xác.

**Schema migration:** Parser output và image-caption schema chuyển sang `2.0.0`. Field `visible_text` bị xóa; consumer chỉ đọc `description_vi` từ mỗi image caption.

### Hiện trạng

Caption LLM mô tả đúng ý ở đa số 171 ảnh và chưa thấy hallucination rõ ràng, nhưng không đảm bảo chép chính xác ký tự.

Bằng chứng:

- `53...`, ảnh 15: công thức nguồn `m_ng / m_h × 100% ≤ 80%`; caption đọc nhầm `g` thành `9`.
- `47...`, ảnh 4: ký hiệu `№/No` bị đọc thành `Nợ`.
- `184...`: nhận ra có QR nhưng không giải mã payload.

### Hướng đã cân nhắc

- Giữ LLM cho `semantic_caption`.
- Dùng OCR chuyên dụng cho `ocr_text` và công cụ QR/barcode cho `machine_readable_codes`.
- Lưu ảnh gốc vào sidecar theo content hash để có thể audit và xử lý lại mà không cần mở DOCX.
- Không trộn text OCR vào caption như thể đó là nội dung đã được xác minh.

### Phạm vi được chấp nhận

- `description_vi` phục vụ semantic retrieval và có thể chứa mô tả chữ quan trọng khi LLM nhìn thấy.
- Không cam kết sao chép chính xác công thức, ký hiệu, QR hoặc barcode.
- Không thêm dependency hay service OCR/QR chuyên dụng khi downstream chưa cần.

## 6. Metadata tài liệu và section — P2

### Hiện trạng

Corpus có 66 core metadata field trên 13 DOCX và 47 section property block chưa được output. Metadata Word có thể cũ hoặc sai, nên không được xem là metadata pháp lý đã xác thực.

### Cần sửa

- Nếu downstream cần, lưu riêng dưới namespace như `source_metadata` và đánh dấu là dữ liệu không tin cậy từ file nguồn.
- Lưu section properties chỉ khi cần tái dựng trang, header/footer hoặc phục vụ audit.
- Metadata pháp lý như số văn bản, cơ quan ban hành và ngày ban hành phải là một bước trích xuất/chuẩn hóa riêng, không lấy mặc định từ core properties.

### Tiêu chí hoàn thành

- Không ghi đè metadata pháp lý bằng metadata kỹ thuật của Word.
- Mỗi giá trị có provenance tới OOXML part tương ứng.

## 7. Layout bảng — P2

### Hiện trạng

Rows, cells, nested tables, `gridSpan` và `vMerge` đang khớp nguồn. Các thuộc tính trình bày như width, border, shading, margin, vertical alignment, header-row marker và row height chưa được lưu.

### Cần sửa

- Chỉ bổ sung thuộc tính layout mà downstream thật sự sử dụng.
- Lấy trực tiếp từ OOXML, không quy đổi bằng heuristic.
- Không làm phức tạp schema RAG nếu mục tiêu chỉ là hiểu nội dung bảng.

### Tiêu chí hoàn thành

- Có test fixture cho từng thuộc tính được quyết định hỗ trợ.
- Nội dung và thứ tự bảng hiện tại không thay đổi.

## Những phần không cần sửa lúc này

- Text, thứ tự paragraph/table, run text, nested table, merge cell, image hash, MIME và source reference đang khớp trong 35 tài liệu đã kiểm tra.
- `description_vi` hiện tại dùng được cho semantic retrieval; không coi đó là OCR chính xác.
- Không cần bảo toàn toàn bộ font, màu, spacing và layout Word nếu downstream chỉ là RAG.
- Header trong `118_2021_ND-CP_477969.docx` không phải lỗi parser: đó là even-page header nhưng tài liệu tắt `evenAndOddHeaders`, nên Word không hiển thị.

## Trình tự triển khai đề xuất

1. [x] Sửa writer để mỗi record JSONL luôn nằm trên một physical line.
2. [x] Trích xuất attachment và lưu asset sidecar.
3. [x] Parse tài liệu con trong attachment cho RAG.
4. [x] Bảo toàn field/hyperlink/bookmark.
5. Không triển khai `style_id` và `outline_level` vì metadata nguồn không đủ tin cậy.
6. Không triển khai OCR/QR; caption chỉ giữ `description_vi`.
7. [ ] Chỉ thêm metadata, section và layout bảng khi downstream xác nhận cần.

## Ngoài phạm vi

- Không refactor toàn bộ parser.
- Không thay schema hiện tại nếu chỉ cần thêm field tương thích ngược.
- Không dùng regex, keyword matching hoặc handcrafted score để suy luận cấu trúc/ngữ nghĩa.
- Không tự động thực thi attachment hoặc xử lý archive không giới hạn.
- Không coi metadata Word hay output LLM là dữ liệu đã được xác thực.
