# Parser Audit: Các hạng mục cần sửa

## Kết luận ngắn

Parser hiện tại **đủ tốt cho RAG dựa trên văn bản, bảng và caption ảnh**. Đối chiếu 35 DOCX với 22.307 record và 171 ảnh không phát hiện nội dung nào bị sai trong phạm vi schema đang hỗ trợ.

Điểm còn thiếu chủ yếu là dữ liệu DOCX chưa được đưa vào output: file đính kèm, quan hệ trích dẫn/liên kết, cấp heading, OCR/QR chính xác và một số metadata. Vì vậy, kết quả “khớp” không có nghĩa là DOCX đã được bảo toàn toàn bộ.

## Thứ tự ưu tiên

| Ưu tiên | Hạng mục | Ảnh hưởng |
| --- | --- | --- |
| P0 | Trích xuất file đính kèm OLE | Đang mất hoàn toàn nội dung phụ lục/quy chuẩn nhúng trong DOCX |
| P0 | Chuẩn hóa ký tự điều khiển khi ghi JSONL | Một số consumer có thể đọc sai số dòng/record |
| P1 | Bảo toàn field, hyperlink và bookmark | Mất quan hệ giữa ký hiệu trích dẫn và đích được tham chiếu |
| P1 | Bảo toàn cấp heading gốc | Mất cấu trúc chương/mục dù text vẫn còn |
| P1 | Tách caption, OCR và QR decoding | Caption đúng ý nhưng không đủ chính xác cho công thức, biểu mẫu và QR |
| P2 | Bổ sung metadata tài liệu và section | Thiếu thông tin nguồn; metadata kỹ thuật có thể không đáng tin |
| P2 | Bổ sung layout bảng khi có nhu cầu | Nội dung bảng đúng nhưng chưa đủ để tái dựng hình thức Word |

## 1. Trích xuất file đính kèm OLE — P0

### Hiện trạng

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

- Sáu nhóm attachment trên đều xuất hiện trong output và hash khớp payload gốc.
- Attachment giữ đúng vị trí/quan hệ với icon hoặc đoạn văn chứa nó.
- Có giới hạn kích thước, số file và độ sâu khi giải nén archive.
- Test bao phủ embedded ZIP, DOCX và file không hỗ trợ.

## 2. Ghi JSONL an toàn — P0

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

- Từ record `[1]` có thể truy ngược đúng bookmark `_ftn1` và nội dung đích.
- URL field giữ được target độc lập với text hiển thị.
- Test bao phủ internal hyperlink, external hyperlink và bookmark không có text.

## 4. Bảo toàn hierarchy của heading — P1

### Hiện trạng

Nội dung heading được giữ như paragraph thường, nhưng cấp heading gốc bị mất.

Bằng chứng:

- `14_2025_TT-BXD_m_664151.docx`: 60 heading có cấu trúc.
- `53_2024_TT-BGTVT_m_624420.docx`: 15 heading có cấu trúc.
- Ví dụ `word/document.xml#/body/409` có style `heading 3`, outline level `3`, text `II. Môn học kỹ thuật lái xe`.

### Cần sửa

- Bổ sung `style_id` và `outline_level` lấy trực tiếp từ OOXML/style definitions.
- Dùng hierarchy gốc để hỗ trợ chunking hoặc dựng cây tài liệu ở bước sau.
- Không suy luận heading từ chữ in hoa, số thứ tự, font hoặc regex.

### Tiêu chí hoàn thành

- 75 heading đã phát hiện trong corpus giữ đúng style và outline level.
- Paragraph thường không bị tự động nâng thành heading.
- Test bao phủ heading trực tiếp và heading kế thừa từ style.

## 5. Tách caption, OCR và QR decoding — P1

### Hiện trạng

Caption LLM mô tả đúng ý ở đa số 171 ảnh và chưa thấy hallucination rõ ràng, nhưng không đảm bảo chép chính xác ký tự.

Bằng chứng:

- `53...`, ảnh 15: công thức nguồn `m_ng / m_h × 100% ≤ 80%`; caption đọc nhầm `g` thành `9`.
- `47...`, ảnh 4: ký hiệu `№/No` bị đọc thành `Nợ`.
- `184...`: nhận ra có QR nhưng không giải mã payload.

### Cần sửa

- Giữ LLM cho `semantic_caption`.
- Dùng OCR chuyên dụng cho `ocr_text` và công cụ QR/barcode cho `machine_readable_codes`.
- Lưu ảnh gốc vào sidecar theo content hash để có thể audit và xử lý lại mà không cần mở DOCX.
- Không trộn text OCR vào caption như thể đó là nội dung đã được xác minh.

### Tiêu chí hoàn thành

- Output phân biệt rõ caption suy luận, OCR và payload QR.
- Công thức/ký hiệu trong bộ test được giữ chính xác hơn caption hiện tại.
- Hash và asset path cho phép truy lại đúng ảnh gốc.

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
- Caption ảnh hiện tại dùng được cho semantic retrieval; chỉ không nên coi là OCR chính xác tuyệt đối.
- Không cần bảo toàn toàn bộ font, màu, spacing và layout Word nếu downstream chỉ là RAG.
- Header trong `118_2021_ND-CP_477969.docx` không phải lỗi parser: đó là even-page header nhưng tài liệu tắt `evenAndOddHeaders`, nên Word không hiển thị.

## Trình tự triển khai đề xuất

1. Sửa writer để mỗi record JSONL luôn nằm trên một physical line.
2. Trích xuất attachment và lưu asset sidecar.
3. Bảo toàn field/hyperlink/bookmark.
4. Bổ sung `style_id` và `outline_level`.
5. Thêm OCR/QR như dữ liệu riêng với caption.
6. Chỉ thêm metadata, section và layout bảng khi downstream xác nhận cần.

## Ngoài phạm vi

- Không refactor toàn bộ parser.
- Không thay schema hiện tại nếu chỉ cần thêm field tương thích ngược.
- Không dùng regex, keyword matching hoặc handcrafted score để suy luận cấu trúc/ngữ nghĩa.
- Không tự động thực thi attachment hoặc xử lý archive không giới hạn.
- Không coi metadata Word hay output LLM là dữ liệu đã được xác thực.
