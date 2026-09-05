class DocumentTemporalOverride {
  affected_provision_paths string[] @description(#"
    Các điều/khoản/điểm chịu hiệu lực riêng, giữ locator tiếng Việt theo nguồn.
  "#)
  valid_from string? @description(#"Ngày bắt đầu riêng dạng YYYY-MM-DD nếu source nêu rõ."#)
  valid_to string? @description(#"Ngày kết thúc riêng dạng YYYY-MM-DD nếu source nêu rõ."#)
  unresolved_basis_vi string? @description(#"
    Căn cứ hiệu lực chưa quy đổi được thành ngày; null khi đã có interval rõ.
  "#)
}

class DocumentMetadata {
  document_number string @description(#"
    Số hiệu của chính văn bản đang đọc.
  "#)
  valid_from string? @description(#"
    Ngày toàn văn bản bắt đầu có hiệu lực, dạng YYYY-MM-DD.
  "#)
  valid_to string? @description(#"
    Ngày toàn văn bản hết hiệu lực, dạng YYYY-MM-DD.
  "#)
  temporal_overrides DocumentTemporalOverride[] @description(#"
    Ngoại lệ hiệu lực áp dụng riêng cho các điều/khoản/điểm của văn bản.
  "#)
}

function ResolveDocumentMetadata(metadata_sections: string[]) -> DocumentMetadata {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Bối cảnh: chuẩn bị nguồn sự thật cho benchmark RAG hỏi đáp về
    xử lý vi phạm giao thông đường bộ tại Việt Nam.

    Nhiệm vụ: lấy metadata và ngoại lệ hiệu lực của chính văn bản. Input chỉ
    gồm các đoạn đã được Step 1A chọn, theo thứ tự tài liệu. Chỉ dùng source
    text được cung cấp.

    - document_number: số hiệu của chính văn bản đang đọc, giữ
      nguyên theo nguồn; không lấy số của
      văn bản được viện dẫn, sửa đổi, thay thế hoặc bãi bỏ.
    - valid_from: ngày toàn văn bản bắt đầu có hiệu lực; không
      nhầm với ngày ký, ban hành, đăng công báo hoặc ngày áp dụng
      riêng của một điều khoản.
    - valid_to: ngày toàn văn bản hết hiệu lực, chỉ khi source
      text nêu rõ; không suy ra từ việc một điều khoản hay văn bản
      khác bị bãi bỏ.
    - temporal_overrides: mọi điều khoản quy định hiệu lực riêng cho một hoặc
      nhiều điều/khoản/điểm. Gom các locator có cùng interval hoặc cùng căn cứ
      chưa xác định vào một entry; không đưa hiệu lực chung vào đây.
      + Nếu source nêu ngày rõ, chuẩn hóa vào valid_from/valid_to và để
        unresolved_basis_vi=null.
      + Nếu hiệu lực phụ thuộc pháp luật/văn bản/sự kiện khác nhưng source chưa
        cho ngày cụ thể, để hai ngày null và ghi ngắn gọn căn cứ đó vào
        unresolved_basis_vi.
      + affected_provision_paths giữ đủ Điều > Khoản > Điểm theo source để bước
        ontology có thể đối chiếu theo nghĩa pháp lý; không tự mở rộng sang quy
        định không được nêu.

    Với văn bản hợp nhất (VBHN), metadata phải thuộc chính văn bản hợp nhất.
    Không lấy ngày hiệu lực của luật, nghị định hoặc văn bản gốc đang được hợp
    nhất làm valid_from/valid_to của VBHN. Nếu source không nêu ngày hiệu lực
    độc lập của chính VBHN thì trả null cho trường tương ứng.

    Chuẩn hóa ngày thành YYYY-MM-DD; trả null nếu không nêu rõ. Không dùng kiến
    thức bên ngoài, suy đoán hoặc để dẫn chiếu ở phần cuối ghi đè danh tính văn
    bản xác định từ phần đầu. Nếu không có ngoại lệ hiệu lực, trả danh sách rỗng.

    {{ _.role("user") }}

    Các đoạn metadata:
    {% for section in metadata_sections %}
    ---
    {{ section }}
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
