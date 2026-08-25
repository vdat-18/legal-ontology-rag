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
}

function ResolveDocumentMetadata(metadata_sections: string[]) -> DocumentMetadata {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Bối cảnh: chuẩn bị nguồn sự thật cho benchmark RAG hỏi đáp về
    xử lý vi phạm giao thông đường bộ tại Việt Nam.

    Nhiệm vụ: lấy metadata cấp văn bản để gắn thống nhất vào mọi
    legal fact. Input chỉ gồm các đoạn đã được Step 1A xác định có metadata
    cấp văn bản, theo thứ tự tài liệu. Chỉ dùng source text được cung cấp.

    - document_number: số hiệu của chính văn bản đang đọc, giữ
      nguyên theo nguồn; không lấy số của
      văn bản được viện dẫn, sửa đổi, thay thế hoặc bãi bỏ.
    - valid_from: ngày toàn văn bản bắt đầu có hiệu lực; không
      nhầm với ngày ký, ban hành, đăng công báo hoặc ngày áp dụng
      riêng của một điều khoản.
    - valid_to: ngày toàn văn bản hết hiệu lực, chỉ khi source
      text nêu rõ; không suy ra từ việc một điều khoản hay văn bản
      khác bị bãi bỏ.

    Với văn bản hợp nhất (VBHN), metadata phải thuộc chính văn bản hợp nhất.
    Không lấy ngày hiệu lực của luật, nghị định hoặc văn bản gốc đang được hợp
    nhất làm valid_from/valid_to của VBHN. Nếu source không nêu ngày hiệu lực
    độc lập của chính VBHN thì trả null cho trường tương ứng.

    Chuẩn hóa ngày thành YYYY-MM-DD; trả null nếu không nêu rõ. Không
    dùng kiến thức bên ngoài, suy đoán hoặc để dẫn chiếu ở phần cuối
    ghi đè danh tính văn bản xác định từ phần đầu.

    {{ _.role("user") }}

    Các đoạn metadata:
    {% for section in metadata_sections %}
    ---
    {{ section }}
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
