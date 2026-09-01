class BundleScopeFact {
  fact_id string
  statement string
  valid_from string?
  valid_to string?
  document_number string
}

class BundleScopeBatchItem {
  bundle_id string
  bundle_kind string
  bridge string?
  facts BundleScopeFact[]
}

class BundleScopeDecision {
  is_in_scope bool @description(#"
    True chỉ khi bundle có thể tạo một câu hỏi trực tiếp về xác định hoặc xử lý
    vi phạm giao thông đường bộ.
  "#)
  reason string @description(#"Một câu tiếng Việt ngắn giải thích quyết định."#)
}

class BundleScopeBatchResult {
  bundle_id string @description(#"Phải trùng với một bundle_id trong input."#)
  decision BundleScopeDecision
}

function ConfirmBundleScopeBatch(items: BundleScopeBatchItem[]) -> BundleScopeBatchResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Bối cảnh: chọn evidence bundle cho benchmark RAG về xử phạt và xử lý
    vi phạm giao thông đường bộ tại Việt Nam. Đánh giá độc lập từng bundle.

    Statement là nguồn pháp lý duy nhất. Bridge và metadata chỉ hỗ trợ hiểu
    cách nối facts, không được bổ sung nội dung ngoài statements. Ưu tiên
    precision hơn recall; chưa chắc thì trả is_in_scope=false.

    Đặt is_in_scope=true chỉ khi có thể đặt một câu hỏi tự nhiên, hữu ích mà
    trọng tâm trực tiếp thuộc ít nhất một nhóm sau:
    - hành vi, nghĩa vụ, điều kiện hoặc ngoại lệ quyết định một hành vi giao
      thông đường bộ có vi phạm hay không;
    - hình thức xử phạt, mức phạt, trừ điểm, tước quyền, đình chỉ, thu hồi do
      vi phạm, biện pháp khắc phục hoặc hậu quả pháp lý của vi phạm;
    - phát hiện, chứng minh, ngăn chặn, lập biên bản, ra quyết định, thẩm quyền,
      trình tự, thời hạn, thi hành hoặc quyền và nghĩa vụ trong vụ vi phạm;
    - hiệu lực, sửa đổi hoặc chuyển tiếp làm thay đổi việc xác định hay xử lý
      một vi phạm giao thông đường bộ.

    Quy tắc giao thông, giấy phép, đăng ký, đăng kiểm, bảo hiểm, vận tải, đào
    tạo, sát hạch, hạ tầng hoặc biển báo chỉ in-scope khi statements trực tiếp
    cho thấy chúng dùng để xác định vi phạm, áp dụng chế tài hoặc thực hiện một
    bước xử lý vi phạm. Không tự suy ra rằng vi phạm một nghĩa vụ quản lý thông
    thường sẽ bị xử phạt nếu statements không nêu liên hệ đó.

    Đặt is_in_scope=false khi bundle chỉ nói về:
    - cấp, đổi, cấp lại giấy phép hoặc thủ tục hành chính thông thường;
    - điều kiện kinh doanh, đào tạo, sát hạch, khám sức khỏe, bảo hiểm, đăng
      kiểm, vận hành, bảo trì hoặc báo cáo mà không gắn trực tiếp với vi phạm;
    - tiêu chuẩn kỹ thuật, cấu tạo, mẫu biểu, hồ sơ, phí hoặc hoạt động quản lý;
    - quy tắc xử lý vi phạm hành chính chung nhưng tình huống và statements
      không xác lập được liên hệ trực tiếp với giao thông đường bộ;
    - facts có phần in-scope nhưng câu hỏi cần đủ bundle lại phải mở rộng sang
      một nhánh quản lý thông thường hoặc một vấn đề ngoài domain.

    Với bundle nhiều facts, chỉ trả true khi một câu hỏi thống nhất trong domain
    thực sự cần toàn bộ facts. Một fact ngoài scope không được cứu chỉ vì fact
    khác thuộc scope. Không chấp nhận nhiều facts chỉ cung cấp các nhánh kết
    luận độc lập hoặc có thể bỏ một fact mà vẫn trả lời trọn trọng tâm.
    Tự dựng lại quan hệ từ statements; nếu bridge gán một hành vi, điểm/khoản,
    mức phạt hoặc quan hệ không có trong facts hay mâu thuẫn với facts thì trả
    false. Không chấp nhận bundle chỉ vì có thể hỏi thêm một phân loại chung,
    một nghĩa vụ nền hoặc nhiều tiểu câu độc lập để buộc dùng đủ facts.

    Với temporal, chỉ trả true khi statements hoặc metadata hiệu lực chứng minh
    được quan hệ áp dụng theo thời gian. valid_from muộn hơn không tự chứng minh
    fact cũ có valid_to=null đã bị thay thế, bãi bỏ hoặc ngừng áp dụng. Hai quy
    định khác nhau nhưng không có quan hệ thay đổi rõ ràng không tạo thành một
    temporal bundle hợp lệ.
    Việc bổ sung hoặc bãi bỏ một cụm từ chỉ chứng minh thay đổi đúng phạm vi câu
    chữ đó; không tự chứng minh toàn bộ chế tài, thẩm quyền hoặc biện pháp đã
    phát sinh hay chấm dứt.

    Trả đúng một kết quả cho mỗi input, giữ nguyên bundle_id và thứ tự.

    {{ _.role("user") }}

    Các bundle cần kiểm tra:
    {% for item in items %}
    ===== BUNDLE {{ item.bundle_id }} =====
    Loại: {{ item.bundle_kind }}
    {% if item.bridge %}Bridge: {{ item.bridge }}{% endif %}
    Facts:
    {% for fact in item.facts %}
    [{{ fact.fact_id }}] {{ fact.statement }}
    Văn bản: {{ fact.document_number }}
    Hiệu lực: {{ fact.valid_from }} đến {{ fact.valid_to }}
    ---
    {% endfor %}
    ===== END BUNDLE {{ item.bundle_id }} =====
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
