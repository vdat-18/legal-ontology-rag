class StructuralContextEntry {
  source_ref string
  text string
}

class ImageCaptionResult {
  description_vi string @description(#"
    Mô tả ngắn gọn, trung lập bằng tiếng Việt về những gì thực sự nhìn thấy trong ảnh.
    Có thể dùng ngữ cảnh cấu trúc để định danh hoặc làm rõ đối tượng, nhưng không trình bày
    nội dung ngữ cảnh như thể nó xuất hiện trực tiếp trong ảnh.
  "#)
  visible_text string? @description(#"
    Chép lại nguyên văn chữ nhìn thấy được trong ảnh. Trả về null nếu không có chữ hoặc
    không đọc được.
  "#)
}

function CaptionImage(structural_context: StructuralContextEntry[], img: image) -> ImageCaptionResult {
  client Router
  prompt #"
    {{ _.role("user") }}

    Quan sát trực tiếp hình ảnh. Không suy diễn quy định pháp luật, danh tính, địa điểm,
    thời gian hoặc thông tin không thể quan sát trực tiếp.

    Ngữ cảnh cấu trúc của vị trí ảnh trong tài liệu:
    {% for entry in structural_context %}
    - {{ entry.source_ref }}: {{ entry.text }}
    {% endfor %}

    Chỉ dùng ngữ cảnh trên để định danh hoặc làm rõ đối tượng nhìn thấy. Không trình bày
    nội dung ngữ cảnh như thể nó xuất hiện trong ảnh.

    Hình ảnh:
    {{ img }}

    {{ ctx.output_format }}
  "#
}
