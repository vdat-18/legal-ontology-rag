enum StructuralContextRelation {
  SameParagraph
  SameTableRow
  Before
  After
}

class StructuralContextEntry {
  source_ref string
  relation StructuralContextRelation
  distance int
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
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Quan sát trực tiếp hình ảnh. Không suy diễn quy định pháp luật, danh tính, địa điểm,
    thời gian hoặc thông tin không thể quan sát trực tiếp.

    Ngữ cảnh cấu trúc của vị trí ảnh trong tài liệu. SameParagraph và SameTableRow là
    ngữ cảnh trực tiếp. Before và After là các đoạn xung quanh; distance càng nhỏ thì
    vị trí càng gần ảnh:
    {% for entry in structural_context %}
    - [{{ entry.relation }}, distance={{ entry.distance }}] {{ entry.source_ref }}: {{ entry.text }}
    {% endfor %}

    Ưu tiên ngữ cảnh trực tiếp. Các đoạn Before và After có thể mô tả nội dung lân cận
    hoặc hình khác, vì vậy chỉ dùng khi phù hợp với những gì thực sự nhìn thấy. Không
    trình bày nội dung ngữ cảnh như thể nó xuất hiện trong ảnh.

    Hình ảnh:
    {{ img }}

    {{ ctx.output_format }}
  "#
}
