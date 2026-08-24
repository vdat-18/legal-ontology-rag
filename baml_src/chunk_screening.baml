class ChunkScreeningItem {
  chunk_id string
  context_text string @description(#"
    Ngữ cảnh cấu trúc do parser tạo; không phải source text.
  "#)
  text string
}

class ChunkScreeningResult {
  chunk_id string @description(#"Phải trùng với một chunk_id trong danh sách đầu vào."#)
  has_penalty_content bool @description(#"
    True nếu target có fact trực tiếp giúp xác định hoặc xử lý vi phạm trong domain.
  "#)
  has_document_metadata bool @description(#"
    True nếu target nêu rõ số hiệu, ngày có hiệu lực hoặc ngày hết hiệu lực của chính văn bản đang đọc.
  "#)
  reason string @description(#"Một câu tiếng Việt ngắn giải thích quyết định."#)
}

function ScreenChunks(chunks: ChunkScreeningItem[]) -> ChunkScreeningResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Bối cảnh: chuẩn bị nguồn sự thật cho benchmark RAG hỏi đáp về
    xử lý vi phạm giao thông đường bộ tại Việt Nam.

    Nhiệm vụ: sàng lọc độc lập từng target chunk trước khi trích legal fact.
    Ưu tiên độ chính xác: chỉ giữ quy tắc pháp lý có liên hệ trực tiếp, rõ
    ràng với việc xác định hoặc xử lý vi phạm giao thông đường bộ.

    Mỗi block là một target độc lập. Khi đánh giá một block, bỏ qua hoàn toàn
    text và fact của mọi block khác trong cùng request. Không dùng thứ tự các
    block để nối, bổ sung hoặc hoàn thiện nội dung target. Fact chỉ xuất hiện
    ở block khác không được dùng làm căn cứ cho kết quả hiện tại.

    Đặt has_penalty_content=true chỉ khi Target text tự nêu ít nhất một nội
    dung có thể dùng trực tiếp trong một vụ việc vi phạm đường bộ:
    - hành vi bị cấm hoặc quy tắc, nghĩa vụ, điều kiện, ngoại lệ trực tiếp
      quyết định hành vi có vi phạm hay không;
    - chủ thể, phương tiện hoặc phạm vi áp dụng của quy tắc đó;
    - chế tài, hậu quả pháp lý hoặc biện pháp khắc phục hậu quả;
    - phát hiện, chứng minh, ngăn chặn, thẩm quyền, trình tự, thời hạn, quyền
      hoặc nghĩa vụ trong quá trình xử lý;
    - hiệu lực hoặc chuyển tiếp trực tiếp thay đổi cách áp dụng các nội dung
      trên.

    Quy định có thể áp dụng cho người tham gia giao thông, chủ xe, đơn vị kinh
    doanh vận tải, cơ sở đào tạo/sát hạch hoặc đơn vị quản lý hạ tầng, nhưng
    chỉ giữ khi Target trực tiếp xác định vi phạm hoặc cách xử lý vi phạm của
    chủ thể đó. Không suy ra rằng mọi yêu cầu bắt buộc đều là fact vi phạm.

    Đặt false nếu Target chỉ mô tả hoạt động quản lý, điều kiện vận hành hoặc
    kiến thức chuyên ngành mà chưa tạo liên hệ pháp lý trực tiếp với một vi
    phạm đường bộ. Đặc biệt không coi các nội dung sau tự chúng là vi phạm:
    - chương trình/thời lượng đào tạo, lỗi và điểm thi sát hạch, điều kiện
      đạt/trượt hoặc tình huống mô phỏng dùng để giảng dạy;
    - kích thước, vật liệu, màu sắc, kết cấu, chế tạo hoặc lắp đặt kỹ thuật;
      vẫn giữ ý nghĩa báo hiệu và nghĩa vụ người tham gia giao thông phải chấp
      hành;
    - phí bảo hiểm, hợp đồng, hồ sơ bồi thường hoặc bảng tỷ lệ thương tật;
      vẫn giữ nghĩa vụ bảo hiểm bắt buộc và hậu quả pháp lý trực tiếp;
    - nội dung riêng của đường sắt, đường thủy, hàng hải, hàng không, ma túy,
      cai nghiện hoặc lĩnh vực khác ngoài giao thông đường bộ;
    - chính sách, tuyên truyền, tổ chức quản lý, dữ liệu/thống kê, tài chính,
      hỗ trợ hoặc thủ tục hành chính độc lập.

    Quy tắc xử lý vi phạm hành chính chung được giữ khi nội dung của nó có thể
    áp dụng trực tiếp cho vụ việc giao thông đường bộ. Loại phần chỉ điều chỉnh
    một lĩnh vực khác hoặc chỉ liệt kê cơ quan, đối tượng, ví dụ ngoài domain.
    Định nghĩa/phân loại chỉ được giữ khi cần áp dụng trực tiếp một quy tắc vi
    phạm. Khi chưa thấy liên hệ trực tiếp này trong Target, đặt false.

    Mục lục, tiêu đề, lời dẫn, căn cứ ban
    hành, danh mục hoặc tên biểu mẫu, trường điền thông tin, dòng giao/nhận,
    ô ký tên và thông tin hành chính tự chúng không đủ. Luôn đặt false cho
    phần chỉ liệt kê trường trống cần điền hoặc chỉ xác nhận đã giao/nhận văn
    bản và thời điểm giao/nhận, kể cả khi các trường đó nhắc tới vi phạm,
    tình tiết, tang vật hay người xử lý. Nhãn trường vẫn là nội dung cần loại,
    kể cả khi dùng từ mang nghĩa bắt buộc như phải thực hiện, phải xác minh
    hoặc phải giám định. Khung quyết định hoặc biên bản chỉ nêu căn cứ, tên
    thủ tục và các trường cần điền vẫn phải đặt false, dù có dẫn điều luật
    hoặc nhắc tới xử phạt. Biểu mẫu chỉ được giữ khi Target nêu một quy tắc,
    điều kiện, quyền, nghĩa vụ, thời hạn hoặc biện pháp pháp lý hoàn chỉnh,
    độc lập với phần thông tin còn trống cần điền.

    Với từng Target, quyết định has_document_metadata trước rồi mới quyết định
    has_penalty_content; hai field hoàn toàn độc lập và có thể khác nhau. Đặt
    metadata=true khi chính Target nêu rõ ít nhất một giá trị metadata của
    toàn văn bản đang đọc: số hiệu chính thức, ngày bắt đầu có hiệu lực hoặc
    ngày hết hiệu lực. Phần đầu văn bản nêu số hiệu chính thức là đủ, dù chỉ
    có thông tin hành chính, ngày ký/ban hành và penalty=false. Câu chỉ nhắc
    đến thời điểm văn bản có hiệu lực nhưng không chứa ngày cụ thể thì chưa
    cung cấp giá trị metadata và phải đặt false.

    Căn cứ cho metadata=true phải xuất hiện trong chính Target text. Không gán
    metadata từ block đứng trước, đứng sau hoặc cùng request; nếu Target chỉ ở
    gần một câu hiệu lực nhưng không tự chứa thông tin đó thì đặt false. Không
    chọn ngày ký/ban hành, ngày áp dụng riêng cho một điều/khoản, metadata của
    văn bản được dẫn chiếu hoặc số hiệu riêng của quy chuẩn, phụ lục, biểu mẫu
    hay tài liệu đính kèm.

    Phải xác định đúng văn bản mà thông tin metadata đang nói tới. Nếu Target
    thuộc phụ lục, mẫu biên bản, mẫu quyết định hoặc mẫu văn bản hành chính,
    câu như "Quyết định này có hiệu lực kể từ ngày ký" chỉ nói về văn bản sẽ
    được lập theo mẫu, không phải văn bản nguồn đang đọc, nên metadata=false.
    Tương tự, số hiệu văn bản nguồn xuất hiện như căn cứ hoặc dẫn chiếu bên
    trong mẫu không phải metadata của Target. Chỉ chọn số hiệu ở phần nhận diện
    chính thức của văn bản nguồn và điều khoản hiệu lực thực sự điều chỉnh toàn
    văn bản nguồn. Luôn đặt false khi số hiệu chỉ được lặp lại trong tiêu đề
    hoặc dòng giới thiệu phụ lục.

    Chỉ Target text làm căn cứ. Structural context chỉ giúp hiểu cấu trúc,
    không tự tạo nội dung pháp lý. Trả đúng một kết quả cho mỗi input, giữ
    nguyên chính xác chunk_id, không tạo hoặc sửa bất kỳ ký tự nào. Reason
    phải là một câu ngắn, hoàn toàn bằng tiếng Việt. Khi hai field khác nhau,
    reason phải thể hiện căn cứ của cả quyết định metadata và penalty.

    {{ _.role("user") }}

    Các chunk cần sàng lọc:
    {% for chunk in chunks %}
    ---
    chunk_id: {{ chunk.chunk_id }}
    {% if chunk.context_text %}
    Structural context:
    {{ chunk.context_text }}
    {% endif %}
    Target text:
    {{ chunk.text }}
    ---
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
