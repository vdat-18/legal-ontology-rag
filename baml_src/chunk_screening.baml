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
    True nếu target giúp xác định số hiệu hoặc hiệu lực của chính văn bản đang đọc.
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
    doanh vận tải, cơ sở đào tạo/sát hạch hoặc đơn vị quản lý hạ tầng. Với các
    chủ thể quản lý, kinh doanh hoặc đào tạo, chỉ giữ khi Target tự nêu hành vi
    vi phạm và hậu quả pháp lý hoặc cách xử lý tương ứng. Không suy ra rằng một
    nghĩa vụ quản lý, điều kiện hoạt động hay từ "phải" tự nó là fact vi phạm.

    Đặt false nếu Target chỉ mô tả hoạt động quản lý, điều kiện vận hành hoặc
    kiến thức chuyên ngành mà chưa tạo liên hệ pháp lý trực tiếp với một vi
    phạm đường bộ. Đánh giá chức năng pháp lý của toàn Target, không quyết định
    chỉ vì xuất hiện từ như "vi phạm", "cấm", "lấn làn" hay "vượt đèn".

    Luôn đặt false cho:
    - chương trình, thời lượng, nội dung đào tạo; lỗi, điểm, đạt/trượt hoặc quy
      trình thi sát hạch; tình huống giả định, mô phỏng hay ví dụ sư phạm, kể
      cả khi tình huống mô tả một hành vi có vẻ vi phạm;
    - nghĩa vụ tổ chức, báo cáo, lưu dữ liệu hoặc vận hành thông thường của cơ
      sở đào tạo/sát hạch nếu Target không tự nêu căn cứ thu hồi, đình chỉ, chế
      tài hoặc biện pháp xử lý do vi phạm nghĩa vụ đó;
    - hình ảnh, bản vẽ, sơ đồ, bảng kích thước hoặc mô tả thuần kỹ thuật về vật
      liệu, màu sắc, kết cấu, chế tạo, lắp đặt báo hiệu hay phương tiện. Vẫn giữ
      phần chữ quy định ý nghĩa, hiệu lực báo hiệu hoặc nghĩa vụ người tham gia
      giao thông phải chấp hành;
    - phí, hợp đồng, hồ sơ, giám định, tạm ứng hoặc thủ tục bồi thường bảo hiểm,
      bảng tỷ lệ thương tật và nghĩa vụ nội bộ của doanh nghiệp bảo hiểm, kể cả
      khi phát sinh từ tai nạn hoặc nhắc đến rượu, bia, ma túy. Chỉ giữ nghĩa vụ
      phải có bảo hiểm bắt buộc để tham gia giao thông, hành vi không có bảo
      hiểm và việc cơ quan có thẩm quyền kiểm tra, xử lý hành vi đó;
    - nội dung riêng của đường sắt, đường thủy, hàng hải, hàng không, ma túy,
      cai nghiện hoặc lĩnh vực khác ngoài giao thông đường bộ;
    - chính sách, tuyên truyền, tổ chức quản lý, dữ liệu/thống kê, tài chính,
      hỗ trợ hoặc thủ tục hành chính độc lập.

    Quy tắc xử lý vi phạm hành chính chung được giữ khi nội dung của nó có thể
    áp dụng trực tiếp cho vụ việc giao thông đường bộ. Danh sách chức danh hoặc
    cơ quan chỉ được giữ khi Target tự gắn chức danh đó với thẩm quyền xử lý vi
    phạm đường bộ; việc một cơ quan giao thông xuất hiện trong danh sách thẩm
    quyền chung pha trộn nhiều lĩnh vực là chưa đủ. Loại phần chỉ điều chỉnh
    một lĩnh vực khác hoặc chỉ liệt kê cơ quan, đối tượng, ví dụ ngoài domain.
    Định nghĩa/phân loại chỉ được giữ khi cần áp dụng trực tiếp một quy tắc vi
    phạm. Khi chưa thấy liên hệ trực tiếp này trong Target, đặt false.

    Mục lục, tiêu đề, lời dẫn, căn cứ ban hành, danh mục hoặc tên biểu mẫu và
    thông tin hành chính tự chúng không đủ. Luôn đặt false cho mẫu quyết định,
    biên bản, thông báo, phiếu, sổ theo dõi hoặc giấy phép chưa được lập; phần
    hướng dẫn điền mẫu; trường trống, dòng giao/nhận và ô ký tên. Quy tắc nhìn
    có vẻ hoàn chỉnh nhưng chỉ là boilerplate của văn bản sẽ được lập theo mẫu
    vẫn phải đặt false. Chỉ phần thuyết minh độc lập bên ngoài mẫu, trực tiếp
    giải thích một quy tắc pháp luật áp dụng chung, mới có thể được giữ.

    Với từng Target, quyết định has_document_metadata trước rồi mới quyết định
    has_penalty_content; hai field hoàn toàn độc lập và có thể khác nhau. Đặt
    metadata=true khi chính Target chứa ít nhất một thông tin giúp Step 1B xác
    định metadata của toàn văn bản đang đọc:
    - số hiệu chính thức tại phần nhận diện của văn bản nguồn;
    - ngày bắt đầu hoặc hết hiệu lực của toàn văn bản;
    - điều khoản hiệu lực tương đối của toàn văn bản như "có hiệu lực kể từ
      ngày ký", dù ngày ký nằm ở metadata candidate khác. Step 1B sẽ kết hợp
      các candidate để suy ra giá trị cuối cùng.

    Phần đầu văn bản nêu số hiệu chính thức là đủ, dù chỉ có thông tin hành
    chính, ngày ký/ban hành và penalty=false. Điều khoản hiệu lực của chính
    văn bản vẫn phải đặt metadata=true khi nằm ở cuối một Target hỗn hợp chứa
    nhiều sửa đổi hoặc nội dung thuộc domain khác.

    Căn cứ cho metadata=true phải xuất hiện trong chính Target text. Không gán
    metadata từ block đứng trước, đứng sau hoặc cùng request; nếu Target chỉ ở
    gần một câu hiệu lực nhưng không tự chứa thông tin đó thì đặt false. Không
    chọn riêng ngày ký/ban hành nếu Target không phải phần nhận diện chính thức;
    ngày áp dụng riêng cho một điều/khoản; metadata của văn bản được dẫn chiếu;
    hoặc số hiệu riêng của quy chuẩn, phụ lục, biểu mẫu hay tài liệu đính kèm.
    Nếu Target đồng thời chứa ngày áp dụng riêng và ngày hiệu lực toàn văn bản,
    metadata=true vì có thông tin hợp lệ về toàn văn bản.

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
