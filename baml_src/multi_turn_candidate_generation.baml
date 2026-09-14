class MultiTurnHistoryInput {
  turn_number int
  question string
  answer string
}

class MultiTurnTargetInput {
  turn_number int
  reference_turn_number int
  facts FactContext[]
  dialogue_act string
  context_dependency string
  answerability string
}

class MultiTurnTargetGenerationInput {
  task_id string
  persona_description string
  history MultiTurnHistoryInput[]
  target MultiTurnTargetInput
}

class GeneratedMultiTurnTarget {
  turn_number int
  query string @description(#"Cau hoi follow-up phu thuoc vao history."#)
  standalone_query string @description(#"Cau hoi day du sau khi khoi phuc thong tin bi luoc bo tu history."#)
  reference_answer string @description(#"Dap an tham chieu dua tren history va facts moi cua target."#)
  reasoning string @description(#"Giai thich ngan cach facts moi dan toi dap an."#)
  missing_information string? @description(#"Chi tiet con thieu doi voi partially_answerable; null doi voi answerable."#)
  required_claims ClaimReference[] @description(#"Cac y bat buoc moi, tro dung fact_id cua target."#)
}

class GeneratedMultiTurnTargetBatchItem {
  task_id string
  target GeneratedMultiTurnTarget
}

function GenerateMultiTurnTargetBatch(
  items: MultiTurnTargetGenerationInput[]
) -> GeneratedMultiTurnTargetBatchItem[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Tao luot cuoi cho ground truth danh gia RAG hoi thoai trong domain xu ly vi
    pham giao thong duong bo tai Viet Nam. History da duoc duyet va phai giu
    nguyen. Moi task chi sinh luot target cuoi cung.

    Muc tieu danh gia:
    - Query phai can history de khoi phuc chu the, hanh vi, doi tuong hoac chu de.
    - History chi cung cap boi canh. Thong tin moi de tra loi trong tam nam trong
      target facts, vi vay he thong van phai retrieval.
    - Khong hoi lai dieu history da tra loi. Khong dua dap an vao query.

    Grounding:
    - Target facts la nguon duy nhat cho moi ket luan moi. Khong them kien thuc,
      con so, dieu kien, ngoai le, moc thoi gian hoac dan chieu khong co trong facts.
    - Co the nhac lai thong tin cu trong history de cau tra loi tu nhien, nhung
      required_claims chi ghi cac ket luan moi duoc ho tro boi target facts.
    - Giu day du cac tu han dinh ve chu the, hanh vi, doi tuong, muc dich, cach
      thuc, thoi diem va ngoai le.
    - valid_to null chi co nghia dau vao khong neu ngay ket thuc; khong duoc tu
      khang dinh quy dinh van con hieu luc.

    Query:
    - Viet tieng Viet tu nhien theo persona. Tuyet doi khong de lo alias nhu
      T5F1, T10F2, hoac thuat ngu noi bo nhu fact, facts, target facts, du lieu
      duoc cung cap hay thong tin duoc cung cap trong bat ky truong text nao.
    - continuity hoi mot khia canh moi cung chu de; deepening hoi sau hon dua tren
      chi tiet trong history; return quay lai chu de cu sau cac luot xen giua.
    - anaphora dung dai tu nhu "vay", "truong hop do", "muc nay". Ellipsis luoc
      phan da ro, vi du "Con xe may thi sao?".
    - query phai phu thuoc history, con standalone_query phai khoi phuc day du
      thong tin de co the dung doc lap cho retrieval.

    Answerability:
    - answerable: target facts tra loi tron trong tam.
    - partially_answerable: tra loi phan co can cu va noi ro dung mot chi tiet
      ma chinh query dang yeu cau nhung target facts chua xac dinh; khong tu them
      mot van de ngoai query chi de tao phan thieu.
    - missing_information phai null doi voi answerable. Voi partially_answerable,
      truong nay phai neu ngan gon dung mot chi tiet con thieu va chi tiet do phai
      duoc hoi truc tiep trong query cung standalone_query.
    - Moi task phai co it nhat mot required_claim. Moi claim tro dung fact_id cua
      target va chi chon tap facts nho nhat du tra loi. Moi fact_id chi duoc dung
      cho mot claim; neu co nhieu y cung mot fact thi gop thanh mot claim.

    Do dai mem:
    - query thuong khong qua 25 tu.
    - reference_answer thuong khong qua 180 tu.
    - reasoning thuong khong qua 80 tu.

    Tra toi da mot ket qua cho moi task. Giu nguyen task_id. Khong tron noi dung
    hoac fact_id giua cac task. Neu khong sinh duoc mot task, bo qua task do thay
    vi tao ket qua sai.

    {{ _.role("user") }}

    Cac target can sinh:
    {% for item in items %}
    ===== TARGET {{ item.task_id }} =====
    Persona: {{ item.persona_description }}
    History:
    {% for turn in item.history %}
    Luot {{ turn.turn_number }}
    Nguoi hoi: {{ turn.question }}
    Tra loi: {{ turn.answer }}
    {% endfor %}

    Luot target: {{ item.target.turn_number }}
    Luot history can tham chieu: {{ item.target.reference_turn_number }}
    Dialogue act: {{ item.target.dialogue_act }}
    Context dependency: {{ item.target.context_dependency }}
    Answerability: {{ item.target.answerability }}
    Target facts moi can retrieval:
    {% for fact in item.target.facts %}
    [{{ fact.fact_id }}] {{ fact.statement }}
    Van ban: {{ fact.document_number }}
    Hieu luc: {{ fact.valid_from }} den {{ fact.valid_to }}
    {% endfor %}
    ===== END TARGET {{ item.task_id }} =====
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
