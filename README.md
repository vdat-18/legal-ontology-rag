# legal-ontology-rag

**Temporal Ontology-Enhanced RAG** cho tra cứu **mức phạt vi phạm giao thông
đường bộ Việt Nam theo thời điểm hiệu lực** (time-aware). Xây trên
[Graphiti](https://github.com/getzep/graphiti) (knowledge graph **bi-temporal**),
dùng **Claude** để trích xuất và **BGE-M3** để nhúng tiếng Việt.

> Ý tưởng cốt lõi: mỗi văn bản được nạp với `reference_time = ngày hiệu lực`.
> Khi nghị định sửa đổi (vd 238/2026) được nạp muộn hơn, Graphiti **tự đóng**
> hiệu lực của quy định cũ — nên cùng một câu hỏi ở hai mốc thời gian sẽ trả về
> đúng mức phạt có hiệu lực tại mốc đó.

---

## Yêu cầu (prerequisites)

| Thành phần | Ghi chú |
| --- | --- |
| **Python 3.11+** | Bắt buộc (dùng `datetime.UTC`). ⚠️ `python3` mặc định của macOS thường là 3.9 — sẽ **không chạy được**. |
| **Neo4j 5.26+** | Backend của Graphiti (chạy nhanh bằng Docker). |
| **ANTHROPIC_API_KEY** | Claude, dùng cho trích xuất thực thể. |
| **OPENAI_API_KEY** *(tuỳ chọn)* | Chỉ cần nếu embedder = `openai`. Dùng `bge` thì không cần. |

---

## Các bước thực hiện

### 1) Lấy code & tạo môi trường ảo (Python 3.11+)

```bash
git clone <repo-url> && cd OntologyRag
python3.11 -m venv .venv && source .venv/bin/activate   # đảm bảo là 3.11+
python -V                                               # phải >= 3.11
```

### 2) Cài đặt

```bash
pip install -e ".[vietnamese]"     # embedder BGE-M3 cục bộ (khuyến nghị cho tiếng Việt)
# hoặc:
pip install -e ".[openai]"         # nếu muốn dùng embedding của OpenAI
```

Lệnh này kéo `graphiti-core`, `loguru`, `pypdf`, `defusedxml`, `python-dotenv`
từ PyPI. (Bản graphiti đã vendor sẵn ở `third_party/graphiti` — nếu muốn dùng bản
đó thay PyPI: `pip install -e third_party/graphiti`.)

### 3) Chạy Neo4j (Docker)

```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5.26
```

Kiểm tra: mở http://localhost:7474 (user `neo4j`, mật khẩu `password`).

### 4) Cấu hình `.env`

```bash
cp .env.example .env
```

Mở `.env` và điền:
- `ANTHROPIC_API_KEY=sk-ant-...`
- `GRAPHITI_EMBEDDER=bge` (không cần key) **hoặc** `openai` + `OPENAI_API_KEY=...`
- `NEO4J_URI/USER/PASSWORD` khớp với bước 3.

> CLI **tự nạp** `.env` (qua python-dotenv) nên không cần `export` thủ công.

### 5) Nạp dữ liệu vào graph (ingest)

```bash
legal-rag ingest --reset
```

- Đọc **trực tiếp từ** `data/raw/road_traffic_legal_documents.zip` (không cần giải nén).
- Chỉ nạp **nhóm văn bản xử phạt** đã lọc trong `data/config/penalty_corpus.json`.
- `--reset` xoá graph cũ trước khi nạp. Bỏ `--reset` để nạp thêm.

### 6) Truy vấn point-in-time

```bash
legal-rag query "vượt đèn đỏ xe máy" 2025-02-01
legal-rag query "vượt đèn đỏ xe máy" 2024-06-01   # so sánh mốc khác
```

Chỉ in ra các fact **còn hiệu lực tại ngày** được hỏi, kèm khoảng hiệu lực.

### 7) Kiểm tra chất lượng (trước khi commit)

```bash
ruff format .
ruff check .
pytest
```

CI (`.github/workflows/code-quality.yml`) chạy `ruff check` + `ruff format --check`.

---

## Cấu trúc project

```
src/legal_rag/            # package ứng dụng (src layout)
├── config.py             # cấu hình theo biến môi trường
├── errors.py             # exception domain
├── ingestion/            # DOCX/PDF -> episode theo Điều, gắn ngày hiệu lực
├── ontology/             # entity/edge types cho LLM trích xuất
├── embeddings/           # embedder tiếng Việt (BGE-M3)
├── graph/                # dựng Graphiti client + orchestration nạp
├── retrieval/            # truy vấn point-in-time (bi-temporal)
└── cli.py                # lệnh `legal-rag ingest` / `legal-rag query`
data/
├── raw/                  # kho văn bản (.zip) — không giải nén vào git
└── config/               # penalty_corpus.json (danh sách văn bản xử phạt đã lọc)
tests/                    # test loader offline (đọc từ zip, không cần Neo4j)
third_party/graphiti/     # graphiti-core vendored (git-ignored để dùng lib)
docs/                     # kế hoạch & tài liệu tham khảo
```

Xem [AGENTS.md](AGENTS.md) cho chuẩn code của repo.

---

## Điều chỉnh nhóm văn bản

Thêm/bớt văn bản trong nhóm xử phạt bằng cách sửa
`data/config/penalty_corpus.json` (không sửa code). Mỗi mục:

```json
{"file": "168_2024_ND-CP_619502.docx", "so_hieu": "168/2024/NĐ-CP",
 "loai": "Nghị định", "effective_date": "2025-01-01", "amends": null}
```

`effective_date: null` → loader tự trích ngày hiệu lực từ văn bản; điền tay để
ghi đè khi bản tự trích chưa chuẩn (thường gặp ở văn bản sửa đổi/hợp nhất).

---

## Xử lý sự cố

| Triệu chứng | Nguyên nhân / cách xử lý |
| --- | --- |
| `ImportError: cannot import name 'UTC'` | Đang chạy Python < 3.11. Tạo venv bằng 3.11+. |
| `ANTHROPIC_API_KEY is not set` | Chưa điền key trong `.env` (hoặc chạy ngoài thư mục chứa `.env`). |
| Kết nối Neo4j lỗi | Container chưa chạy / sai `NEO4J_URI`/mật khẩu. Kiểm tra bước 3. |
| Nạp báo *Missing document* | Thiếu `data/raw/road_traffic_legal_documents.zip`. |
| Tải model BGE-M3 lâu lần đầu | Bình thường (~2GB). Hoặc đổi `GRAPHITI_EMBEDDER=openai`. |
