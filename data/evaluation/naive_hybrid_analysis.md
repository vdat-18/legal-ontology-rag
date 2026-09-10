# Naive RAG Hybrid Evaluation — Analysis Report

- **Date:** 2026-09-09
- **Config:** pipeline=naive, search_mode=hybrid, limit=10 (pool 30/branch, RRF k=60), concurrency=30
- **Judge model:** gpt-5.6-terra (BAML LLM judge)
- **Samples:** 1000 (ground truth: `data/benchmark_synthesis/groundtruth.jsonl`)
- **Outputs:** `naive_hybrid_predictions.jsonl`, `naive_hybrid_metrics.json`, `naive_hybrid_generation.json`

## 1. Metrics Summary

### Retrieval

| Metric | Value | Note |
|--------|-------|------|
| hit@1 / @5 / @10 | 30.5% / 56.3% / **67.9%** | 321/1000 câu zero-hit trong top 10 |
| recall@10 | 0.465 | Trung bình 2.4 gold chunks/câu (1–8) |
| precision@10 | 0.110 | Ceiling cấu trúc = 2.4/10 = 0.24 → đạt ~46% ceiling |

### Generation (LLM judge)

| Metric | Value |
|--------|-------|
| faithfulness | 0.941 |
| correctness | 3.10 / 5 |
| answer_relevancy | 3.87 / 5 |
| temporal_correctness | 0.729 (271 câu = 0) |

### Tính nhất quán pipeline

- correctness khi **có hit** (n=679): **3.58**
- correctness khi **miss** (n=321): **2.08**
- correctness ≥ 4.5: 186/1000; faithfulness ≤ 0.5: 8/1000
- → Hệ thống coherent; **retrieval là bottleneck chính**, không phải generation.

## 2. Phát hiện quan trọng nhất: query không dấu làm retrieval sập

Zero-hit rate theo `query_style`:

| query_style | zero-hit |
|-------------|----------|
| polite / formal / standard / conversational | 17–22% |
| keyword_style | 26% |
| misspelled | 30% |
| **no_diacritics** | **89%** |

- 143 câu không dấu chiếm **~127/321 miss (39% tổng số miss)**.
- Nguyên nhân: BM25 match token chính xác ("tien" ≠ "tiền"); bge_m3 embedding suy giảm nặng trên văn bản bỏ dấu.
- Đây cũng giải thích tại sao `bundle_kind=single_hop` có zero-hit cao nhất (48%) — style no_diacritics dồn vào nhóm này, không phải câu dễ harder hơn.
- **Đề xuất:** khôi phục dấu query (diacritic restoration) hoặc index thêm bản không dấu cho BM25. Ước tính hit@10 tăng lên ~75–77% nếu xử lý.

## 3. Ground Truth Assessment

**Đúng phần lớn:**
- Chunk ID format khớp hoàn toàn (hash); các row có hit chứng minh join hoạt động.
- 321 câu zero-hit **không câu nào** retrieve trúng document của gold chunks → miss là miss thật, không phải lỗi ID mapping.

**2 điểm lưu ý:**
1. **26 câu hit=0 nhưng correctness≥4** — retrieval tìm chunk khác cùng chứa thông tin đúng (VD: câu bảo hiểm bắt buộc) nhưng gold list hẹp → retrieval bị under-credit. Ảnh hưởng nhỏ (26/1000) nhưng systematic.
2. 333 câu **temporal bundle** có expected answer dạng tầng lớp ("Theo NĐ 168/2024... Theo NĐ 336/2025 từ ngày...") — benchmark thiết kế khó cho naive RAG by design; đây chính là điểm ontology RAG phải thắng.

## 4. Judge Prompt Issues (đã xử lý)

1. **temporal_correctness binary ép chấm câu abstain** — 884/1000 answer có dạng "chưa thể khẳng định..." vẫn bị chấm temporal=0 dù không áp sai luật. **ĐÃ FIX 2026-09-09:** `temporal_correctness` đổi thành nullable, null khi answer không khẳng định quy định pháp luật nào; aggregate tính mean chỉ trên giá trị non-null. → Cần re-judge để số liệu phản ánh đúng.
2. **Faithfulness 0.94 bị "phồng" do abstention** — answer không khẳng định gì thì không có claim vi phạm. Không sai, nhưng phải đọc kèm abstention rate (88%) khi diễn giải.

## 5. Dataset Metadata

- `bundle_kind`: single_hop 334, temporal 333, multi_hop 333
- `reasoning_hops`: 1→334, 2→417, 3/4/5→83 mỗi loại
- `query_style`: 7 style × ~143 câu (standard, polite, formal, no_diacritics, misspelled, conversational, keyword_style)
- `interaction_type`: single_turn 100%; `generator_model`: gpt-5.6-terra
- expected_answer: mean 504 chars, max 1739 — verbose, multi-layered

## 6. Khuyến nghị (thứ tự ưu tiên)

1. **Xử lý no_diacritics query** — 39% tổng miss; quick win lớn nhất cho retrieval.
2. ✅ ~~Sửa temporal judge N/A cho abstain~~ (đã xong)
3. ~~Dọn docs stale~~ — đã sạch; `docs/*.md` ghi đúng lịch sử removal QUEST/MRR.
4. (Tùy chọn) Thêm MRR/NDCG để đo chất lượng ranking; multi-gold expansion cho 26 case under-credit.

## 7. Re-judge command (sau fix temporal nullable)

```powershell
uv run python -m legal_rag.evaluation.generation data/evaluation/naive_hybrid_predictions.jsonl --output data/evaluation/naive_hybrid_generation_v2.json --batch-size 5 --concurrency 30
```
