"use client";

import { useState } from "react";
import type { QueryResponse } from "./types";

/**
 * Show how an answer was reached, from the pipeline's own trace.
 *
 * These are the real retrieval steps, not a narrated chain of thought: the question the graph
 * was actually searched with, the facts that came back, and the chunk each fact was extracted
 * from. That makes an answer checkable — a wrong answer is usually a wrong rewrite or a fact
 * retrieved from the wrong document, and both are visible here.
 */
export function Reasoning({ response }: { response: QueryResponse }) {
  const [open, setOpen] = useState(false);
  const rewritten = response.rewritten_query !== response.query;

  return (
    <div className="reasoning">
      <button className="reasoning-toggle" onClick={() => setOpen(!open)} aria-expanded={open}>
        {open ? "▾" : "▸"} Suy luận · {response.sources.length} fact
        {rewritten ? " · câu hỏi đã được viết lại" : ""}
      </button>

      {open && (
        <div className="reasoning-body">
          <section>
            <h4>1. Truy vấn dùng để tìm trong đồ thị</h4>
            {rewritten ? (
              <>
                <p className="muted">Câu hỏi gốc: {response.query}</p>
                <p className="rewritten">→ {response.rewritten_query}</p>
                <p className="muted small">
                  Đồ thị lưu thuật ngữ pháp lý chính thức, nên câu hỏi thông tục được viết lại
                  trước khi tìm kiếm.
                </p>
              </>
            ) : (
              <p className="muted">Dùng nguyên văn câu hỏi.</p>
            )}
          </section>

          <section>
            <h4>2. Facts truy xuất từ đồ thị ({response.sources.length})</h4>
            {response.sources.length === 0 ? (
              <p className="muted">Không tìm thấy fact nào — câu trả lời không có căn cứ.</p>
            ) : (
              <ol className="facts">
                {response.sources.map((fact, index) => (
                  <li key={index}>
                    <p>{fact.fact}</p>
                    <ul className="sources">
                      {fact.sources.map((source) => (
                        <li key={source.chunk_id}>
                          <span className="doc">{source.source_name}</span>
                          <code>{source.chunk_id.slice(0, 12)}…</code>
                        </li>
                      ))}
                    </ul>
                  </li>
                ))}
              </ol>
            )}
          </section>

          {response.unindexed_citations.length > 0 && (
            <section className="warning">
              <h4>⚠ Trích dẫn ngoài phạm vi đã index</h4>
              <p>
                Câu trả lời nhắc tới {response.unindexed_citations.join(", ")}, chưa có trong đồ
                thị. Thông tin liên quan có thể thiếu.
              </p>
            </section>
          )}
        </div>
      )}
    </div>
  );
}
