"use client";

import { useState } from "react";
import type { QueryResponse } from "./types";

/**
 * Show how an answer was reached, from the pipeline's own trace.
 *
 * These are the real facts retrieved from the graph, not a narrated chain of thought, so an
 * answer can be checked against what actually grounded it rather than taken on trust.
 */
export function Reasoning({ response }: { response: QueryResponse }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="reasoning">
      <button className="reasoning-toggle" onClick={() => setOpen(!open)} aria-expanded={open}>
        {open ? "▾" : "▸"} Suy luận · {response.sources.length} fact
      </button>

      {open && (
        <div className="reasoning-body">
          <section>
            <h4>Facts truy xuất từ đồ thị ({response.sources.length})</h4>
            {response.sources.length === 0 ? (
              <p className="muted">Không tìm thấy fact nào — câu trả lời không có căn cứ.</p>
            ) : (
              <ol className="facts">
                {response.sources.map((fact, index) => (
                  <li key={index}>
                    <p>{fact.fact}</p>
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
