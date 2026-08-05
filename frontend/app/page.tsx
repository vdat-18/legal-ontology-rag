"use client";

import { useState } from "react";
import { Reasoning } from "./Reasoning";
import type { QueryResponse, Turn } from "./types";

const API_URL = process.env.NEXT_PUBLIC_GRAPHITI_API_URL ?? "http://127.0.0.1:8100";

const EXAMPLES = [
  "Vượt đèn đỏ phạt bao nhiêu tiền?",
  "Nồng độ cồn vượt mức bị xử phạt thế nào?",
  "Xóa dấu vết hiện trường tai nạn ô tô bị phạt bao nhiêu?",
];

async function askGraph(question: string): Promise<QueryResponse> {
  const response = await fetch(`${API_URL}/graphiti/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: question }),
  });
  if (!response.ok) {
    // Surface the server's own message; a bare status hides whether the graph or the LLM failed.
    throw new Error(`API ${response.status}: ${(await response.text()).slice(0, 300)}`);
  }
  return response.json();
}

export default function Home() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(false);

  async function submit(question: string) {
    const trimmed = question.trim();
    if (!trimmed || pending) return;

    setDraft("");
    setPending(true);
    const index = turns.length;
    setTurns((current) => [...current, { question: trimmed }]);

    try {
      const response = await askGraph(trimmed);
      setTurns((current) => current.map((turn, i) => (i === index ? { ...turn, response } : turn)));
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setTurns((current) => current.map((turn, i) => (i === index ? { ...turn, error: message } : turn)));
    } finally {
      setPending(false);
    }
  }

  return (
    <main>
      <header>
        <h1>⚖️ Hỏi đáp luật giao thông</h1>
        <p className="muted small">
          Trả lời từ đồ thị tri thức thời gian (Neo4j + Graphiti). Mỗi câu trả lời kèm vết suy
          luận và nguồn trích dẫn.
        </p>
      </header>

      {turns.length === 0 && (
        <div className="examples">
          {EXAMPLES.map((example) => (
            <button key={example} onClick={() => submit(example)} disabled={pending}>
              {example}
            </button>
          ))}
        </div>
      )}

      <div className="transcript">
        {turns.map((turn, index) => (
          <div key={index} className="turn">
            <div className="bubble question">{turn.question}</div>

            {turn.error && <div className="bubble error">Lỗi: {turn.error}</div>}

            {!turn.response && !turn.error && <div className="bubble thinking">Đang tra cứu đồ thị…</div>}

            {turn.response && (
              <div className="bubble answer">
                <p>{turn.response.answer}</p>
                <Reasoning response={turn.response} />
              </div>
            )}
          </div>
        ))}
      </div>

      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault();
          submit(draft);
        }}
      >
        <input
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          placeholder="Nhập câu hỏi về luật giao thông…"
          disabled={pending}
          autoFocus
        />
        <button type="submit" disabled={pending || !draft.trim()}>
          {pending ? "…" : "Gửi"}
        </button>
      </form>
    </main>
  );
}
