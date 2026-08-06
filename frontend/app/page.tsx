"use client";

import { useEffect, useState } from "react";
import { Reasoning } from "./Reasoning";
import type { HistoryTurn, QueryResponse, Turn } from "./types";

const EXAMPLES = [
  "Vượt đèn đỏ phạt bao nhiêu tiền?",
  "Nồng độ cồn vượt mức bị xử phạt thế nào?",
  "Xóa dấu vết hiện trường tai nạn ô tô bị phạt bao nhiêu?",
];

async function askGraph(question: string, sessionId: string): Promise<QueryResponse> {
  // Same-origin proxy route, so one tunnel exposes the whole app and CORS never applies.
  const response = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: question, session_id: sessionId }),
  });
  if (!response.ok) {
    // Surface the server's own message; a bare status hides whether the graph or the LLM failed.
    throw new Error(`API ${response.status}: ${(await response.text()).slice(0, 300)}`);
  }
  return response.json();
}

const SESSION_KEY = "graphiti-chat-session";

/** Reuse this browser's session id across reloads, minting one on first visit. */
function loadSessionId(): string {
  const stored = window.localStorage.getItem(SESSION_KEY);
  if (stored) return stored;
  const created = crypto.randomUUID();
  window.localStorage.setItem(SESSION_KEY, created);
  return created;
}

export default function Home() {
  // Kept in localStorage so a reload rejoins the same conversation: the server keys context on
  // this id, and a fresh id would silently start a new one while the old transcript lingered.
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(false);

  useEffect(() => {
    // localStorage and crypto are browser-only, so the id is resolved after mount rather than
    // during render, which would break the server-rendered pass.
    const id = loadSessionId();
    setSessionId(id);

    fetch(`/api/history/${id}`)
      .then((response) => (response.ok ? response.json() : { turns: [] }))
      .then((data: { turns?: HistoryTurn[] }) => {
        setTurns((data.turns ?? []).map((turn) => ({ question: turn.question, restoredAnswer: turn.answer })));
      })
      .catch(() => setTurns([]));
  }, []);

  async function submit(question: string) {
    const trimmed = question.trim();
    if (!trimmed || pending || !sessionId) return;

    setDraft("");
    setPending(true);
    const index = turns.length;
    setTurns((current) => [...current, { question: trimmed }]);

    try {
      const response = await askGraph(trimmed, sessionId);
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

      <div className="transcript" aria-live="polite" aria-busy={pending}>
        {turns.map((turn, index) => (
          <div key={index} className="turn">
            <div className="bubble question">{turn.question}</div>

            {turn.error && (
              <div className="bubble error" role="alert">
                Lỗi: {turn.error}
              </div>
            )}

            {!turn.response && !turn.restoredAnswer && !turn.error && (
              <div className="bubble thinking" role="status">
                <span>Đang suy nghĩ</span>
                <span className="dots" aria-hidden="true">
                  <i />
                  <i />
                  <i />
                </span>
              </div>
            )}

            {turn.response && (
              <div className="bubble answer">
                <p>{turn.response.answer}</p>
                <Reasoning response={turn.response} />
              </div>
            )}

            {/* Only under the newest answer: older suggestions refer to a point the
                conversation has already moved past. */}
            {turn.response && index === turns.length - 1 && turn.response.suggestions.length > 0 && (
              <div className="suggestions">
                {turn.response.suggestions.map((suggestion) => (
                  <button key={suggestion} onClick={() => submit(suggestion)} disabled={pending}>
                    {suggestion}
                  </button>
                ))}
              </div>
            )}

            {turn.restoredAnswer && (
              <div className="bubble answer">
                <p>{turn.restoredAnswer}</p>
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
          aria-label="Câu hỏi về luật giao thông"
          disabled={pending || !sessionId}
          autoFocus
        />
        <button type="submit" disabled={pending || !draft.trim() || !sessionId}>
          {pending ? "…" : "Gửi"}
        </button>
      </form>
    </main>
  );
}
