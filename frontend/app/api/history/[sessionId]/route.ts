import { NextResponse } from "next/server";

/** Proxy a session's transcript from the knowledge-graph API, same-origin like /api/query. */

const API_URL = process.env.GRAPHITI_API_URL ?? "http://127.0.0.1:8100";

export async function GET(_request: Request, context: { params: Promise<{ sessionId: string }> }) {
  const { sessionId } = await context.params;

  let response: Response;
  try {
    response = await fetch(`${API_URL}/graphiti/history/${encodeURIComponent(sessionId)}`);
  } catch {
    // A reload with the backend down should show an empty chat, not a crashed page.
    return NextResponse.json({ session_id: sessionId, turns: [] });
  }

  const body = await response.text();
  return new NextResponse(body, {
    status: response.status,
    headers: { "Content-Type": response.headers.get("Content-Type") ?? "application/json" },
  });
}
