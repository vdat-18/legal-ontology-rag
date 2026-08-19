import { NextResponse } from "next/server";

/**
 * Proxy one question to the knowledge-graph API from the server.
 *
 * The browser calls this same-origin route instead of the API directly, so the backend never has
 * to be reachable from the visitor's machine and no CORS allow list is involved. That is what lets
 * a single tunnel expose the whole app: the API URL is resolved here, on the server, rather than
 * being baked into the client bundle where it would point at the visitor's own localhost.
 */

const API_URL = process.env.GRAPHITI_API_URL ?? "http://127.0.0.1:8100";

export async function POST(request: Request) {
  const payload = await request.json();

  let response: Response;
  try {
    response = await fetch(`${API_URL}/graphiti/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    // The backend is a separate process; say so plainly instead of a bare network error.
    return NextResponse.json({ detail: `Không kết nối được tới API tại ${API_URL}` }, { status: 502 });
  }

  const body = await response.text();
  return new NextResponse(body, {
    status: response.status,
    headers: { "Content-Type": response.headers.get("Content-Type") ?? "application/json" },
  });
}
