import { fetchJson } from "./core";

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------
export interface ChatResponse {
  answer?: string;
  sql?: string;
  data?: Record<string, unknown>[];
  columns?: string[];
  row_count?: number;
  error?: string;
}

export async function sendChatMessage(question: string, domain: string): Promise<ChatResponse> {
  return fetchJson("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, domain }),
  });
}
