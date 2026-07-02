// SKU Chatbot API client — spec docs/specs/06-ai-platform/07-sku-chatbot.md.
// Streaming is Server-Sent Events; fetchJson can't read a stream, so streamSkuChat
// uses a ReadableStream reader and yields parsed events.
import { fetchJson } from "./core";

// ---------------------------------------------------------------------------
// Types (mirror api/routers/intelligence/sku_chat.py)
// ---------------------------------------------------------------------------
export interface SkuChatConfig {
  auth_mode: string;
  models: Record<string, string>;
  routing: { default_tier: string; allow_user_override: boolean };
  guardrails: Record<string, number>;
  tools: string[];
}

export interface SkuChatHistoryMessage {
  role: "user" | "assistant";
  content: string;
}

export interface SkuChatSession {
  session_id: string;
  item_id: string;
  customer_group: string;
  loc: string;
}

export interface SkuChatStoredMessage {
  id: number;
  role: string;
  content: string;
  model: string | null;
  tier: string | null;
  created_at: string;
}

export interface SkuChatSessionDetail extends SkuChatSession {
  created_by: string | null;
  created_at: string;
  last_active_at: string;
  messages: SkuChatStoredMessage[];
}

export interface StreamSkuChatParams {
  question: string;
  item_id: string;
  loc: string;
  customer_group?: string;
  session_id?: string | null;
  history?: SkuChatHistoryMessage[];
  model_tier?: string | null;
  model?: string | null;
  page_focus?: string | null;
}

export interface ChampionAdjustMonth {
  forecast_month: string;
  champion_qty: number;
  ai_qty: number;
  pct_change: number | null;
}

export interface ChampionAdjustPreview {
  recommendation_code: string;
  rec_pct_change: number | null;
  confidence: number | null;
  rationale: string;
  months?: ChampionAdjustMonth[];
}

// SSE event union emitted by /sku-chat/stream.
export type SkuChatEvent =
  | { type: "meta"; tier: string; model: string; session_id?: string }
  | { type: "text"; chunk: string }
  | { type: "tool"; name: string | null; input: unknown }
  | { type: "result"; text: string | null; cost_usd: number | null; usage: unknown }
  | { type: "error"; message: string; truncated?: boolean }
  | {
      type: "approval_request";
      approval_id: string;
      item_id: string;
      loc: string;
      preview: ChampionAdjustPreview;
    };

// ---------------------------------------------------------------------------
// Query keys
// ---------------------------------------------------------------------------
export const skuChatKeys = {
  config: () => ["sku-chat", "config"] as const,
  session: (sessionId: string) => ["sku-chat", "session", sessionId] as const,
};

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------
export async function fetchSkuChatConfig(): Promise<SkuChatConfig> {
  return fetchJson<SkuChatConfig>("/sku-chat/config");
}

export async function createSkuChatSession(params: {
  item_id: string;
  loc: string;
  customer_group?: string;
}): Promise<SkuChatSession> {
  return fetchJson<SkuChatSession>("/sku-chat/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
}

export async function fetchSkuChatSession(sessionId: string): Promise<SkuChatSessionDetail> {
  return fetchJson<SkuChatSessionDetail>(
    `/sku-chat/session/${encodeURIComponent(sessionId)}`,
  );
}

export interface AdjustmentDecisionResult {
  approval_id: string;
  status: string;
  result?: unknown;
}

/** Approve (apply) or reject a champion-forecast adjustment the agent staged. */
export async function decideSkuChatAdjustment(
  approvalId: string,
  decision: "approve" | "reject",
): Promise<AdjustmentDecisionResult> {
  return fetchJson<AdjustmentDecisionResult>(
    `/sku-chat/adjustment/${encodeURIComponent(approvalId)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision }),
    },
  );
}

/**
 * Stream one chat turn. Yields parsed SSE events as they arrive.
 * Pass an AbortSignal to cancel an in-flight turn.
 */
export async function* streamSkuChat(
  params: StreamSkuChatParams,
  signal?: AbortSignal,
): AsyncGenerator<SkuChatEvent> {
  const res = await fetch("/sku-chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
    signal,
  });
  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `sku-chat stream failed: ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx: number;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 2);
        if (frame.startsWith("data:")) {
          const json = frame.slice(5).trim();
          if (json) {
            yield JSON.parse(json) as SkuChatEvent;
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
