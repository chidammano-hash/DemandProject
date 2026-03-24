/**
 * Tuning Chat — API types, query keys, and fetchers for AI-powered LGBM tuning sessions.
 */

// --- Types ---

export type ChatMessageType =
  | "text"
  | "recommendation"
  | "run_started"
  | "run_completed"
  | "run_failed"
  | "analysis"
  | "error";

export type ChatRole = "user" | "assistant" | "system";

export interface ChatMessage {
  message_id: number;
  session_id: string;
  role: ChatRole;
  content: string;
  message_type: ChatMessageType;
  metadata: Record<string, unknown> | null;
  created_at: string;
}

export interface ChatSession {
  session_id: string;
  title: string;
  status: "active" | "archived";
  created_at: string;
  updated_at: string;
  message_count?: number;
}

export interface TuningRecommendation {
  strategy_label: string;
  description: string;
  overrides: Record<string, number | string>;
  expected_impact: string;
  risk_assessment: string;
  base_on_run_id: number | null;
}

export interface RunStatusResult {
  run_id: number;
  status: "running" | "completed" | "failed";
  started_at: string | null;
  completed_at: string | null;
  elapsed_seconds?: number;
  results?: {
    accuracy_pct: number;
    wape: number;
    bias: number;
    n_predictions: number;
    n_dfus: number;
  };
}

// --- Query keys ---

export const tuningChatKeys = {
  sessions: () => ["tuning-chat-sessions"] as const,
  session: (id: string) => ["tuning-chat-session", id] as const,
  runStatus: (sessionId: string, runId: number) =>
    ["tuning-chat-run-status", sessionId, runId] as const,
};

// --- Fetchers ---

export async function fetchChatSessions(
  params?: { status?: string; limit?: number },
): Promise<{ sessions: ChatSession[] }> {
  const qs = new URLSearchParams();
  if (params?.status) qs.set("status", params.status);
  if (params?.limit) qs.set("limit", String(params.limit));
  const url = `/lgbm-tuning/chat/sessions${qs.toString() ? `?${qs}` : ""}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetchChatSessions: ${res.status}`);
  return res.json();
}

export async function fetchChatSession(
  sessionId: string,
): Promise<{ session: ChatSession; messages: ChatMessage[] }> {
  const res = await fetch(`/lgbm-tuning/chat/sessions/${sessionId}`);
  if (!res.ok) throw new Error(`fetchChatSession: ${res.status}`);
  return res.json();
}

export async function createChatSession(
  title?: string,
): Promise<{ session: { session_id: string; title: string } }> {
  const res = await fetch("/lgbm-tuning/chat/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: title || "New Tuning Session" }),
  });
  if (!res.ok) throw new Error(`createChatSession: ${res.status}`);
  return res.json();
}

export async function sendTuningChatMessage(
  sessionId: string,
  content: string,
): Promise<{ messages: ChatMessage[] }> {
  const res = await fetch(
    `/lgbm-tuning/chat/sessions/${sessionId}/messages`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content }),
    },
  );
  if (!res.ok) throw new Error(`sendTuningChatMessage: ${res.status}`);
  return res.json();
}

export async function confirmTuningRun(
  sessionId: string,
  recommendationMessageId: number,
  overrideParams?: Record<string, unknown>,
): Promise<{ run_id: number; status: string; strategy_label: string }> {
  const res = await fetch(
    `/lgbm-tuning/chat/sessions/${sessionId}/confirm-run`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        recommendation_message_id: recommendationMessageId,
        override_params: overrideParams,
      }),
    },
  );
  if (!res.ok) throw new Error(`confirmTuningRun: ${res.status}`);
  return res.json();
}

export async function fetchRunStatus(
  sessionId: string,
  runId: number,
): Promise<RunStatusResult> {
  const res = await fetch(
    `/lgbm-tuning/chat/sessions/${sessionId}/run-status/${runId}`,
  );
  if (!res.ok) throw new Error(`fetchRunStatus: ${res.status}`);
  return res.json();
}
