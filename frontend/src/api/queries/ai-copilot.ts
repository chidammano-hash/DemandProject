import { fetchJson } from "./core";

export interface CopilotContextInput {
  page: string;
  item_id?: string;
  customer_group?: string;
  loc?: string;
  opportunity_id?: string;
  exception_id?: string;
  workflow_run_id?: string;
}

export interface CopilotSession {
  session_id: string;
  owner_id: string;
  context: CopilotContextInput;
  status: string;
}

export interface CopilotCitation {
  evidence_id: string;
  claim: string;
  source: string;
  business_key: string;
  freshness: string;
  content_hash: string;
  values: Record<string, unknown>;
}

export interface CopilotTurn {
  turn_id: string;
  answer: string;
  citations: CopilotCitation[];
  action_request: null;
}

export function createCopilotSession(
  context: CopilotContextInput,
  idempotencyKey: string
): Promise<CopilotSession> {
  return fetchJson<CopilotSession>("/ai-copilot/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json", "Idempotency-Key": idempotencyKey },
    body: JSON.stringify(context),
  });
}

export function runCopilotTurn(
  sessionId: string,
  prompt: string,
  idempotencyKey: string
): Promise<CopilotTurn> {
  return fetchJson<CopilotTurn>(`/ai-copilot/sessions/${encodeURIComponent(sessionId)}/turns`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Idempotency-Key": idempotencyKey },
    body: JSON.stringify({ prompt }),
  });
}
