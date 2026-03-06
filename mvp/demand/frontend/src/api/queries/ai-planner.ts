import type {
  InsightListResponse,
  MemoListResponse,
  AnalyzeResponse,
  InsightSeverity,
  InsightStatus,
  InsightType,
} from "@/types/ai_planner";
import { fetchJson, submitJob } from "./core";

// ---------------------------------------------------------------------------
// AI Planner queries (IPAIfeature1)
// ---------------------------------------------------------------------------
export interface AiInsightParams {
  severity?: InsightSeverity;
  status?: InsightStatus;
  insight_type?: InsightType;
  item_no?: string;
  loc?: string;
  page?: number;
  page_size?: number;
}

export async function fetchAiInsights(params: AiInsightParams = {}): Promise<InsightListResponse> {
  const qs = new URLSearchParams();
  if (params.severity) qs.set("severity", params.severity);
  if (params.status) qs.set("status", params.status);
  if (params.insight_type) qs.set("insight_type", params.insight_type);
  if (params.item_no) qs.set("item_no", params.item_no);
  if (params.loc) qs.set("loc", params.loc);
  if (params.page) qs.set("page", String(params.page));
  if (params.page_size) qs.set("page_size", String(params.page_size));
  return fetchJson(`/ai-planner/insights?${qs}`);
}

export async function fetchAiMemos(params: { scope?: string; limit?: number } = {}): Promise<MemoListResponse> {
  const qs = new URLSearchParams();
  if (params.scope) qs.set("scope", params.scope);
  if (params.limit) qs.set("limit", String(params.limit));
  return fetchJson(`/ai-planner/memos?${qs}`);
}

export async function triggerPortfolioScan(): Promise<{ job_id: string; status: string }> {
  return submitJob("generate_ai_insights", {}, "AI Portfolio Scan");
}

export async function triggerDfuAnalyze(item_no: string, loc: string): Promise<AnalyzeResponse> {
  return fetchJson("/ai-planner/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ item_no, loc }),
  });
}

export async function updateInsightStatus(
  insight_id: number,
  status: InsightStatus,
): Promise<{ insight_id: number; status: string }> {
  return fetchJson(`/ai-planner/insights/${insight_id}/status`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status }),
  });
}
