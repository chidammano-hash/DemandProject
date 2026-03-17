import type {
  InsightListResponse,
  MemoListResponse,
  AnalyzeResponse,
  InsightSeverity,
  InsightStatus,
  InsightType,
} from "@/types/ai-planner";
import { fetchJson, submitJob } from "./core";
import { buildSearchParams } from "./helpers";

// ---------------------------------------------------------------------------
// AI Planner queries (IPAIfeature1)
// ---------------------------------------------------------------------------
export interface AiInsightParams {
  severity?: InsightSeverity;
  status?: InsightStatus;
  insight_type?: InsightType;
  item_no?: string;
  loc?: string;
  brand?: string;
  category?: string;
  market?: string;
  channel?: string;
  page?: number;
  page_size?: number;
}

export async function fetchAiInsights(params: AiInsightParams = {}): Promise<InsightListResponse> {
  const qs = buildSearchParams({
    severity: params.severity,
    status: params.status,
    insight_type: params.insight_type,
    item_no: params.item_no,
    loc: params.loc,
    brand: params.brand,
    category: params.category,
    market: params.market,
    channel: params.channel,
    page: params.page,
    page_size: params.page_size,
  });
  return fetchJson(`/ai-planner/insights?${qs}`);
}

export async function fetchAiMemos(params: { scope?: string; limit?: number } = {}): Promise<MemoListResponse> {
  const qs = buildSearchParams({
    scope: params.scope,
    limit: params.limit,
  });
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
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status }),
  });
}

export interface AutoAcceptRequest {
  min_severity: InsightSeverity;
  insight_types: InsightType[];
  dry_run: boolean;
}

export interface AutoAcceptResponse {
  accepted: number;
  dry_run: boolean;
  insight_ids: number[];
}

export async function triggerAutoAccept(req: AutoAcceptRequest): Promise<AutoAcceptResponse> {
  return fetchJson("/ai-planner/auto-accept", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

// PL-012: Snooze an insight for N days
export async function snoozeInsight(
  insight_id: number,
  days: number,
): Promise<{ insight_id: number; status: string; snoozed_until: string | null }> {
  return fetchJson(`/ai-planner/insights/${insight_id}/snooze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ days }),
  });
}
