// IPAIfeature1 — AI Planning Agent types

export type InsightSeverity = "critical" | "high" | "medium" | "low";
export type InsightStatus = "open" | "acknowledged" | "resolved";
export type InsightType =
  | "stockout_risk"
  | "excess_inventory"
  | "forecast_bias"
  | "policy_gap"
  | "champion_degradation";

export interface AiInsight {
  insight_id: number;
  insight_type: InsightType;
  severity: InsightSeverity;
  item_id: string;
  loc: string;
  abc_vol: string | null;
  cluster_assignment: string | null;
  summary: string;
  recommendation: string;
  reasoning: string | null;
  financial_impact_estimate: number | null;
  dos: number | null;
  total_lt_days: number | null;
  champion_wape: number | null;
  forecast_bias_pct: number | null;
  current_policy_id: string | null;
  eoq_effective: number | null;
  status: InsightStatus;
  acknowledged_at: string | null;
  resolved_at: string | null;
  model_version: string | null;
  scan_run_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface AiPlanningMemo {
  memo_id: number;
  period: string;
  scope: "portfolio" | "sku";
  item_id: string | null;
  loc: string | null;
  narrative_text: string;
  content_json: Record<string, unknown>;
  model_version: string | null;
  created_at: string;
}

export interface InsightListResponse {
  insights: AiInsight[];
  total: number;
  page: number;
  page_size: number;
}

export interface MemoListResponse {
  memos: AiPlanningMemo[];
}

export interface PortfolioScanResponse {
  status: string;
  scan_run_id: string;
  message: string;
}

export interface AnalyzeResponse {
  insights: AiInsight[];
  scan_run_id: string;
}
