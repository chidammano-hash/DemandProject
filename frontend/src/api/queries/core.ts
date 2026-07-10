import type {
  ShapFilterParams,
} from "@/types/shap";

// ---------------------------------------------------------------------------
// Query key factories
// ---------------------------------------------------------------------------
export const queryKeys = {
  domains: () => ["domains"] as const,
  domainMeta: (domain: string) => ["domain-meta", domain] as const,
  domainPage: (domain: string, params: Record<string, unknown>) => ["domain-page", domain, params] as const,
  domainSuggest: (domain: string, field: string, q: string, filters?: string) => ["domain-suggest", domain, field, q, filters] as const,
  forecastModels: () => ["forecast-models"] as const,
  skuClusters: (source: string) => ["sku-clusters", source] as const,
  clusterProfiles: () => ["cluster-profiles"] as const,
  clusteringDefaults: () => ["clustering-defaults"] as const,
  clusteringScenario: (id: string) => ["clustering-scenario", id] as const,
  scenarioEstimate: (params: Record<string, unknown>) => ["scenario-estimate", params] as const,
  scenarioStatus: (id: string) => ["scenario-status", id] as const,
  samplePair: (domain: string) => ["sample-pair", domain] as const,
  accuracySlice: (params: Record<string, unknown>) => ["accuracy-slice", params] as const,
  lagCurve: (params: Record<string, unknown>) => ["lag-curve", params] as const,
  competitionConfig: () => ["competition-config"] as const,
  competitionSummary: () => ["competition-summary"] as const,
  skuAnalysis: (params: Record<string, unknown>) => ["sku-analysis", params] as const,
  inventoryPosition: (params: Record<string, unknown>) => ["inventory-position", params] as const,
  inventoryKpis: (params: Record<string, unknown>) => ["inventory-kpis", params] as const,
  inventoryTrend: (params: Record<string, unknown>) => ["inventory-trend", params] as const,
  inventoryItemDetail: (params: Record<string, unknown>) => ["inventory-item-detail", params] as const,
  // Inventory backtest keys (Feature 37)
  invBacktestSummary: (params: Record<string, unknown>) => ["inv-backtest-summary", params] as const,
  invBacktestTrend: (params: Record<string, unknown>) => ["inv-backtest-trend", params] as const,
  invBacktestRootCause: (params: Record<string, unknown>) => ["inv-backtest-root-cause", params] as const,
  invBacktestDetail: (params: Record<string, unknown>) => ["inv-backtest-detail", params] as const,
  // Dashboard & filter keys (Feature 36)
  planningDate: () => ["planning-date"] as const,
  distinctValues: (domain: string, column: string) => ["distinct-values", domain, column] as const,
  dashboardKpis: (params: Record<string, unknown>) => ["dashboard-kpis", params] as const,
  dashboardAlerts: (params: Record<string, unknown>) => ["dashboard-alerts", params] as const,
  dashboardTopMovers: (params: Record<string, unknown>) => ["dashboard-top-movers", params] as const,
  dashboardHeatmap: (params: Record<string, unknown>) => ["dashboard-heatmap", params] as const,
  dashboardTrend: (params: Record<string, unknown>) => ["dashboard-trend", params] as const,
  customerMap: (groupBy: string) => ["customer-map", groupBy] as const,
  // Job scheduler keys (Feature 39)
  jobTypes: () => ["job-types"] as const,
  jobsAll: () => ["jobs"] as const,
  jobs: (params: Record<string, unknown>) => ["jobs", params] as const,
  jobDetail: (id: string) => ["job-detail", id] as const,
  activeJobs: () => ["active-jobs"] as const,
  jobStats: () => ["job-stats"] as const,
  jobLogs: (id: string) => ["job-logs", id] as const,
  jobSchedules: () => ["job-schedules"] as const,
  scenarioHistory: () => ["scenario-history"] as const,
  // Inventory Planning keys (IPfeature1+)
  variabilitySummary: (params: Record<string, unknown>) => ["variability-summary", params] as const,
  variabilityDetail: (params: Record<string, unknown>) => ["variability-detail", params] as const,
  ltSummary: (params: Record<string, unknown>) => ["lt-summary", params] as const,
  ltProfile: (params: Record<string, unknown>) => ["lt-profile", params] as const,
  eoqSummary: (params: Record<string, unknown>) => ["eoq-summary", params] as const,
  eoqDetail: (params: Record<string, unknown>) => ["eoq-detail", params] as const,
  eoqSensitivity: (params: Record<string, unknown>) => ["eoq-sensitivity", params] as const,
  // IPfeature5 — Replenishment Policy Management
  policyList: (f?: Record<string, unknown>) => ["policy-list", f ?? {}] as const,
  policyAssignments: (params: Record<string, unknown>) => ["policy-assignments", params] as const,
  policyCompliance: (f?: Record<string, unknown>) => ["policy-compliance", f ?? {}] as const,
  // SHAP feature importance keys (Feature 42)
  shapModels: () => ["shap-models"] as const,
  shapSummary: (modelId: string, topN: number, filters?: ShapFilterParams) => ["shap-summary", modelId, topN, filters ?? {}] as const,
  shapTimeframes: (modelId: string) => ["shap-timeframes", modelId] as const,
  shapTimeframeDetail: (modelId: string, idx: number, topN: number, cluster?: string, filters?: ShapFilterParams) => ["shap-timeframe-detail", modelId, idx, topN, cluster ?? "all", filters ?? {}] as const,
  shapClusters: (modelId: string) => ["shap-clusters", modelId] as const,
  skuShap: (modelId: string, itemNo: string, loc: string, topN: number) => ["sku-shap", modelId, itemNo, loc, topN] as const,
  // AI Planner keys (IPAIfeature1)
  aiInsights: (params: Record<string, unknown>) => ["ai-insights", params] as const,
  aiMemos: (params: Record<string, unknown>) => ["ai-memos", params] as const,
  // Production Forecast keys (F1.1)
  productionForecast: (params: Record<string, unknown>) => ["production-forecast", params] as const,
  productionForecastSummary: (params: Record<string, unknown>) => ["production-forecast-summary", params] as const,
  productionForecastVersions: () => ["production-forecast-versions"] as const,
  // Open PO Integration keys (F1.3)
  openPOs: (params: Record<string, unknown>) => ["open-pos", params] as const,
  openPOSummary: () => ["open-po-summary"] as const,
  pastDuePOs: () => ["past-due-pos"] as const,
  // Order Recommendation keys (F2.1)
  plannedOrders: (params: Record<string, unknown>) => ["planned-orders", params] as const,
  plannedOrdersSummary: () => ["planned-orders-summary"] as const,
};

// ---------------------------------------------------------------------------
// Stale time constants (ms)
// ---------------------------------------------------------------------------
export const STALE = {
  FOREVER: Infinity,
  TEN_MIN: 10 * 60_000,
  FIVE_MIN: 5 * 60_000,
  TWO_MIN: 2 * 60_000,
  ONE_MIN: 60_000,
  THIRTY_SEC: 30_000,
  NONE: 0,
} as const;

function formatErrorDetail(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const messages = detail
      .filter((entry): entry is { loc?: unknown; msg?: unknown } => Boolean(entry) && typeof entry === "object")
      .map((entry) => {
        const location = Array.isArray(entry.loc) ? entry.loc.join(".") : "request";
        const message = typeof entry.msg === "string" ? entry.msg : "Invalid value";
        return `${location}: ${message}`;
      });
    return messages.join("; ") || "Request validation failed";
  }
  return "Request failed";
}

// ---------------------------------------------------------------------------
// Fetch helpers
// ---------------------------------------------------------------------------
export async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    // Parse FastAPI `{detail}` so formatApiError can sanitize a clean message,
    // and attach the HTTP status so the global handler maps it to friendly copy
    // (404 → "That record could not be found.") instead of leaking the raw body.
    let detail: unknown = text;
    try {
      detail = JSON.parse(text);
    } catch {
      /* non-JSON body — keep the raw text */
    }
    const message = detail && typeof detail === "object" && "detail" in detail
      ? formatErrorDetail((detail as { detail: unknown }).detail)
      : text || `HTTP ${res.status}`;
    const err = new Error(message);
    Object.assign(err, { status: res.status, detail });
    throw err;
  }
  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Re-exports from domain-specific modules (backward compatibility)
// ---------------------------------------------------------------------------
export { fetchDomains, fetchDomainMeta, fetchDomainPage, fetchDomainSuggest, fetchSamplePair, fetchForecastModels } from "./domains";
export type { PageParams } from "./domains";

export {
  fetchSkuClusters, fetchClusterProfiles, fetchClusteringDefaults,
  fetchClusterCoreFeatures,
  fetchScenarioEstimate, fetchScenarioStatus, runClusteringScenario, promoteScenario,
} from "./clustering";
export type {
  ClusteringDefaultsPayload, ClusterCoreFeaturesPayload, ClusteringScenarioParams, ScenarioProfile,
  PCAScatterPoint, PCAScatterData, ClusteringScenarioResult,
  ScenarioEstimate, ScenarioStatusResponse,
} from "./clustering";

export { fetchAccuracySlice, fetchLagCurve, SLICE_DEFAULT_LIMIT } from "./accuracy";
export type { SliceParams, LagCurveParams } from "./accuracy";

export {
  fetchCompetitionConfig, fetchCompetitionSummary,
  saveCompetitionConfig, runCompetition,
} from "./competition";
export type { CompetitionConfig, ChampionSummary } from "./competition";

export { fetchSkuAnalysis, fetchMarketIntel } from "./sku-analysis";
export type { SkuAnalysisParams } from "./sku-analysis";

export {
  fetchInventoryPosition, fetchInventoryKpis,
  fetchInventoryTrend, fetchInventoryItemDetail,
} from "./inventory";
export type { InventoryPositionParams } from "./inventory";

export {
  fetchDistinctValues, fetchPlanningDate, fetchDashboardKpis,
  fetchDashboardAlerts, fetchDashboardTopMovers, fetchDashboardHeatmap,
  fetchDashboardTrend, fetchCustomerMap,
  fetchPipelineReadiness, pipelineReadinessKeys,
} from "./dashboard";
export type {
  CascadeFilterParams, DashboardFilterParams,
  PlanningDateInfo, TrendPoint, HeatmapGrain, CustomerMapLocation,
  PipelineReadiness, PipelineReadinessCheck, PipelineReadinessAction,
} from "./dashboard";

export {
  fetchInvBacktestSummary, fetchInvBacktestTrend,
  fetchInvBacktestRootCause, fetchInvBacktestDetail,
} from "./inventory-backtest";
export type { InvBacktestFilterParams, InvBacktestDetailParams } from "./inventory-backtest";

export {
  fetchSeasonalityProfiles, fetchSeasonalityProfileNames,
  seasonalityProfileKeys,
} from "./seasonality";
export type { SeasonalityProfilesPayload } from "./seasonality";

export {
  fetchJobTypes, fetchJobs, fetchJobDetail, fetchActiveJobs,
  submitJob, cancelJob, deleteJob, fetchScenarioHistory,
  fetchJobLogs, fetchJobStats, fetchJobSchedules,
  createSchedule, deleteSchedule, submitPipeline,
  planOperationalWorkflows, runNamedPipeline,
} from "./jobs";
export type {
  Job, JobType, JobListPayload, JobTypesPayload, ActiveJobsPayload,
  JobStats, JobSchedule, JobSchedulesPayload, JobLogsPayload,
  WorkflowPlanAnswer, WorkflowQuestion, WorkflowStep,
  WorkflowRecommendation, WorkflowPlan,
} from "./jobs";

export {
  fetchShapModels, fetchShapSummary, fetchShapTimeframes,
  fetchShapTimeframeDetail, fetchShapClusters, fetchSkuShap,
} from "./shap";
export type { ShapClustersPayload } from "./shap";
