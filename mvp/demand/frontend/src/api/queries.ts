import type {
  DomainMeta,
  DomainPage,
  SuggestPayload,
  SamplePairPayload,
  DfuClustersPayload,
  ClusterProfilesPayload,
  AccuracySlicePayload,
  LagCurvePayload,
  DfuAnalysisPayload,
  DfuAnalysisMode,
  MarketIntelPayload,
  InventoryPositionPayload,
  InventoryKpis,
  InventoryTrendPayload,
  InventoryItemDetailPayload,
  InvBacktestSummaryPayload,
  InvBacktestTrendPayload,
  InvBacktestRootCausePayload,
  InvBacktestDetailPayload,
} from "@/types";
import type {
  InsightListResponse,
  MemoListResponse,
  AnalyzeResponse,
  InsightSeverity,
  InsightStatus,
  InsightType,
} from "@/types/ai_planner";
import type {
  ShapModelsPayload,
  ShapSummaryPayload,
  ShapTimeframesPayload,
  ShapTimeframeDetailPayload,
} from "@/types/shap";
import type {
  DashboardKpis,
  Alert,
  Mover,
  HeatmapRow,
  DistinctValuesPayload,
} from "@/types/theme";

// ---------------------------------------------------------------------------
// Query key factories
// ---------------------------------------------------------------------------
export const queryKeys = {
  domains: () => ["domains"] as const,
  domainMeta: (domain: string) => ["domain-meta", domain] as const,
  domainPage: (domain: string, params: Record<string, unknown>) => ["domain-page", domain, params] as const,
  domainSuggest: (domain: string, field: string, q: string, filters?: string) => ["domain-suggest", domain, field, q, filters] as const,
  forecastModels: () => ["forecast-models"] as const,
  dfuClusters: (source: string) => ["dfu-clusters", source] as const,
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
  dfuAnalysis: (params: Record<string, unknown>) => ["dfu-analysis", params] as const,
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
  distinctValues: (domain: string, column: string) => ["distinct-values", domain, column] as const,
  dashboardKpis: (params: Record<string, unknown>) => ["dashboard-kpis", params] as const,
  dashboardAlerts: (params: Record<string, unknown>) => ["dashboard-alerts", params] as const,
  dashboardTopMovers: (params: Record<string, unknown>) => ["dashboard-top-movers", params] as const,
  dashboardHeatmap: (params: Record<string, unknown>) => ["dashboard-heatmap", params] as const,
  // Job scheduler keys (Feature 39)
  jobTypes: () => ["job-types"] as const,
  jobs: (params: Record<string, unknown>) => ["jobs", params] as const,
  jobDetail: (id: string) => ["job-detail", id] as const,
  activeJobs: () => ["active-jobs"] as const,
  jobStats: () => ["job-stats"] as const,
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
  policyList: () => ["policy-list"] as const,
  policyAssignments: (params: Record<string, unknown>) => ["policy-assignments", params] as const,
  policyCompliance: () => ["policy-compliance"] as const,
  // SHAP feature importance keys (Feature 42)
  shapModels: () => ["shap-models"] as const,
  shapSummary: (modelId: string, topN: number) => ["shap-summary", modelId, topN] as const,
  shapTimeframes: (modelId: string) => ["shap-timeframes", modelId] as const,
  shapTimeframeDetail: (modelId: string, idx: number, topN: number) => ["shap-timeframe-detail", modelId, idx, topN] as const,
  // AI Planner keys (IPAIfeature1)
  aiInsights: (params: Record<string, unknown>) => ["ai-insights", params] as const,
  aiMemos: (params: Record<string, unknown>) => ["ai-memos", params] as const,
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

// ---------------------------------------------------------------------------
// Fetch helpers
// ---------------------------------------------------------------------------
async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Domain queries
// ---------------------------------------------------------------------------
export async function fetchDomains(): Promise<{ domains: string[] }> {
  return fetchJson("/domains");
}

export async function fetchDomainMeta(domain: string): Promise<DomainMeta> {
  return fetchJson(`/domains/${encodeURIComponent(domain)}/meta`);
}

export interface PageParams {
  limit: number;
  offset: number;
  q: string;
  sort_by: string;
  sort_dir: string;
  filters?: Record<string, string>;
}

export async function fetchDomainPage(domain: string, params: PageParams): Promise<DomainPage> {
  const qs = new URLSearchParams({
    limit: String(params.limit),
    offset: String(params.offset),
    q: params.q,
    sort_by: params.sort_by,
    sort_dir: params.sort_dir,
  });
  if (params.filters && Object.keys(params.filters).length > 0) {
    qs.set("filters", JSON.stringify(params.filters));
  }
  return fetchJson(`/domains/${encodeURIComponent(domain)}/page?${qs}`);
}

export async function fetchDomainSuggest(
  domain: string,
  field: string,
  q: string,
  filters?: Record<string, string>,
  limit = 12,
): Promise<string[]> {
  const qs = new URLSearchParams({ field, q, limit: String(limit) });
  if (filters && Object.keys(filters).length > 0) {
    qs.set("filters", JSON.stringify(filters));
  }
  const payload = await fetchJson<SuggestPayload>(`/domains/${encodeURIComponent(domain)}/suggest?${qs}`);
  return Array.from(new Set((payload.values || []).filter(Boolean))).slice(0, limit);
}

export async function fetchSamplePair(domain: string): Promise<SamplePairPayload> {
  return fetchJson(`/domains/${encodeURIComponent(domain)}/sample-pair`);
}

// ---------------------------------------------------------------------------
// Forecast / model queries
// ---------------------------------------------------------------------------
export async function fetchForecastModels(): Promise<string[]> {
  const payload = await fetchJson<{ models?: string[] }>("/domains/forecast/models");
  return payload.models || [];
}

// ---------------------------------------------------------------------------
// Clustering queries
// ---------------------------------------------------------------------------
export async function fetchDfuClusters(source: string): Promise<DfuClustersPayload> {
  return fetchJson(`/domains/dfu/clusters?source=${source}`);
}

export async function fetchClusterProfiles(): Promise<ClusterProfilesPayload> {
  return fetchJson("/domains/dfu/clusters/profiles");
}

export interface ClusteringDefaultsPayload {
  feature_params: { time_window_months: number; min_months_history: number };
  model_params: {
    k_range: number[];
    min_cluster_size_pct: number;
    use_pca: boolean;
    pca_components: number | null;
    skip_gap: boolean;
    all_features: boolean;
  };
  label_params: {
    volume_high: number;
    volume_low: number;
    cv_steady: number;
    cv_volatile: number;
    seasonality_threshold: number;
    zero_demand_threshold: number;
  };
}

export async function fetchClusteringDefaults(): Promise<ClusteringDefaultsPayload> {
  return fetchJson("/clustering/defaults");
}

export interface ClusteringScenarioParams {
  feature_params?: ClusteringDefaultsPayload["feature_params"];
  model_params?: ClusteringDefaultsPayload["model_params"];
  label_params?: ClusteringDefaultsPayload["label_params"];
  relabel_only?: boolean;
  previous_scenario_id?: string;
}

export interface ScenarioProfile {
  label: string;
  count: number;
  pct_of_total: number;
  mean_demand: number;
  cv_demand: number;
  seasonality_strength: number;
  trend_slope: number;
  growth_rate: number;
  zero_demand_pct: number;
}

export interface ClusteringScenarioResult {
  scenario_id: string;
  status: "completed" | "failed";
  runtime_seconds: number;
  params: Record<string, unknown>;
  result: {
    optimal_k: number;
    silhouette_score: number;
    inertia: number;
    total_dfus: number;
    k_selection_results: {
      k_values: number[];
      inertias: number[];
      silhouette_scores: number[];
      gap_stats?: number[] | null;
    };
    profiles: ScenarioProfile[];
    feature_importance?: { feature: string; variance_ratio: number }[];
  } | null;
  error?: string | null;
}

export interface ScenarioEstimate {
  estimated_seconds: number;
  dfu_count: number;
  training_sample: number;
  sampled: boolean;
  k_range: number;
  skip_gap: boolean;
}

export interface ScenarioStatusResponse {
  scenario_id: string;
  status: "running" | "completed" | "failed";
  elapsed_seconds?: number;
  runtime_seconds?: number;
  result?: ClusteringScenarioResult;
  error?: string;
}

export async function fetchScenarioEstimate(params: {
  k_min: number;
  k_max: number;
  skip_gap: boolean;
}): Promise<ScenarioEstimate> {
  const qs = new URLSearchParams({
    k_min: String(params.k_min),
    k_max: String(params.k_max),
    skip_gap: String(params.skip_gap),
  });
  return fetchJson(`/clustering/scenario/estimate?${qs}`);
}

export async function fetchScenarioStatus(scenarioId: string): Promise<ScenarioStatusResponse> {
  return fetchJson(`/clustering/scenario/${encodeURIComponent(scenarioId)}/status`);
}

export async function runClusteringScenario(
  params: ClusteringScenarioParams,
): Promise<{ scenario_id: string; status: string; job_id?: string }> {
  return fetchJson("/clustering/scenario", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
}

export async function promoteScenario(scenarioId: string): Promise<unknown> {
  return fetchJson(`/clustering/scenario/${encodeURIComponent(scenarioId)}/promote`, {
    method: "POST",
  });
}

// ---------------------------------------------------------------------------
// Accuracy queries
// ---------------------------------------------------------------------------
export interface SliceParams {
  group_by: string;
  lag: number;
  models: string;
  month_from: string;
  common_dfus: boolean;
  include_dfu_count: boolean;
  item?: string;
  location?: string;
  seasonality_profile?: string;
}

export async function fetchAccuracySlice(params: SliceParams): Promise<AccuracySlicePayload> {
  const qs = new URLSearchParams({ group_by: params.group_by, lag: String(params.lag) });
  if (params.models.trim()) qs.set("models", params.models.trim());
  if (params.month_from) qs.set("month_from", params.month_from);
  if (params.common_dfus) qs.set("common_dfus", "true");
  if (params.include_dfu_count) qs.set("include_dfu_count", "true");
  if (params.item) qs.set("item", params.item);
  if (params.location) qs.set("location", params.location);
  if (params.seasonality_profile) qs.set("seasonality_profile", params.seasonality_profile);
  return fetchJson(`/forecast/accuracy/slice?${qs}`);
}

export interface LagCurveParams {
  models: string;
  month_from: string;
  common_dfus: boolean;
  include_dfu_count: boolean;
  item?: string;
  location?: string;
  seasonality_profile?: string;
}

export async function fetchLagCurve(params: LagCurveParams): Promise<LagCurvePayload> {
  const qs = new URLSearchParams();
  if (params.models.trim()) qs.set("models", params.models.trim());
  if (params.month_from) qs.set("month_from", params.month_from);
  if (params.common_dfus) qs.set("common_dfus", "true");
  if (params.include_dfu_count) qs.set("include_dfu_count", "true");
  if (params.item) qs.set("item", params.item);
  if (params.location) qs.set("location", params.location);
  if (params.seasonality_profile) qs.set("seasonality_profile", params.seasonality_profile);
  return fetchJson(`/forecast/accuracy/lag-curve?${qs}`);
}

// ---------------------------------------------------------------------------
// Competition queries
// ---------------------------------------------------------------------------
export interface CompetitionConfig {
  metric: string;
  lag: string;
  min_dfu_rows: number;
  champion_model_id: string;
  models: string[];
  strategy?: string;
  strategy_params?: Record<string, unknown>;
}

export interface ChampionSummary {
  total_dfus: number;
  total_dfu_months?: number;
  total_champion_rows: number;
  model_wins: Record<string, number>;
  overall_champion_wape: number | null;
  overall_champion_accuracy_pct: number | null;
  run_ts: string;
  total_ceiling_rows?: number;
  ceiling_model_wins?: Record<string, number>;
  overall_ceiling_wape?: number | null;
  overall_ceiling_accuracy_pct?: number | null;
}

export async function fetchCompetitionConfig(): Promise<{ config: CompetitionConfig; available_models: string[] } | null> {
  try {
    return await fetchJson("/competition/config");
  } catch { return null; }
}

export async function fetchCompetitionSummary(): Promise<{ summary: ChampionSummary } | null> {
  try {
    return await fetchJson("/competition/summary");
  } catch { return null; }
}

export async function saveCompetitionConfig(config: CompetitionConfig): Promise<void> {
  await fetchJson("/competition/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
}

export async function runCompetition(): Promise<ChampionSummary> {
  return fetchJson("/competition/run", { method: "POST" });
}

// ---------------------------------------------------------------------------
// DFU Analysis queries
// ---------------------------------------------------------------------------
export interface DfuAnalysisParams {
  mode: DfuAnalysisMode;
  item: string;
  location: string;
  points: number;
}

export async function fetchDfuAnalysis(params: DfuAnalysisParams): Promise<DfuAnalysisPayload> {
  const qs = new URLSearchParams({
    mode: params.mode,
    item: params.item.trim(),
    location: params.location.trim(),
    points: String(params.points),
  });
  return fetchJson(`/dfu/analysis?${qs}`);
}

// ---------------------------------------------------------------------------
// Market Intelligence
// ---------------------------------------------------------------------------
export async function fetchMarketIntel(item: string, location: string): Promise<MarketIntelPayload> {
  return fetchJson("/market-intelligence", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ item_no: item.trim(), location_id: location.trim() }),
  });
}

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

// ---------------------------------------------------------------------------
// Inventory queries
// ---------------------------------------------------------------------------
export interface InventoryPositionParams {
  item?: string;
  location?: string;
  limit: number;
  offset: number;
  sort_by: string;
  sort_dir: string;
}

export async function fetchInventoryPosition(params: InventoryPositionParams): Promise<InventoryPositionPayload> {
  const qs = new URLSearchParams({
    limit: String(params.limit),
    offset: String(params.offset),
    sort_by: params.sort_by,
    sort_dir: params.sort_dir,
  });
  if (params.item?.trim()) qs.set("item", params.item.trim());
  if (params.location?.trim()) qs.set("location", params.location.trim());
  return fetchJson(`/inventory/position?${qs}`);
}

export async function fetchInventoryKpis(params: { item?: string; location?: string; months?: number }): Promise<InventoryKpis> {
  const qs = new URLSearchParams();
  if (params.item?.trim()) qs.set("item", params.item.trim());
  if (params.location?.trim()) qs.set("location", params.location.trim());
  if (params.months) qs.set("months", String(params.months));
  return fetchJson(`/inventory/kpis?${qs}`);
}

export async function fetchInventoryTrend(params: { item?: string; location?: string; months?: number }): Promise<InventoryTrendPayload> {
  const qs = new URLSearchParams();
  if (params.item?.trim()) qs.set("item", params.item.trim());
  if (params.location?.trim()) qs.set("location", params.location.trim());
  if (params.months) qs.set("months", String(params.months));
  return fetchJson(`/inventory/trend?${qs}`);
}

export async function fetchInventoryItemDetail(params: { item: string; location: string; months?: number }): Promise<InventoryItemDetailPayload> {
  const qs = new URLSearchParams({ item: params.item.trim(), location: params.location.trim() });
  if (params.months) qs.set("months", String(params.months));
  return fetchJson(`/inventory/item-detail?${qs}`);
}

// ---------------------------------------------------------------------------
// Distinct values (Feature 36 — global filter dropdowns)
// ---------------------------------------------------------------------------
export async function fetchDistinctValues(
  domain: string,
  column: string,
  search?: string,
  limit = 100,
): Promise<DistinctValuesPayload> {
  const qs = new URLSearchParams({ column, limit: String(limit) });
  if (search) qs.set("search", search);
  return fetchJson(`/domains/${encodeURIComponent(domain)}/distinct?${qs}`);
}

// ---------------------------------------------------------------------------
// Dashboard queries (Feature 36)
// ---------------------------------------------------------------------------
export interface DashboardFilterParams {
  brand?: string[];
  category?: string[];
  market?: string[];
  channel?: string[];
  item?: string[];
  location?: string[];
}

function appendFilterParams(qs: URLSearchParams, params?: DashboardFilterParams) {
  if (!params) return;
  if (params.brand?.length) qs.set("brand", params.brand.join(","));
  if (params.category?.length) qs.set("category", params.category.join(","));
  if (params.market?.length) qs.set("market", params.market.join(","));
  if (params.channel?.length) qs.set("channel", params.channel.join(","));
  if (params.item?.length) qs.set("item", params.item.join(","));
  if (params.location?.length) qs.set("location", params.location.join(","));
}

export async function fetchDashboardKpis(
  window = 3,
  filters?: DashboardFilterParams,
): Promise<DashboardKpis> {
  const qs = new URLSearchParams({ window: String(window) });
  appendFilterParams(qs, filters);
  return fetchJson(`/dashboard/kpis?${qs}`);
}

export async function fetchDashboardAlerts(
  limit = 10,
  filters?: DashboardFilterParams,
): Promise<{ alerts: Alert[] }> {
  const qs = new URLSearchParams({ limit: String(limit) });
  appendFilterParams(qs, filters);
  return fetchJson(`/dashboard/alerts?${qs}`);
}

export async function fetchDashboardTopMovers(
  limit = 5,
  direction: "up" | "down" | "both" = "both",
  filters?: DashboardFilterParams,
): Promise<{ movers: Mover[] }> {
  const qs = new URLSearchParams({ limit: String(limit), direction });
  appendFilterParams(qs, filters);
  return fetchJson(`/dashboard/top-movers?${qs}`);
}

export async function fetchDashboardHeatmap(
  grain: "category" | "brand" | "location" = "category",
  periods = 4,
  filters?: DashboardFilterParams,
): Promise<{ rows: HeatmapRow[]; period_labels: string[]; metric: string }> {
  const qs = new URLSearchParams({ grain, periods: String(periods) });
  appendFilterParams(qs, filters);
  return fetchJson(`/dashboard/heatmap?${qs}`);
}

// ---------------------------------------------------------------------------
// Inventory Backtest queries (Feature 37)
// ---------------------------------------------------------------------------
export interface InvBacktestFilterParams {
  models?: string;
  month_from?: string;
  month_to?: string;
  item?: string;
  location?: string;
  cluster_assignment?: string;
  abc_vol?: string;
  region?: string;
  excess_dos_threshold?: number;
}

function appendInvBacktestParams(qs: URLSearchParams, params: InvBacktestFilterParams) {
  if (params.models?.trim()) qs.set("models", params.models.trim());
  if (params.month_from?.trim()) qs.set("month_from", params.month_from.trim());
  if (params.month_to?.trim()) qs.set("month_to", params.month_to.trim());
  if (params.item?.trim()) qs.set("item", params.item.trim());
  if (params.location?.trim()) qs.set("location", params.location.trim());
  if (params.cluster_assignment?.trim()) qs.set("cluster_assignment", params.cluster_assignment.trim());
  if (params.abc_vol?.trim()) qs.set("abc_vol", params.abc_vol.trim());
  if (params.region?.trim()) qs.set("region", params.region.trim());
  if (params.excess_dos_threshold != null) qs.set("excess_dos_threshold", String(params.excess_dos_threshold));
}

export async function fetchInvBacktestSummary(params: InvBacktestFilterParams): Promise<InvBacktestSummaryPayload> {
  const qs = new URLSearchParams();
  appendInvBacktestParams(qs, params);
  return fetchJson(`/inventory-backtest/summary?${qs}`);
}

export async function fetchInvBacktestTrend(params: InvBacktestFilterParams): Promise<InvBacktestTrendPayload> {
  const qs = new URLSearchParams();
  appendInvBacktestParams(qs, params);
  return fetchJson(`/inventory-backtest/trend?${qs}`);
}

export async function fetchInvBacktestRootCause(
  params: InvBacktestFilterParams & { model_id: string },
): Promise<InvBacktestRootCausePayload> {
  const qs = new URLSearchParams({ model_id: params.model_id });
  appendInvBacktestParams(qs, params);
  return fetchJson(`/inventory-backtest/root-cause?${qs}`);
}

export interface InvBacktestDetailParams extends InvBacktestFilterParams {
  event_type?: string;
  limit: number;
  offset: number;
  sort_by: string;
  sort_dir: string;
}

export async function fetchInvBacktestDetail(params: InvBacktestDetailParams): Promise<InvBacktestDetailPayload> {
  const qs = new URLSearchParams({
    limit: String(params.limit),
    offset: String(params.offset),
    sort_by: params.sort_by,
    sort_dir: params.sort_dir,
  });
  if (params.event_type && params.event_type !== "all") qs.set("event_type", params.event_type);
  appendInvBacktestParams(qs, params);
  return fetchJson(`/inventory-backtest/detail?${qs}`);
}

// ---------------------------------------------------------------------------
// Seasonality profiles
// ---------------------------------------------------------------------------
export interface SeasonalityProfilesPayload {
  profiles: { profile: string; count: number }[];
}

export async function fetchSeasonalityProfiles(): Promise<SeasonalityProfilesPayload> {
  return fetchJson("/domains/dfu/seasonality-profiles");
}

// ---------------------------------------------------------------------------
// Job scheduler queries (Feature 39)
// ---------------------------------------------------------------------------
import type {
  Job, JobType, JobListPayload, JobTypesPayload, ActiveJobsPayload,
  JobStats, JobSchedule, JobSchedulesPayload,
} from "@/types/jobs";

export type {
  Job, JobType, JobListPayload, JobTypesPayload, ActiveJobsPayload,
  JobStats, JobSchedule, JobSchedulesPayload,
};

export async function fetchJobTypes(): Promise<JobTypesPayload> {
  return fetchJson("/jobs/types");
}

export async function fetchJobs(params: {
  status?: string;
  job_type?: string;
  limit?: number;
  offset?: number;
}): Promise<JobListPayload> {
  const qs = new URLSearchParams();
  if (params.status) qs.set("status", params.status);
  if (params.job_type) qs.set("job_type", params.job_type);
  if (params.limit != null) qs.set("limit", String(params.limit));
  if (params.offset != null) qs.set("offset", String(params.offset));
  return fetchJson(`/jobs?${qs}`);
}

export async function fetchJobDetail(jobId: string): Promise<Job> {
  return fetchJson(`/jobs/${encodeURIComponent(jobId)}`);
}

export async function fetchActiveJobs(): Promise<ActiveJobsPayload> {
  return fetchJson("/jobs/active");
}

export async function submitJob(
  jobType: string,
  params: Record<string, unknown> = {},
  label?: string,
): Promise<{ job_id: string; status: string }> {
  return fetchJson("/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_type: jobType, params, label }),
  });
}

export async function cancelJob(jobId: string): Promise<{ job_id: string; status: string }> {
  return fetchJson(`/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
}

export async function deleteJob(jobId: string): Promise<{ deleted: boolean }> {
  return fetchJson(`/jobs/${encodeURIComponent(jobId)}`, { method: "DELETE" });
}

export async function fetchScenarioHistory(limit = 10): Promise<Job[]> {
  const data = await fetchJson<JobListPayload>(`/jobs?job_type=cluster_scenario&status=completed&limit=${limit}`);
  return data.jobs;
}

export async function fetchJobStats(): Promise<JobStats> {
  return fetchJson("/jobs/stats");
}

export async function fetchJobSchedules(): Promise<JobSchedulesPayload> {
  return fetchJson("/jobs/schedules");
}

export async function createSchedule(
  jobType: string,
  params: Record<string, unknown> = {},
  label?: string,
  cron?: string,
  intervalMinutes?: number,
): Promise<{ schedule_id: string; status: string }> {
  return fetchJson("/jobs/schedule", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      job_type: jobType,
      params,
      label,
      cron,
      interval_minutes: intervalMinutes,
    }),
  });
}

// ---------------------------------------------------------------------------
// SHAP feature importance queries (Feature 42)
// ---------------------------------------------------------------------------
export async function fetchShapModels(): Promise<ShapModelsPayload> {
  return fetchJson("/forecast/shap/models");
}

export async function fetchShapSummary(modelId: string, topN = 15): Promise<ShapSummaryPayload> {
  return fetchJson(`/forecast/shap/${encodeURIComponent(modelId)}/summary?top_n=${topN}`);
}

export async function fetchShapTimeframes(modelId: string): Promise<ShapTimeframesPayload> {
  return fetchJson(`/forecast/shap/${encodeURIComponent(modelId)}/timeframes`);
}

export async function fetchShapTimeframeDetail(
  modelId: string,
  idx: number,
  topN = 15,
): Promise<ShapTimeframeDetailPayload> {
  return fetchJson(`/forecast/shap/${encodeURIComponent(modelId)}/timeframe/${idx}?top_n=${topN}`);
}

export async function deleteSchedule(scheduleId: string): Promise<{ deleted: boolean }> {
  return fetchJson(`/jobs/schedules/${encodeURIComponent(scheduleId)}`, { method: "DELETE" });
}

export async function submitPipeline(
  steps: { job_type: string; params?: Record<string, unknown>; label?: string }[],
  label?: string,
): Promise<{ pipeline_id: string; status: string; steps: number }> {
  return fetchJson("/jobs/pipeline", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ steps, label }),
  });
}

// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature1: Demand Variability
// ---------------------------------------------------------------------------

export interface VariabilitySummaryPayload {
  total_dfus: number;
  by_class: { low: number; medium: number; high: number; lumpy: number };
  cv_percentiles: { p25: number | null; p50: number | null; p75: number | null; p95: number | null };
  avg_cv: number | null;
  avg_intermittency_ratio: number | null;
  top_volatile: {
    item_no: string;
    loc: string;
    abc_vol: string | null;
    cluster_assignment: string | null;
    demand_mean: number | null;
    demand_std: number | null;
    demand_cv: number | null;
    demand_mad: number | null;
    intermittency_ratio: number | null;
    variability_class: string | null;
  }[];
}

export interface VariabilityDetailRow {
  item_no: string;
  loc: string;
  abc_vol: string | null;
  cluster_assignment: string | null;
  demand_mean: number | null;
  demand_std: number | null;
  demand_cv: number | null;
  demand_mad: number | null;
  demand_p50: number | null;
  demand_p90: number | null;
  demand_skewness: number | null;
  demand_kurtosis: number | null;
  zero_demand_months: number | null;
  total_demand_months: number | null;
  intermittency_ratio: number | null;
  variability_class: string | null;
  demand_profile_ts: string | null;
}

export interface VariabilityDetailPayload {
  total: number;
  rows: VariabilityDetailRow[];
}

export async function fetchVariabilitySummary(params: {
  abc_vol?: string;
  cluster_assignment?: string;
}): Promise<VariabilitySummaryPayload> {
  const qs = new URLSearchParams();
  if (params.abc_vol?.trim()) qs.set("abc_vol", params.abc_vol.trim());
  if (params.cluster_assignment?.trim()) qs.set("cluster_assignment", params.cluster_assignment.trim());
  return fetchJson(`/inv-planning/variability/summary?${qs}`);
}

export async function fetchVariabilityDetail(params: {
  item?: string;
  location?: string;
  abc_vol?: string;
  variability_class?: string;
  limit?: number;
  offset?: number;
  sort_by?: string;
  sort_dir?: string;
}): Promise<VariabilityDetailPayload> {
  const qs = new URLSearchParams({
    limit: String(params.limit ?? 50),
    offset: String(params.offset ?? 0),
    sort_by: params.sort_by ?? "demand_cv",
    sort_dir: params.sort_dir ?? "desc",
  });
  if (params.item?.trim()) qs.set("item", params.item.trim());
  if (params.location?.trim()) qs.set("location", params.location.trim());
  if (params.abc_vol?.trim()) qs.set("abc_vol", params.abc_vol.trim());
  if (params.variability_class?.trim()) qs.set("variability_class", params.variability_class.trim());
  return fetchJson(`/inv-planning/variability/detail?${qs}`);
}

// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature2: Lead Time Variability
// ---------------------------------------------------------------------------

export interface LtSummaryPayload {
  total_profiles: number;
  by_class: { stable: number; moderate: number; volatile: number };
  avg_lt_cv: number | null;
  avg_lt_mean_days: number | null;
  lt_cv_p50: number | null;
  lt_cv_p95: number | null;
  top_volatile: {
    item_no: string;
    loc: string;
    lt_mean_days: number | null;
    lt_std_days: number | null;
    lt_cv: number | null;
    lt_min_days: number | null;
    lt_max_days: number | null;
    observation_count: number | null;
    lt_variability_class: string | null;
  }[];
}

export interface LtProfileRow {
  item_no: string;
  loc: string;
  lt_mean_days: number | null;
  lt_std_days: number | null;
  lt_cv: number | null;
  lt_min_days: number | null;
  lt_max_days: number | null;
  lt_p25_days: number | null;
  lt_p50_days: number | null;
  lt_p75_days: number | null;
  lt_p95_days: number | null;
  observation_count: number | null;
  observation_months: number | null;
  lt_variability_class: string | null;
  computed_at: string | null;
}

export interface LtProfilePayload {
  total: number;
  rows: LtProfileRow[];
}

export async function fetchLtSummary(params: {
  abc_vol?: string;
}): Promise<LtSummaryPayload> {
  const qs = new URLSearchParams();
  if (params.abc_vol?.trim()) qs.set("abc_vol", params.abc_vol.trim());
  return fetchJson(`/inv-planning/lead-time/summary?${qs}`);
}

export async function fetchLtProfile(params: {
  item?: string;
  location?: string;
  lt_variability_class?: string;
  limit?: number;
  offset?: number;
  sort_by?: string;
  sort_dir?: string;
}): Promise<LtProfilePayload> {
  const qs = new URLSearchParams({
    limit: String(params.limit ?? 50),
    offset: String(params.offset ?? 0),
    sort_by: params.sort_by ?? "lt_cv",
    sort_dir: params.sort_dir ?? "desc",
  });
  if (params.item?.trim()) qs.set("item", params.item.trim());
  if (params.location?.trim()) qs.set("location", params.location.trim());
  if (params.lt_variability_class?.trim()) qs.set("lt_variability_class", params.lt_variability_class.trim());
  return fetchJson(`/inv-planning/lead-time/profile?${qs}`);
}

// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature4: EOQ & Cycle Stock
// ---------------------------------------------------------------------------

export interface EoqAbcEntry {
  abc_vol: string;
  count: number;
  avg_eoq: number | null;
  total_cycle_stock: number | null;
  total_annual_cost: number | null;
  avg_order_frequency: number | null;
}

export interface EoqSummaryPayload {
  total_dfus: number;
  avg_effective_eoq: number | null;
  total_cycle_stock: number | null;
  avg_order_frequency: number | null;
  total_annual_cost: number | null;
  by_abc: EoqAbcEntry[];
}

export interface EoqDetailRow {
  item_no: string;
  loc: string;
  abc_vol: string | null;
  demand_mean_monthly: number | null;
  annual_demand: number | null;
  ordering_cost: number | null;
  holding_cost_pct: number | null;
  unit_cost: number | null;
  moq: number | null;
  eoq: number | null;
  effective_eoq: number | null;
  eoq_cycle_stock: number | null;
  order_frequency: number | null;
  annual_holding_cost: number | null;
  annual_order_cost: number | null;
  total_annual_cost: number | null;
  computed_at: string | null;
}

export interface EoqDetailPayload {
  total: number;
  limit: number;
  offset: number;
  rows: EoqDetailRow[];
}

export interface EoqSensitivityPoint {
  ordering_cost: number;
  eoq: number;
  effective_eoq: number;
  total_annual_cost: number;
}

export interface EoqSensitivityPayload {
  item_no: string | null;
  loc: string | null;
  avg_demand_monthly: number;
  curve: EoqSensitivityPoint[];
}

export async function fetchEoqSummary(params: { abc_vol?: string }): Promise<EoqSummaryPayload> {
  const qs = new URLSearchParams();
  if (params.abc_vol?.trim()) qs.set("abc_vol", params.abc_vol.trim());
  return fetchJson(`/inv-planning/eoq/summary?${qs}`);
}

export async function fetchEoqDetail(params: {
  item?: string;
  loc?: string;
  abc_vol?: string;
  sort_by?: string;
  sort_dir?: string;
  limit?: number;
  offset?: number;
}): Promise<EoqDetailPayload> {
  const qs = new URLSearchParams({
    limit: String(params.limit ?? 50),
    offset: String(params.offset ?? 0),
    sort_by: params.sort_by ?? "total_annual_cost",
    sort_dir: params.sort_dir ?? "desc",
  });
  if (params.item?.trim()) qs.set("item", params.item.trim());
  if (params.loc?.trim()) qs.set("loc", params.loc.trim());
  if (params.abc_vol?.trim()) qs.set("abc_vol", params.abc_vol.trim());
  return fetchJson(`/inv-planning/eoq/detail?${qs}`);
}

export async function fetchEoqSensitivity(params: {
  item?: string;
  loc?: string;
}): Promise<EoqSensitivityPayload> {
  const qs = new URLSearchParams();
  if (params.item?.trim()) qs.set("item", params.item.trim());
  if (params.loc?.trim()) qs.set("loc", params.loc.trim());
  return fetchJson(`/inv-planning/eoq/sensitivity?${qs}`);
}

// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature5: Replenishment Policy Management
// ---------------------------------------------------------------------------

export interface ReplenishmentPolicy {
  policy_id: string;
  policy_name: string;
  policy_type: "continuous_rop" | "periodic_review" | "min_max" | "manual";
  segment: string | null;
  review_cycle_days: number | null;
  service_level: number | null;
  use_eoq: boolean;
  use_safety_stock: boolean;
  active: boolean;
  dfu_count: number;
}

export interface PolicyListPayload {
  policies: ReplenishmentPolicy[];
}

export interface PolicyAssignmentRow {
  item_no: string;
  loc: string;
  policy_id: string;
  policy_name: string;
  policy_type: string;
  override_reason: string | null;
  assigned_by: string;
  effective_date: string | null;
}

export interface PolicyAssignmentsPayload {
  total: number;
  rows: PolicyAssignmentRow[];
}

export interface PolicyComplianceByPolicy {
  policy_name: string;
  policy_type: string;
  dfu_count: number;
  below_ss_pct: number | null;
  avg_ss_coverage: number | null;
  avg_dos: number | null;
}

export interface PolicyCompliancePayload {
  total_dfus: number;
  assigned_count: number;
  unassigned_count: number;
  assignment_pct: number;
  by_policy: Record<string, PolicyComplianceByPolicy>;
}

export interface PolicyAssignResult {
  assigned_count: number;
  failed_count: number;
  already_assigned_count: number;
}

export async function fetchPolicies(): Promise<PolicyListPayload> {
  return fetchJson("/inv-planning/policies");
}

export async function createPolicy(body: {
  policy_id: string;
  policy_name: string;
  policy_type: string;
  segment?: string;
  review_cycle_days?: number;
  service_level?: number;
  use_eoq?: boolean;
  use_safety_stock?: boolean;
  notes?: string;
}): Promise<ReplenishmentPolicy> {
  return fetchJson("/inv-planning/policies", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function updatePolicy(
  policyId: string,
  body: Partial<{
    policy_name: string;
    policy_type: string;
    segment: string;
    review_cycle_days: number;
    service_level: number;
    use_eoq: boolean;
    use_safety_stock: boolean;
    active: boolean;
    notes: string;
  }>,
): Promise<ReplenishmentPolicy> {
  return fetchJson(`/inv-planning/policies/${encodeURIComponent(policyId)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function fetchPolicyAssignments(params: {
  item?: string;
  location?: string;
  policy_id?: string;
  assigned_by?: string;
  limit?: number;
  offset?: number;
}): Promise<PolicyAssignmentsPayload> {
  const qs = new URLSearchParams({
    limit: String(params.limit ?? 50),
    offset: String(params.offset ?? 0),
  });
  if (params.item?.trim()) qs.set("item", params.item.trim());
  if (params.location?.trim()) qs.set("location", params.location.trim());
  if (params.policy_id?.trim()) qs.set("policy_id", params.policy_id.trim());
  if (params.assigned_by?.trim()) qs.set("assigned_by", params.assigned_by.trim());
  return fetchJson(`/inv-planning/policy-assignments?${qs}`);
}

export async function assignPolicy(body: {
  item_no?: string;
  loc?: string;
  policy_id?: string;
  override_reason?: string;
  segment?: string;
}): Promise<PolicyAssignResult> {
  return fetchJson("/inv-planning/policy-assignments/assign", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function fetchPolicyCompliance(): Promise<PolicyCompliancePayload> {
  return fetchJson("/inv-planning/policy-assignments/compliance");
}

// ---------------------------------------------------------------------------
// IPfeature6 — Inventory Health Score
// ---------------------------------------------------------------------------

export const healthKeys = {
  summary: (filters?: HealthSummaryFilters) => ["health-summary", filters ?? {}] as const,
  detail:  (params?: HealthDetailParams)   => ["health-detail",  params ?? {}]   as const,
  heatmap: (groupX?: string, groupY?: string) => ["health-heatmap", groupX ?? "abc_vol", groupY ?? "variability_class"] as const,
};

export interface HealthSummaryFilters {
  abc_vol?: string;
  cluster_assignment?: string;
  region?: string;
  variability_class?: string;
}

export interface HealthTierBreakdown {
  healthy: number;
  monitor: number;
  at_risk: number;
  critical: number;
}

export interface HealthComponentAvgs {
  ss_coverage: number | null;
  dos_target: number | null;
  stockout_risk: number | null;
  forecast_accuracy: number | null;
}

export interface HealthHistogramBucket {
  bucket: string;
  count: number;
}

export interface HealthSummaryPayload {
  total_dfus: number;
  by_tier: HealthTierBreakdown;
  avg_health_score: number | null;
  component_avgs: HealthComponentAvgs;
  score_histogram: HealthHistogramBucket[];
}

export interface HealthDetailParams {
  item?: string;
  location?: string;
  health_tier?: string;
  abc_vol?: string;
  cluster_assignment?: string;
  variability_class?: string;
  limit?: number;
  offset?: number;
  sort_by?: string;
  sort_dir?: string;
}

export interface HealthDetailRow {
  item_no: string;
  loc: string;
  abc_vol: string | null;
  variability_class: string | null;
  cluster_assignment: string | null;
  health_score: number;
  health_tier: string;
  score_ss_coverage: number;
  score_dos_target: number;
  score_stockout_risk: number;
  score_forecast_accuracy: number;
  ss_coverage: number | null;
  current_dos: number | null;
  target_dos_min: number | null;
  target_dos_max: number | null;
  is_below_ss: boolean | null;
  recent_wape: number | null;
  stockout_count_3m: number | null;
}

export interface HealthDetailPayload {
  total: number;
  rows: HealthDetailRow[];
}

export interface HealthHeatmapCell {
  x: string;
  y: string;
  avg_health_score: number | null;
  count: number;
  critical_count: number;
}

export interface HealthHeatmapPayload {
  x_labels: string[];
  y_labels: string[];
  cells: HealthHeatmapCell[];
}

export async function fetchHealthSummary(
  filters: HealthSummaryFilters = {},
): Promise<HealthSummaryPayload> {
  const qs = new URLSearchParams();
  if (filters.abc_vol) qs.set("abc_vol", filters.abc_vol);
  if (filters.cluster_assignment) qs.set("cluster_assignment", filters.cluster_assignment);
  if (filters.region) qs.set("region", filters.region);
  if (filters.variability_class) qs.set("variability_class", filters.variability_class);
  const q = qs.toString();
  return fetchJson(`/inv-planning/health/summary${q ? "?" + q : ""}`);
}

export async function fetchHealthDetail(
  params: HealthDetailParams = {},
): Promise<HealthDetailPayload> {
  const qs = new URLSearchParams({
    limit: String(params.limit ?? 100),
    offset: String(params.offset ?? 0),
    sort_by: params.sort_by ?? "health_score",
    sort_dir: params.sort_dir ?? "asc",
  });
  if (params.item) qs.set("item", params.item);
  if (params.location) qs.set("location", params.location);
  if (params.health_tier) qs.set("health_tier", params.health_tier);
  if (params.abc_vol) qs.set("abc_vol", params.abc_vol);
  if (params.cluster_assignment) qs.set("cluster_assignment", params.cluster_assignment);
  if (params.variability_class) qs.set("variability_class", params.variability_class);
  return fetchJson(`/inv-planning/health/detail?${qs}`);
}

export async function fetchHealthHeatmap(
  group_x = "abc_vol",
  group_y = "variability_class",
): Promise<HealthHeatmapPayload> {
  return fetchJson(
    `/inv-planning/health/heatmap?group_x=${encodeURIComponent(group_x)}&group_y=${encodeURIComponent(group_y)}`,
  );
}


// ---------------------------------------------------------------------------
// IPfeature7 — Exception Queue & Replenishment Recommendations
// ---------------------------------------------------------------------------

export interface ExceptionSummaryFilters {
  status?: string;
}

export interface ExceptionListParams {
  exception_type?: string;
  severity?: string;
  status?: string;
  item?: string;
  location?: string;
  sort_by?: string;
  sort_dir?: string;
  limit?: number;
  offset?: number;
}

export interface ExceptionRow {
  exception_id: string;
  item_no: string;
  loc: string;
  exception_date: string;
  exception_type: string;
  severity: string;
  current_qty_on_hand: number | null;
  current_dos: number | null;
  ss_combined: number | null;
  reorder_point: number | null;
  recommended_order_qty: number | null;
  recommended_order_by: string | null;
  expected_receipt_date: string | null;
  estimated_order_value: number | null;
  policy_id: string | null;
  status: string;
  acknowledged_by: string | null;
  notes: string | null;
}

export interface ExceptionListPayload {
  total: number;
  limit: number;
  offset: number;
  rows: ExceptionRow[];
}

export interface ExceptionSummaryPayload {
  open_count: number;
  by_type: Record<string, number>;
  by_severity: { critical: number; high: number; medium: number; low: number };
  total_recommended_order_value: number;
  oldest_open_days: number;
}

export interface ExceptionGeneratePayload {
  generated_count: number;
  skipped_dedup: number;
  by_type: Record<string, number>;
}

export const exceptionKeys = {
  list:    (p?: ExceptionListParams)      => ["exception-list",    p ?? {}] as const,
  summary: (f?: ExceptionSummaryFilters)  => ["exception-summary", f ?? {}] as const,
};

export async function fetchExceptions(
  params: ExceptionListParams = {},
): Promise<ExceptionListPayload> {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  });
  const q = qs.toString();
  return fetchJson(`/inv-planning/exceptions${q ? `?${q}` : ""}`);
}

export async function fetchExceptionSummary(
  filters: ExceptionSummaryFilters = {},
): Promise<ExceptionSummaryPayload> {
  const qs = new URLSearchParams();
  if (filters.status) qs.set("status", filters.status);
  const q = qs.toString();
  return fetchJson(`/inv-planning/exceptions/summary${q ? `?${q}` : ""}`);
}

export async function acknowledgeException(
  exceptionId: string,
  acknowledgedBy: string,
  notes?: string,
): Promise<ExceptionRow> {
  return fetchJson(`/inv-planning/exceptions/${encodeURIComponent(exceptionId)}/acknowledge`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ acknowledged_by: acknowledgedBy, notes }),
  });
}

export async function updateExceptionStatus(
  exceptionId: string,
  status: "ordered" | "resolved",
  notes?: string,
): Promise<ExceptionRow> {
  return fetchJson(`/inv-planning/exceptions/${encodeURIComponent(exceptionId)}/status`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status, notes }),
  });
}

export async function generateExceptions(): Promise<ExceptionGeneratePayload> {
  return fetchJson("/inv-planning/exceptions/generate", { method: "POST" });
}

// ---------------------------------------------------------------------------
// IPfeature8: Fill Rate Analytics
// ---------------------------------------------------------------------------

export interface FillRateSummaryPayload {
  portfolio_fill_rate: number | null;
  total_ordered: number;
  total_shipped: number;
  total_shortage_qty: number;
  partial_fulfillment_events: number;
  by_abc: Record<string, { avg_fill_rate: number | null; total_shortage_qty: number; events: number }>;
  worst_items: Array<{ item_no: string; loc: string; fill_rate: number | null; shortage_qty: number | null; abc_vol: string | null }>;
  trend: Array<{ month_start: string; portfolio_fill_rate: number | null; total_shortage_qty: number }>;
}

export interface FillRateTrendRow {
  month_start: string;
  fill_rate: number | null;
  total_ordered: number;
  total_shipped: number;
  shortage_qty: number;
}

export interface FillRateDetailRow {
  item_no: string;
  loc: string;
  month_start: string;
  total_ordered: number | null;
  total_shipped: number | null;
  fill_rate: number | null;
  shortage_qty: number | null;
  had_partial_fulfillment: boolean;
  abc_vol: string | null;
  cluster_assignment: string | null;
  region: string | null;
}

export const fillRateKeys = {
  summary: (f?: Record<string, unknown>) => ["fill-rate-summary", f ?? {}] as const,
  trend:   (f?: Record<string, unknown>) => ["fill-rate-trend", f ?? {}] as const,
  detail:  (f?: Record<string, unknown>) => ["fill-rate-detail", f ?? {}] as const,
};

export async function fetchFillRateSummary(
  params: Record<string, unknown> = {},
): Promise<FillRateSummaryPayload> {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  });
  const q = qs.toString();
  return fetchJson(`/fill-rate/summary${q ? `?${q}` : ""}`);
}

export async function fetchFillRateTrend(
  params: Record<string, unknown> = {},
): Promise<{ months: FillRateTrendRow[] }> {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  });
  const q = qs.toString();
  return fetchJson(`/fill-rate/trend${q ? `?${q}` : ""}`);
}

export async function fetchFillRateDetail(
  params: Record<string, unknown> = {},
): Promise<{ total: number; rows: FillRateDetailRow[] }> {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  });
  const q = qs.toString();
  return fetchJson(`/fill-rate/detail${q ? `?${q}` : ""}`);
}

// ---------------------------------------------------------------------------
// IPfeature11: ABC-XYZ Classification
// ---------------------------------------------------------------------------

export interface AbcXyzCell {
  abc_vol: string;
  xyz_class: string;
  segment: string;
  dfu_count: number;
  avg_service_level: number | null;
  avg_dos_min: number | null;
  avg_dos_max: number | null;
}

export interface AbcXyzDetailRow {
  dmdunit: string;
  dmdgroup: string;
  loc: string;
  abc_vol: string | null;
  xyz_class: string | null;
  abc_xyz_segment: string | null;
  demand_cv: number | null;
  intermittency_ratio: number | null;
  abc_xyz_dos_min: number | null;
  abc_xyz_dos_max: number | null;
  abc_xyz_service_level: number | null;
}

export const abcXyzKeys = {
  matrix:  () => ["abc-xyz-matrix"] as const,
  summary: () => ["abc-xyz-summary"] as const,
  detail:  (f?: Record<string, unknown>) => ["abc-xyz-detail", f ?? {}] as const,
};

export const fetchAbcXyzMatrix = (): Promise<{ cells: AbcXyzCell[]; total_classified: number }> =>
  fetchJson("/inv-planning/abc-xyz/matrix");

export const fetchAbcXyzSummary = (): Promise<Record<string, number | null>> =>
  fetchJson("/inv-planning/abc-xyz/summary");

export async function fetchAbcXyzDetail(
  params: Record<string, unknown> = {},
): Promise<{ total: number; rows: AbcXyzDetailRow[] }> {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  });
  const q = qs.toString();
  return fetchJson(`/inv-planning/abc-xyz/detail${q ? `?${q}` : ""}`);
}

// ---------------------------------------------------------------------------
// IPfeature12: Supplier Performance
// ---------------------------------------------------------------------------

export interface SupplierRow {
  supplier_no: string;
  supplier_name: string | null;
  sku_loc_count: number;
  distinct_items: number;
  avg_lt_mean_days: number | null;
  avg_lt_cv: number | null;
  avg_lt_std_days: number | null;
  pct_stable_lt: number | null;
  pct_volatile_lt: number | null;
  total_safety_stock_units: number | null;
  total_ss_value: number | null;
  supplier_reliability_score: number | null;
}

export const supplierKeys = {
  summary: () => ["supplier-perf-summary"] as const,
  detail:  (f?: Record<string, unknown>) => ["supplier-perf-detail", f ?? {}] as const,
  items:   (supplierNo: string) => ["supplier-perf-items", supplierNo] as const,
};

export const fetchSupplierSummary = (): Promise<Record<string, number | null>> =>
  fetchJson("/inv-planning/supplier-performance/summary");

export async function fetchSupplierDetail(
  params: Record<string, unknown> = {},
): Promise<{ total: number; rows: SupplierRow[] }> {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  });
  const q = qs.toString();
  return fetchJson(`/inv-planning/supplier-performance/detail${q ? `?${q}` : ""}`);
}

// ---------------------------------------------------------------------------
// IPfeature14: Intra-Month Stockout Detection
// ---------------------------------------------------------------------------

export interface IntramonthStockoutRow {
  item_no: string;
  loc: string;
  month_start: string;
  snapshot_days: number;
  stockout_days: number;
  stockout_day_rate: number | null;
  min_qty_on_hand: number | null;
  max_qty_on_hand: number | null;
  avg_qty_on_hand: number | null;
  est_lost_sales: number | null;
  had_full_stockout: boolean;
  had_extended_stockout: boolean;
  abc_vol: string | null;
  abc_xyz_segment: string | null;
  cluster_assignment: string | null;
}

export const intramonthKeys = {
  summary: (f?: Record<string, unknown>) => ["intramonth-summary", f ?? {}] as const,
  detail:  (f?: Record<string, unknown>) => ["intramonth-detail", f ?? {}] as const,
};

export async function fetchIntramonthSummary(
  params: Record<string, unknown> = {},
): Promise<Record<string, number | null>> {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  });
  const q = qs.toString();
  return fetchJson(`/inv-planning/intramonth-stockouts/summary${q ? `?${q}` : ""}`);
}

export async function fetchIntramonthDetail(
  params: Record<string, unknown> = {},
): Promise<{ total: number; rows: IntramonthStockoutRow[] }> {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  });
  const q = qs.toString();
  return fetchJson(`/inv-planning/intramonth-stockouts/detail${q ? `?${q}` : ""}`);
}

// ---------------------------------------------------------------------------
// IPfeature15: Control Tower
// ---------------------------------------------------------------------------

export interface ControlTowerKpis {
  computed_at: string | null;
  health: {
    total_dfus: number; healthy_count: number; monitor_count: number;
    at_risk_count: number; critical_count: number;
    avg_health_score: number | null; avg_ss_coverage: number | null;
    below_ss_count: number; below_ss_pct: number | null; avg_portfolio_dos: number | null;
  };
  exceptions: {
    open_exceptions_total: number; critical_exceptions: number;
    high_exceptions: number; recommended_order_value: number | null;
  };
  fill_rate: { portfolio_fill_rate_3m: number | null; total_shortage_qty_3m: number | null };
  demand_signals: { urgent_demand_signals: number; projected_stockouts_today: number };
  intramonth: { items_with_stockout_this_month: number; extended_stockouts_this_month: number };
}

export interface ControlTowerAlert {
  alert_id: string;
  source: string;
  severity: string;
  item_no: string;
  loc: string;
  alert_type: string;
  description: string;
  action: string;
  alert_ts: string | null;
  abc_vol: string | null;
}

export interface ControlTowerCriticalItem {
  item_no: string; loc: string; abc_vol: string | null; abc_xyz_segment: string | null;
  health_score: number | null; health_tier: string | null;
  ss_coverage: number | null; is_below_ss: boolean;
  current_dos: number | null; target_dos_min: number | null; target_dos_max: number | null;
  open_exception_count: number; recommended_order_qty: number | null;
  fill_rate_last_3m: number | null; stockout_days_this_month: number;
}

export const controlTowerKeys = {
  kpis:       () => ["ct-kpis"] as const,
  alerts:     (f?: Record<string, unknown>) => ["ct-alerts", f ?? {}] as const,
  topCritical:(limit?: number) => ["ct-top-critical", limit ?? 10] as const,
  trend:      (months?: number) => ["ct-trend", months ?? 6] as const,
};

export const fetchControlTowerKpis = (): Promise<ControlTowerKpis> =>
  fetchJson("/control-tower/kpis");

export async function fetchControlTowerAlerts(
  params: { limit?: number; severity?: string } = {},
): Promise<{ total: number; alerts: ControlTowerAlert[] }> {
  const qs = new URLSearchParams();
  if (params.limit) qs.set("limit", String(params.limit));
  if (params.severity) qs.set("severity", params.severity);
  const q = qs.toString();
  return fetchJson(`/control-tower/alerts${q ? `?${q}` : ""}`);
}

export const fetchControlTowerTopCritical = (
  limit = 10,
): Promise<{ items: ControlTowerCriticalItem[] }> =>
  fetchJson(`/control-tower/top-critical?limit=${limit}`);

export const fetchControlTowerTrend = (
  months = 6,
): Promise<{ trend: Array<Record<string, number | string | null>> }> =>
  fetchJson(`/control-tower/trend?months=${months}`);

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

// ---------------------------------------------------------------------------
// Seasonality Profiles — Feature 32 filter support
// ---------------------------------------------------------------------------
export const seasonalityProfileKeys = {
  list: () => ["seasonality-profiles"] as const,
};

/** Returns a plain string[] of distinct seasonality profile names for use in filter dropdowns. */
export async function fetchSeasonalityProfileNames(): Promise<string[]> {
  const data = await fetchSeasonalityProfiles();
  return (data.profiles ?? []).map((p) => p.profile);
}

// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature3: Safety Stock
// ---------------------------------------------------------------------------

export const safetyStockKeys = {
  summary: (filters?: Record<string, string>) => ["safety-stock", "summary", filters] as const,
  detail: (params?: Record<string, unknown>) => ["safety-stock", "detail", params] as const,
  waterfall: (itemNo: string, loc: string) => ["safety-stock", "waterfall", itemNo, loc] as const,
  config: () => ["safety-stock", "config"] as const,
};

export interface SafetyStockSummary {
  total_dfus: number;
  below_ss_count: number;
  avg_ss_coverage: number;
  avg_ss_days: number;
  by_abc: Array<{ abc_vol: string; count: number; below_ss_count: number; avg_coverage: number }>;
}

export interface SafetyStockRow {
  item_no: string;
  loc: string;
  ss_combined: number;
  ss_coverage: number;
  is_below_ss: boolean;
  reorder_point: number;
  abc_vol: string;
  ss_demand_only: number;
  ss_lt_only: number;
}

export interface SafetyStockWaterfall {
  item_no: string;
  loc: string;
  ss_demand_only: number;
  ss_lt_only: number;
  ss_combined: number;
  reorder_point: number;
  avg_daily_demand: number;
  lt_mean_days: number;
}

export async function fetchSafetyStockSummary(filters?: Record<string, string>): Promise<SafetyStockSummary> {
  const params = new URLSearchParams(filters ?? {});
  const res = await fetch(`/inv-planning/safety-stock/summary?${params}`);
  if (!res.ok) throw new Error("Failed to fetch safety stock summary");
  return res.json();
}

export async function fetchSafetyStockDetail(params?: {
  is_below_ss?: boolean;
  abc_vol?: string;
  item?: string;
  loc?: string;
  limit?: number;
  offset?: number;
}): Promise<{ total: number; rows: SafetyStockRow[] }> {
  const p = new URLSearchParams();
  if (params?.is_below_ss !== undefined) p.set("is_below_ss", String(params.is_below_ss));
  if (params?.abc_vol) p.set("abc_vol", params.abc_vol);
  if (params?.item) p.set("item", params.item);
  if (params?.loc) p.set("loc", params.loc);
  if (params?.limit !== undefined) p.set("limit", String(params.limit));
  if (params?.offset !== undefined) p.set("offset", String(params.offset));
  const res = await fetch(`/inv-planning/safety-stock/detail?${p}`);
  if (!res.ok) throw new Error("Failed to fetch safety stock detail");
  return res.json();
}

export async function fetchSafetyStockWaterfall(itemNo: string, loc: string): Promise<SafetyStockWaterfall> {
  const res = await fetch(
    `/inv-planning/safety-stock/waterfall?item_no=${encodeURIComponent(itemNo)}&loc=${encodeURIComponent(loc)}`,
  );
  if (!res.ok) throw new Error("Failed to fetch safety stock waterfall");
  return res.json();
}

export async function fetchSafetyStockConfig(): Promise<Record<string, unknown>> {
  const res = await fetch("/inv-planning/safety-stock/config");
  if (!res.ok) throw new Error("Failed to fetch safety stock config");
  return res.json();
}

// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature9: Demand Signals
// ---------------------------------------------------------------------------

export const demandSignalsKeys = {
  summary: (date?: string) => ["demand-signals", "summary", date] as const,
  list: (params?: Record<string, unknown>) => ["demand-signals", "list", params] as const,
  item: (itemNo: string, loc: string) => ["demand-signals", "item", itemNo, loc] as const,
};

export interface DemandSignalSummary {
  signal_date: string;
  above_plan_count: number;
  below_plan_count: number;
  on_plan_count: number;
  urgent_count: number;
  watch_count: number;
  projected_stockouts: number;
}

export interface DemandSignalRow {
  item_no: string;
  loc: string;
  signal_date: string;
  signal_type: string;
  signal_strength: number;
  demand_vs_forecast_pct: number;
  projected_stockout: boolean;
  alert_priority: string;
  mtd_actual: number;
  projected_monthly: number;
  forecast_monthly: number;
  current_on_hand: number;
  is_below_ss: boolean;
}

export async function fetchDemandSignalsSummary(date?: string): Promise<DemandSignalSummary> {
  const p = date ? `?signal_date=${date}` : "";
  const res = await fetch(`/inv-planning/demand-signals/summary${p}`);
  if (!res.ok) throw new Error("Failed to fetch demand signals summary");
  return res.json();
}

export async function fetchDemandSignals(params?: {
  signal_type?: string;
  alert_priority?: string;
  item?: string;
  loc?: string;
  limit?: number;
  offset?: number;
}): Promise<{ total: number; rows: DemandSignalRow[] }> {
  const p = new URLSearchParams();
  if (params?.signal_type) p.set("signal_type", params.signal_type);
  if (params?.alert_priority) p.set("alert_priority", params.alert_priority);
  if (params?.item) p.set("item", params.item);
  if (params?.loc) p.set("loc", params.loc);
  if (params?.limit !== undefined) p.set("limit", String(params.limit));
  if (params?.offset !== undefined) p.set("offset", String(params.offset));
  const res = await fetch(`/inv-planning/demand-signals?${p}`);
  if (!res.ok) throw new Error("Failed to fetch demand signals");
  return res.json();
}

export async function fetchDemandSignalItem(itemNo: string, loc: string): Promise<DemandSignalRow> {
  const res = await fetch(
    `/inv-planning/demand-signals/item?item_no=${encodeURIComponent(itemNo)}&loc=${encodeURIComponent(loc)}`,
  );
  if (!res.ok) throw new Error("Failed to fetch demand signal item");
  return res.json();
}

// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature10: Safety Stock Monte Carlo Simulation
// ---------------------------------------------------------------------------

export const simulationKeys = {
  results: (params?: Record<string, unknown>) => ["simulation", "results", params] as const,
  compare: (itemNo: string, loc: string) => ["simulation", "compare", itemNo, loc] as const,
  status: (simRunId: string) => ["simulation", "status", simRunId] as const,
};

export interface SimulationResult {
  sim_run_id: string;
  item_no: string;
  loc: string;
  simulation_date: string;
  n_simulations: number;
  target_csl: number;
  recommended_ss: number;
  recommended_ss_days: number;
  analytical_ss: number;
  sim_vs_analytical_pct: number;
  run_duration_secs: number;
  results_by_ss_level: Array<{ ss_qty: number; csl: number }>;
}

export async function fetchSimulationResults(params?: {
  item?: string;
  loc?: string;
  limit?: number;
  offset?: number;
}): Promise<{ total: number; rows: SimulationResult[] }> {
  const p = new URLSearchParams();
  if (params?.item) p.set("item", params.item);
  if (params?.loc) p.set("loc", params.loc);
  if (params?.limit !== undefined) p.set("limit", String(params.limit));
  const res = await fetch(`/inv-planning/simulation/results?${p}`);
  if (!res.ok) throw new Error("Failed to fetch simulation results");
  return res.json();
}

export async function fetchSimulationCompare(itemNo: string, loc: string): Promise<SimulationResult[]> {
  const res = await fetch(
    `/inv-planning/simulation/compare?item_no=${encodeURIComponent(itemNo)}&loc=${encodeURIComponent(loc)}`,
  );
  if (!res.ok) throw new Error("Failed to fetch simulation compare");
  return res.json();
}

export async function runSimulation(body: {
  item_no: string;
  loc: string;
  n_simulations?: number;
  target_csl?: number;
}): Promise<SimulationResult> {
  const res = await fetch("/inv-planning/simulation/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error("Failed to run simulation");
  return res.json();
}

// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature13: Investment Plan
// ---------------------------------------------------------------------------

export const investmentKeys = {
  summary: (planId?: string) => ["investment", "summary", planId] as const,
  detail: (params?: Record<string, unknown>) => ["investment", "detail", params] as const,
  frontier: (planId?: string) => ["investment", "frontier", planId] as const,
};

export interface InvestmentSummary {
  plan_id: string;
  computation_date: string;
  total_items: number;
  total_current_investment: number;
  total_recommended_investment: number;
  total_investment_gap: number;
  avg_current_csl: number;
  avg_recommended_csl: number;
}

export interface InvestmentRow {
  item_no: string;
  loc: string;
  abc_vol: string;
  abc_xyz_segment: string;
  current_ss_qty: number;
  current_ss_value: number;
  current_csl: number;
  recommended_ss_qty: number;
  recommended_ss_value: number;
  recommended_csl: number;
  ss_increment_qty: number;
  investment_increment: number;
  csl_increment: number;
  marginal_roi: number;
  investment_rank: number;
  cumulative_investment: number;
}

export interface FrontierPoint {
  plan_id: string;
  budget_point: number;
  items_funded: number;
  achievable_csl: number;
  marginal_item: string;
}

export async function fetchInvestmentSummary(planId?: string): Promise<InvestmentSummary> {
  const p = planId ? `?plan_id=${planId}` : "";
  const res = await fetch(`/inv-planning/investment/summary${p}`);
  if (!res.ok) throw new Error("Failed to fetch investment summary");
  return res.json();
}

export async function fetchInvestmentDetail(params?: {
  plan_id?: string;
  abc_vol?: string;
  limit?: number;
  offset?: number;
}): Promise<{ total: number; rows: InvestmentRow[] }> {
  const p = new URLSearchParams();
  if (params?.plan_id) p.set("plan_id", params.plan_id);
  if (params?.abc_vol) p.set("abc_vol", params.abc_vol);
  if (params?.limit !== undefined) p.set("limit", String(params.limit));
  if (params?.offset !== undefined) p.set("offset", String(params.offset));
  const res = await fetch(`/inv-planning/investment/detail?${p}`);
  if (!res.ok) throw new Error("Failed to fetch investment detail");
  return res.json();
}

export async function fetchInvestmentFrontier(planId?: string): Promise<FrontierPoint[]> {
  const p = planId ? `?plan_id=${planId}` : "";
  const res = await fetch(`/inv-planning/investment/efficient-frontier${p}`);
  if (!res.ok) throw new Error("Failed to fetch investment frontier");
  return res.json();
}

export async function runInvestmentPlan(params?: {
  budget?: number;
  target_csl?: number;
}): Promise<{ plan_id: string; total_items: number; total_investment_gap: number }> {
  const p = new URLSearchParams();
  if (params?.budget !== undefined) p.set("budget", String(params.budget));
  if (params?.target_csl !== undefined) p.set("target_csl", String(params.target_csl));
  const res = await fetch(`/inv-planning/investment/plan?${p}`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to run investment plan");
  return res.json();
}
