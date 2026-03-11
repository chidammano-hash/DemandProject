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
  ShapModelsPayload,
  ShapSummaryPayload,
  ShapTimeframesPayload,
  ShapTimeframeDetailPayload,
  DfuShapPayload,
} from "@/types/shap";
import type {
  DashboardKpis,
  Alert,
  Mover,
  HeatmapRow,
  DistinctValuesPayload,
} from "@/types/theme";
import type {
  Job, JobType, JobListPayload, JobTypesPayload, ActiveJobsPayload,
  JobStats, JobSchedule, JobSchedulesPayload,
} from "@/types/jobs";

export type {
  Job, JobType, JobListPayload, JobTypesPayload, ActiveJobsPayload,
  JobStats, JobSchedule, JobSchedulesPayload,
};

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
  planningDate: () => ["planning-date"] as const,
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
  dfuShap: (modelId: string, itemNo: string, loc: string, topN: number) => ["dfu-shap", modelId, itemNo, loc, topN] as const,
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

// ---------------------------------------------------------------------------
// Fetch helpers
// ---------------------------------------------------------------------------
export async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
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
      ch_scores?: number[];
      combined_scores?: number[];
      feasible_mask?: boolean[];
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
}): Promise<ScenarioEstimate> {
  const qs = new URLSearchParams({
    k_min: String(params.k_min),
    k_max: String(params.k_max),
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

// ---------------------------------------------------------------------------
// Planning date
// ---------------------------------------------------------------------------

export interface PlanningDateInfo {
  planning_date: string;   // ISO date string e.g. "2026-02-24"
  system_date: string;     // ISO date string e.g. "2026-03-09"
  is_frozen: boolean;      // true when planning_date !== system_date
  days_behind: number;     // system_date - planning_date in days
}

export async function fetchPlanningDate(): Promise<PlanningDateInfo> {
  return fetchJson("/dashboard/planning-date");
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

export async function fetchDfuShap(
  modelId: string,
  itemNo: string,
  loc: string,
  topN = 10,
): Promise<DfuShapPayload> {
  const qs = new URLSearchParams({
    item_no: itemNo,
    loc,
    top_n: String(topN),
  });
  return fetchJson(`/forecast/shap/${encodeURIComponent(modelId)}/dfu?${qs}`);
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
