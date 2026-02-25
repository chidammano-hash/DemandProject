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
} from "@/types";

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
    };
    profiles: ScenarioProfile[];
  } | null;
  error: string | null;
}

export async function runClusteringScenario(
  params: ClusteringScenarioParams,
): Promise<ClusteringScenarioResult> {
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
}

export async function fetchAccuracySlice(params: SliceParams): Promise<AccuracySlicePayload> {
  const qs = new URLSearchParams({ group_by: params.group_by, lag: String(params.lag) });
  if (params.models.trim()) qs.set("models", params.models.trim());
  if (params.month_from) qs.set("month_from", params.month_from);
  if (params.common_dfus) qs.set("common_dfus", "true");
  if (params.include_dfu_count) qs.set("include_dfu_count", "true");
  return fetchJson(`/forecast/accuracy/slice?${qs}`);
}

export interface LagCurveParams {
  models: string;
  month_from: string;
  common_dfus: boolean;
  include_dfu_count: boolean;
}

export async function fetchLagCurve(params: LagCurveParams): Promise<LagCurvePayload> {
  const qs = new URLSearchParams();
  if (params.models.trim()) qs.set("models", params.models.trim());
  if (params.month_from) qs.set("month_from", params.month_from);
  if (params.common_dfus) qs.set("common_dfus", "true");
  if (params.include_dfu_count) qs.set("include_dfu_count", "true");
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
}

export interface ChampionSummary {
  total_dfus: number;
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
