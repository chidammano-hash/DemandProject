/**
 * Shared constants, types, and helpers for Portfolio Analysis tab.
 */
import type { CascadeFilterParams } from "@/api/queries";

// ---------------------------------------------------------------------------
// Panel toggle config
// ---------------------------------------------------------------------------
export const PANEL_DEFAULTS: Record<string, boolean> = {
  kpis: true,
  forecastChart: true,
  heatmap: true,
  accuracy: true,
  lagCurve: true,
  champion: false,
  shap: false,
  bias: false,
};

export const PANELS = [
  { key: "kpis", label: "KPIs" },
  { key: "forecastChart", label: "Forecast vs Actual" },
  { key: "heatmap", label: "Heatmap" },
  { key: "accuracy", label: "Accuracy" },
  { key: "lagCurve", label: "Lag Curve" },
  { key: "champion", label: "Champion" },
  { key: "shap", label: "SHAP" },
  { key: "bias", label: "Bias" },
] as const;

// ---------------------------------------------------------------------------
// Filter types
// ---------------------------------------------------------------------------
export interface FilterConfig {
  key: "brand" | "category" | "market" | "channel" | "item" | "location" | "cluster";
  label: string;
  domain: string;
  column: string;
  searchable?: boolean;
}

export const FILTERS: FilterConfig[] = [
  { key: "brand", label: "Brand", domain: "item", column: "brand_name" },
  { key: "category", label: "Category", domain: "item", column: "class_" },
  { key: "item", label: "Item", domain: "item", column: "item_no", searchable: true },
  { key: "location", label: "Location", domain: "location", column: "location_id", searchable: true },
  { key: "market", label: "Market", domain: "location", column: "state_id" },
  { key: "channel", label: "Channel", domain: "customer", column: "rpt_channel_desc" },
  { key: "cluster", label: "Cluster", domain: "dfu", column: "cluster_assignment" },
];

export interface LocalFilters {
  brand: string[];
  category: string[];
  market: string[];
  channel: string[];
  item: string[];
  location: string[];
  cluster: string[];
  timeGrain: "month" | "quarter";
}

export const EMPTY_FILTERS: LocalFilters = {
  brand: [], category: [], market: [], channel: [], item: [], location: [], cluster: [],
  timeGrain: "month",
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
export function buildCascade(filters: LocalFilters, excludeKey: FilterConfig["key"]): CascadeFilterParams | undefined {
  const c: CascadeFilterParams = {};
  if (excludeKey !== "brand" && filters.brand.length > 0) c.brand = filters.brand.join(",");
  if (excludeKey !== "category" && filters.category.length > 0) c.category = filters.category.join(",");
  if (excludeKey !== "item" && filters.item.length > 0) c.item = filters.item.join(",");
  if (excludeKey !== "location" && filters.location.length > 0) c.location = filters.location.join(",");
  if (excludeKey !== "market" && filters.market.length > 0) c.market = filters.market.join(",");
  if (excludeKey !== "channel" && filters.channel.length > 0) c.channel = filters.channel.join(",");
  if (excludeKey !== "cluster" && filters.cluster.length > 0) c.cluster = filters.cluster.join(",");
  return Object.keys(c).length > 0 ? c : undefined;
}

export function hasActiveFilters(f: LocalFilters): boolean {
  return f.brand.length > 0 || f.category.length > 0 || f.item.length > 0 || f.location.length > 0 || f.market.length > 0 || f.channel.length > 0 || f.cluster.length > 0;
}

export function formatNumberCompact(n: number | null): string {
  if (n == null) return "N/A";
  if (Math.abs(n) >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (Math.abs(n) >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toFixed(0);
}

export function trendDirection(delta: number | null): "up" | "down" | "flat" {
  if (delta == null || delta === 0) return "flat";
  return delta > 0 ? "up" : "down";
}

export const HEATMAP_SCALE = ["#16a34a", "#65a30d", "#eab308", "#f97316", "#dc2626"];
