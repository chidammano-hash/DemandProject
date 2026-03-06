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
