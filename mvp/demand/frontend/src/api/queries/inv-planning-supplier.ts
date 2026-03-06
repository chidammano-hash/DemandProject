import { fetchJson } from "./core";

// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature12: Supplier Performance
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
