import { fetchJson } from "./core";

// ---------------------------------------------------------------------------
// F1.1 — Production Forecast (future-period ML predictions)
// ---------------------------------------------------------------------------

export interface ProductionForecastPoint {
  forecast_month: string;
  forecast_qty: number | null;
  forecast_qty_lower: number | null;
  forecast_qty_upper: number | null;
  model_id: string;
  cluster_id: number | null;
  horizon_months: number;
  is_recursive: boolean;
  lag_source: string | null;
}

export interface ProductionForecastPayload {
  item_no: string;
  loc: string;
  plan_version: string;
  model_id: string;
  generated_at: string | null;
  horizon_months: number;
  is_recursive: boolean;
  forecasts: ProductionForecastPoint[];
}

export interface ProductionForecastAbcRow {
  abc_class: string;
  dfu_count: number;
  forecast_qty: number;
}

export interface ProductionForecastSummaryPayload {
  plan_version: string | null;
  horizon_months: number;
  total_dfu_count: number;
  total_forecast_qty: number;
  generated_at: string | null;
  by_abc_class: ProductionForecastAbcRow[];
}

export interface ProductionForecastVersion {
  plan_version: string;
  dfu_count: number;
  total_rows: number;
  generated_at: string | null;
}

export interface ProductionForecastVersionsPayload {
  versions: ProductionForecastVersion[];
}

export async function fetchProductionForecast(params: {
  item_no: string;
  loc: string;
  horizon?: number;
  plan_version?: string;
}): Promise<ProductionForecastPayload> {
  const qs = new URLSearchParams({
    item_no: params.item_no,
    loc: params.loc,
    horizon: String(params.horizon ?? 12),
  });
  if (params.plan_version) qs.set("plan_version", params.plan_version);
  return fetchJson(`/forecast/production?${qs}`);
}

export async function fetchProductionForecastSummary(params: {
  plan_version?: string;
  horizon_months?: number;
  brand?: string;
  category?: string;
}): Promise<ProductionForecastSummaryPayload> {
  const qs = new URLSearchParams({
    horizon_months: String(params.horizon_months ?? 3),
  });
  if (params.plan_version) qs.set("plan_version", params.plan_version);
  if (params.brand?.trim()) qs.set("brand", params.brand.trim());
  if (params.category?.trim()) qs.set("category", params.category.trim());
  return fetchJson(`/forecast/production/summary?${qs}`);
}

export async function fetchProductionForecastVersions(): Promise<ProductionForecastVersionsPayload> {
  return fetchJson("/forecast/production/versions");
}
