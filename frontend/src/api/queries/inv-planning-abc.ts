import { fetchJson } from "./core";

// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature11: ABC-XYZ Classification
// ---------------------------------------------------------------------------

export interface AbcXyzCell {
  abc_vol: string;
  xyz_class: string;
  segment: string;
  sku_count: number;
  avg_service_level: number | null;
  avg_dos_min: number | null;
  avg_dos_max: number | null;
}

export interface AbcXyzDetailRow {
  item_id: string;
  customer_group: string;
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
  matrix:  (f?: Record<string, unknown>) => ["abc-xyz-matrix", f ?? {}] as const,
  summary: (f?: Record<string, unknown>) => ["abc-xyz-summary", f ?? {}] as const,
  detail:  (f?: Record<string, unknown>) => ["abc-xyz-detail", f ?? {}] as const,
};

export async function fetchAbcXyzMatrix(
  params: Record<string, unknown> = {},
): Promise<{ cells: AbcXyzCell[]; total_classified: number }> {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  });
  const q = qs.toString();
  return fetchJson(`/inv-planning/abc-xyz/matrix${q ? `?${q}` : ""}`);
}

export async function fetchAbcXyzSummary(
  params: Record<string, unknown> = {},
): Promise<Record<string, number | null>> {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  });
  const q = qs.toString();
  return fetchJson(`/inv-planning/abc-xyz/summary${q ? `?${q}` : ""}`);
}

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
