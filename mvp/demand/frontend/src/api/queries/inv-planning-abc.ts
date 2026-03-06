import { fetchJson } from "./core";

// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature11: ABC-XYZ Classification
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
