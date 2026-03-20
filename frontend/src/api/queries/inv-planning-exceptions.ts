import { fetchJson } from "./core";

// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature7: Exception Queue & Replenishment Recommendations
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
    method: "POST",
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
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status, notes }),
  });
}

export async function generateExceptions(): Promise<ExceptionGeneratePayload> {
  return fetchJson("/inv-planning/exceptions/generate", { method: "POST" });
}
