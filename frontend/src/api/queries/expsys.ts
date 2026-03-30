/**
 * Expert System Backtest (ExpSys) API queries.
 *
 * Fetches pre-computed lag accuracy results from the ExpSys backtest run
 * and status information about the latest run.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ExpSysSegmentStats {
  [segment: string]: number; // accuracy_pct per demand archetype
}

export interface ExpSysLagStats {
  accuracy_pct: number;
  wape: number;
  n_dfus: number;
  n_dfu_months: number;
  per_segment: ExpSysSegmentStats;
}

export interface ExpSysAccuracyReport {
  by_lag: Record<string, ExpSysLagStats>; // key = "0".."4"
  execution_lag: ExpSysLagStats | null;
}

export interface ExpSysStatus {
  has_report: boolean;
  report_updated_at: string | null;
  completed_timeframes: number;
  checkpoint_labels: string[];
  db_row_count: number;
}

// ---------------------------------------------------------------------------
// Query keys
// ---------------------------------------------------------------------------

export const expSysKeys = {
  lagAccuracy: (modelId?: string) =>
    ["expsys", "lag-accuracy", modelId ?? "ExpSys"] as const,
  status: () => ["expsys", "status"] as const,
};

// ---------------------------------------------------------------------------
// Fetch functions
// ---------------------------------------------------------------------------

export async function fetchExpSysLagAccuracy(
  modelId = "ExpSys",
): Promise<ExpSysAccuracyReport> {
  const params = modelId !== "ExpSys" ? `?model_id=${encodeURIComponent(modelId)}` : "";
  const res = await fetch(`/expsys/lag-accuracy${params}`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function fetchExpSysStatus(): Promise<ExpSysStatus> {
  const res = await fetch("/expsys/status");
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}
