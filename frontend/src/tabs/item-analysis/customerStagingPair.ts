import type {
  StagingForecastPoint,
  StagingForecastsPayload,
} from "@/api/queries/production-forecast";

const BOTTOM_UP_MODEL = "customer_bottom_up";
const BLEND_MODEL = "customer_bottom_up_blend";

export interface PairedCustomerStaging {
  payload: StagingForecastsPayload | null;
  bottomUpMonths: Set<string>;
  blendMonths: Set<string>;
}

function exactRows(
  rows: StagingForecastPoint[] | undefined,
  runId: string | null,
  sourceModelId: string
): StagingForecastPoint[] {
  if (!runId) return [];
  return (rows ?? []).filter(
    (row) => row.source_run_id === runId && row.source_model_id === sourceModelId
  );
}

/** Keep customer staging only when it matches the exact blend/shadow pair. */
export function pairCustomerStaging(
  source: StagingForecastsPayload | null | undefined,
  blendRunId: string | null,
  bottomUpRunId: string | null
): PairedCustomerStaging {
  const bottomUpRows = exactRows(source?.models[BOTTOM_UP_MODEL], bottomUpRunId, BOTTOM_UP_MODEL);
  const blendRows = exactRows(source?.models[BLEND_MODEL], blendRunId, BLEND_MODEL);
  const models = Object.fromEntries(
    Object.entries(source?.models ?? {}).filter(
      ([modelId]) => modelId !== BOTTOM_UP_MODEL && modelId !== BLEND_MODEL
    )
  );
  if (bottomUpRows.length > 0) models[BOTTOM_UP_MODEL] = bottomUpRows;
  if (blendRows.length > 0) models[BLEND_MODEL] = blendRows;
  return {
    payload: source ? { ...source, models } : null,
    bottomUpMonths: new Set(bottomUpRows.map((row) => row.forecast_month.slice(0, 7))),
    blendMonths: new Set(blendRows.map((row) => row.forecast_month.slice(0, 7))),
  };
}

/** Replace generic customer staging values with rows from the validated pair. */
export function applyPairedCustomerStaging(
  series: Record<string, unknown>[],
  paired: PairedCustomerStaging
): Record<string, unknown>[] {
  const bottomUpByMonth = new Map(
    (paired.payload?.models[BOTTOM_UP_MODEL] ?? []).map((row) => [
      row.forecast_month.slice(0, 7),
      row.forecast_qty,
    ])
  );
  const blendByMonth = new Map(
    (paired.payload?.models[BLEND_MODEL] ?? []).map((row) => [
      row.forecast_month.slice(0, 7),
      row.forecast_qty,
    ])
  );
  return series.map((point) => {
    const next = { ...point };
    const month = String(point.month).slice(0, 7);
    delete next[`staging_${BOTTOM_UP_MODEL}`];
    delete next[`staging_${BLEND_MODEL}`];
    if (bottomUpByMonth.has(month)) {
      next[`staging_${BOTTOM_UP_MODEL}`] = bottomUpByMonth.get(month) ?? null;
    }
    if (blendByMonth.has(month)) {
      next[`staging_${BLEND_MODEL}`] = blendByMonth.get(month) ?? null;
    }
    return next;
  });
}
