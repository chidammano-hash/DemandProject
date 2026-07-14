import type { CustomerBlendSeriesMonth } from "@/api/queries/customerForecast";

export const CUSTOMER_BLEND_KEYS = {
  bottomUp: "customer_bottom_up_qty",
  sourceChampion: "customer_source_champion_qty",
  blend: "customer_blend_qty",
} as const;

export interface CustomerBlendChartPoint extends Record<string, unknown> {
  month: string;
  customer_bottom_up_qty: number | null;
  customer_source_champion_qty: number;
  customer_blend_qty: number;
}

export type CustomerBlendOverlayStatus = "idle" | "loading" | "error" | "empty" | "ready";

export function toCustomerBlendChartPoints(
  months: CustomerBlendSeriesMonth[]
): CustomerBlendChartPoint[] {
  return months.map((month) => ({
    month: month.forecast_month,
    customer_bottom_up_qty: month.normalized_customer_qty,
    customer_source_champion_qty: month.champion_qty,
    customer_blend_qty: month.blended_qty,
  }));
}

function monthKey(value: unknown): string {
  return String(value ?? "").slice(0, 7);
}

/** Overlay customer blend fields on existing chart points and retain blend-only months. */
export function mergeCustomerBlendOverlay(
  base: Record<string, unknown>[],
  overlay: CustomerBlendChartPoint[]
): Record<string, unknown>[] {
  const merged = base.map((point) => ({ ...point }));
  const indexByMonth = new Map(
    merged.map((point, index) => [monthKey(point.month), index] as const)
  );

  for (const overlayPoint of overlay) {
    const key = monthKey(overlayPoint.month);
    const existingIndex = indexByMonth.get(key);
    if (existingIndex === undefined) {
      indexByMonth.set(key, merged.length);
      merged.push({ ...overlayPoint });
      continue;
    }
    const existing = merged[existingIndex];
    merged[existingIndex] = {
      ...existing,
      ...overlayPoint,
      month: existing.month,
    };
  }

  return merged.sort((left, right) => String(left.month).localeCompare(String(right.month)));
}

export function hasCustomerBlendData(points: Record<string, unknown>[]): boolean {
  return points.some((point) => point[CUSTOMER_BLEND_KEYS.blend] != null);
}
