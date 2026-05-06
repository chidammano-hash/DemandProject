/**
 * SKU Features tab — formatting and badge helpers.
 */
import { TREND_LABELS } from "./constants";

export function formatNumber(v: number | null | undefined, decimals = 2): string {
  if (v == null) return "—";
  return v.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: decimals,
  });
}

export function formatPct(v: number | null | undefined): string {
  if (v == null) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

export function relativeTime(ts: string): string {
  const diff = Date.now() - new Date(ts).getTime();
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return "Just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

export function trendLabel(v: number | null): string {
  if (v == null) return "—";
  return TREND_LABELS[String(v)] ?? String(v);
}

export type BadgeCategory = "seasonality" | "variability" | "trend";

export function badgeClass(value: string | null, category: BadgeCategory): string {
  if (!value) return "bg-muted text-muted-foreground";
  const map: Record<string, Record<string, string>> = {
    seasonality: {
      none: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
      low: "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300",
      moderate: "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300",
      strong: "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300",
    },
    variability: {
      smooth: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300",
      erratic: "bg-orange-100 text-orange-700 dark:bg-orange-900/40 dark:text-orange-300",
      intermittent: "bg-violet-100 text-violet-700 dark:bg-violet-900/40 dark:text-violet-300",
      lumpy: "bg-rose-100 text-rose-700 dark:bg-rose-900/40 dark:text-rose-300",
    },
    trend: {
      declining: "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300",
      flat: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
      growing: "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300",
    },
  };
  return map[category]?.[value] ?? "bg-muted text-muted-foreground";
}

/** Convert summary.distributions Record<string, number> to chart data array */
export function recordToChartData(
  record: Record<string, number> | undefined,
): { label: string; count: number }[] {
  if (!record) return [];
  return Object.entries(record).map(([label, count]) => ({ label, count }));
}
