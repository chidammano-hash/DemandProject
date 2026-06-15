/**
 * Formatting helpers for the Today's Plan banner.
 *
 * Kept in a dedicated module so the rounding/no-data rules can be unit-tested
 * and shared, ensuring the banner's "At Risk" tile rounds identically to the
 * Action Feed KPI (U8.1) and that unpopulated daily-briefing fields are not
 * rendered as real zeros (U8.2).
 */

/**
 * Compact currency for the "At Risk" tile.
 *
 * Sub-$10K values keep one decimal (so 3598.89 → "$3.6K", matching the Action
 * Feed which shows $3.6K — same metric, identical string, no upward overstating
 * to "$4K"). Values >= $10K drop the decimal to stay compact. Null/zero render
 * the "--" placeholder.
 */
export function formatCompactCurrency(value: number | null | undefined): string {
  if (value == null || value === 0) return "--";
  const thousands = value / 1000;
  const digits = Math.abs(thousands) < 10 ? 1 : 0;
  return `$${thousands.toFixed(digits)}K`;
}

/**
 * Format the planning/data as-of date for the banner header (U1.1).
 *
 * Takes the daily-briefing `date` (an ISO `YYYY-MM-DD` string, e.g.
 * "2026-04-02") and renders it as "Apr 2, 2026". Parsed as a LOCAL date — not
 * via `new Date("2026-04-02")` which is interpreted as UTC midnight and can
 * shift the day backwards in negative-offset timezones. Returns "" for a
 * missing/invalid value so the caller renders no date until the briefing
 * resolves (never the browser wall clock).
 */
export function formatAsOfDate(value: string | null | undefined): string {
  if (!value) return "";
  const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(value);
  if (!match) return "";
  const [, y, m, d] = match;
  const date = new Date(Number(y), Number(m) - 1, Number(d));
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

/**
 * Whether a daily-briefing stat carries real signal.
 *
 * The live `/inv-planning/daily-briefing` payload leaves `total_skus`,
 * `excess_count`, and `total_excess_value` unpopulated (0/null) while
 * `below_ss_count` is a real 3,152 — so a literal "0 SKUs · 3,152 at risk" is
 * self-contradictory. A 0/null stat is treated as no-data and degraded to "—"
 * (mirroring the existing `avg_health_score != null` guard on the same row).
 */
export function shouldRenderStat(value: number | null | undefined): boolean {
  return value != null && value !== 0;
}
