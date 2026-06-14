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
