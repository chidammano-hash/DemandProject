/**
 * TodaysPlanBanner — compact landing summary at the top of the Inventory Planning tab.
 *
 * Shows urgent/high priority counts, financial-at-risk total, and the top 3 action
 * items from the unified action feed. Collapsible so planners can dismiss after review.
 *
 * Issue #14 — "Today's Plan" Landing Dashboard
 */

import { useQuery } from "@tanstack/react-query";
import { Badge } from "@/components/ui/badge";
import {
  insightKeys,
  fetchActionFeed,
  fetchDailyBriefing,
  STALE_INSIGHTS,
  type ActionFeedItem,
} from "@/api/queries";
import { ChevronUp } from "lucide-react";
import { formatAsOfDate, formatCompactCurrency, shouldRenderStat } from "./todaysPlanFormat";

// ---------------------------------------------------------------------------
// Priority badge sub-component
// ---------------------------------------------------------------------------
const COLOR_MAP: Record<string, string> = {
  red: "bg-red-50 text-red-700 border-red-200 dark:bg-red-950/30 dark:text-red-400 dark:border-red-800",
  amber: "bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-950/30 dark:text-amber-400 dark:border-amber-800",
  blue: "bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950/30 dark:text-blue-400 dark:border-blue-800",
};

function PriorityBadge({
  label,
  count,
  value,
  color,
  title,
}: {
  label: string;
  count?: number;
  value?: string;
  color: string;
  /** Optional tooltip naming the metric's basis (e.g. the at-risk window). */
  title?: string;
}) {
  // U2.2 — comma-format the integer count so the priority ribbon matches the
  // comma-formatted Action-Feed KPIs stacked directly below it on the same tab.
  const display = value ?? (count != null ? count.toLocaleString() : "0");
  return (
    <div
      className={`rounded-md border px-3 py-1.5 ${COLOR_MAP[color] ?? COLOR_MAP.blue}`}
      title={title}
    >
      <p className="text-[10px] text-muted-foreground">{label}</p>
      <p className="text-lg font-bold leading-tight">{display}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export function TodaysPlanBanner({ onCollapse }: { onCollapse: () => void }) {
  const { data } = useQuery({
    queryKey: insightKeys.actionFeed(),
    queryFn: fetchActionFeed,
    staleTime: STALE_INSIGHTS.ONE_MIN,
  });

  const { data: briefing } = useQuery({
    queryKey: insightKeys.dailyBriefing(),
    queryFn: fetchDailyBriefing,
    staleTime: STALE_INSIGHTS.FIVE_MIN,
  });

  const summary = data?.summary;
  const topActions: ActionFeedItem[] = data?.actions?.slice(0, 3) ?? [];
  const stats = briefing?.stats;

  // U1.1 — stamp the banner with the planning/data as-of date the action feed
  // and KPIs are computed against (briefing.date, e.g. "2026-04-02"), NOT the
  // browser wall clock. Deriving a user-facing data anchor from `new Date()`
  // makes "Today's Plan" imply same-day data when the figures are frozen to a
  // prior planning date. Falls back to "Today's Plan" with no date until the
  // briefing resolves.
  const asOfLabel = formatAsOfDate(briefing?.date);

  return (
    <div className="rounded-lg border bg-card p-4 mb-4 max-h-[170px] overflow-hidden">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold">Today&apos;s Plan</h2>
        <div className="flex items-center gap-2">
          {asOfLabel && (
            <span className="text-[10px] text-muted-foreground">
              Plan as of {asOfLabel}
            </span>
          )}
          <button
            onClick={onCollapse}
            className="text-muted-foreground hover:text-foreground transition-colors p-0.5 rounded"
            title="Collapse plan banner"
          >
            <ChevronUp size={14} />
          </button>
        </div>
      </div>

      <div className="flex gap-4">
        {/* Priority ribbon */}
        <div className="flex gap-2 flex-shrink-0">
          <PriorityBadge label="Urgent" count={summary?.critical ?? 0} color="red" />
          <PriorityBadge label="High" count={summary?.high ?? 0} color="amber" />
          {/* F2.1 — name the at-risk basis on the chip (label + tooltip) so this
              banner figure is self-explaining and isn't read as the same metric
              as the Command Center "Order Value at Risk" tile (a different,
              larger number). The basis text matches the Action Feed panel
              sublabel on this same tab. */}
          <PriorityBadge
            label="$ at Risk"
            value={formatCompactCurrency(summary?.financial_at_risk)}
            color="blue"
            title={summary?.financial_at_risk_basis}
          />
        </div>

        {/* Top 3 actions */}
        {topActions.length > 0 && (
          <div className="flex-1 min-w-0 space-y-1">
            <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
              Top Actions
            </p>
            {topActions.map((action) => (
              <div key={action.id} className="flex items-center gap-2 text-xs">
                <Badge
                  variant={action.severity === "critical" ? "default" : "outline"}
                  className="text-[9px] flex-shrink-0"
                >
                  {action.severity}
                </Badge>
                <span className="truncate">{action.title}</span>
                {action.financial_impact != null && action.financial_impact > 0 && (
                  <span className="ml-auto font-medium text-red-600 dark:text-red-400 flex-shrink-0">
                    ${action.financial_impact.toLocaleString(undefined, {
                      maximumFractionDigits: 0,
                    })}
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Stats row from daily briefing */}
      {stats && (
        <div className="flex gap-4 text-xs text-muted-foreground mt-2">
          <span>
            {shouldRenderStat(stats.total_skus) ? stats.total_skus.toLocaleString() : "—"} SKUs
          </span>
          <span className="text-red-600 dark:text-red-400">
            {stats.below_ss_count.toLocaleString()} at risk
          </span>
          {shouldRenderStat(stats.excess_count) && (
            <span className="text-amber-600 dark:text-amber-400">
              {stats.excess_count.toLocaleString()} excess (
              {formatCompactCurrency(stats.total_excess_value)})
            </span>
          )}
          {stats.avg_health_score != null && (
            <span>
              Health: {stats.avg_health_score.toFixed(0)}/100
            </span>
          )}
        </div>
      )}
    </div>
  );
}
