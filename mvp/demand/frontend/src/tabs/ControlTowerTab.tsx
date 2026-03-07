/**
 * IPfeature15 — Unified Inventory Control Tower
 *
 * 5-zone command center: KPI strip, health overview, exception queue,
 * top-critical items, and 6-month trend chart.
 */
import { useState, useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from "recharts";
import {
  controlTowerKeys,
  fetchControlTowerKpis,
  fetchControlTowerAlerts,
  fetchControlTowerTopCritical,
  fetchControlTowerTrend,
  STALE,
  type ControlTowerKpis,
  type ControlTowerAlert,
  type ControlTowerCriticalItem,
} from "@/api/queries";
import { formatFixed as fmt } from "@/lib/formatters";

function pct(n: number | null | undefined, scale = 1): string {
  if (n == null) return "—";
  return `${(Number(n) * scale).toFixed(1)}%`;
}

// Severity color mapping
const SEV_COLORS: Record<string, string> = {
  critical: "bg-red-100 text-red-800 border-red-200",
  high:     "bg-orange-100 text-orange-800 border-orange-200",
  medium:   "bg-amber-100 text-amber-800 border-amber-200",
  low:      "bg-gray-100 text-gray-700 border-gray-200",
};

const SOURCE_LABELS: Record<string, string> = {
  exception:     "EXCEPTION",
  demand_signal: "SIGNAL",
  health_drop:   "HEALTH",
};

// PL-014: trend window options (months)
const TREND_WINDOWS = [
  { label: "3M",  value: 3 },
  { label: "6M",  value: 6 },
  { label: "12M", value: 12 },
  { label: "18M", value: 18 },
] as const;

export default function ControlTowerTab({ onNavigate }: { onNavigate?: (tab: string) => void }) {
  const queryClient = useQueryClient();
  const [trendMonths, setTrendMonths] = useState(6); // PL-014
  const [showMoreKpis, setShowMoreKpis] = useState(false);

  // Reduce auto-refresh interval when tab is hidden (10 min instead of 2 min)
  const [isTabVisible, setIsTabVisible] = useState(!document.hidden);
  useEffect(() => {
    const handler = () => setIsTabVisible(!document.hidden);
    document.addEventListener("visibilitychange", handler);
    return () => document.removeEventListener("visibilitychange", handler);
  }, []);
  const refreshInterval = isTabVisible ? 120_000 : 600_000;

  const { data: kpis, isLoading: kpisLoading } = useQuery({
    queryKey: controlTowerKeys.kpis(),
    queryFn: fetchControlTowerKpis,
    staleTime: STALE,
    refetchInterval: refreshInterval,
  });

  const { data: alertsData } = useQuery({
    queryKey: controlTowerKeys.alerts({ limit: 10 }),
    queryFn: () => fetchControlTowerAlerts({ limit: 10 }),
    staleTime: STALE,
    refetchInterval: refreshInterval,
  });

  const { data: topCritical } = useQuery({
    queryKey: controlTowerKeys.topCritical(10),
    queryFn: () => fetchControlTowerTopCritical(10),
    staleTime: STALE,
    refetchInterval: refreshInterval,
  });

  const { data: trendData } = useQuery({
    queryKey: controlTowerKeys.trend(trendMonths),
    queryFn: () => fetchControlTowerTrend(trendMonths),
    staleTime: STALE,
    refetchInterval: 3_600_000,
  });

  const handleRefresh = () => {
    queryClient.invalidateQueries({ queryKey: ["ct-"] });
  };

  const h = kpis?.health;
  const ex = kpis?.exceptions;
  const fr = kpis?.fill_rate;
  const ds = kpis?.demand_signals;
  const im = kpis?.intramonth;

  return (
    <div className="flex flex-col gap-4 p-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Inventory Control Tower</h2>
        <button
          className="text-xs rounded border px-3 py-1 hover:bg-muted"
          onClick={handleRefresh}
        >
          Refresh Now
        </button>
      </div>

      {kpisLoading && <p className="text-sm text-muted-foreground">Loading control tower data...</p>}

      {/* ZONE 1: KPI Strip — 4 primary KPIs (MAX_PRIMARY_KPIS = 4) */}
      <div className="space-y-2">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <KpiCard
            label="Portfolio Health"
            value={h?.avg_health_score != null ? `${h.avg_health_score.toFixed(0)}/100` : "—"}
            color={h?.avg_health_score == null ? undefined : h.avg_health_score >= 80 ? "green" : h.avg_health_score >= 60 ? "amber" : "red"}
          />
          <KpiCard
            label="Open Exceptions"
            value={`${ex?.open_exceptions_total ?? 0}`}
            badge={ex?.critical_exceptions ? `${ex.critical_exceptions} critical` : undefined}
            badgeColor="red"
            color={ex?.critical_exceptions ? "red" : undefined}
          />
          <KpiCard
            label="Fill Rate (3m)"
            value={fr?.portfolio_fill_rate_3m != null ? pct(fr.portfolio_fill_rate_3m, 100) : "—"}
            color={fr?.portfolio_fill_rate_3m == null ? undefined : fr.portfolio_fill_rate_3m >= 0.95 ? "green" : fr.portfolio_fill_rate_3m >= 0.90 ? "amber" : "red"}
          />
          <KpiCard
            label="$ at Risk"
            value={ex?.critical_exceptions ? `${ex.critical_exceptions} critical` : "—"}
            color={ex?.critical_exceptions ? "red" : "green"}
          />
        </div>

        {/* Secondary KPIs — shown on demand */}
        {showMoreKpis && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <KpiCard
              label="Stockout Items (MTD)"
              value={`${im?.items_with_stockout_this_month ?? 0}`}
              color={im?.extended_stockouts_this_month ? "red" : undefined}
            />
            <KpiCard
              label="Below SS %"
              value={h?.below_ss_pct != null ? pct(h.below_ss_pct, 100) : "—"}
              color={h?.below_ss_pct == null ? undefined : h.below_ss_pct < 0.05 ? "green" : h.below_ss_pct < 0.20 ? "amber" : "red"}
            />
            <KpiCard
              label="Urgent Signals"
              value={`${ds?.urgent_demand_signals ?? 0}`}
              color={ds?.urgent_demand_signals ? "red" : "green"}
            />
            <KpiCard
              label="Extended Stockouts"
              value={`${im?.extended_stockouts_this_month ?? 0}`}
              color={im?.extended_stockouts_this_month ? "red" : undefined}
            />
          </div>
        )}

        <button
          onClick={() => setShowMoreKpis((v) => !v)}
          className="text-[11px] text-muted-foreground hover:text-foreground underline-offset-2 hover:underline"
        >
          {showMoreKpis ? "▲ Fewer metrics" : "▼ More metrics"}
        </button>
      </div>

      {/* ZONE 2 + ZONE 3: Health Overview & Exception Queue */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* ZONE 2: Health Distribution */}
        <div className="rounded-lg border bg-card p-4 space-y-3">
          <h3 className="text-sm font-semibold">Health Distribution</h3>
          {h && (
            <div className="space-y-2">
              {[
                { tier: "healthy", count: h.healthy_count, color: "bg-green-500" },
                { tier: "monitor", count: h.monitor_count, color: "bg-amber-500" },
                { tier: "at_risk", count: h.at_risk_count, color: "bg-orange-500" },
                { tier: "critical", count: h.critical_count, color: "bg-red-600" },
              ].map(({ tier, count, color }) => (
                <div key={tier} className="flex items-center gap-2 text-xs">
                  <div className={`w-2 h-2 rounded-full ${color}`} />
                  <span className="capitalize w-16">{tier.replace("_", " ")}</span>
                  <div className="flex-1 rounded-full bg-muted h-2">
                    <div
                      className={`${color} h-2 rounded-full`}
                      style={{ width: `${h.total_dfus > 0 ? (count / h.total_dfus) * 100 : 0}%` }}
                    />
                  </div>
                  <span className="w-10 text-right">{count.toLocaleString()}</span>
                </div>
              ))}
              <p className="text-xs text-muted-foreground mt-2">
                Avg health: {h.avg_health_score?.toFixed(1) ?? "—"} |
                Below SS: {h.below_ss_count.toLocaleString()} DFUs
              </p>
            </div>
          )}
        </div>

        {/* ZONE 3: Exception Queue */}
        <div className="rounded-lg border bg-card p-4 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold">
              Open Exceptions ({ex?.open_exceptions_total ?? 0})
            </h3>
            {onNavigate && (
              <button
                onClick={() => onNavigate("exceptions")}
                className="text-xs text-primary hover:underline"
              >
                View All →
              </button>
            )}
          </div>
          {alertsData?.alerts && alertsData.alerts.length > 0 ? (
            <div className="space-y-2 max-h-52 overflow-y-auto">
              {alertsData.alerts.map((alert: ControlTowerAlert) => (
                <div
                  key={alert.alert_id}
                  className={`rounded border p-2 text-xs ${SEV_COLORS[alert.severity] ?? ""}`}
                >
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-bold text-[10px]">
                      {SOURCE_LABELS[alert.source] ?? alert.source}
                    </span>
                    <span className="font-medium">{alert.item_no} @ {alert.loc}</span>
                    <span className="ml-auto capitalize font-semibold">{alert.severity}</span>
                  </div>
                  <p className="mt-1 text-muted-foreground leading-tight">{alert.description}</p>
                  <p className="mt-0.5 italic">{alert.action}</p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-green-600 dark:text-green-400">
              ✓ No open exceptions matching current thresholds.
              {onNavigate && (
                <button onClick={() => onNavigate("invPlanning")} className="ml-1 underline">
                  Adjust in Inv. Planning →
                </button>
              )}
            </p>
          )}
        </div>
      </div>

      {/* ZONE 4: Top-10 Critical Items */}
      <div className="rounded-lg border bg-card p-4 space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold">Top Critical Items (worst health first)</h3>
          {onNavigate && (
            <button
              onClick={() => onNavigate("inventory")}
              className="text-xs text-primary hover:underline"
            >
              View Inventory →
            </button>
          )}
        </div>
        {topCritical?.items && topCritical.items.length > 0 ? (
          <div className="flex gap-3 overflow-x-auto pb-2">
            {topCritical.items.map((item: ControlTowerCriticalItem) => (
              <div
                key={`${item.item_no}-${item.loc}`}
                className={`min-w-[180px] rounded-lg border p-3 text-xs space-y-1 ${
                  item.health_tier === "critical" ? "border-red-400 bg-red-50 dark:bg-red-900/20" :
                  item.health_tier === "at_risk"  ? "border-orange-400 bg-orange-50 dark:bg-orange-900/20" :
                  "border-amber-300 bg-amber-50 dark:bg-amber-900/20"
                }`}
              >
                <p className="font-bold truncate">{item.item_no}</p>
                <p className="text-muted-foreground">{item.loc}</p>
                <p>
                  Health:{" "}
                  <span className="font-semibold text-red-600">
                    {item.health_score?.toFixed(0) ?? "—"}/100
                  </span>
                </p>
                <p>DOS: {fmt(item.current_dos)} / {fmt(item.target_dos_min)}–{fmt(item.target_dos_max)}d</p>
                <p>SS Coverage: {pct(item.ss_coverage, 100)}</p>
                {item.is_below_ss && (
                  <p className="text-red-600 font-semibold">⚠ Below SS</p>
                )}
                {item.recommended_order_qty != null && (
                  <p>Order: {Math.round(item.recommended_order_qty).toLocaleString()} units</p>
                )}
                {item.fill_rate_last_3m != null && (
                  <p>Fill Rate: {pct(item.fill_rate_last_3m, 100)}</p>
                )}
                {item.stockout_days_this_month > 0 && (
                  <p className="text-red-600">Stockout: {item.stockout_days_this_month}d MTD</p>
                )}
                {onNavigate && (
                  <button
                    onClick={() => onNavigate("dfuAnalysis")}
                    className="mt-1 w-full text-center rounded border border-current px-2 py-0.5 text-[10px] font-medium hover:opacity-80"
                  >
                    View Detail →
                  </button>
                )}
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-green-600 dark:text-green-400">
            ✓ No critical items — all DFUs within health thresholds.
          </p>
        )}
      </div>

      {/* ZONE 5: 6-Month Trend */}
      <div className="rounded-lg border bg-card p-4 space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold">Portfolio Trend</h3>
          <div className="flex gap-1">
            {TREND_WINDOWS.map((w) => (
              <button
                key={w.value}
                onClick={() => setTrendMonths(w.value)}
                className={`rounded px-2 py-0.5 text-[10px] font-medium transition-colors ${
                  trendMonths === w.value
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted text-muted-foreground hover:bg-muted/80"
                }`}
              >
                {w.label}
              </button>
            ))}
          </div>
        </div>
        {trendData?.trend && trendData.trend.length > 0 ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={trendData.trend}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month_start" tick={{ fontSize: 10 }} />
              <YAxis yAxisId="left" domain={[0, 100]} tick={{ fontSize: 10 }} />
              <YAxis yAxisId="right" orientation="right" domain={[0, 1]}
                tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 10 }} />
              <Tooltip />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              <Line yAxisId="left" type="monotone" dataKey="avg_health_score"
                stroke="#3b82f6" dot={false} strokeWidth={2} name="Avg Health Score" />
              <Line yAxisId="right" type="monotone" dataKey="fill_rate"
                stroke="#10b981" dot={false} strokeWidth={2} name="Fill Rate" />
              <Line yAxisId="right" type="monotone" dataKey="stockout_day_rate"
                stroke="#ef4444" dot={false} strokeWidth={1.5} name="Stockout Day Rate" strokeDasharray="4 4" />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-xs text-muted-foreground">No trend data available.</p>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// KPI Card helper
// ---------------------------------------------------------------------------
function KpiCard({
  label,
  value,
  badge,
  badgeColor = "gray",
  color,
}: {
  label: string;
  value: string;
  badge?: string;
  badgeColor?: string;
  color?: "green" | "amber" | "red";
}) {
  const textColor =
    color === "green" ? "text-green-600"
    : color === "amber" ? "text-amber-600"
    : color === "red"  ? "text-red-600"
    : "";

  return (
    <div className="rounded-lg border bg-card p-3">
      <p className="text-xs text-muted-foreground truncate">{label}</p>
      <p className={`text-xl font-bold ${textColor}`}>{value}</p>
      {badge && (
        <span className={`text-[10px] font-medium ${badgeColor === "red" ? "text-red-600" : "text-muted-foreground"}`}>
          {badge}
        </span>
      )}
    </div>
  );
}
