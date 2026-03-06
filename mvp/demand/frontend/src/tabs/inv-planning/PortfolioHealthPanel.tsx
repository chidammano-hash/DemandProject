import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  healthKeys,
  fetchHealthSummary,
  fetchHealthDetail,
  fetchHealthHeatmap,
  STALE,
  type HealthDetailRow,
} from "@/api/queries";

const PAGE = 50;

const TIER_COLORS: Record<string, string> = {
  healthy:  "#22c55e",
  monitor:  "#3b82f6",
  at_risk:  "#f59e0b",
  critical: "#ef4444",
};

const TIER_LABEL: Record<string, string> = {
  healthy:  "Healthy",
  monitor:  "Monitor",
  at_risk:  "At Risk",
  critical: "Critical",
};

export function PortfolioHealthPanel() {
  const [healthTierFilter, setHealthTierFilter] = useState("");
  const [healthDetailOffset, setHealthDetailOffset] = useState(0);

  const { data: healthSummary, isLoading: healthSummaryLoading } = useQuery({
    queryKey: healthKeys.summary(),
    queryFn: () => fetchHealthSummary(),
    staleTime: STALE.FIVE_MIN,
  });

  const { data: healthDetail, isLoading: healthDetailLoading } = useQuery({
    queryKey: healthKeys.detail({ health_tier: healthTierFilter || undefined, limit: PAGE, offset: healthDetailOffset }),
    queryFn: () =>
      fetchHealthDetail({
        health_tier: healthTierFilter || undefined,
        limit: PAGE,
        offset: healthDetailOffset,
        sort_by: "health_score",
        sort_dir: "asc",
      }),
    staleTime: STALE.FIVE_MIN,
  });

  const { data: healthHeatmap } = useQuery({
    queryKey: healthKeys.heatmap("abc_vol", "variability_class"),
    queryFn: () => fetchHealthHeatmap("abc_vol", "variability_class"),
    staleTime: STALE.TEN_MIN,
  });

  const tierPieData = healthSummary
    ? (["healthy", "monitor", "at_risk", "critical"] as const)
        .map((t) => ({ name: TIER_LABEL[t], value: healthSummary.by_tier[t], tier: t }))
        .filter((d) => d.value > 0)
    : [];

  const healthDetailPages = healthDetail ? Math.ceil(healthDetail.total / PAGE) : 0;
  const healthDetailPage = Math.floor(healthDetailOffset / PAGE) + 1;

  return (
    <div>
      <h3 className="text-base font-semibold text-foreground mb-3">Portfolio Health Score</h3>

      {/* Health KPI cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
        {(["healthy", "monitor", "at_risk", "critical"] as const).map((tier) => {
          const count = healthSummary?.by_tier[tier] ?? 0;
          const total = healthSummary?.total_dfus ?? 0;
          const pct = total > 0 ? ((count / total) * 100).toFixed(1) : "0.0";
          const colors: Record<string, string> = {
            healthy:  "border-green-200 bg-green-50 dark:bg-green-950",
            monitor:  "border-blue-200 bg-blue-50 dark:bg-blue-950",
            at_risk:  "border-amber-200 bg-amber-50 dark:bg-amber-950",
            critical: "border-red-200 bg-red-50 dark:bg-red-950",
          };
          const textColors: Record<string, string> = {
            healthy:  "text-green-700 dark:text-green-300",
            monitor:  "text-blue-700 dark:text-blue-300",
            at_risk:  "text-amber-700 dark:text-amber-300",
            critical: "text-red-700 dark:text-red-300",
          };
          return (
            <button
              key={tier}
              className={`rounded-lg border p-3 text-left cursor-pointer transition-opacity ${colors[tier]} ${healthTierFilter === tier ? "ring-2 ring-offset-1 ring-current opacity-100" : "opacity-90 hover:opacity-100"}`}
              onClick={() => {
                setHealthTierFilter(healthTierFilter === tier ? "" : tier);
                setHealthDetailOffset(0);
              }}
            >
              <p className={`text-xs font-medium uppercase tracking-wide ${textColors[tier]}`}>
                {TIER_LABEL[tier]}
              </p>
              <p className={`text-2xl font-bold mt-1 ${textColors[tier]}`}>
                {healthSummaryLoading ? "…" : count.toLocaleString()}
              </p>
              <p className={`text-xs mt-0.5 ${textColors[tier]} opacity-80`}>{pct}% of portfolio</p>
            </button>
          );
        })}
      </div>

      {/* Health donut + component scores */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <div className="rounded-lg border bg-card p-4">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
            Health Distribution
          </h4>
          {healthSummaryLoading ? (
            <div className="text-xs text-muted-foreground">Loading…</div>
          ) : tierPieData.length > 0 ? (
            <div className="flex items-center gap-4">
              <div className="h-40 w-40 shrink-0">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={tierPieData}
                      cx="50%"
                      cy="50%"
                      innerRadius={40}
                      outerRadius={64}
                      paddingAngle={2}
                      dataKey="value"
                    >
                      {tierPieData.map((entry) => (
                        <Cell key={entry.tier} fill={TIER_COLORS[entry.tier]} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(v: number) => [v.toLocaleString(), "DFUs"]} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="flex flex-col gap-1.5 text-xs">
                {tierPieData.map((d) => (
                  <div key={d.tier} className="flex items-center gap-1.5">
                    <span
                      className="w-2.5 h-2.5 rounded-sm shrink-0"
                      style={{ background: TIER_COLORS[d.tier] }}
                    />
                    <span className="text-foreground">{d.name}</span>
                    <span className="text-muted-foreground ml-auto pl-4">{d.value.toLocaleString()}</span>
                  </div>
                ))}
                <div className="mt-1 pt-1 border-t text-muted-foreground">
                  Avg score: <span className="font-semibold text-foreground">{healthSummary?.avg_health_score?.toFixed(1) ?? "—"}</span>
                </div>
              </div>
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">No health score data. Run health-schema + health-refresh.</p>
          )}
        </div>

        <div className="rounded-lg border bg-card p-4">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
            Score Components (avg pts / 25)
          </h4>
          {healthSummaryLoading ? (
            <div className="text-xs text-muted-foreground">Loading…</div>
          ) : healthSummary?.component_avgs ? (
            <div className="flex flex-col gap-2">
              {(
                [
                  ["SS Coverage",       healthSummary.component_avgs.ss_coverage],
                  ["DOS Target",        healthSummary.component_avgs.dos_target],
                  ["Stockout Risk",     healthSummary.component_avgs.stockout_risk],
                  ["Forecast Accuracy", healthSummary.component_avgs.forecast_accuracy],
                ] as [string, number | null][]
              ).map(([label, val]) => {
                const pct = val != null ? Math.min(100, (val / 25) * 100) : 0;
                return (
                  <div key={label}>
                    <div className="flex justify-between text-xs mb-0.5">
                      <span className="text-muted-foreground">{label}</span>
                      <span className="font-medium text-foreground">{val?.toFixed(1) ?? "—"}</span>
                    </div>
                    <div className="h-1.5 rounded-full bg-muted overflow-hidden">
                      <div
                        className="h-full rounded-full bg-blue-500"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">No data.</p>
          )}
        </div>
      </div>

      {/* Heatmap: ABC x Variability */}
      {healthHeatmap && healthHeatmap.cells.length > 0 && (
        <div className="rounded-lg border bg-card p-4 mb-4">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
            Avg Health Score by ABC Class x Variability
          </h4>
          <div className="overflow-x-auto">
            <table className="text-xs w-full">
              <thead>
                <tr className="border-b text-muted-foreground">
                  <th className="text-left py-1 pr-3">Variability \ ABC</th>
                  {healthHeatmap.x_labels.map((x) => (
                    <th key={x} className="text-center py-1 px-2">{x}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {healthHeatmap.y_labels.map((y) => (
                  <tr key={y} className="border-b last:border-0">
                    <td className="py-1 pr-3 font-medium">{y}</td>
                    {healthHeatmap.x_labels.map((x) => {
                      const cell = healthHeatmap.cells.find((c) => c.x === x && c.y === y);
                      const score = cell?.avg_health_score;
                      const bg =
                        score == null ? ""
                        : score >= 80 ? "bg-green-100 text-green-800"
                        : score >= 60 ? "bg-blue-100 text-blue-800"
                        : score >= 40 ? "bg-amber-100 text-amber-800"
                        : "bg-red-100 text-red-800";
                      return (
                        <td key={x} className={`text-center py-1 px-2 rounded font-medium ${bg}`}>
                          {score != null ? score.toFixed(0) : "—"}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Health detail table */}
      <div className="rounded-lg border bg-card p-4">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-sm font-semibold text-foreground">
            Health Detail
            {healthTierFilter && (
              <span className={`ml-2 px-1.5 py-0.5 rounded text-xs font-medium ${
                healthTierFilter === "healthy"  ? "bg-green-100 text-green-800" :
                healthTierFilter === "monitor"  ? "bg-blue-100 text-blue-800" :
                healthTierFilter === "at_risk"  ? "bg-amber-100 text-amber-800" :
                "bg-red-100 text-red-800"
              }`}>
                {TIER_LABEL[healthTierFilter]}
              </span>
            )}
          </h4>
          {healthTierFilter && (
            <button
              className="text-xs text-muted-foreground underline"
              onClick={() => { setHealthTierFilter(""); setHealthDetailOffset(0); }}
            >
              Clear filter
            </button>
          )}
        </div>
        {healthDetailLoading ? (
          <div className="text-xs text-muted-foreground">Loading…</div>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="text-xs w-full">
                <thead>
                  <tr className="border-b text-muted-foreground">
                    <th className="text-left py-1 pr-3">Item</th>
                    <th className="text-left py-1 pr-3">Location</th>
                    <th className="text-center py-1 pr-3">Score</th>
                    <th className="text-center py-1 pr-3">Tier</th>
                    <th className="text-right py-1 pr-3">SS Cov</th>
                    <th className="text-right py-1 pr-3">DOS Tgt</th>
                    <th className="text-right py-1 pr-3">Stockout</th>
                    <th className="text-right py-1">Fcst Acc</th>
                  </tr>
                </thead>
                <tbody>
                  {(healthDetail?.rows ?? []).map((row: HealthDetailRow) => {
                    const tierBg: Record<string, string> = {
                      healthy:  "bg-green-100 text-green-800",
                      monitor:  "bg-blue-100 text-blue-800",
                      at_risk:  "bg-amber-100 text-amber-800",
                      critical: "bg-red-100 text-red-800",
                    };
                    return (
                      <tr key={`${row.item_no}-${row.loc}`} className="border-b last:border-0">
                        <td className="py-1 pr-3 font-mono">{row.item_no}</td>
                        <td className="py-1 pr-3">{row.loc}</td>
                        <td className="text-center py-1 pr-3 font-bold">{row.health_score}</td>
                        <td className="text-center py-1 pr-3">
                          <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${tierBg[row.health_tier] ?? ""}`}>
                            {TIER_LABEL[row.health_tier] ?? row.health_tier}
                          </span>
                        </td>
                        <td className="text-right py-1 pr-3">{row.score_ss_coverage}</td>
                        <td className="text-right py-1 pr-3">{row.score_dos_target}</td>
                        <td className="text-right py-1 pr-3">{row.score_stockout_risk}</td>
                        <td className="text-right py-1">{row.score_forecast_accuracy}</td>
                      </tr>
                    );
                  })}
                  {(healthDetail?.rows ?? []).length === 0 && (
                    <tr>
                      <td colSpan={8} className="text-center py-4 text-muted-foreground">
                        No records. Run <code>make health-schema health-refresh</code> to populate.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
            {healthDetailPages > 1 && (
              <div className="flex items-center gap-2 mt-3 text-xs">
                <button
                  disabled={healthDetailOffset === 0}
                  onClick={() => setHealthDetailOffset(Math.max(0, healthDetailOffset - PAGE))}
                  className="px-2 py-1 rounded border disabled:opacity-40"
                >
                  Prev
                </button>
                <span className="text-muted-foreground">
                  Page {healthDetailPage} of {healthDetailPages}
                  {healthDetail && ` · ${healthDetail.total.toLocaleString()} total`}
                </span>
                <button
                  disabled={healthDetailPage >= healthDetailPages}
                  onClick={() => setHealthDetailOffset(healthDetailOffset + PAGE)}
                  className="px-2 py-1 rounded border disabled:opacity-40"
                >
                  Next
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
