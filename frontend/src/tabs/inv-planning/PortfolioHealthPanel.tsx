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
  type HealthSummaryFilters,
} from "@/api/queries";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";
import { EmptyState } from "@/components/EmptyState";
import { severityBadgeClass } from "@/lib/severityBadge";

// U5.1 — map a 0–100 health score / tier to a themed status pill (success /
// info / warning / critical), each carrying a `dark:` tint so the chip stays
// legible in Dark theme (was hand-rolled `bg-*-100 text-*-800`, Light-only).
function healthScoreBadge(score: number | null | undefined): string {
  if (score == null) return "";
  if (score >= 80) return severityBadgeClass("success");
  if (score >= 60) return severityBadgeClass("info");
  if (score >= 40) return severityBadgeClass("warning");
  return severityBadgeClass("critical");
}

const HEALTH_TIER_BADGE: Record<string, string> = {
  healthy: severityBadgeClass("success"),
  monitor: severityBadgeClass("info"),
  at_risk: severityBadgeClass("warning"),
  critical: severityBadgeClass("critical"),
};
import { RecommendedActionCard } from "@/components/RecommendedActionCard";
import { TableSkeleton } from "@/components/Skeleton";
import { Activity, HelpCircle } from "lucide-react";

const PAGE = 50;

const SCORE_COMPONENTS: Record<string, { label: string; businessLabel: string; max: number; description: string }> = {
  ss_coverage: {
    label: "SS Coverage",
    businessLabel: "Buffer Adequacy",
    max: 25,
    description: "How well on-hand inventory covers the safety buffer target. 25/25 = fully covered with margin.",
  },
  dos_adequacy: {
    label: "DOS Adequacy",
    businessLabel: "Supply Coverage",
    max: 25,
    description: "How many days your current inventory can sustain demand. 25/25 = meets or exceeds target days.",
  },
  fill_rate: {
    label: "Fill Rate",
    businessLabel: "Order Fill Rate",
    max: 25,
    description: "Percentage of customer orders fulfilled completely on time. 25/25 = meets service level target.",
  },
  policy_compliance: {
    label: "Policy Compliance",
    businessLabel: "Policy Compliance",
    max: 25,
    description: "Percentage of items following their assigned ordering policy. 25/25 = 95%+ compliant.",
  },
  // aliases used by the existing component_avgs keys
  dos_target: {
    label: "DOS Target",
    businessLabel: "Supply Coverage",
    max: 25,
    description: "How many days your current inventory can sustain demand. 25/25 = meets or exceeds target days.",
  },
  stockout_risk: {
    label: "Stockout Risk",
    businessLabel: "Stockout Protection",
    max: 25,
    description: "How well-protected you are against running out of stock. 25/25 = near-zero stockout risk.",
  },
  forecast_accuracy: {
    label: "Forecast Accuracy",
    businessLabel: "Forecast Fit",
    max: 25,
    description: "How closely demand forecasts match actual sales. 25/25 = forecast error within acceptable range.",
  },
};

function scoreColor(val: number | null | undefined): string {
  if (val == null) return "";
  if (val >= 20) return "text-green-600";
  if (val >= 15) return "text-blue-600";
  if (val >= 10) return "text-amber-600";
  return "text-red-600";
}

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

  const { filters } = useGlobalFilterContext();
  const gf = {
    brand: filters.brand.length > 0 ? filters.brand.join(",") : undefined,
    category: filters.category.length > 0 ? filters.category.join(",") : undefined,
    market: filters.market.length > 0 ? filters.market.join(",") : undefined,
    item: filters.item.length === 1 ? filters.item[0] : undefined,
    location: filters.location.length === 1 ? filters.location[0] : undefined,
  };

  const healthFilters: HealthSummaryFilters = {};
  const { data: healthSummary, isLoading: healthSummaryLoading } = useQuery({
    queryKey: healthKeys.summary(healthFilters),
    queryFn: () => fetchHealthSummary(healthFilters),
    staleTime: STALE.FIVE_MIN,
  });

  const { data: healthDetail, isLoading: healthDetailLoading } = useQuery({
    queryKey: healthKeys.detail({ ...gf, health_tier: healthTierFilter || undefined, limit: PAGE, offset: healthDetailOffset }),
    queryFn: () =>
      fetchHealthDetail({
        item: gf.item,
        location: gf.location,
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
      <h3 className="text-base font-semibold text-foreground mb-3">Portfolio Risk Overview</h3>

      {/* Health KPI cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
        {(["healthy", "monitor", "at_risk", "critical"] as const).map((tier) => {
          const count = healthSummary?.by_tier[tier] ?? 0;
          const total = healthSummary?.total_skus ?? 0;
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

      {/* Recommended actions based on current data */}
      {!healthSummaryLoading && healthSummary?.avg_health_score != null && healthSummary.avg_health_score < 60 && (
        <RecommendedActionCard
          severity="critical"
          title={`Portfolio health is ${Math.round(healthSummary.avg_health_score)}/100 — below acceptable threshold`}
          action="Focus on improving buffer adequacy and reducing stockout events"
        />
      )}

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
                    <Tooltip formatter={(v: number) => [v.toLocaleString(), "SKUs"]} />
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
                <div className="mt-1 pt-1 border-t text-muted-foreground flex items-center gap-1 flex-wrap">
                  <span className="flex items-center gap-1">
                    Portfolio Risk
                    <span title="Overall inventory health across all items. Based on buffer adequacy, supply coverage, stockout protection, and forecast fit." className="cursor-help">
                      <HelpCircle className="h-3 w-3 text-muted-foreground/60" strokeWidth={1.5} />
                    </span>
                    :
                  </span>{" "}
                  {(() => {
                    const score = healthSummary?.avg_health_score;
                    if (score == null) return <span className="font-semibold text-foreground">—</span>;
                    const riskLevel = score >= 80 ? "Low" : score >= 60 ? "Medium" : score >= 40 ? "High" : "Critical";
                    const riskColor = healthScoreBadge(score);
                    return (
                      <span className={`px-1.5 py-0.5 rounded text-xs font-semibold ${riskColor}`} title={`Health score: ${score.toFixed(1)}/100`}>
                        {riskLevel}
                      </span>
                    );
                  })()}
                </div>
              </div>
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">No health score data. Run health-schema + health-refresh.</p>
          )}
        </div>

        <div className="rounded-lg border bg-card p-4">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
            Risk Factor Breakdown
          </h4>
          {healthSummaryLoading ? (
            <div className="text-xs text-muted-foreground">Loading…</div>
          ) : healthSummary?.component_avgs ? (
            <>
              <div className="text-xs text-muted-foreground p-2 rounded bg-muted/30 border mb-3">
                <span className="font-medium text-foreground">How healthy is each area? </span>
                <span className="text-green-600">● Excellent</span> ·{" "}
                <span className="text-blue-600 ml-1">● Good</span> ·{" "}
                <span className="text-amber-600 ml-1">● Needs Attention</span> ·{" "}
                <span className="text-red-600 ml-1">● Critical</span>
                <span className="ml-2 text-muted-foreground">(4 areas, each scored out of 25)</span>
              </div>
              <div className="flex flex-col gap-2">
                {(
                  [
                    ["ss_coverage",      "Buffer Adequacy",       healthSummary.component_avgs.ss_coverage],
                    ["dos_target",       "Supply Coverage",        healthSummary.component_avgs.dos_target],
                    ["stockout_risk",    "Stockout Protection",     healthSummary.component_avgs.stockout_risk],
                    ["forecast_accuracy","Forecast Fit", healthSummary.component_avgs.forecast_accuracy],
                  ] as [string, string, number | null][]
                ).map(([key, label, val]) => {
                  const pct = val != null ? Math.min(100, (val / 25) * 100) : 0;
                  const valColor = scoreColor(val);
                  return (
                    <div
                      key={key}
                      title={SCORE_COMPONENTS[key]?.description ?? ""}
                    >
                      <div className="flex justify-between text-xs mb-0.5">
                        <span className="text-muted-foreground">{label}</span>
                        <span className={`font-medium ${valColor || "text-foreground"}`}>
                          {val?.toFixed(1) ?? "—"}
                        </span>
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
            </>
          ) : (
            <p className="text-xs text-muted-foreground">No data.</p>
          )}
        </div>
      </div>

      {/* Heatmap: ABC x Variability */}
      {(healthHeatmap?.cells?.length ?? 0) > 0 && healthHeatmap && (
        <div className="rounded-lg border bg-card p-4 mb-4">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
            Risk by ABC Class x Demand Variability
          </h4>
          <div className="text-xs text-muted-foreground p-2 rounded bg-muted/30 border mb-3">
            <span className="font-medium text-foreground">Risk Levels: </span>
            <span className="text-green-700">● Low Risk (80-100)</span> ·{" "}
            <span className="text-blue-700 ml-1">● Medium Risk (60-79)</span> ·{" "}
            <span className="text-amber-700 ml-1">● High Risk (40-59)</span> ·{" "}
            <span className="text-red-700 ml-1">● Critical (0-39)</span>
          </div>
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
                      const bg = healthScoreBadge(score);
                      const tier =
                        score == null ? ""
                        : score >= 80 ? "Healthy"
                        : score >= 60 ? "Monitor"
                        : score >= 40 ? "At Risk"
                        : "Critical";
                      const cellTitle =
                        score != null
                          ? `${y} × ${x}: avg health score ${score.toFixed(0)} (${tier})`
                          : `${y} × ${x}: no data`;
                      return (
                        <td
                          key={x}
                          className={`text-center py-1 px-2 rounded font-medium ${bg}`}
                          title={cellTitle}
                        >
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
              <span className={`ml-2 px-1.5 py-0.5 rounded text-xs font-medium ${HEALTH_TIER_BADGE[healthTierFilter] ?? severityBadgeClass("critical")}`}>
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
          <TableSkeleton rows={8} cols={8} />
        ) : (healthDetail?.rows ?? []).length === 0 ? (
          <EmptyState
            icon={Activity}
            title="Health scores not yet computed"
            description="Portfolio Health scores combine SS Coverage, Days of Supply adherence, Stockout Risk, and Forecast Accuracy into a 0–100 score per item-location."
            steps={[
              { label: "Apply schema (first time only)", command: "make health-schema" },
              { label: "Refresh health score view", command: "make health-refresh" },
            ]}
          />
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="text-xs w-full">
                <thead>
                  <tr className="border-b text-muted-foreground">
                    <th className="text-left py-1 pr-3">Item</th>
                    <th className="text-left py-1 pr-3">Location</th>
                    <th className="text-center py-1 pr-3" title="Overall health score out of 100">Score</th>
                    <th className="text-center py-1 pr-3">Risk Level</th>
                    <th className="text-right py-1 pr-3" title="Buffer Adequacy: how well safety stock covers the target">Buffer</th>
                    <th className="text-right py-1 pr-3" title="Supply Coverage: days of inventory on hand vs target">Supply</th>
                    <th className="text-right py-1 pr-3" title="Stockout Protection: risk of running out of stock">Stockout</th>
                    <th className="text-right py-1" title="Forecast Fit: how closely forecasts match actual demand">Forecast</th>
                  </tr>
                </thead>
                <tbody>
                  {(healthDetail?.rows ?? []).map((row: HealthDetailRow) => {
                    const tierBg = HEALTH_TIER_BADGE;
                    return (
                      <tr key={`${row.item_id}-${row.loc}`} className="border-b last:border-0">
                        <td className="py-1 pr-3 font-mono">{row.item_id}</td>
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
