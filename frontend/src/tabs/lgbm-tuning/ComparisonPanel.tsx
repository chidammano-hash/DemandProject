import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ArrowUp, ArrowDown, Minus } from "lucide-react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ReferenceLine,
} from "recharts";

import {
  lgbmTuningKeys,
  fetchTuningComparison,
  STALE,
  type TuningComparison,
  type ClusterComparison,
  type MonthComparison,
  type ParamDiff,
  type ParamCommon,
  type FeatureDiffs,
  type ConfigDiff,
  type ConfigCommon,
} from "@/api/queries";
import { useChartColors } from "@/hooks/useChartColors";
import { formatPct, formatFixed } from "@/lib/formatters";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { LoadingElement } from "@/components/LoadingElement";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------
type ComparisonPanelProps = {
  baselineId: number;
  candidateId: number;
};

// ---------------------------------------------------------------------------
// Delta indicator
// ---------------------------------------------------------------------------
function DeltaIndicator({ baseline, candidate, lowerIsBetter = false }: {
  baseline: number | null;
  candidate: number | null;
  lowerIsBetter?: boolean;
}) {
  if (baseline == null || candidate == null) {
    return <span className="text-xs text-muted-foreground">--</span>;
  }
  const delta = candidate - baseline;
  if (Math.abs(delta) < 0.01) {
    return (
      <span className="inline-flex items-center gap-0.5 text-xs text-muted-foreground">
        <Minus className="h-3 w-3" />
        <span className="tabular-nums">0.0</span>
      </span>
    );
  }

  const improved = lowerIsBetter ? delta < 0 : delta > 0;
  const Icon = delta > 0 ? ArrowUp : ArrowDown;
  const color = improved
    ? "text-emerald-600 dark:text-emerald-400"
    : "text-red-600 dark:text-red-400";

  return (
    <span className={cn("inline-flex items-center gap-0.5 text-xs", color)}>
      <Icon className="h-3 w-3" />
      <span className="tabular-nums">
        {delta > 0 ? "+" : ""}
        {delta.toFixed(2)}
      </span>
    </span>
  );
}

// ---------------------------------------------------------------------------
// Metric card (side-by-side)
// ---------------------------------------------------------------------------
function MetricCard({ label, baseVal, candVal, lowerIsBetter = false }: {
  label: string;
  baseVal: number | null;
  candVal: number | null;
  lowerIsBetter?: boolean;
}) {
  return (
    <div className="rounded-lg border border-border/60 bg-card px-4 py-3">
      <p className="text-xs text-muted-foreground mb-1">{label}</p>
      <div className="grid grid-cols-3 items-end gap-2">
        <div>
          <p className="text-[10px] text-muted-foreground/70">Baseline</p>
          <p className="text-sm font-semibold tabular-nums">
            {baseVal != null ? formatFixed(baseVal, 2) : "--"}
          </p>
        </div>
        <div>
          <p className="text-[10px] text-muted-foreground/70">Candidate</p>
          <p className="text-sm font-semibold tabular-nums">
            {candVal != null ? formatFixed(candVal, 2) : "--"}
          </p>
        </div>
        <div className="text-right">
          <DeltaIndicator baseline={baseVal} candidate={candVal} lowerIsBetter={lowerIsBetter} />
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Verdict badge
// ---------------------------------------------------------------------------
function VerdictBadge({ verdict }: { verdict: string }) {
  const v = verdict.toUpperCase();
  const variant =
    v === "IMPROVED"
      ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300"
      : v === "DEGRADED"
        ? "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300"
        : "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300";

  return (
    <Badge className={cn("text-xs font-semibold px-3 py-1", variant)}>
      {v}
    </Badge>
  );
}

// ---------------------------------------------------------------------------
// Grouped bar chart (reusable for cluster + month)
// ---------------------------------------------------------------------------
function ComparisonBarChart({
  data,
  xKey,
  baselineLabel,
  candidateLabel,
  chartColors,
  trendColors,
  baselineRef,
  height = 280,
}: {
  data: Array<Record<string, unknown>>;
  xKey: string;
  baselineLabel: string;
  candidateLabel: string;
  chartColors: ReturnType<typeof useChartColors>["chartColors"];
  trendColors: string[];
  baselineRef?: number | null;
  height?: number;
}) {
  if (data.length === 0) return null;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={data}
        margin={{ top: 8, right: 16, left: 0, bottom: 4 }}
      >
        <CartesianGrid
          strokeDasharray="3 3"
          stroke={chartColors.grid}
          vertical={false}
        />
        <XAxis
          dataKey={xKey}
          tick={{ fontSize: 10, fill: chartColors.axis }}
          axisLine={{ stroke: chartColors.grid }}
          tickLine={false}
          interval={0}
          angle={data.length > 12 ? -45 : 0}
          textAnchor={data.length > 12 ? "end" : "middle"}
          height={data.length > 12 ? 60 : 30}
        />
        <YAxis
          tick={{ fontSize: 11, fill: chartColors.axis }}
          axisLine={false}
          tickLine={false}
          domain={["auto", "auto"]}
          tickFormatter={(v: number) => formatPct(v)}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: chartColors.tooltip_bg,
            border: `1px solid ${chartColors.tooltip_border}`,
            borderRadius: 6,
            fontSize: 12,
          }}
          formatter={(value: number, name: string) => [
            formatPct(value),
            name,
          ]}
        />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {baselineRef != null && (
          <ReferenceLine
            y={baselineRef}
            stroke={chartColors.axis}
            strokeDasharray="4 4"
            label={{
              value: `Baseline avg: ${formatPct(baselineRef)}`,
              position: "insideTopRight",
              fontSize: 10,
              fill: chartColors.axis,
            }}
          />
        )}
        <Bar
          dataKey="baseline_accuracy"
          name={baselineLabel}
          fill={trendColors[0]}
          radius={[3, 3, 0, 0]}
          maxBarSize={28}
        />
        <Bar
          dataKey="candidate_accuracy"
          name={candidateLabel}
          fill={trendColors[1]}
          radius={[3, 3, 0, 0]}
          maxBarSize={28}
        />
      </BarChart>
    </ResponsiveContainer>
  );
}

// ---------------------------------------------------------------------------
// Cluster delta table
// ---------------------------------------------------------------------------
function ClusterDeltaTable({ items, title }: {
  items: ClusterComparison[];
  title: string;
}) {
  if (items.length === 0) return null;

  // Sort by delta_accuracy descending (most improved first)
  const sorted = [...items].sort((a, b) => (b.delta_accuracy ?? 0) - (a.delta_accuracy ?? 0));

  return (
    <div>
      <p className="text-xs text-muted-foreground mb-1.5">{title}</p>
      <div className="overflow-x-auto max-h-[240px] overflow-y-auto border rounded-md">
        <table className="w-full text-xs">
          <thead className="bg-muted/50 sticky top-0">
            <tr>
              <th className="text-left px-2 py-1.5 font-medium">Cluster</th>
              <th className="text-right px-2 py-1.5 font-medium">Base Acc%</th>
              <th className="text-right px-2 py-1.5 font-medium">Cand Acc%</th>
              <th className="text-right px-2 py-1.5 font-medium">Delta</th>
              <th className="text-right px-2 py-1.5 font-medium">DFUs</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((item) => {
              const delta = item.delta_accuracy;
              const deltaColor =
                delta != null && delta > 0.05
                  ? "text-emerald-600 dark:text-emerald-400"
                  : delta != null && delta < -0.05
                    ? "text-red-600 dark:text-red-400"
                    : "text-muted-foreground";

              return (
                <tr key={item.cluster} className="border-t border-border/40 hover:bg-muted/30">
                  <td className="px-2 py-1 truncate max-w-[120px]" title={item.cluster}>
                    {item.cluster}
                  </td>
                  <td className="text-right px-2 py-1 tabular-nums">
                    {item.baseline_accuracy != null ? formatFixed(item.baseline_accuracy, 1) : "--"}
                  </td>
                  <td className="text-right px-2 py-1 tabular-nums">
                    {item.candidate_accuracy != null ? formatFixed(item.candidate_accuracy, 1) : "--"}
                  </td>
                  <td className={cn("text-right px-2 py-1 tabular-nums font-medium", deltaColor)}>
                    {delta != null ? `${delta > 0 ? "+" : ""}${delta.toFixed(1)}` : "--"}
                  </td>
                  <td className="text-right px-2 py-1 tabular-nums text-muted-foreground">
                    {item.baseline_n_dfus ?? "--"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab selector
// ---------------------------------------------------------------------------
type ComparisonView = "summary" | "params" | "ml_cluster" | "business_cluster" | "month";

const VIEW_LABELS: Record<ComparisonView, string> = {
  summary: "Summary",
  params: "Params",
  ml_cluster: "ML Cluster",
  business_cluster: "Business Cluster",
  month: "Month",
};

// ---------------------------------------------------------------------------
// ComparisonPanel
// ---------------------------------------------------------------------------
export function ComparisonPanel({ baselineId, candidateId }: ComparisonPanelProps) {
  const { chartColors, trendColors } = useChartColors();
  const [view, setView] = useState<ComparisonView>("summary");

  const { data, isLoading, isError, error } = useQuery<TuningComparison>({
    queryKey: lgbmTuningKeys.compare(baselineId, candidateId),
    queryFn: () => fetchTuningComparison(baselineId, candidateId),
    staleTime: STALE.FIVE_MIN,
    enabled: baselineId > 0 && candidateId > 0,
  });

  // -- Loading / error states -----------------------------------------------
  if (isLoading) {
    return (
      <Card>
        <CardContent className="py-12">
          <LoadingElement message="Loading comparison..." />
        </CardContent>
      </Card>
    );
  }

  if (isError) {
    return (
      <Card>
        <CardContent className="py-12 text-center text-sm text-destructive">
          Failed to load comparison: {(error as Error).message}
        </CardContent>
      </Card>
    );
  }

  if (!data) return null;

  const { baseline, candidate, verdict } = data;
  const per_timeframe = data.per_timeframe ?? [];
  const ml_clusters = data.per_cluster?.ml_cluster ?? [];
  const biz_clusters = data.per_cluster?.business_cluster ?? [];
  const per_month = data.per_month ?? [];
  const param_diffs = data.param_diffs ?? [];
  const param_common = data.param_common ?? [];
  const feature_diffs = data.feature_diffs;
  const config_diffs = data.config_diffs ?? [];
  const config_common = data.config_common ?? [];
  const baselineHasBreakdowns = data.baseline_has_breakdowns ?? true;
  const candidateHasBreakdowns = data.candidate_has_breakdowns ?? true;

  const baselineLabel = `Baseline (#${baseline.run_id})`;
  const candidateLabel = `Candidate (#${candidate.run_id})`;

  // Check which views have data
  const hasClusterData = ml_clusters.length > 0 || biz_clusters.length > 0;
  const hasMonthData = per_month.length > 0;
  const hasParamData = param_diffs.length > 0 || param_common.length > 0
    || config_diffs.length > 0 || config_common.length > 0
    || (feature_diffs && (feature_diffs.added.length > 0 || feature_diffs.removed.length > 0 || feature_diffs.common_count > 0));

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">
            Run #{baseline.run_id} vs Run #{candidate.run_id}
          </CardTitle>
          <VerdictBadge verdict={verdict} />
        </div>

        {/* View tabs */}
        <div className="flex gap-1 mt-2">
          {(Object.keys(VIEW_LABELS) as ComparisonView[]).map((key) => {
            const disabled =
              (key === "params" && !hasParamData) ||
              (key === "ml_cluster" && ml_clusters.length === 0) ||
              (key === "business_cluster" && biz_clusters.length === 0) ||
              (key === "month" && per_month.length === 0);

            return (
              <button
                key={key}
                onClick={() => setView(key)}
                disabled={disabled}
                className={cn(
                  "px-2.5 py-1 text-xs rounded-md transition-colors",
                  view === key
                    ? "bg-primary text-primary-foreground"
                    : disabled
                      ? "text-muted-foreground/40 cursor-not-allowed"
                      : "text-muted-foreground hover:bg-muted",
                )}
              >
                {VIEW_LABELS[key]}
              </button>
            );
          })}
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Side-by-side KPI metrics (always visible) */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <MetricCard
            label="Accuracy %"
            baseVal={baseline.accuracy_pct}
            candVal={candidate.accuracy_pct}
          />
          <MetricCard
            label="WAPE"
            baseVal={baseline.wape}
            candVal={candidate.wape}
            lowerIsBetter
          />
          <MetricCard
            label="Bias"
            baseVal={baseline.bias}
            candVal={candidate.bias}
            lowerIsBetter
          />
        </div>

        {/* Summary view: per-timeframe chart */}
        {view === "summary" && per_timeframe.length > 0 && (
          <div>
            <p className="text-xs text-muted-foreground mb-2">
              Per-Timeframe Accuracy
            </p>
            <ComparisonBarChart
              data={per_timeframe}
              xKey="timeframe"
              baselineLabel={baselineLabel}
              candidateLabel={candidateLabel}
              chartColors={chartColors}
              trendColors={trendColors}
              baselineRef={baseline.accuracy_pct}
            />
          </div>
        )}

        {view === "summary" && !hasClusterData && !hasMonthData && per_timeframe.length === 0 && (
          <p className="text-xs text-muted-foreground text-center py-4">
            No per-timeframe data available. Register runs with cluster/month breakdowns for detailed comparisons.
          </p>
        )}

        {/* Params view */}
        {view === "params" && hasParamData && (
          <div className="space-y-4">
            {/* Changed parameters */}
            {param_diffs.length > 0 && (
              <div>
                <p className="text-xs font-medium text-foreground mb-2">
                  Changed Parameters
                </p>
                <div className="overflow-x-auto border rounded-md">
                  <table className="w-full text-xs">
                    <thead className="bg-muted/50">
                      <tr>
                        <th className="text-left px-3 py-2 font-medium">Parameter</th>
                        <th className="text-right px-3 py-2 font-medium">
                          Baseline (#{baseline.run_id})
                        </th>
                        <th className="text-right px-3 py-2 font-medium">
                          Candidate (#{candidate.run_id})
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {param_diffs.map((d: ParamDiff) => (
                        <tr
                          key={d.param}
                          className="border-t border-border/40 bg-amber-50/50 dark:bg-amber-900/10 hover:bg-amber-100/50 dark:hover:bg-amber-900/20"
                        >
                          <td className="px-3 py-1.5 font-mono text-xs font-medium">
                            {d.param}
                          </td>
                          <td className="text-right px-3 py-1.5 tabular-nums text-muted-foreground">
                            {d.baseline != null ? String(d.baseline) : "--"}
                          </td>
                          <td className="text-right px-3 py-1.5 tabular-nums font-semibold">
                            {d.candidate != null ? String(d.candidate) : "--"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {param_diffs.length === 0 && (
              <p className="text-xs text-muted-foreground text-center py-2">
                Both runs used identical parameters.
              </p>
            )}

            {/* Common parameters */}
            {param_common.length > 0 && (
              <div>
                <p className="text-xs font-medium text-muted-foreground mb-2">
                  Common Parameters
                </p>
                <div className="overflow-x-auto border rounded-md">
                  <table className="w-full text-xs">
                    <thead className="bg-muted/50">
                      <tr>
                        <th className="text-left px-3 py-2 font-medium">Parameter</th>
                        <th className="text-right px-3 py-2 font-medium">Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {param_common.map((p: ParamCommon) => (
                        <tr
                          key={p.param}
                          className="border-t border-border/40 hover:bg-muted/30"
                        >
                          <td className="px-3 py-1.5 font-mono text-xs text-muted-foreground">
                            {p.param}
                          </td>
                          <td className="text-right px-3 py-1.5 tabular-nums text-muted-foreground">
                            {p.value != null ? String(p.value) : "--"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Config diffs (cluster_strategy, recursive, shap, etc.) */}
            {(config_diffs.length > 0 || config_common.length > 0) && (
              <div>
                <p className="text-xs font-medium text-foreground mb-2">
                  Training Configuration
                </p>
                <div className="overflow-x-auto border rounded-md">
                  <table className="w-full text-xs">
                    <thead className="bg-muted/50">
                      <tr>
                        <th className="text-left px-3 py-2 font-medium">Setting</th>
                        <th className="text-right px-3 py-2 font-medium">
                          Baseline (#{baseline.run_id})
                        </th>
                        <th className="text-right px-3 py-2 font-medium">
                          Candidate (#{candidate.run_id})
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {config_diffs.map((d: ConfigDiff) => (
                        <tr
                          key={d.setting}
                          className="border-t border-border/40 bg-amber-50/50 dark:bg-amber-900/10 hover:bg-amber-100/50 dark:hover:bg-amber-900/20"
                        >
                          <td className="px-3 py-1.5 font-mono text-xs font-medium">
                            {d.setting}
                          </td>
                          <td className="text-right px-3 py-1.5 tabular-nums text-muted-foreground">
                            {d.baseline != null ? String(d.baseline) : "--"}
                          </td>
                          <td className="text-right px-3 py-1.5 tabular-nums font-semibold">
                            {d.candidate != null ? String(d.candidate) : "--"}
                          </td>
                        </tr>
                      ))}
                      {config_common.map((c: ConfigCommon) => (
                        <tr
                          key={c.setting}
                          className="border-t border-border/40 hover:bg-muted/30"
                        >
                          <td className="px-3 py-1.5 font-mono text-xs text-muted-foreground">
                            {c.setting}
                          </td>
                          <td className="text-right px-3 py-1.5 tabular-nums text-muted-foreground" colSpan={2}>
                            {c.value != null ? String(c.value) : "--"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Feature diffs */}
            {feature_diffs && (
              <div>
                <p className="text-xs font-medium text-foreground mb-2">
                  Features ({feature_diffs.baseline_count} baseline, {feature_diffs.candidate_count} candidate, {feature_diffs.common_count} common)
                </p>
                {feature_diffs.added.length > 0 && (
                  <div className="mb-2">
                    <p className="text-[10px] font-medium text-emerald-600 dark:text-emerald-400 mb-1">
                      + Added ({feature_diffs.added.length})
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {feature_diffs.added.map((f) => (
                        <span
                          key={f}
                          className="inline-block px-1.5 py-0.5 text-[10px] font-mono rounded bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300"
                        >
                          {f}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                {feature_diffs.removed.length > 0 && (
                  <div className="mb-2">
                    <p className="text-[10px] font-medium text-red-600 dark:text-red-400 mb-1">
                      - Removed ({feature_diffs.removed.length})
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {feature_diffs.removed.map((f) => (
                        <span
                          key={f}
                          className="inline-block px-1.5 py-0.5 text-[10px] font-mono rounded bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300"
                        >
                          {f}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                {feature_diffs.added.length === 0 && feature_diffs.removed.length === 0 && (
                  <p className="text-xs text-muted-foreground">
                    Both runs used identical features ({feature_diffs.common_count} features).
                  </p>
                )}
              </div>
            )}
          </div>
        )}

        {view === "params" && !hasParamData && (
          <p className="text-xs text-muted-foreground text-center py-4">
            No parameter data available for these runs.
          </p>
        )}

        {/* Missing breakdown data warning */}
        {(view === "ml_cluster" || view === "business_cluster" || view === "month") &&
          (!baselineHasBreakdowns || !candidateHasBreakdowns) && (
          <div className="rounded-md border border-amber-300 bg-amber-50 dark:border-amber-700 dark:bg-amber-950/30 px-3 py-2 text-xs text-amber-800 dark:text-amber-300 mb-2">
            {!baselineHasBreakdowns && !candidateHasBreakdowns
              ? "Neither run has cluster/month breakdown data. Re-run backtests and re-register to populate breakdowns."
              : `Run #${!candidateHasBreakdowns ? candidate.run_id : baseline.run_id} does not have cluster/month breakdown data. Only Run #${!candidateHasBreakdowns ? baseline.run_id : candidate.run_id}'s data is shown. Re-run the backtest and re-register to populate breakdown data.`}
          </div>
        )}

        {/* ML Cluster view */}
        {view === "ml_cluster" && ml_clusters.length > 0 && (
          <div className="space-y-3">
            <ComparisonBarChart
              data={ml_clusters}
              xKey="cluster"
              baselineLabel={baselineLabel}
              candidateLabel={candidateLabel}
              chartColors={chartColors}
              trendColors={trendColors}
              baselineRef={baseline.accuracy_pct}
              height={Math.min(400, Math.max(250, ml_clusters.length * 28))}
            />
            <ClusterDeltaTable items={ml_clusters} title="ML Cluster Accuracy Delta" />
          </div>
        )}

        {/* Business Cluster view */}
        {view === "business_cluster" && biz_clusters.length > 0 && (
          <div className="space-y-3">
            <ComparisonBarChart
              data={biz_clusters}
              xKey="cluster"
              baselineLabel={baselineLabel}
              candidateLabel={candidateLabel}
              chartColors={chartColors}
              trendColors={trendColors}
              baselineRef={baseline.accuracy_pct}
              height={Math.min(400, Math.max(250, biz_clusters.length * 28))}
            />
            <ClusterDeltaTable items={biz_clusters} title="Business Cluster Accuracy Delta" />
          </div>
        )}

        {/* Month view */}
        {view === "month" && per_month.length > 0 && (
          <div className="space-y-3">
            <p className="text-xs text-muted-foreground mb-2">
              Monthly Accuracy Comparison
            </p>
            <ComparisonBarChart
              data={per_month.map((m: MonthComparison) => ({
                ...m,
                month_label: m.month.slice(0, 7),
              }))}
              xKey="month_label"
              baselineLabel={baselineLabel}
              candidateLabel={candidateLabel}
              chartColors={chartColors}
              trendColors={trendColors}
              baselineRef={baseline.accuracy_pct}
              height={320}
            />
            {/* Month delta table */}
            <div className="overflow-x-auto max-h-[240px] overflow-y-auto border rounded-md">
              <table className="w-full text-xs">
                <thead className="bg-muted/50 sticky top-0">
                  <tr>
                    <th className="text-left px-2 py-1.5 font-medium">Month</th>
                    <th className="text-right px-2 py-1.5 font-medium">Base Acc%</th>
                    <th className="text-right px-2 py-1.5 font-medium">Cand Acc%</th>
                    <th className="text-right px-2 py-1.5 font-medium">Delta</th>
                    <th className="text-right px-2 py-1.5 font-medium">Base WAPE</th>
                    <th className="text-right px-2 py-1.5 font-medium">Cand WAPE</th>
                  </tr>
                </thead>
                <tbody>
                  {per_month.map((m: MonthComparison) => {
                    const delta = m.delta_accuracy;
                    const deltaColor =
                      delta != null && delta > 0.05
                        ? "text-emerald-600 dark:text-emerald-400"
                        : delta != null && delta < -0.05
                          ? "text-red-600 dark:text-red-400"
                          : "text-muted-foreground";
                    return (
                      <tr key={m.month} className="border-t border-border/40 hover:bg-muted/30">
                        <td className="px-2 py-1">{m.month.slice(0, 7)}</td>
                        <td className="text-right px-2 py-1 tabular-nums">
                          {m.baseline_accuracy != null ? formatFixed(m.baseline_accuracy, 1) : "--"}
                        </td>
                        <td className="text-right px-2 py-1 tabular-nums">
                          {m.candidate_accuracy != null ? formatFixed(m.candidate_accuracy, 1) : "--"}
                        </td>
                        <td className={cn("text-right px-2 py-1 tabular-nums font-medium", deltaColor)}>
                          {delta != null ? `${delta > 0 ? "+" : ""}${delta.toFixed(1)}` : "--"}
                        </td>
                        <td className="text-right px-2 py-1 tabular-nums text-muted-foreground">
                          {m.baseline_wape != null ? formatFixed(m.baseline_wape, 1) : "--"}
                        </td>
                        <td className="text-right px-2 py-1 tabular-nums text-muted-foreground">
                          {m.candidate_wape != null ? formatFixed(m.candidate_wape, 1) : "--"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
