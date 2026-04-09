import { useQuery } from "@tanstack/react-query";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  LabelList,
} from "recharts";
import {
  insightKeys,
  fetchServiceLevelWaterfall,
  fetchServiceLevelBridge,
  STALE_INSIGHTS,
} from "@/api/queries";
import type {
  WaterfallBridgeStep,
  WaterfallBridgeClassData,
} from "@/api/queries";
import { KpiCard } from "@/components/KpiCard";
import { EmptyState } from "@/components/EmptyState";
import { formatPct } from "@/lib/formatters";
import { cn } from "@/lib/utils";
import { Layers } from "lucide-react";

const PANEL_KPI = "rounded-lg bg-muted/30 p-3";

// ---------------------------------------------------------------------------
// Section 1 — Target vs Actual Bridge (Issue #16)
// ---------------------------------------------------------------------------

function BridgeChart() {
  const { data, isLoading, error } = useQuery({
    queryKey: insightKeys.serviceLevelBridge(),
    queryFn: () => fetchServiceLevelBridge(),
    staleTime: STALE_INSIGHTS.FIVE_MIN,
  });

  if (error) {
    return (
      <div className="text-xs text-red-600 p-4">
        Failed to load service level bridge: {(error as Error).message}
      </div>
    );
  }

  if (isLoading) {
    return <p className="text-xs text-muted-foreground">Loading bridge data...</p>;
  }

  if (!data || data.target == null || data.actual == null || data.steps.length === 0) {
    return (
      <EmptyState
        icon={Layers}
        title="No service level bridge data"
        description="The bridge chart shows how each ABC class contributes to the gap between the portfolio service level target and the actual achieved fill rate."
        steps={[
          { label: "Compute safety stock", command: "make ss-compute" },
          { label: "Refresh fill rate MV", command: "make refresh-mvs-tiered" },
        ]}
      />
    );
  }

  const steps: WaterfallBridgeStep[] = data.steps;
  const byClass: WaterfallBridgeClassData[] = data.by_class ?? [];

  // For the bridge bar chart we track cumulative running total
  // "total" bars show from 0, delta bars are stacked on the running total
  let running = 0;
  const chartBars = steps.map((step) => {
    if (step.type === "total") {
      const bar = { label: step.label, base: 0, delta: step.value, fill: "", type: step.type };
      running = step.value;
      return bar;
    }
    const bar = {
      label: step.label,
      base: step.value > 0 ? running : running + step.value,
      delta: Math.abs(step.value),
      fill: "",
      type: step.type,
    };
    running += step.value;
    return bar;
  });

  // Assign colors
  for (const bar of chartBars) {
    if (bar.type === "total") bar.fill = "hsl(var(--primary))";
    else if (bar.type === "positive") bar.fill = "#10b981";
    else bar.fill = "#ef4444";
  }

  return (
    <div className="space-y-4">
      {/* KPI row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KpiCard
          className={PANEL_KPI}
          label="Portfolio Target"
          value={formatPct(data.target * 100)}
        />
        <KpiCard
          className={PANEL_KPI}
          label="Portfolio Actual"
          value={formatPct(data.actual * 100)}
          colorClass={
            data.actual >= (data.target ?? 0)
              ? "text-green-600"
              : data.actual >= (data.target ?? 0) - 0.03
                ? "text-yellow-600"
                : "text-red-600"
          }
        />
        <KpiCard
          className={PANEL_KPI}
          label="Gap"
          value={`${data.actual >= data.target ? "+" : ""}${formatPct((data.actual - data.target) * 100)}`}
          colorClass={data.actual >= data.target ? "text-green-600" : "text-red-600"}
        />
        <KpiCard
          className={PANEL_KPI}
          label="Period"
          value={data.month ?? "Latest"}
        />
      </div>

      {/* Bridge bar chart */}
      <div style={{ height: "calc(min(300px, 32vh))" }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartBars}
            margin={{ top: 20, right: 20, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" className="stroke-border/40" />
            <XAxis
              dataKey="label"
              tick={{ fontSize: 10 }}
              className="text-muted-foreground"
            />
            <YAxis
              tick={{ fontSize: 11 }}
              className="text-muted-foreground"
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
              domain={[
                (dataMin: number) => Math.max(0, Math.floor((dataMin - 0.05) * 20) / 20),
                (dataMax: number) => Math.min(1, Math.ceil((dataMax + 0.02) * 20) / 20),
              ]}
            />
            <Tooltip
              contentStyle={{
                fontSize: 11,
                borderRadius: 8,
                border: "1px solid var(--border)",
                backgroundColor: "var(--card)",
              }}
              formatter={(value: number, name: string) => {
                if (name === "base") return [null, null];
                return [`${(value * 100).toFixed(2)}%`, "Value"];
              }}
            />
            {/* Invisible base bar for waterfall stacking */}
            <Bar dataKey="base" stackId="bridge" fill="transparent" />
            {/* Delta bar */}
            <Bar dataKey="delta" stackId="bridge" radius={[2, 2, 0, 0]}>
              {chartBars.map((entry, idx) => (
                <Cell key={idx} fill={entry.fill} />
              ))}
              <LabelList
                dataKey="delta"
                position="top"
                style={{ fontSize: 10, fill: "var(--foreground)" }}
                formatter={(v: number) => `${(v * 100).toFixed(1)}%`}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Simple div-based horizontal bridge (accessible, no chart lib needed) */}
      <div className="space-y-2">
        {steps.map((step) => (
          <div key={step.label} className="flex items-center gap-2">
            <span className="w-28 text-xs truncate" title={step.label}>
              {step.label}
            </span>
            <div className="flex-1 h-6 bg-muted rounded relative">
              <div
                className={cn(
                  "h-6 rounded",
                  step.type === "total"
                    ? "bg-primary"
                    : step.type === "positive"
                      ? "bg-emerald-500"
                      : "bg-red-500",
                )}
                style={{
                  width: step.type === "total"
                    ? `${Math.min(step.value * 100, 100)}%`
                    : `${Math.min(Math.abs(step.value) * 1000, 100)}%`,
                }}
              />
            </div>
            <span className="text-xs font-mono w-16 text-right">
              {step.type === "total"
                ? `${(step.value * 100).toFixed(1)}%`
                : `${step.value > 0 ? "+" : ""}${(step.value * 100).toFixed(2)}%`}
            </span>
          </div>
        ))}
      </div>

      {/* Per-class detail table */}
      {byClass.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b text-muted-foreground">
                <th className="text-left py-1 pr-3">Class</th>
                <th className="text-right py-1 px-2">SKUs</th>
                <th className="text-right py-1 px-2">Target</th>
                <th className="text-right py-1 px-2">Actual</th>
                <th className="text-right py-1 px-2">Gap</th>
                <th className="text-right py-1 px-2">Weight</th>
                <th className="text-right py-1 pl-2">Wtd Gap</th>
              </tr>
            </thead>
            <tbody>
              {byClass.map((c) => (
                <tr key={c.abc_class} className="border-b border-border/30">
                  <td className="py-1 pr-3 font-medium">{c.abc_class}</td>
                  <td className="text-right py-1 px-2">{c.sku_count}</td>
                  <td className="text-right py-1 px-2">{(c.target_sl * 100).toFixed(1)}%</td>
                  <td className="text-right py-1 px-2">{(c.avg_fill_rate * 100).toFixed(1)}%</td>
                  <td
                    className={cn(
                      "text-right py-1 px-2 font-mono",
                      c.gap >= 0 ? "text-green-600" : "text-red-600",
                    )}
                  >
                    {c.gap >= 0 ? "+" : ""}{(c.gap * 100).toFixed(2)}%
                  </td>
                  <td className="text-right py-1 px-2">{(c.weight * 100).toFixed(1)}%</td>
                  <td
                    className={cn(
                      "text-right py-1 pl-2 font-mono",
                      c.weighted_gap >= 0 ? "text-green-600" : "text-red-600",
                    )}
                  >
                    {c.weighted_gap >= 0 ? "+" : ""}{(c.weighted_gap * 100).toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Section 2 — Lever Decomposition (existing waterfall)
// ---------------------------------------------------------------------------

const LEVER_COLORS: Record<string, string> = {
  base_forecast: "#94a3b8",
  base_forecast_accuracy: "#94a3b8",
  ss_buffer: "#3b82f6",
  ss_buffer_contribution: "#3b82f6",
  lt_buffer: "#8b5cf6",
  lt_buffer_contribution: "#8b5cf6",
  sensing: "#10b981",
  sensing_adjustment: "#10b981",
  achieved: "#059669",
};

const LEVER_LABELS: Record<string, string> = {
  base_forecast: "Base Forecast",
  base_forecast_accuracy: "Forecast Accuracy",
  ss_buffer: "Safety Stock Buffer",
  ss_buffer_contribution: "Safety Stock Buffer",
  lt_buffer: "Lead Time Buffer",
  lt_buffer_contribution: "Lead Time Reliability",
  sensing: "Demand Sensing",
  sensing_adjustment: "Demand Sensing",
  achieved: "Achieved CSL",
};

const LEVER_DESCRIPTIONS: Record<string, string> = {
  base_forecast: "Baseline service level from statistical forecast accuracy alone.",
  base_forecast_accuracy: "Baseline service level from statistical forecast accuracy alone.",
  ss_buffer: "Additional service level gained from safety stock inventory.",
  ss_buffer_contribution: "Additional service level gained from safety stock inventory.",
  lt_buffer: "Service level improvement from lead time buffer.",
  lt_buffer_contribution: "Service level improvement accounting for supplier delivery reliability.",
  sensing: "Incremental service level from demand sensing adjustments.",
  sensing_adjustment: "Incremental service level from demand sensing signal adjustments.",
};

interface LeverWaterfallBar {
  lever: string;
  label: string;
  contribution: number;
  cumulative: number;
  base: number;
  fill: string;
}

function LeverWaterfall() {
  const { data, isLoading, error } = useQuery({
    queryKey: insightKeys.serviceLevelWaterfall(),
    queryFn: fetchServiceLevelWaterfall,
    staleTime: STALE_INSIGHTS.FIVE_MIN,
  });

  if (error) {
    return (
      <div className="text-xs text-red-600 p-4">
        Failed to load lever waterfall: {(error as Error).message}
      </div>
    );
  }

  // Build waterfall bars from steps or segments (backend may return either key)
  const waterfallData: LeverWaterfallBar[] = [];
  const rawSteps = data?.steps ?? data?.segments ?? [];
  if (rawSteps.length > 0) {
    for (const seg of rawSteps) {
      const lever = seg.lever ?? seg.label ?? "";
      const contribution = seg.contribution ?? seg.contribution_pct ?? 0;
      const cumulative = seg.cumulative ?? seg.cumulative_pct ?? 0;
      waterfallData.push({
        lever,
        label: LEVER_LABELS[lever] ?? lever,
        contribution,
        cumulative,
        base: cumulative - contribution,
        fill: seg.color || LEVER_COLORS[lever] || "#94a3b8",
      });
    }
  }

  if (isLoading) {
    return <p className="text-xs text-muted-foreground">Loading lever data...</p>;
  }

  if (waterfallData.length === 0) {
    return (
      <p className="text-xs text-muted-foreground italic">
        No lever decomposition data available.
      </p>
    );
  }

  return (
    <div className="space-y-4">
      {/* Waterfall chart */}
      <div style={{ height: "calc(min(280px, 30vh))" }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={waterfallData}
            margin={{ top: 20, right: 20, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" className="stroke-border/40" />
            <XAxis
              dataKey="label"
              tick={{ fontSize: 10 }}
              className="text-muted-foreground"
            />
            <YAxis
              tick={{ fontSize: 11 }}
              className="text-muted-foreground"
              tickFormatter={(v: number) => `${v}%`}
              domain={[0, 100]}
            />
            <Tooltip
              contentStyle={{
                fontSize: 11,
                borderRadius: 8,
                border: "1px solid var(--border)",
                backgroundColor: "var(--card)",
              }}
              formatter={(value: number, name: string) => [
                `${formatPct(value)}`,
                name === "base" ? "Base" : "Contribution",
              ]}
            />
            {/* Invisible base bar */}
            <Bar dataKey="base" stackId="waterfall" fill="transparent" />
            {/* Contribution bar */}
            <Bar dataKey="contribution" stackId="waterfall" radius={[2, 2, 0, 0]}>
              {waterfallData.map((entry, idx) => (
                <Cell key={idx} fill={entry.fill} />
              ))}
              <LabelList
                dataKey="contribution"
                position="top"
                formatter={(v: number) => `+${formatPct(v)}`}
                style={{ fontSize: 10, fill: "var(--foreground)" }}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Lever descriptions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {waterfallData.map((bar) => (
          <div
            key={bar.lever}
            className="flex items-start gap-2 rounded-lg border p-3"
          >
            <div
              className="mt-1 h-3 w-3 rounded shrink-0"
              style={{ backgroundColor: bar.fill }}
            />
            <div>
              <p className="text-xs font-medium">{bar.label}</p>
              <p className="text-xs text-muted-foreground mt-0.5">
                {LEVER_DESCRIPTIONS[bar.lever] ?? `Contributes ${formatPct(bar.contribution)} to CSL.`}
              </p>
              <p className="text-xs font-mono mt-1">
                +{formatPct(bar.contribution)} (cumulative: {formatPct(bar.cumulative)})
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Panel
// ---------------------------------------------------------------------------

export function ServiceLevelWaterfallPanel() {
  return (
    <div className="space-y-6">
      {/* Section 1: Target vs Actual Bridge */}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold">Target vs Actual Bridge</h3>
        <p className="text-xs text-muted-foreground bg-muted/20 border rounded px-3 py-2">
          Waterfall bridge showing how the portfolio service level target decomposes by ABC class
          into the achieved actual. Green bars indicate classes exceeding their target; red bars
          indicate underperformance.
        </p>
        <BridgeChart />
      </div>

      {/* Section 2: Lever Decomposition */}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold">Lever Decomposition</h3>
        <p className="text-xs text-muted-foreground bg-muted/20 border rounded px-3 py-2">
          How each planning lever (forecast accuracy, safety stock, lead time buffer, demand sensing)
          contributes to the achieved customer service level.
        </p>
        <LeverWaterfall />
      </div>
    </div>
  );
}
