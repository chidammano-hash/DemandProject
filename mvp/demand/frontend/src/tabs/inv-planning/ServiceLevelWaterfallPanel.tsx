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
  STALE_INSIGHTS,
} from "@/api/queries";
import { KpiCard } from "@/components/KpiCard";
import { EmptyState } from "@/components/EmptyState";
import { formatPct } from "@/lib/formatters";
import { Layers } from "lucide-react";

const PANEL_KPI = "rounded-lg bg-muted/30 p-3";

const LEVER_COLORS: Record<string, string> = {
  base_forecast: "#94a3b8",
  ss_buffer: "#3b82f6",
  lt_buffer: "#8b5cf6",
  sensing: "#10b981",
  achieved: "#059669",
};

const LEVER_LABELS: Record<string, string> = {
  base_forecast: "Base Forecast",
  ss_buffer: "Safety Stock Buffer",
  lt_buffer: "Lead Time Buffer",
  sensing: "Demand Sensing",
  achieved: "Achieved CSL",
};

const LEVER_DESCRIPTIONS: Record<string, string> = {
  base_forecast: "Baseline service level from statistical forecast accuracy alone, before any inventory buffers are applied.",
  ss_buffer: "Additional service level gained from safety stock inventory, protecting against demand variability.",
  lt_buffer: "Service level improvement from lead time buffer, accounting for supplier delivery uncertainty.",
  sensing: "Incremental service level from demand sensing adjustments that react to recent demand signals.",
};

interface WaterfallBar {
  lever: string;
  label: string;
  contribution: number;
  cumulative: number;
  base: number;
  fill: string;
}

export function ServiceLevelWaterfallPanel() {
  const { data, isLoading, error } = useQuery({
    queryKey: insightKeys.serviceLevelWaterfall(),
    queryFn: fetchServiceLevelWaterfall,
    staleTime: STALE_INSIGHTS.FIVE_MIN,
  });

  if (error) {
    return (
      <div className="text-xs text-red-600 p-4">
        Failed to load service level waterfall: {(error as Error).message}
      </div>
    );
  }

  // Build waterfall bars from segments
  const waterfallData: WaterfallBar[] = [];
  if (data?.segments) {
    for (const seg of data.segments) {
      waterfallData.push({
        lever: seg.lever,
        label: LEVER_LABELS[seg.lever] ?? seg.lever,
        contribution: seg.contribution,
        cumulative: seg.cumulative,
        base: seg.cumulative - seg.contribution,
        fill: seg.color || LEVER_COLORS[seg.lever] || "#94a3b8",
      });
    }
  }

  return (
    <div className="space-y-4">
      <div className="text-xs text-muted-foreground bg-muted/20 border rounded px-3 py-2">
        Service level waterfall showing how each planning lever contributes to the achieved customer service level (CSL).
        Each bar segment represents the incremental CSL improvement from that lever.
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KpiCard
          className={PANEL_KPI}
          label="Base Forecast CSL"
          value={isLoading ? "..." : formatPct(data?.base_forecast_csl)}
        />
        <KpiCard
          className={PANEL_KPI}
          label="+SS Buffer"
          value={isLoading ? "..." : `+${formatPct(data?.ss_buffer_contribution)}`}
        />
        <KpiCard
          className={PANEL_KPI}
          label="+LT Buffer"
          value={isLoading ? "..." : `+${formatPct(data?.lt_buffer_contribution)}`}
        />
        <KpiCard
          className={PANEL_KPI}
          label="Achieved CSL"
          value={isLoading ? "..." : formatPct(data?.achieved_csl)}
          colorClass={
            (data?.achieved_csl ?? 0) >= 95
              ? "text-green-600"
              : (data?.achieved_csl ?? 0) >= 85
                ? "text-yellow-600"
                : "text-red-600"
          }
          tooltip={{
            title: "Achieved Customer Service Level",
            description: "The final CSL after all buffers and sensing adjustments. Target is typically 95%+.",
          }}
        />
      </div>

      {isLoading ? (
        <p className="text-xs text-muted-foreground">Loading waterfall data...</p>
      ) : waterfallData.length === 0 ? (
        <EmptyState
          icon={Layers}
          title="No service level data available"
          description="The waterfall chart decomposes the achieved customer service level into contributions from each planning lever: forecast, safety stock, lead time buffer, and demand sensing."
          steps={[
            { label: "Run backtest pipeline", command: "make backtest-all" },
            { label: "Compute safety stock", command: "make ss-compute" },
          ]}
        />
      ) : (
        <>
          {/* Waterfall chart */}
          <div style={{ height: "calc(min(320px, 35vh))" }}>
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
        </>
      )}
    </div>
  );
}
