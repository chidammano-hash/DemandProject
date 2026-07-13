import { useQuery } from "@tanstack/react-query";
import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import {
  fillRateKeys,
  fetchFillRateSummary,
  fetchFillRateTrend,
  fetchFillRateGapAnalysis,
  STALE,
} from "@/api/queries";
import type { GapDecompositionItem } from "@/api/queries";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";
import { KpiCard } from "@/components/KpiCard";
import { EmptyState } from "@/components/EmptyState";
import { formatInt, formatPct } from "@/lib/formatters";
import { TrendingUp } from "lucide-react";

const PANEL_KPI = "rounded-lg bg-muted/30 p-3";

export function FillRatePanel() {
  const { filters } = useGlobalFilterContext();
  const gf = {
    brand: filters.brand.length > 0 ? filters.brand.join(",") : undefined,
    category: filters.category.length > 0 ? filters.category.join(",") : undefined,
    market: filters.market.length > 0 ? filters.market.join(",") : undefined,
    item: filters.item.length === 1 ? filters.item[0] : undefined,
    location: filters.location.length === 1 ? filters.location[0] : undefined,
  };

  const { data: summary, isLoading } = useQuery({
    queryKey: fillRateKeys.summary(gf),
    queryFn: () => fetchFillRateSummary(gf),
    staleTime: STALE.FIVE_MIN,
  });
  const { data: trendData } = useQuery({
    queryKey: fillRateKeys.trend(gf),
    queryFn: () => fetchFillRateTrend(gf),
    staleTime: STALE.FIVE_MIN,
  });
  const { data: gapData } = useQuery({
    queryKey: fillRateKeys.gapAnalysis(gf),
    queryFn: () => fetchFillRateGapAnalysis(gf as { month?: string; abc_vol?: string }),
    staleTime: STALE.FIVE_MIN,
  });
  const trendMonths = trendData?.months ?? [];

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KpiCard
          className={PANEL_KPI}
          label="Portfolio Fill Rate"
          value={formatPct((summary?.portfolio_fill_rate ?? 0) * 100)}
          severity={(summary?.portfolio_fill_rate ?? 0) >= 0.98 ? "best" : (summary?.portfolio_fill_rate ?? 0) < 0.90 ? "warning" : "neutral"}
          sublabel="Target: 98%"
          sparkline={trendMonths.length
            ? trendMonths.slice(-6).map((month) => (month.fill_rate ?? 0) * 100)
            : undefined}
        />
        <KpiCard className={PANEL_KPI} label="Total Ordered" value={formatInt(summary?.total_ordered)} />
        <KpiCard className={PANEL_KPI} label="Total Shortage" value={formatInt(summary?.total_shortage_qty)} colorClass="text-red-600" severity={(summary?.total_shortage_qty ?? 0) > 0 ? "warning" : "neutral"} />
        <KpiCard className={PANEL_KPI} label="Partial Fulfillment Events" value={formatInt(summary?.partial_fulfillment_events)} />
      </div>
      {isLoading && <p className="text-xs text-muted-foreground">Loading fill rate data...</p>}
      {!isLoading && trendMonths.length === 0 && (
        <EmptyState
          icon={TrendingUp}
          title="No fill rate data available"
          description="Fill rate measures what percentage of ordered quantities were shipped on time. The monthly trend shows fill rate, total orders, and shortage quantities over time."
          steps={[
            { label: "Apply schema (first time only)", command: "make fill-rate-schema" },
            { label: "Refresh fill rate materialized view", command: "make fill-rate-refresh" },
          ]}
        />
      )}
      {trendMonths.length > 0 && (
        <div>
          <p className="text-xs font-medium mb-2">Monthly Fill Rate Trend (Target: 98%)</p>
          <ResponsiveContainer width="100%" height={160}>
            <ComposedChart data={trendMonths}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month_start" tick={{ fontSize: 10 }} />
              <YAxis tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} domain={[0, 1]} tick={{ fontSize: 10 }} />
              <Tooltip
                formatter={(v: number, name: string) => [`${(v * 100).toFixed(1)}%`, name === "fill_rate" ? "Fill Rate" : name]}
                labelFormatter={(l: string) => `Month: ${l}`}
              />
              <Line type="monotone" dataKey="fill_rate" stroke="#3b82f6" dot={false} strokeWidth={2} />
              <ReferenceLine y={0.98} stroke="#ef4444" strokeDasharray="4 2" label={{ value: "Target 98%", position: "insideTopRight", fontSize: 9, fill: "#ef4444" }} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}
      {gapData && gapData.decomposition.length > 0 && (
        <GapWaterfall
          targetFillRate={gapData.target_fill_rate}
          actualFillRate={gapData.actual_fill_rate}
          gapPct={gapData.gap_pct}
          decomposition={gapData.decomposition}
          month={gapData.month}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Gap Waterfall Decomposition Sub-component
// ---------------------------------------------------------------------------

const CAUSE_COLORS: Record<string, string> = {
  "Safety Stock Shortfall": "bg-red-500",
  "Demand Spike (>20% above forecast)": "bg-orange-500",
  "Lead Time Delay": "bg-yellow-500",
  "Other / Data Gap": "bg-gray-400",
};

function GapWaterfall({
  targetFillRate,
  actualFillRate,
  gapPct,
  decomposition,
  month,
}: {
  targetFillRate: number;
  actualFillRate: number | null;
  gapPct: number | null;
  decomposition: GapDecompositionItem[];
  month: string | null;
}) {
  // Find the max absolute impact for scaling the bars
  const maxImpact = Math.max(...decomposition.map((d) => Math.abs(d.impact_pct)), 0.01);

  return (
    <div className="rounded-lg border p-4 space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-xs font-medium">
          Fill Rate Gap Decomposition{month ? ` (${month})` : ""}
        </p>
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <span>Target: {formatPct(targetFillRate * 100)}</span>
          <span>Actual: {actualFillRate != null ? formatPct(actualFillRate * 100) : "N/A"}</span>
          <span className={gapPct != null && gapPct < 0 ? "text-red-600 font-medium" : "text-green-600 font-medium"}>
            Gap: {gapPct != null ? `${gapPct > 0 ? "+" : ""}${gapPct.toFixed(2)}%` : "N/A"}
          </span>
        </div>
      </div>
      <div className="space-y-2">
        {decomposition.map((d) => (
          <div key={d.cause} className="flex items-center gap-2">
            <div className="w-44 text-xs truncate" title={d.cause}>{d.cause}</div>
            <div className="flex-1 bg-muted rounded h-4 overflow-hidden">
              <div
                className={`${CAUSE_COLORS[d.cause] ?? "bg-gray-400"} h-4 rounded transition-all`}
                style={{ width: `${Math.min((Math.abs(d.impact_pct) / maxImpact) * 100, 100)}%` }}
              />
            </div>
            <span className="text-xs font-mono w-16 text-right">{d.impact_pct}%</span>
            <span className="text-xs text-muted-foreground w-20 text-right">{d.sku_count} SKUs</span>
            <span className="text-xs text-muted-foreground w-24 text-right">{formatInt(d.shortage_qty)} qty</span>
          </div>
        ))}
      </div>
    </div>
  );
}
