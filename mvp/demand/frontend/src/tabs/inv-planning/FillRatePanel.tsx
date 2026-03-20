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
  STALE,
} from "@/api/queries";
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

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KpiCard
          className={PANEL_KPI}
          label="Portfolio Fill Rate"
          value={formatPct((summary?.portfolio_fill_rate ?? 0) * 100)}
          severity={(summary?.portfolio_fill_rate ?? 0) >= 0.98 ? "best" : (summary?.portfolio_fill_rate ?? 0) < 0.90 ? "warning" : "neutral"}
          sublabel="Target: 98%"
          sparkline={trendData?.months?.length
            ? trendData.months.slice(-6).map((m: { fill_rate: number }) => m.fill_rate * 100)
            : undefined}
        />
        <KpiCard className={PANEL_KPI} label="Total Ordered" value={formatInt(summary?.total_ordered)} />
        <KpiCard className={PANEL_KPI} label="Total Shortage" value={formatInt(summary?.total_shortage_qty)} colorClass="text-red-600" severity={(summary?.total_shortage_qty ?? 0) > 0 ? "warning" : "neutral"} />
        <KpiCard className={PANEL_KPI} label="Partial Fulfillment Events" value={formatInt(summary?.partial_fulfillment_events)} />
      </div>
      {isLoading && <p className="text-xs text-muted-foreground">Loading fill rate data...</p>}
      {!isLoading && (!trendData?.months || trendData.months.length === 0) && (
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
      {trendData?.months?.length > 0 && (
        <div>
          <p className="text-xs font-medium mb-2">Monthly Fill Rate Trend (Target: 98%)</p>
          <ResponsiveContainer width="100%" height={160}>
            <ComposedChart data={trendData.months}>
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
    </div>
  );
}
