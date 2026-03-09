import { useQuery } from "@tanstack/react-query";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  fillRateKeys,
  fetchFillRateSummary,
  fetchFillRateTrend,
  STALE,
} from "@/api/queries";
import { KpiCard } from "@/components/KpiCard";
import { formatInt, formatPct } from "@/lib/formatters";

const PANEL_KPI = "rounded-lg bg-muted/30 p-3";

export function FillRatePanel() {
  const { data: summary, isLoading } = useQuery({
    queryKey: fillRateKeys.summary(),
    queryFn: () => fetchFillRateSummary(),
    staleTime: STALE.FIVE_MIN,
  });
  const { data: trendData } = useQuery({
    queryKey: fillRateKeys.trend(),
    queryFn: () => fetchFillRateTrend(),
    staleTime: STALE.FIVE_MIN,
  });

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KpiCard className={PANEL_KPI} label="Portfolio Fill Rate" value={formatPct((summary?.portfolio_fill_rate ?? 0) * 100)} />
        <KpiCard className={PANEL_KPI} label="Total Ordered" value={formatInt(summary?.total_ordered)} />
        <KpiCard className={PANEL_KPI} label="Total Shortage" value={formatInt(summary?.total_shortage_qty)} colorClass="text-red-600" />
        <KpiCard className={PANEL_KPI} label="Partial Fulfillment Events" value={formatInt(summary?.partial_fulfillment_events)} />
      </div>
      {isLoading && <p className="text-xs text-muted-foreground">Loading fill rate data...</p>}
      {trendData?.months && trendData.months.length > 0 && (
        <div>
          <p className="text-xs font-medium mb-2">Monthly Fill Rate Trend</p>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={trendData.months}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month_start" tick={{ fontSize: 10 }} />
              <YAxis tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} domain={[0, 1]} tick={{ fontSize: 10 }} />
              <Tooltip formatter={(v: number) => `${(v * 100).toFixed(1)}%`} />
              <Line type="monotone" dataKey="fill_rate" stroke="#3b82f6" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
