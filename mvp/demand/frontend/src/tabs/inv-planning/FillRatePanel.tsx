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

export function FillRatePanel() {
  const { data: summary, isLoading } = useQuery({
    queryKey: fillRateKeys.summary(),
    queryFn: () => fetchFillRateSummary(),
    staleTime: STALE,
  });
  const { data: trendData } = useQuery({
    queryKey: fillRateKeys.trend(),
    queryFn: () => fetchFillRateTrend(),
    staleTime: STALE,
  });

  const pct = (n: number | null | undefined) =>
    n == null ? "—" : `${(Number(n) * 100).toFixed(1)}%`;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Portfolio Fill Rate</p>
          <p className="text-xl font-bold">{pct(summary?.portfolio_fill_rate)}</p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Total Ordered</p>
          <p className="text-xl font-bold">{Math.round(Number(summary?.total_ordered ?? 0)).toLocaleString()}</p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Total Shortage</p>
          <p className="text-xl font-bold text-red-600">{Math.round(Number(summary?.total_shortage_qty ?? 0)).toLocaleString()}</p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Partial Fulfillment Events</p>
          <p className="text-xl font-bold">{(summary?.partial_fulfillment_events ?? 0).toLocaleString()}</p>
        </div>
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
