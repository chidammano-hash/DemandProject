import { useQuery } from "@tanstack/react-query";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import {
  insightKeys,
  fetchCashFlowTimeline,
  STALE_INSIGHTS,
  type CashFlowMonth,
} from "@/api/queries";
import { KpiCard } from "@/components/KpiCard";
import { EmptyState } from "@/components/EmptyState";
import { formatCurrency } from "@/lib/formatters";
import { DollarSign } from "lucide-react";

const PANEL_KPI = "rounded-lg bg-muted/30 p-3";

export function CashFlowPanel() {
  const { data, isLoading, error } = useQuery({
    queryKey: insightKeys.cashFlow(),
    queryFn: fetchCashFlowTimeline,
    staleTime: STALE_INSIGHTS.FIVE_MIN,
  });

  if (error) {
    return (
      <div className="text-xs text-red-600 p-4">
        Failed to load cash flow timeline: {(error as Error).message}
      </div>
    );
  }

  const summary = data?.summary;
  const months = data?.months ?? [];

  return (
    <div className="space-y-4">
      <div className="text-xs text-muted-foreground bg-muted/20 border rounded px-3 py-2">
        Cash flow timeline showing projected monthly inventory outflow broken down by PO commitments,
        planned orders, and safety stock investments. Helps finance and planning teams align on working capital needs.
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        <KpiCard
          className={PANEL_KPI}
          label="Total 6-Month Outflow"
          value={isLoading ? "..." : formatCurrency(summary?.total_6m_outflow)}
        />
        <KpiCard
          className={PANEL_KPI}
          label="Largest Month"
          value={isLoading ? "..." : summary?.largest_month ?? "—"}
        />
        <KpiCard
          className={PANEL_KPI}
          label="Avg Monthly"
          value={isLoading ? "..." : formatCurrency(summary?.avg_monthly)}
        />
      </div>

      {isLoading ? (
        <p className="text-xs text-muted-foreground">Loading cash flow data...</p>
      ) : months.length === 0 ? (
        <EmptyState
          icon={DollarSign}
          title="No cash flow data available"
          description="The cash flow timeline combines PO commitments, planned orders, and safety stock investment projections. Ensure order recommendations and safety stock have been computed."
          steps={[
            { label: "Compute safety stock", command: "make ss-compute" },
            { label: "Generate planned orders", command: "make planned-orders" },
          ]}
        />
      ) : (
        <>
          {/* Stacked bar chart */}
          <div style={{ height: "calc(min(360px, 40vh))" }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={months} margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-border/40" />
                <XAxis
                  dataKey="month"
                  tick={{ fontSize: 11 }}
                  className="text-muted-foreground"
                />
                <YAxis
                  tick={{ fontSize: 11 }}
                  className="text-muted-foreground"
                  tickFormatter={(v: number) => formatCurrency(v)}
                />
                <Tooltip
                  contentStyle={{
                    fontSize: 11,
                    borderRadius: 8,
                    border: "1px solid var(--border)",
                    backgroundColor: "var(--card)",
                  }}
                  formatter={(value: number, name: string) => [
                    formatCurrency(value),
                    name,
                  ]}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar
                  dataKey="po_committed"
                  name="PO Committed"
                  stackId="a"
                  fill="#3b82f6"
                  radius={[0, 0, 0, 0]}
                />
                <Bar
                  dataKey="planned_orders"
                  name="Planned Orders"
                  stackId="a"
                  fill="#8b5cf6"
                  radius={[0, 0, 0, 0]}
                />
                <Bar
                  dataKey="ss_investment"
                  name="SS Investment"
                  stackId="a"
                  fill="#f59e0b"
                  radius={[2, 2, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Detail table */}
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b text-muted-foreground">
                  <th className="text-left py-1 pr-3">Month</th>
                  <th className="text-right py-1 pr-3">PO Committed</th>
                  <th className="text-right py-1 pr-3">Planned Orders</th>
                  <th className="text-right py-1 pr-3">SS Investment</th>
                  <th className="text-right py-1 font-semibold">Total</th>
                </tr>
              </thead>
              <tbody>
                {months.map((m: CashFlowMonth) => (
                  <tr key={m.month} className="border-b last:border-0 hover:bg-muted/40">
                    <td className="py-1 pr-3 font-medium">{m.month}</td>
                    <td className="py-1 pr-3 text-right font-mono tabular-nums">
                      {formatCurrency(m.po_committed)}
                    </td>
                    <td className="py-1 pr-3 text-right font-mono tabular-nums">
                      {formatCurrency(m.planned_orders)}
                    </td>
                    <td className="py-1 pr-3 text-right font-mono tabular-nums">
                      {formatCurrency(m.ss_investment)}
                    </td>
                    <td className="py-1 text-right font-mono tabular-nums font-semibold">
                      {formatCurrency(m.total)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
