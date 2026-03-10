/**
 * FinancialPlanPanel — F4.1 Financial Inventory Plan
 * Shows inventory value, carrying cost projections, budget utilization, and excess inventory.
 */

import { useQuery } from "@tanstack/react-query";
import { Banknote, DollarSign, TrendingDown, AlertCircle, BarChart2 } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { EmptyState } from "@/components/EmptyState";
import {
  financialPlanKeys,
  fetchBudgetStatus,
  fetchWorkingCapitalTrend,
  STALE_EVO,
} from "@/api/queries/evolution";

const fmtCurrency = (v: number | null | undefined) =>
  v == null ? "—" : `$${(v / 1_000).toFixed(1)}K`;

const fmtM = (v: number | null | undefined) =>
  v == null ? "—" : `$${(v / 1_000_000).toFixed(2)}M`;

export function FinancialPlanPanel() {
  const { data: budgets, isLoading: budgetLoading } = useQuery({
    queryKey: financialPlanKeys.budget({}),
    queryFn: () => fetchBudgetStatus(),
    staleTime: STALE_EVO.FIVE_MIN,
  });

  const { data: trend, isLoading: trendLoading } = useQuery({
    queryKey: financialPlanKeys.workingCapital({}),
    queryFn: () => fetchWorkingCapitalTrend(),
    staleTime: STALE_EVO.FIVE_MIN,
  });

  const budgetList = budgets?.budgets ?? [];
  const trendMonths = trend?.months ?? [];

  // Summary totals
  const totalValue = trendMonths[0]?.inventory_value ?? null;
  const totalCarrying = trendMonths.reduce((s, m) => s + (m.carrying_cost ?? 0), 0);
  const totalExcess = trendMonths.reduce((s, m) => s + (m.excess_value ?? 0), 0);
  const breachedCount = budgetList.filter((b) => b.is_breached).length;

  return (
    <div className="space-y-6 p-4">
      {/* Carrying cost formula explanation */}
      <div className="text-xs text-muted-foreground bg-muted/20 border rounded px-3 py-2 mb-3">
        <strong className="text-foreground">Inventory Carrying Cost</strong> = Average On-Hand Value × 25% per annum ÷ 12 (monthly). The 25% rate includes warehouse costs, capital cost, obsolescence risk, and insurance. Reducing excess inventory directly lowers this cost.
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: "Inventory Value", value: fmtM(totalValue), icon: <DollarSign size={16} /> },
          { label: "Carrying Cost (6-mo)", value: fmtM(totalCarrying), icon: <TrendingDown size={16} /> },
          { label: "Excess Inventory", value: fmtM(totalExcess), icon: <AlertCircle size={16} />, warn: totalExcess > 0 },
          { label: "Budget Breaches", value: breachedCount.toString(), icon: <BarChart2 size={16} />, warn: breachedCount > 0 },
        ].map((c) => (
          <Card key={c.label} className={c.warn ? "border-amber-400" : ""}>
            <CardHeader className="pb-2 flex flex-row items-center justify-between">
              <CardTitle className="text-sm font-medium text-muted-foreground">{c.label}</CardTitle>
              <span className={c.warn ? "text-amber-500" : "text-muted-foreground"}>{c.icon}</span>
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold">{c.value}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Empty state: shown when both budget rows and working capital data are empty and not loading */}
      {!budgetLoading && !trendLoading && !budgetList.length && !trendMonths.length && (
        <EmptyState
          icon={Banknote}
          title="No financial inventory plan"
          description="The financial plan tracks inventory value, carrying cost (holding cost × on-hand value), excess inventory value, and budget utilisation by category over a rolling horizon."
          steps={[
            { label: "Load inventory snapshots", command: "make load-inventory" },
            { label: "Compute investment plan", command: "make investment-plan" },
            { label: "Compute financial inventory plan", command: "make financial-plan-compute" },
          ]}
        />
      )}

      {/* Working capital trend chart */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Working Capital Timeline</CardTitle>
        </CardHeader>
        <CardContent>
          {trendLoading ? (
            <p className="text-sm text-muted-foreground">Loading…</p>
          ) : !trendMonths.length ? (
            <p className="text-sm text-muted-foreground">No trend data available.</p>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={trendMonths} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.08)" />
                <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                <YAxis tickFormatter={(v) => `$${(v / 1000).toFixed(0)}K`} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v: number) => fmtCurrency(v)} />
                <Legend />
                <Line type="monotone" dataKey="inventory_value" name="Inv Value" stroke="#3b82f6" dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="carrying_cost" name="Carrying Cost" stroke="#f59e0b" dot={false} strokeWidth={1.5} />
                <Line type="monotone" dataKey="excess_value" name="Excess Value" stroke="#ef4444" dot={false} strokeWidth={1.5} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </CardContent>
      </Card>

      {/* Budget status table */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Budget Status</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {budgetLoading ? (
            <p className="p-4 text-sm text-muted-foreground">Loading…</p>
          ) : !budgetList.length ? (
            <p className="p-4 text-sm text-muted-foreground">No budgets configured.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-muted/50 text-xs uppercase text-muted-foreground">
                  <tr>
                    {["Category", "Budget Cap", "Committed", "Utilization", "Status"].map((h) => (
                      <th key={h} className="px-3 py-2 text-left font-medium">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {budgetList.map((b, i) => {
                    const budgetStatus = b.is_breached || b.utilization_pct > 100 ? "BREACHED" : b.utilization_pct > 80 ? "AT RISK" : "OK";
                    const budgetStatusClass = budgetStatus === "BREACHED" ? "text-red-600 bg-red-50" : budgetStatus === "AT RISK" ? "text-amber-600 bg-amber-50" : "text-green-600 bg-green-50";
                    return (
                      <tr key={i} className={`border-t hover:bg-muted/30 ${budgetStatus === "BREACHED" ? "bg-red-50" : budgetStatus === "AT RISK" ? "bg-amber-50/40" : ""}`}>
                        <td className="px-3 py-2 font-medium">{b.category}</td>
                        <td className="px-3 py-2 text-xs">{fmtCurrency(b.budget_cap)}</td>
                        <td className="px-3 py-2 text-xs">{fmtCurrency(b.committed_spend)}</td>
                        <td className="px-3 py-2">
                          <div className="flex items-center gap-2" title="Budget utilization: committed spend ÷ total budget cap. ≥80% = AT RISK, >100% = BREACHED">
                            <div className="flex-1 bg-gray-200 rounded-full h-1.5 min-w-[60px]">
                              <div
                                className={`h-1.5 rounded-full ${budgetStatus === "BREACHED" ? "bg-red-500" : budgetStatus === "AT RISK" ? "bg-amber-500" : "bg-green-500"}`}
                                style={{ width: `${Math.min(100, b.utilization_pct)}%` }}
                              />
                            </div>
                            <span className="text-xs font-medium">{b.utilization_pct.toFixed(1)}%</span>
                          </div>
                        </td>
                        <td className="px-3 py-2">
                          <span
                            className={`text-xs px-1.5 py-0.5 rounded font-medium ${budgetStatusClass}`}
                            title={budgetStatus === "AT RISK" ? "Approaching budget cap (>80%). Monitor closely and consider adjusting planned orders." : undefined}
                          >
                            {budgetStatus}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
