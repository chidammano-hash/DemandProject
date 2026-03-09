import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
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
  investmentKeys,
  fetchInvestmentSummary,
  fetchInvestmentDetail,
  fetchInvestmentFrontier,
  runInvestmentPlan,
  STALE,
  type InvestmentRow,
} from "@/api/queries";

import { KpiCard } from "@/components/KpiCard";
import { formatInt, formatPct } from "@/lib/formatters";

const PANEL_KPI = "rounded-lg bg-muted/30 p-3";

export function InvestmentPanel() {
  const queryClient = useQueryClient();
  const [invOffset, setInvOffset] = useState(0);
  const [runStatus, setRunStatus] = useState("");

  const { data: summary, isLoading: summaryLoading } = useQuery({
    queryKey: investmentKeys.summary(),
    queryFn: () => fetchInvestmentSummary(),
    staleTime: STALE.TEN_MIN,
  });

  const { data: detail, isLoading: detailLoading } = useQuery({
    queryKey: investmentKeys.detail({ limit: 20, offset: invOffset }),
    queryFn: () => fetchInvestmentDetail({ limit: 20, offset: invOffset }),
    staleTime: STALE.TEN_MIN,
  });

  const { data: frontier } = useQuery({
    queryKey: investmentKeys.frontier(),
    queryFn: () => fetchInvestmentFrontier(),
    staleTime: STALE.TEN_MIN,
  });

  const runPlanMutation = useMutation({
    mutationFn: () => runInvestmentPlan(),
    onSuccess: () => {
      setRunStatus("Plan computed successfully.");
      queryClient.invalidateQueries({ queryKey: investmentKeys.summary() });
      queryClient.invalidateQueries({ queryKey: investmentKeys.detail() });
      queryClient.invalidateQueries({ queryKey: investmentKeys.frontier() });
    },
    onError: () => setRunStatus("Failed to compute plan. Check auth settings."),
  });

  const totalPages = detail ? Math.ceil(detail.total / 20) : 0;
  const currentPage = Math.floor(invOffset / 20) + 1;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        {runStatus && <span className="text-xs text-muted-foreground">{runStatus}</span>}
        <button
          className="h-8 px-4 text-xs rounded bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          disabled={runPlanMutation.isPending}
          onClick={() => { setRunStatus(""); runPlanMutation.mutate(); }}
        >
          {runPlanMutation.isPending ? "Computing..." : "Run Plan"}
        </button>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KpiCard
          className={PANEL_KPI}
          label="Investment Gap"
          value={summaryLoading ? "..." : summary?.total_investment_gap != null ? `$${formatInt(summary.total_investment_gap)}` : "-"}
          colorClass="text-amber-600"
        />
        <KpiCard
          className={PANEL_KPI}
          label="Current Portfolio CSL"
          value={summaryLoading ? "..." : formatPct(summary?.avg_current_csl != null ? summary.avg_current_csl * 100 : null)}
        />
        <KpiCard
          className={PANEL_KPI}
          label="Target Portfolio CSL"
          value={summaryLoading ? "..." : formatPct(summary?.avg_recommended_csl != null ? summary.avg_recommended_csl * 100 : null)}
          colorClass="text-green-600"
        />
        <KpiCard
          className={PANEL_KPI}
          label="DFUs Analyzed"
          value={summaryLoading ? "..." : (summary?.total_items ?? 0).toLocaleString()}
        />
      </div>

      {frontier && frontier.length > 0 && (
        <div>
          <p className="text-xs font-medium mb-2">Efficient Frontier</p>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={frontier} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis
                  dataKey="cumulative_investment"
                  tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`}
                  tick={{ fontSize: 10 }}
                />
                <YAxis
                  dataKey="achievable_csl"
                  tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                  domain={[0, 1]}
                  tick={{ fontSize: 10 }}
                />
                <Tooltip
                  formatter={(v: number, name: string) =>
                    name === "achievable_csl"
                      ? [`${(v * 100).toFixed(1)}%`, "Achievable CSL"]
                      : [`$${v.toLocaleString()}`, name]
                  }
                />
                <Line
                  type="monotone"
                  dataKey="achievable_csl"
                  stroke="hsl(142, 70%, 45%)"
                  dot={false}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {detailLoading ? (
        <p className="text-xs text-muted-foreground">Loading...</p>
      ) : (
        <>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b text-muted-foreground">
                  <th className="text-right py-1 pr-2">Rank</th>
                  <th className="text-left py-1 pr-2">Item No</th>
                  <th className="text-left py-1 pr-2">Loc</th>
                  <th className="text-center py-1 pr-2">ABC</th>
                  <th className="text-right py-1 pr-2">Current CSL</th>
                  <th className="text-right py-1 pr-2">Target CSL</th>
                  <th className="text-right py-1 pr-2">Inv. Gap ($)</th>
                  <th className="text-right py-1">Marginal ROI</th>
                </tr>
              </thead>
              <tbody>
                {(detail?.rows ?? []).length === 0 ? (
                  <tr>
                    <td colSpan={8} className="py-4 text-center text-muted-foreground">
                      No data. Click Run Plan to compute the investment plan.
                    </td>
                  </tr>
                ) : (
                  (detail?.rows ?? []).map((r: InvestmentRow) => (
                    <tr key={`${r.investment_rank}`} className="border-b last:border-0 hover:bg-muted/30">
                      <td className="py-1 pr-2 text-right text-muted-foreground">{r.investment_rank}</td>
                      <td className="py-1 pr-2 font-mono">{r.item_no}</td>
                      <td className="py-1 pr-2">{r.loc}</td>
                      <td className="py-1 pr-2 text-center">{r.abc_vol ?? "-"}</td>
                      <td className="py-1 pr-2 text-right">
                        {r.current_csl != null ? `${(r.current_csl * 100).toFixed(1)}%` : "-"}
                      </td>
                      <td className="py-1 pr-2 text-right text-green-600">
                        {r.recommended_csl != null ? `${(r.recommended_csl * 100).toFixed(1)}%` : "-"}
                      </td>
                      <td className="py-1 pr-2 text-right text-amber-600">
                        {r.investment_increment != null ? `$${formatInt(r.investment_increment)}` : "-"}
                      </td>
                      <td className="py-1 text-right">
                        {r.marginal_roi != null ? r.marginal_roi.toFixed(2) : "-"}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
          {totalPages > 1 && (
            <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
              <button
                className="px-2 py-1 rounded border disabled:opacity-40"
                disabled={invOffset === 0}
                onClick={() => setInvOffset(Math.max(0, invOffset - 20))}
              >
                Prev
              </button>
              <span>
                Page {currentPage} / {totalPages} - {detail?.total.toLocaleString()} items
              </span>
              <button
                className="px-2 py-1 rounded border disabled:opacity-40"
                disabled={currentPage >= totalPages}
                onClick={() => setInvOffset(invOffset + 20)}
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
