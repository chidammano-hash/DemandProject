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
import { DollarSign } from "lucide-react";
import {
  investmentKeys,
  fetchInvestmentSummary,
  fetchInvestmentDetail,
  fetchInvestmentFrontier,
  runInvestmentPlan,
  STALE,
  type InvestmentRow,
} from "@/api/queries";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";

import { KpiCard } from "@/components/KpiCard";
import { EmptyState } from "@/components/EmptyState";
import { formatInt, formatPct } from "@/lib/formatters";

const PANEL_KPI = "rounded-lg bg-muted/30 p-3";

export function InvestmentPanel() {
  const queryClient = useQueryClient();
  const [invOffset, setInvOffset] = useState(0);
  const [runStatus, setRunStatus] = useState("");

  const { filters } = useGlobalFilterContext();
  const gf = {
    brand: filters.brand.length > 0 ? filters.brand.join(",") : undefined,
    category: filters.category.length > 0 ? filters.category.join(",") : undefined,
    market: filters.market.length > 0 ? filters.market.join(",") : undefined,
    item: filters.item.length === 1 ? filters.item[0] : undefined,
    location: filters.location.length === 1 ? filters.location[0] : undefined,
  };

  const { data: summary, isLoading: summaryLoading } = useQuery({
    queryKey: investmentKeys.summary(undefined, gf),
    queryFn: () => fetchInvestmentSummary(gf),
    staleTime: STALE.TEN_MIN,
  });

  const { data: detail, isLoading: detailLoading } = useQuery({
    queryKey: investmentKeys.detail({ ...gf, limit: 20, offset: invOffset }),
    queryFn: () => fetchInvestmentDetail({ ...gf, limit: 20, offset: invOffset }),
    staleTime: STALE.TEN_MIN,
  });

  const { data: frontier } = useQuery({
    queryKey: investmentKeys.frontier(undefined, gf),
    queryFn: () => fetchInvestmentFrontier(gf),
    staleTime: STALE.TEN_MIN,
  });

  const runPlanMutation = useMutation({
    mutationFn: () => runInvestmentPlan(),
    onSuccess: () => {
      setRunStatus("Plan computed successfully.");
      queryClient.invalidateQueries({ queryKey: ["investment"] });
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
          <div className="text-xs text-muted-foreground bg-muted/20 border rounded px-3 py-2 mb-3">
            <strong className="text-foreground">Efficient Frontier</strong>: Each point shows the portfolio fill rate achievable at a given capital investment. Use the curve to answer "What service level can we reach with a $2M budget?" or "How much investment do we need to reach 98% SL?"
          </div>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={frontier} margin={{ top: 4, right: 16, left: 16, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis
                  dataKey="cumulative_investment"
                  tick={{ fontSize: 10 }}
                  label={{ value: "Total Capital Investment ($)", position: "insideBottomRight", offset: -5, fontSize: 9, fill: "hsl(var(--muted-foreground))" }}
                  tickFormatter={(v: number) => v >= 1000000 ? `$${(v/1000000).toFixed(1)}M` : v >= 1000 ? `$${(v/1000).toFixed(0)}K` : `$${v}`}
                />
                <YAxis
                  dataKey="achievable_csl"
                  tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                  domain={[0, 1]}
                  tick={{ fontSize: 10 }}
                  label={{ value: "Portfolio Service Level (%)", angle: -90, position: "insideLeft", fontSize: 9, fill: "hsl(var(--muted-foreground))" }}
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

      {detailLoading || summaryLoading ? (
        <p className="text-xs text-muted-foreground">Loading...</p>
      ) : (detail?.rows ?? []).length === 0 && !frontier?.length && !summary ? (
        <EmptyState
          icon={DollarSign}
          title="No investment plan computed"
          description="The investment plan computes an efficient frontier: each point shows the portfolio service level achievable at a given capital investment. Use this to answer 'What fill rate can we reach with a $2M inventory budget?'"
          steps={[
            { label: "Compute safety stock targets first", command: "make ss-compute" },
            { label: "Apply schema (first time only)", command: "make investment-schema" },
            { label: "Compute efficient frontier", command: "make investment-plan" },
          ]}
        />
      ) : (
        <>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b text-muted-foreground">
                  <th className="text-right py-1 pr-2" title="Priority rank: allocate capital starting from rank 1 (highest marginal ROI) for maximum service level improvement">Rank</th>
                  <th className="text-left py-1 pr-2">Item No</th>
                  <th className="text-left py-1 pr-2">Loc</th>
                  <th className="text-center py-1 pr-2">ABC</th>
                  <th className="text-right py-1 pr-2" title="Current fill rate for this item-location">Current Fill Rate</th>
                  <th className="text-right py-1 pr-2" title="Recommended fill rate based on ABC class and policy">Target Fill Rate</th>
                  <th className="text-right py-1 pr-2" title="Additional capital needed to reach this DFU's target service level">Additional SS Investment ($)</th>
                  <th className="text-right py-1" title="Fill rate improvement per dollar invested. Higher = more efficient capital allocation. Rank items by this to maximize service level within budget.">Marginal ROI</th>
                </tr>
              </thead>
              <tbody>
                {(detail?.rows ?? []).map((r: InvestmentRow) => (
                  <tr key={`${r.investment_rank}`} className={`border-b last:border-0 hover:bg-muted/30 ${
                    r.investment_rank <= Math.ceil((detail?.total ?? 0) / 4) ? "bg-green-50 dark:bg-green-950/20" : ""
                  }`}>
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
                      {r.investment_increment != null ? (r.investment_increment >= 1000000 ? `$${(r.investment_increment/1000000).toFixed(1)}M` : r.investment_increment >= 1000 ? `$${(r.investment_increment/1000).toFixed(0)}K` : `$${r.investment_increment.toFixed(0)}`) : "—"}
                    </td>
                    <td className="py-1 text-right">
                      {r.marginal_roi != null ? r.marginal_roi.toFixed(2) : "-"}
                    </td>
                  </tr>
                ))}
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
