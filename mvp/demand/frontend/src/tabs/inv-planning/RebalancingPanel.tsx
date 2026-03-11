import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ZAxis,
  ReferenceLine,
} from "recharts";
import { Repeat2 } from "lucide-react";
import {
  rebalancingKeys,
  fetchRebalancingKpis,
  fetchRebalancingPlans,
  fetchPlanTransfers,
  computeRebalancingPlan,
  approveTransfer,
  rejectTransfer,
  approveAllTransfers,
  STALE,
  type RebalancingTransfer,
} from "@/api/queries";
import { KpiCard } from "@/components/KpiCard";
import { EmptyState } from "@/components/EmptyState";
import { formatFixed, formatInt } from "@/lib/formatters";

const PAGE = 50;

const URGENCY_BADGE: Record<string, string> = {
  critical: "bg-red-100 text-red-800",
  high: "bg-amber-100 text-amber-800",
  medium: "bg-yellow-100 text-yellow-800",
  low: "bg-neutral-100 text-neutral-600",
};

const URGENCY_ROW_BG: Record<string, string> = {
  critical: "bg-red-50 dark:bg-red-950/20",
  high: "bg-amber-50 dark:bg-amber-950/20",
  medium: "bg-yellow-50 dark:bg-yellow-950/20",
  low: "",
};

const STATUS_BADGE: Record<string, string> = {
  recommended: "bg-blue-100 text-blue-800",
  approved: "bg-green-100 text-green-800",
  rejected: "bg-red-100 text-red-800",
  hold: "bg-yellow-100 text-yellow-800",
};

const PANEL_KPI = "rounded-lg bg-muted/30 p-3";

export function RebalancingPanel() {
  const queryClient = useQueryClient();
  const [offset, setOffset] = useState(0);
  const [solver, setSolver] = useState("greedy");
  const [horizon, setHorizon] = useState(4);
  const [computeStatus, setComputeStatus] = useState("");
  const [rejectModal, setRejectModal] = useState<{ id: string } | null>(null);
  const [rejectReason, setRejectReason] = useState("");

  // KPIs
  const { data: kpis, isLoading: kpisLoading } = useQuery({
    queryKey: rebalancingKeys.kpis(),
    queryFn: fetchRebalancingKpis,
    staleTime: STALE.TWO_MIN,
  });

  // Latest plan
  const { data: plans } = useQuery({
    queryKey: rebalancingKeys.plans({ limit: 1 }),
    queryFn: () => fetchRebalancingPlans({ limit: 1 }),
    staleTime: STALE.TWO_MIN,
  });

  const latestPlan = plans?.rows?.[0];

  // Transfers for latest plan
  const { data: transfers, isLoading: transfersLoading } = useQuery({
    queryKey: rebalancingKeys.transfers(latestPlan?.plan_id ?? "", { limit: PAGE, offset }),
    queryFn: () => fetchPlanTransfers(latestPlan!.plan_id, { limit: PAGE, offset }),
    enabled: !!latestPlan?.plan_id,
    staleTime: STALE.TWO_MIN,
  });

  // Compute mutation
  const computeMutation = useMutation({
    mutationFn: () => computeRebalancingPlan({ solver, horizon_weeks: horizon }),
    onSuccess: () => {
      setComputeStatus("Computation started. Refresh in a few seconds.");
      queryClient.invalidateQueries({ queryKey: ["rebalancing-kpis"] });
      queryClient.invalidateQueries({ queryKey: ["rebalancing-plans"] });
    },
    onError: () => setComputeStatus("Failed. Check API key / server logs."),
  });

  // Approve mutation
  const approveMutation = useMutation({
    mutationFn: (transferId: string) =>
      approveTransfer(transferId, { approved_by: "planner" }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["rebalancing-transfers"] });
    },
  });

  // Reject mutation
  const doReject = useMutation({
    mutationFn: ({ id, reason }: { id: string; reason: string }) =>
      rejectTransfer(id, { rejection_reason: reason }),
    onSuccess: () => {
      setRejectModal(null);
      setRejectReason("");
      queryClient.invalidateQueries({ queryKey: ["rebalancing-transfers"] });
    },
  });

  // Approve all mutation
  const approveAllMutation = useMutation({
    mutationFn: () =>
      approveAllTransfers(latestPlan!.plan_id, { approved_by: "planner" }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["rebalancing-transfers"] });
      queryClient.invalidateQueries({ queryKey: ["rebalancing-plans"] });
    },
  });

  const totalPages = transfers ? Math.ceil(transfers.total / PAGE) : 0;
  const currentPage = Math.floor(offset / PAGE) + 1;

  // Chart data: top 10 transfers by qty for bar chart
  const barData = (transfers?.rows ?? [])
    .filter((t) => t.recommended_qty != null)
    .slice(0, 10)
    .map((t) => ({
      label: `${t.source_loc}→${t.dest_loc}`,
      qty: t.recommended_qty ?? 0,
      urgency: t.urgency,
      item: t.item_no,
    }));

  // Scatter data for cost-benefit
  const scatterData = (transfers?.rows ?? [])
    .filter((t) => t.transfer_cost != null && t.net_benefit != null)
    .map((t) => ({
      cost: t.transfer_cost ?? 0,
      benefit: t.net_benefit ?? 0,
      qty: t.recommended_qty ?? 1,
      urgency: t.urgency,
      item: t.item_no,
    }));

  const urgencyColor = (u: string) =>
    u === "critical" ? "#ef4444" : u === "high" ? "#f59e0b" : u === "medium" ? "#eab308" : "#6b7280";

  return (
    <div className="space-y-4">
      {/* Info banner */}
      <div className="text-xs text-muted-foreground bg-muted/20 border rounded px-3 py-2">
        <strong className="text-foreground">Inventory Rebalancing</strong> identifies items
        with simultaneous excess at one location and shortage at another, then recommends
        cost-optimal transfers across the distribution network to improve service levels
        while minimizing transfer costs.
      </div>

      {/* KPI cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KpiCard
          className={PANEL_KPI}
          label="Transfer Opportunities"
          value={kpisLoading ? "..." : formatInt(kpis?.imbalanced_items ?? 0)}
          colorClass="text-blue-600"
        />
        <KpiCard
          className={PANEL_KPI}
          label="Est. Cost Savings"
          value={kpisLoading ? "..." : kpis?.latest_plan?.total_avoided_stockout_value != null
            ? `$${formatInt(kpis.latest_plan.total_avoided_stockout_value)}`
            : "-"}
          colorClass="text-green-600"
        />
        <KpiCard
          className={PANEL_KPI}
          label="Urgent Transfers"
          value={kpisLoading ? "..." : formatInt(
            (transfers?.rows ?? []).filter((t) => t.urgency === "critical" || t.urgency === "high").length,
          )}
          colorClass="text-amber-600"
        />
        <KpiCard
          className={PANEL_KPI}
          label="Network Balance"
          value={kpisLoading ? "..." : kpis?.network_balance_score != null
            ? `${kpis.network_balance_score.toFixed(1)}%`
            : "-"}
        />
      </div>

      {/* Action bar */}
      <div className="flex items-center gap-3 flex-wrap">
        <select
          className="h-8 px-2 text-xs rounded border bg-background"
          value={solver}
          onChange={(e) => setSolver(e.target.value)}
        >
          <option value="greedy">Greedy Solver</option>
          <option value="lp">LP Solver</option>
        </select>
        <label className="flex items-center gap-1 text-xs text-muted-foreground">
          Horizon:
          <input
            type="number"
            min={1}
            max={12}
            className="w-14 h-8 px-2 text-xs rounded border bg-background"
            value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value))}
          />
          <span>weeks</span>
        </label>
        <button
          className="h-8 px-4 text-xs rounded bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          disabled={computeMutation.isPending}
          onClick={() => { setComputeStatus(""); computeMutation.mutate(); }}
        >
          {computeMutation.isPending ? "Computing..." : "Compute Plan"}
        </button>
        {latestPlan && latestPlan.status === "draft" && (
          <button
            className="h-8 px-4 text-xs rounded bg-green-600 text-white hover:bg-green-700 disabled:opacity-50"
            disabled={approveAllMutation.isPending}
            onClick={() => approveAllMutation.mutate()}
          >
            {approveAllMutation.isPending ? "Approving..." : "Approve All"}
          </button>
        )}
        {computeStatus && (
          <span className="text-xs text-muted-foreground">{computeStatus}</span>
        )}
      </div>

      {/* Content: empty state or plan data */}
      {!latestPlan && !kpisLoading ? (
        <EmptyState
          icon={Repeat2}
          title="No rebalancing plan computed"
          description="The rebalancing engine detects spatial inventory imbalances and recommends cost-optimal cross-location transfers."
          steps={[
            { label: "Apply schema (first time only)", command: "make rebalancing-schema" },
            { label: "Compute rebalancing plan", command: "make rebalancing-compute" },
          ]}
        />
      ) : (
        <>
          {/* Network flow bar chart */}
          {barData.length > 0 && (
            <div>
              <p className="text-xs font-medium mb-2">Top Transfer Routes</p>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={barData} layout="vertical" margin={{ top: 4, right: 16, left: 80, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis type="number" tick={{ fontSize: 10 }} />
                    <YAxis dataKey="label" type="category" tick={{ fontSize: 9 }} width={75} />
                    <Tooltip
                      formatter={(v: number) => [v.toLocaleString(), "Qty"]}
                      labelFormatter={(l: string) => `Route: ${l}`}
                    />
                    <Bar dataKey="qty" fill="hsl(217, 91%, 60%)" radius={[0, 3, 3, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Transfers table */}
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b text-muted-foreground">
                  <th className="text-left py-1 pr-2">Item</th>
                  <th className="text-left py-1 pr-2">Source</th>
                  <th className="text-left py-1 pr-2">Dest</th>
                  <th className="text-right py-1 pr-2">Qty</th>
                  <th className="text-right py-1 pr-2">Cost</th>
                  <th className="text-right py-1 pr-2">Benefit</th>
                  <th className="text-right py-1 pr-2">ROI</th>
                  <th className="text-center py-1 pr-2">Urgency</th>
                  <th className="text-center py-1 pr-2">Status</th>
                  <th className="text-right py-1">Actions</th>
                </tr>
              </thead>
              <tbody>
                {transfersLoading ? (
                  <tr><td colSpan={10} className="py-4 text-center text-muted-foreground">Loading...</td></tr>
                ) : (transfers?.rows ?? []).length === 0 ? (
                  <tr><td colSpan={10} className="py-4 text-center text-muted-foreground">No transfers in this plan.</td></tr>
                ) : (
                  (transfers?.rows ?? []).map((t: RebalancingTransfer) => (
                    <tr key={t.transfer_id} className={`border-b last:border-0 hover:bg-muted/30 ${URGENCY_ROW_BG[t.urgency] ?? ""}`}>
                      <td className="py-1 pr-2 font-mono">{t.item_no}</td>
                      <td className="py-1 pr-2">{t.source_loc}</td>
                      <td className="py-1 pr-2">{t.dest_loc}</td>
                      <td className="py-1 pr-2 text-right">{formatFixed(t.recommended_qty, 0)}</td>
                      <td className="py-1 pr-2 text-right text-red-600">${formatFixed(t.transfer_cost, 2)}</td>
                      <td className="py-1 pr-2 text-right text-green-600">${formatFixed(t.net_benefit, 2)}</td>
                      <td className="py-1 pr-2 text-right">{t.roi != null ? t.roi.toFixed(2) : "-"}</td>
                      <td className="py-1 pr-2 text-center">
                        <span className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-medium ${URGENCY_BADGE[t.urgency] ?? ""}`}>
                          {t.urgency}
                        </span>
                      </td>
                      <td className="py-1 pr-2 text-center">
                        <span className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-medium ${STATUS_BADGE[t.status] ?? "bg-neutral-100 text-neutral-600"}`}>
                          {t.status}
                        </span>
                      </td>
                      <td className="py-1 text-right">
                        {t.status === "recommended" && (
                          <div className="flex gap-1 justify-end">
                            <button
                              className="px-2 py-0.5 rounded text-[10px] bg-green-100 text-green-800 hover:bg-green-200"
                              onClick={() => approveMutation.mutate(t.transfer_id)}
                              disabled={approveMutation.isPending}
                            >
                              Approve
                            </button>
                            <button
                              className="px-2 py-0.5 rounded text-[10px] bg-red-100 text-red-800 hover:bg-red-200"
                              onClick={() => { setRejectModal({ id: t.transfer_id }); setRejectReason(""); }}
                            >
                              Reject
                            </button>
                          </div>
                        )}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <button
                className="px-2 py-1 rounded border disabled:opacity-40"
                disabled={offset === 0}
                onClick={() => setOffset(Math.max(0, offset - PAGE))}
              >
                Prev
              </button>
              <span>Page {currentPage} / {totalPages} — {transfers?.total.toLocaleString()} transfers</span>
              <button
                className="px-2 py-1 rounded border disabled:opacity-40"
                disabled={currentPage >= totalPages}
                onClick={() => setOffset(offset + PAGE)}
              >
                Next
              </button>
            </div>
          )}

          {/* Cost-Benefit scatter */}
          {scatterData.length > 0 && (
            <div>
              <p className="text-xs font-medium mb-2">Cost vs. Benefit</p>
              <div className="text-xs text-muted-foreground bg-muted/20 border rounded px-3 py-2 mb-3">
                Each dot is one transfer. Dots above the break-even line (y=0) yield positive ROI.
                Size = transfer quantity. Position = transfer cost (x) vs net benefit (y).
              </div>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 4, right: 16, left: 16, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis
                      dataKey="cost"
                      type="number"
                      tick={{ fontSize: 10 }}
                      label={{ value: "Transfer Cost ($)", position: "insideBottomRight", offset: -5, fontSize: 9, fill: "hsl(var(--muted-foreground))" }}
                    />
                    <YAxis
                      dataKey="benefit"
                      type="number"
                      tick={{ fontSize: 10 }}
                      label={{ value: "Net Benefit ($)", angle: -90, position: "insideLeft", fontSize: 9, fill: "hsl(var(--muted-foreground))" }}
                    />
                    <ZAxis dataKey="qty" range={[30, 300]} />
                    <ReferenceLine y={0} stroke="hsl(var(--muted-foreground))" strokeDasharray="3 3" label={{ value: "Break-even", fontSize: 9, fill: "hsl(var(--muted-foreground))" }} />
                    <Tooltip
                      formatter={(v: number, name: string) => [
                        name === "benefit" ? `$${v.toFixed(2)}` : `$${v.toFixed(2)}`,
                        name === "benefit" ? "Net Benefit" : "Cost",
                      ]}
                    />
                    <Scatter
                      data={scatterData}
                      fill="hsl(217, 91%, 60%)"
                      fillOpacity={0.7}
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </>
      )}

      {/* Reject modal */}
      {rejectModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
          <div className="bg-card border rounded-lg shadow-lg p-6 w-96 max-w-[90vw]">
            <h3 className="text-sm font-semibold mb-3">Reject Transfer</h3>
            <textarea
              className="w-full h-20 px-3 py-2 text-xs border rounded bg-background resize-none"
              placeholder="Rejection reason (required)..."
              value={rejectReason}
              onChange={(e) => setRejectReason(e.target.value)}
            />
            <div className="flex gap-2 mt-3 justify-end">
              <button
                className="px-3 py-1.5 text-xs rounded border hover:bg-muted"
                onClick={() => setRejectModal(null)}
              >
                Cancel
              </button>
              <button
                className="px-3 py-1.5 text-xs rounded bg-red-600 text-white hover:bg-red-700 disabled:opacity-50"
                disabled={!rejectReason.trim() || doReject.isPending}
                onClick={() => doReject.mutate({ id: rejectModal.id, reason: rejectReason })}
              >
                {doReject.isPending ? "Rejecting..." : "Reject"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
