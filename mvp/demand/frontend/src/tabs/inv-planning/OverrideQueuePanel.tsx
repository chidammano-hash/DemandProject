/**
 * F2.3 — Override Queue Panel
 *
 * Shows all pending planner overrides awaiting approval,
 * with Approve/Reject workflow for demand managers.
 */

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchOverrides,
  fetchOverrideSummary,
  approveOverride,
  rejectOverride,
  STALE,
} from "@/api/queries";

type OverrideRow = {
  override_id: number;
  item_no: string;
  loc: string;
  override_month: string;
  override_type: string;
  override_multiplier?: number | null;
  override_qty?: number | null;
  estimated_impact_units?: number | null;
  estimated_impact_value?: number | null;
  override_reason: string;
  created_by: string;
  status: string;
  requires_approval: boolean;
};

function formatCurrency(v: number | null | undefined): string {
  if (v == null) return "—";
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(1)}M`;
  if (v >= 1_000) return `$${(v / 1_000).toFixed(1)}K`;
  return `$${v.toFixed(0)}`;
}

const STATUS_OPTIONS = [
  { label: "Pending Approval", value: "pending_approval" },
  { label: "Approved", value: "approved" },
  { label: "Rejected", value: "rejected" },
  { label: "All", value: "" },
];

const TYPE_BADGE: Record<string, string> = {
  PROMO: "bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-300",
  LAUNCH: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300",
  PHASE_OUT: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300",
  MARKET_EVENT: "bg-cyan-100 text-cyan-700 dark:bg-cyan-900/30 dark:text-cyan-300",
  CAPACITY_LOCK: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300",
  MANUAL: "bg-muted text-muted-foreground",
};

export function OverrideQueuePanel() {
  const qc = useQueryClient();
  const [statusFilter, setStatusFilter] = useState("pending_approval");
  const [rejectModalId, setRejectModalId] = useState<number | null>(null);
  const [rejectReason, setRejectReason] = useState("");

  const summaryQ = useQuery({
    queryKey: ["override-summary"],
    queryFn: fetchOverrideSummary,
    staleTime: STALE.TWO_MIN,
  });

  const overridesQ = useQuery({
    queryKey: ["overrides", { status: statusFilter }],
    queryFn: () => fetchOverrides({ status: statusFilter || undefined }),
    staleTime: STALE.ONE_MIN,
  });

  const approveMutation = useMutation({
    mutationFn: ({ id, approvedBy }: { id: number; approvedBy: string }) =>
      approveOverride(id, approvedBy),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["overrides"] });
      qc.invalidateQueries({ queryKey: ["override-summary"] });
    },
  });

  const rejectMutation = useMutation({
    mutationFn: ({ id, reason }: { id: number; reason: string }) =>
      rejectOverride(id, reason),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["overrides"] });
      qc.invalidateQueries({ queryKey: ["override-summary"] });
      setRejectModalId(null);
      setRejectReason("");
    },
  });

  const summary = summaryQ.data;
  const overrides = overridesQ.data;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div>
        <h3 className="font-semibold text-base">Override Queue</h3>
        <p className="text-xs text-muted-foreground mt-0.5">
          Planner overrides to the statistical demand plan. Approve or reject pending items.
        </p>
      </div>

      {/* Summary KPIs */}
      {summary && (
        <div className="grid grid-cols-3 md:grid-cols-5 gap-3">
          {[
            { label: "Pending", value: summary.by_status?.pending_approval ?? 0, warn: true },
            { label: "Approved", value: summary.by_status?.approved ?? 0, warn: false },
            { label: "Rejected", value: summary.by_status?.rejected ?? 0, warn: false },
            {
              label: "Total Uplift",
              value: `${(summary.total_uplift_units ?? 0).toLocaleString()} units`,
              warn: false,
            },
            {
              label: "Impact Value",
              value: formatCurrency(summary.total_uplift_value ?? null),
              warn: false,
            },
          ].map((kpi) => (
            <div
              key={kpi.label}
              className={`rounded-lg border p-3 ${kpi.warn && Number(kpi.value) > 0 ? "border-amber-400/50 bg-amber-50/50 dark:bg-amber-900/10" : "bg-card"}`}
            >
              <p className="text-xs text-muted-foreground">{kpi.label}</p>
              <p
                className={`text-xl font-semibold mt-1 ${kpi.warn && Number(kpi.value) > 0 ? "text-amber-600 dark:text-amber-400" : ""}`}
              >
                {kpi.value}
              </p>
            </div>
          ))}
        </div>
      )}

      {/* Filter */}
      <div className="flex gap-2 flex-wrap items-center">
        {STATUS_OPTIONS.map((opt) => (
          <button
            key={opt.value}
            onClick={() => setStatusFilter(opt.value)}
            className={`rounded px-3 py-1 text-xs font-medium border ${
              statusFilter === opt.value
                ? "bg-primary text-primary-foreground border-primary"
                : "bg-background hover:bg-muted"
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {/* Table */}
      {overridesQ.isLoading ? (
        <div className="text-xs text-muted-foreground py-4 text-center">Loading...</div>
      ) : !overrides?.overrides?.length ? (
        <div className="rounded-lg border bg-card p-8 text-center text-sm text-muted-foreground">
          No overrides found for the selected filter.
        </div>
      ) : (
        <div className="rounded-lg border overflow-auto">
          <table className="w-full text-xs">
            <thead className="bg-muted/50">
              <tr>
                {[
                  "Item",
                  "Loc",
                  "Month",
                  "Type",
                  "Multiplier",
                  "Impact Units",
                  "Impact Value",
                  "Reason",
                  "Submitted By",
                  "Status",
                  "Actions",
                ].map((h) => (
                  <th
                    key={h}
                    className="px-3 py-2 text-left font-medium text-muted-foreground whitespace-nowrap"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y">
              {overrides.overrides.map((ov: OverrideRow) => (
                <tr key={ov.override_id} className="hover:bg-muted/30">
                  <td className="px-3 py-2 font-mono">{ov.item_no}</td>
                  <td className="px-3 py-2">{ov.loc}</td>
                  <td className="px-3 py-2 whitespace-nowrap">
                    {ov.override_month
                      ? new Date(ov.override_month + "T00:00:00").toLocaleDateString("en-US", {
                          month: "short",
                          year: "2-digit",
                        })
                      : "—"}
                  </td>
                  <td className="px-3 py-2">
                    <span
                      className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${
                        TYPE_BADGE[ov.override_type] ?? "bg-muted text-muted-foreground"
                      }`}
                    >
                      {ov.override_type}
                    </span>
                  </td>
                  <td className="px-3 py-2">
                    {ov.override_multiplier != null
                      ? `×${ov.override_multiplier.toFixed(2)}`
                      : ov.override_qty != null
                      ? `=${ov.override_qty.toLocaleString()}`
                      : "—"}
                  </td>
                  <td className="px-3 py-2">
                    {ov.estimated_impact_units != null
                      ? `+${ov.estimated_impact_units.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
                      : "—"}
                  </td>
                  <td className="px-3 py-2">{formatCurrency(ov.estimated_impact_value)}</td>
                  <td
                    className="px-3 py-2 text-muted-foreground max-w-[200px] truncate"
                    title={ov.override_reason}
                  >
                    {ov.override_reason}
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">{ov.created_by}</td>
                  <td className="px-3 py-2">
                    <span
                      className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${
                        ov.status === "pending_approval"
                          ? "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300"
                          : ov.status === "approved"
                          ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300"
                          : "bg-muted text-muted-foreground"
                      }`}
                    >
                      {ov.status}
                    </span>
                  </td>
                  <td className="px-3 py-2">
                    {ov.status === "pending_approval" && (
                      <div className="flex gap-1">
                        <button
                          onClick={() =>
                            approveMutation.mutate({
                              id: ov.override_id,
                              approvedBy: "manager",
                            })
                          }
                          disabled={approveMutation.isPending}
                          className="rounded bg-emerald-600 px-2 py-0.5 text-white text-xs hover:bg-emerald-700 disabled:opacity-50"
                        >
                          Approve
                        </button>
                        <button
                          onClick={() => {
                            setRejectModalId(ov.override_id);
                            setRejectReason("");
                          }}
                          className="rounded bg-red-600 px-2 py-0.5 text-white text-xs hover:bg-red-700"
                        >
                          Reject
                        </button>
                      </div>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Reject Modal */}
      {rejectModalId !== null && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-card rounded-lg border p-4 shadow-lg w-80">
            <h4 className="font-semibold text-sm mb-2">
              Reject Override #{rejectModalId}
            </h4>
            <textarea
              className="w-full rounded border bg-background p-2 text-xs h-20 resize-none"
              placeholder="Rejection reason (required)"
              value={rejectReason}
              onChange={(e) => setRejectReason(e.target.value)}
            />
            <div className="flex gap-2 mt-3 justify-end">
              <button
                onClick={() => setRejectModalId(null)}
                className="rounded border px-3 py-1 text-xs hover:bg-muted"
              >
                Cancel
              </button>
              <button
                onClick={() =>
                  rejectMutation.mutate({ id: rejectModalId!, reason: rejectReason })
                }
                disabled={!rejectReason.trim() || rejectMutation.isPending}
                className="rounded bg-red-600 px-3 py-1 text-xs text-white hover:bg-red-700 disabled:opacity-50"
              >
                Confirm Reject
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
