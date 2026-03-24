/**
 * F2.4 — Procurement Workflow & Order Release Panel
 *
 * Two-column layout: PO queue (left) + PO detail with timeline (right).
 * Supports the full DS purchase order lifecycle:
 *   proposed → planner_approved → buyer_released → po_sent → ...
 */

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchPurchaseOrders,
  approvePurchaseOrder,
  releasePurchaseOrder,
  exportPOsCSV,
  fetchPOTimeline,
  STALE,
} from "@/api/queries";
import type { PurchaseOrderLine } from "@/api/queries";
import { EmptyState } from "@/components/EmptyState";
import { ShoppingCart } from "lucide-react";

function fmt(v: number | null | undefined): string {
  if (v == null) return "—";
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(2)}M`;
  if (v >= 1_000) return `$${(v / 1_000).toFixed(1)}K`;
  return `$${v.toFixed(2)}`;
}

function fmtDate(s: string | null): string {
  if (!s) return "—";
  return new Date(s + "T00:00:00").toLocaleDateString("en-US", {
    month: "short", day: "numeric", year: "2-digit",
  });
}

const STATUS_COLOR: Record<string, string> = {
  proposed: "bg-muted text-muted-foreground",
  planner_approved: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300",
  buyer_released: "bg-sky-100 text-sky-700 dark:bg-sky-900/30 dark:text-sky-300",
  po_sent: "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300",
  supplier_confirmed: "border border-green-500 text-green-600",
  partially_received: "bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-300",
  fully_received: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300",
  closed: "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400",
  cancelled: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300",
};

const STATUS_FILTERS = [
  { label: "All", value: "" },
  { label: "Proposed", value: "proposed" },
  { label: "Approved", value: "planner_approved" },
  { label: "Released", value: "buyer_released" },
  { label: "Sent", value: "po_sent" },
];

export function ProcurementPanel() {
  const qc = useQueryClient();
  const [statusFilter, setStatusFilter] = useState("");
  const [selectedPO, setSelectedPO] = useState<string | null>(null);
  const [releaseModalPO, setReleaseModalPO] = useState<PurchaseOrderLine | null>(null);

  const ordersQ = useQuery({
    queryKey: ["purchase-orders", { status: statusFilter }],
    queryFn: () => fetchPurchaseOrders({ status: statusFilter || undefined }),
    staleTime: STALE.ONE_MIN,
  });

  const timelineQ = useQuery({
    queryKey: ["po-timeline", selectedPO],
    queryFn: () => fetchPOTimeline(selectedPO!),
    enabled: !!selectedPO,
    staleTime: STALE.ONE_MIN,
  });

  const approveMutation = useMutation({
    mutationFn: (po: PurchaseOrderLine) =>
      approvePurchaseOrder(po.po_number, { approved_by: "planner" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["purchase-orders"] }),
  });

  const releaseMutation = useMutation({
    mutationFn: (po: PurchaseOrderLine) =>
      releasePurchaseOrder(po.po_number, { released_by: "buyer" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["purchase-orders"] });
      setReleaseModalPO(null);
    },
  });

  const exportMutation = useMutation({
    mutationFn: (poNumbers: string[]) =>
      exportPOsCSV({ po_numbers: poNumbers, exported_by: "user" }),
    onSuccess: (data) => {
      // Trigger browser download
      const blob = new Blob([data.csv_content], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = data.filename;
      a.click();
      URL.revokeObjectURL(url);
    },
  });

  const orders = ordersQ.data?.orders ?? [];
  const selectedOrder = orders.find((o) => o.po_number === selectedPO) ?? null;
  const timeline = timelineQ.data?.timeline ?? [];

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h3 className="font-semibold text-base">Procurement Workflow</h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Purchase orders generated from replenishment exceptions. Approve, release, and export to ERP.
          </p>
        </div>
        {ordersQ.data && (
          <div className="text-right">
            <p className="text-xs text-muted-foreground">Total Open Value</p>
            <p className="text-lg font-semibold">{fmt(ordersQ.data.total_value)}</p>
          </div>
        )}
      </div>

      {/* Status filters */}
      <div className="flex gap-2 flex-wrap">
        {STATUS_FILTERS.map((f) => (
          <button
            key={f.value}
            onClick={() => setStatusFilter(f.value)}
            className={`rounded px-3 py-1 text-xs font-medium border ${
              statusFilter === f.value
                ? "bg-primary text-primary-foreground border-primary"
                : "bg-background hover:bg-muted"
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {/* Two-column layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 min-h-[400px]">
        {/* LEFT — Order Queue */}
        <div className="rounded-lg border overflow-auto">
          {ordersQ.isLoading ? (
            <div className="p-6 text-center text-xs text-muted-foreground">Loading...</div>
          ) : orders.length === 0 ? (
            <EmptyState
              icon={ShoppingCart}
              title="No purchase orders in workflow"
              description="The procurement workflow manages proposed → planner-approved → buyer-released purchase orders generated from planned replenishment. Each order goes through a two-step approval before being released to the supplier."
              steps={[
                { label: "Generate planned orders first", command: "make planned-orders-generate" },
                { label: "Approve planned orders to create POs", command: "(approve via Planned Orders panel)" },
              ]}
            />
          ) : (
            <div className="divide-y">
              {orders.map((po) => (
                <div
                  key={`${po.po_number}-${po.line_number}`}
                  onClick={() => setSelectedPO(po.po_number)}
                  className={`p-3 cursor-pointer hover:bg-muted/50 ${
                    selectedPO === po.po_number ? "bg-muted/70" : ""
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <p className="font-mono text-xs font-medium truncate">{po.po_number}</p>
                      <p className="text-xs text-muted-foreground truncate">
                        {po.supplier_name || po.supplier_id || "—"}
                      </p>
                    </div>
                    <span
                      className={`shrink-0 rounded-full px-2 py-0.5 text-xs font-medium ${
                        STATUS_COLOR[po.status] ?? "bg-muted text-muted-foreground"
                      }`}
                    >
                      {po.status.replace(/_/g, " ")}
                    </span>
                  </div>
                  <div className="mt-1.5 flex items-center justify-between">
                    <span className="text-xs text-muted-foreground">
                      {po.item_id} · {po.loc}
                    </span>
                    <span className="text-xs font-medium">{fmt(po.total_value)}</span>
                  </div>
                  {po.status === "proposed" && (
                    <div className="mt-2 flex gap-1">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          approveMutation.mutate(po);
                        }}
                        disabled={approveMutation.isPending}
                        className="rounded bg-blue-600 px-2 py-0.5 text-white text-xs hover:bg-blue-700 disabled:opacity-50"
                      >
                        Approve
                      </button>
                    </div>
                  )}
                  {po.status === "planner_approved" && (
                    <div className="mt-2">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setReleaseModalPO(po);
                        }}
                        className="rounded bg-sky-600 px-2 py-0.5 text-white text-xs hover:bg-sky-700"
                      >
                        Release
                      </button>
                    </div>
                  )}
                  {po.status === "buyer_released" && (
                    <div className="mt-2">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          exportMutation.mutate([po.po_number]);
                        }}
                        disabled={exportMutation.isPending}
                        className="rounded bg-amber-600 px-2 py-0.5 text-white text-xs hover:bg-amber-700 disabled:opacity-50"
                      >
                        Export CSV
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* RIGHT — Detail + Timeline */}
        {selectedOrder ? (
          <div className="rounded-lg border p-4 space-y-4 overflow-auto">
            <div>
              <p className="font-mono text-sm font-semibold">{selectedOrder.po_number}</p>
              <p className="text-xs text-muted-foreground">
                {selectedOrder.item_description || selectedOrder.item_id} · {selectedOrder.loc}
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3">
              {[
                { label: "Supplier", value: selectedOrder.supplier_name || selectedOrder.supplier_id || "—" },
                { label: "Ordered Qty", value: selectedOrder.ordered_qty != null ? `${selectedOrder.ordered_qty.toLocaleString()} ${selectedOrder.currency ?? ""}` : "—" },
                { label: "Unit Cost", value: selectedOrder.unit_cost != null ? `$${selectedOrder.unit_cost.toFixed(4)}` : "—" },
                { label: "Total Value", value: fmt(selectedOrder.total_value) },
                { label: "PO Date", value: fmtDate(selectedOrder.po_date) },
                { label: "Req. Delivery", value: fmtDate(selectedOrder.requested_delivery_date) },
              ].map((kv) => (
                <div key={kv.label}>
                  <p className="text-xs text-muted-foreground">{kv.label}</p>
                  <p className="text-sm font-medium">{kv.value}</p>
                </div>
              ))}
            </div>

            <div>
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                Timeline
              </p>
              {timelineQ.isLoading ? (
                <p className="text-xs text-muted-foreground">Loading timeline...</p>
              ) : timeline.length === 0 ? (
                <p className="text-xs text-muted-foreground">No events yet.</p>
              ) : (
                <ol className="space-y-2">
                  {timeline.map((ev, i) => (
                    <li key={i} className="flex gap-2">
                      <div className="mt-0.5 h-2 w-2 shrink-0 rounded-full bg-primary ring-2 ring-background ring-offset-1" />
                      <div>
                        <p className="text-xs font-medium capitalize">
                          {ev.action.replace(/_/g, " ")}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {ev.performed_by}
                          {ev.performed_at
                            ? ` · ${new Date(ev.performed_at).toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}`
                            : ""}
                        </p>
                        {ev.note && (
                          <p className="text-xs text-muted-foreground italic">{ev.note}</p>
                        )}
                      </div>
                    </li>
                  ))}
                </ol>
              )}
            </div>
          </div>
        ) : (
          <div className="rounded-lg border bg-muted/20 flex items-center justify-center">
            <p className="text-sm text-muted-foreground">Select an order to view details</p>
          </div>
        )}
      </div>

      {/* Release Confirmation Modal */}
      {releaseModalPO && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-card rounded-lg border p-6 shadow-lg w-96 space-y-4">
            <h4 className="font-semibold text-sm">
              Release Purchase Order {releaseModalPO.po_number}
            </h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Supplier</span>
                <span>{releaseModalPO.supplier_name || releaseModalPO.supplier_id || "—"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Item</span>
                <span className="font-mono text-xs">{releaseModalPO.item_id}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Ordered Qty</span>
                <span>{releaseModalPO.ordered_qty?.toLocaleString() ?? "—"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Total Value</span>
                <span className="font-semibold">{fmt(releaseModalPO.total_value)}</span>
              </div>
            </div>
            <p className="text-xs text-muted-foreground">
              Once released, the buyer will be notified to review and send to ERP.
            </p>
            <div className="flex gap-2 justify-end">
              <button
                onClick={() => setReleaseModalPO(null)}
                className="rounded border px-3 py-1 text-xs hover:bg-muted"
              >
                Cancel
              </button>
              <button
                onClick={() => releaseMutation.mutate(releaseModalPO)}
                disabled={releaseMutation.isPending}
                className="rounded bg-sky-600 px-3 py-1 text-xs text-white hover:bg-sky-700 disabled:opacity-50"
              >
                Confirm Release
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
