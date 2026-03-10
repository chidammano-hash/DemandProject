/**
 * F1.3 — Open Purchase Orders panel
 *
 * Displays portfolio-level PO KPIs, filterable PO line table, and past-due alerts.
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, CheckCircle, Package, Clock, FileText } from "lucide-react";
import { fetchOpenPOs, fetchOpenPOSummary, fetchPastDuePOs } from "@/api/queries/supply";
import { queryKeys } from "@/api/queries/core";
import { EmptyState } from "@/components/EmptyState";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmtCurrency(v: number | null): string {
  if (v == null) return "—";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    notation: v >= 1_000_000 ? "compact" : "standard",
    maximumFractionDigits: 0,
  }).format(v);
}

function fmtNum(v: number | null | undefined): string {
  if (v == null) return "—";
  return new Intl.NumberFormat("en-US").format(v);
}

function deliveryColor(days_past_due: number): string {
  if (days_past_due === 0) return "text-green-600";
  if (days_past_due <= 7) return "text-yellow-600";
  return "text-red-600";
}

// ---------------------------------------------------------------------------
// KPI Card
// ---------------------------------------------------------------------------

function KpiCard({
  label,
  value,
  sub,
  warning,
}: {
  label: string;
  value: string;
  sub?: string;
  warning?: boolean;
}) {
  return (
    <div
      className={`rounded-lg border p-4 flex flex-col gap-1 ${
        warning ? "border-red-300 bg-red-50 dark:bg-red-900/10" : "bg-card"
      }`}
    >
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className={`text-xl font-bold ${warning ? "text-red-600" : "text-foreground"}`}>
        {value}
      </span>
      {sub && <span className="text-xs text-muted-foreground">{sub}</span>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

export function OpenPOPanel() {
  const [itemFilter, setItemFilter] = useState("");
  const [locFilter, setLocFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState("open,partially_received");
  const [pastDueOnly, setPastDueOnly] = useState(false);
  const [page, setPage] = useState(1);

  const PAGE_SIZE = 50;

  const summaryQuery = useQuery({
    queryKey: queryKeys.openPOSummary(),
    queryFn: fetchOpenPOSummary,
    staleTime: 60_000,
  });

  const posQuery = useQuery({
    queryKey: queryKeys.openPOs({ itemFilter, locFilter, statusFilter, pastDueOnly, page }),
    queryFn: () =>
      fetchOpenPOs({
        item_no: itemFilter || undefined,
        loc: locFilter || undefined,
        status: statusFilter,
        past_due_only: pastDueOnly,
        page,
        page_size: PAGE_SIZE,
      }),
    staleTime: 60_000,
  });

  const pastDueQuery = useQuery({
    queryKey: queryKeys.pastDuePOs(),
    queryFn: () => fetchPastDuePOs({ min_days_past_due: 7, page_size: 5 }),
    staleTime: 60_000,
  });

  const summary = summaryQuery.data;
  const pos = posQuery.data;

  return (
    <div className="space-y-6">
      {/* KPI Row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <KpiCard
          label="Total Open Value"
          value={fmtCurrency(summary?.total_open_value_usd ?? null)}
          sub={`${fmtNum(summary?.total_open_lines)} lines`}
        />
        <KpiCard
          label="Open Qty"
          value={fmtNum(summary?.total_open_qty_by_status?.open ?? null)}
          sub="fully open"
        />
        <KpiCard
          label="Past-Due Lines"
          value={fmtNum(summary?.past_due_lines ?? null)}
          sub={fmtCurrency(summary?.past_due_value_usd ?? null)}
          warning={(summary?.past_due_lines ?? 0) > 0}
        />
        <KpiCard
          label="Active Suppliers"
          value={fmtNum(summary?.suppliers_with_open_pos ?? null)}
          sub={
            summary?.last_loaded_at
              ? `Loaded ${new Date(summary.last_loaded_at).toLocaleDateString()}`
              : "No data loaded"
          }
        />
      </div>

      {/* PO Status Legend */}
      <div className="text-xs text-muted-foreground p-2 rounded bg-muted/30 border mb-3">
        <span className="font-medium text-foreground">PO Status: </span>
        <span className="text-blue-600">● Open</span> · <span className="text-amber-600">● In Transit</span> · <span className="text-yellow-600">● Partial</span> · <span className="text-green-600">● Received</span>
      </div>

      {/* Delivery Status Legend */}
      <div className="text-xs text-muted-foreground p-2 rounded bg-muted/30 border mb-3">
        <span className="font-medium text-foreground">Delivery Status: </span>
        <span className="text-green-600">● On Time (0d)</span> · <span className="text-amber-600">● At Risk (1–7d late, monitor)</span> · <span className="text-red-600">● Late (&gt;7d, escalate)</span>
      </div>

      {/* Past-due alert strip */}
      {(pastDueQuery.data?.total ?? 0) > 0 && (
        <div className="rounded-lg border border-yellow-300 bg-yellow-50 dark:bg-yellow-900/10 p-3">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className="h-4 w-4 text-yellow-600" />
            <span className="text-sm font-medium text-yellow-800 dark:text-yellow-200">
              {pastDueQuery.data!.total} past-due PO lines
            </span>
          </div>
          <div className="space-y-1">
            {pastDueQuery.data!.items.map((pd) => (
              <div key={`${pd.po_number}`} className="text-xs text-yellow-700 dark:text-yellow-300 flex gap-3">
                <span className="font-mono">{pd.po_number}</span>
                <span>{pd.item_no} @ {pd.loc}</span>
                <span className="text-red-600 font-medium">{pd.days_past_due}d overdue</span>
                <span>{pd.supplier_name}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-wrap gap-3 items-end">
        <div className="flex flex-col gap-1">
          <label className="text-xs text-muted-foreground">Item No</label>
          <input
            className="border rounded px-2 py-1 text-sm w-32 bg-background"
            value={itemFilter}
            onChange={(e) => { setItemFilter(e.target.value); setPage(1); }}
            placeholder="Filter..."
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-muted-foreground">Location</label>
          <input
            className="border rounded px-2 py-1 text-sm w-32 bg-background"
            value={locFilter}
            onChange={(e) => { setLocFilter(e.target.value); setPage(1); }}
            placeholder="Filter..."
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-muted-foreground">Status</label>
          <select
            className="border rounded px-2 py-1 text-sm bg-background"
            value={statusFilter}
            onChange={(e) => { setStatusFilter(e.target.value); setPage(1); }}
          >
            <option value="open,partially_received">Open + Partial</option>
            <option value="open">Open only</option>
            <option value="partially_received">Partial only</option>
          </select>
        </div>
        <div className="flex items-center gap-2 pt-4">
          <input
            type="checkbox"
            id="pastDueOnly"
            checked={pastDueOnly}
            onChange={(e) => { setPastDueOnly(e.target.checked); setPage(1); }}
          />
          <label htmlFor="pastDueOnly" className="text-sm cursor-pointer">Past-Due Only</label>
        </div>
      </div>

      {/* PO Table */}
      {!posQuery.isLoading && pos?.items.length === 0 ? (
        <EmptyState
          icon={FileText}
          title="No open purchase orders loaded"
          description="Open POs track confirmed in-flight supplier orders with their promised delivery dates. This data feeds the inventory projection scenarios (with-PO scenario) and planned order confidence scores."
          steps={[
            { label: "Load open PO data from inventory files", command: "make load-inventory" },
            { label: "Or manually upload a PO CSV", command: "make load-open-pos" },
          ]}
        />
      ) : (
      <div className="rounded-lg border overflow-auto">
        <table className="w-full text-xs">
          <thead className="bg-muted/40">
            <tr>
              {["PO #", "Item", "Loc", "Supplier", "Open Qty", "Value", "Delivery", "Status"].map((h) => (
                <th key={h} className="px-3 py-2 text-left font-medium text-muted-foreground whitespace-nowrap">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {posQuery.isLoading && (
              <tr>
                <td colSpan={8} className="px-3 py-6 text-center text-muted-foreground">
                  Loading...
                </td>
              </tr>
            )}
            {pos?.items.map((po) => (
              <tr key={`${po.po_number}-${po.po_line_number}`} className="border-t hover:bg-muted/20">
                <td className="px-3 py-2 font-mono">{po.po_number}-{po.po_line_number}</td>
                <td className="px-3 py-2">{po.item_no}</td>
                <td className="px-3 py-2">{po.loc}</td>
                <td className="px-3 py-2 max-w-[140px] truncate">{po.supplier_name ?? po.supplier_id ?? "—"}</td>
                <td className="px-3 py-2 text-right">{fmtNum(po.open_qty)}</td>
                <td className="px-3 py-2 text-right">{fmtCurrency(po.line_value)}</td>
                <td className={`px-3 py-2 whitespace-nowrap ${deliveryColor(po.days_past_due)}`}>
                  {po.effective_delivery_date ?? "—"}
                  {po.days_past_due > 0 && (
                    <span className="ml-1 text-red-500 font-medium">
                      ⚠ {po.days_past_due}d
                    </span>
                  )}
                </td>
                <td className="px-3 py-2 capitalize">{po.line_status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      )}

      {/* Pagination */}
      {pos && pos.total > PAGE_SIZE && (
        <div className="flex items-center gap-2 text-sm">
          <button
            className="px-3 py-1 border rounded disabled:opacity-40"
            disabled={page <= 1}
            onClick={() => setPage((p) => p - 1)}
          >
            Previous
          </button>
          <span className="text-muted-foreground">
            Page {page} of {Math.ceil(pos.total / PAGE_SIZE)} ({fmtNum(pos.total)} rows)
          </span>
          <button
            className="px-3 py-1 border rounded disabled:opacity-40"
            disabled={page * PAGE_SIZE >= pos.total}
            onClick={() => setPage((p) => p + 1)}
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
