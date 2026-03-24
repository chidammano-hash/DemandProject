import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  fetchPORows,
  fetchPOSummary,
  fetchPOAging,
  STALE,
} from "@/api/queries";
import { KpiCard } from "@/components/KpiCard";
import { EmptyState } from "@/components/EmptyState";
import { FileText } from "lucide-react";

const PANEL_KPI = "rounded-lg bg-muted/30 p-3";

function formatCurrency(v: number | null): string {
  if (v == null) return "-";
  return "$" + v.toLocaleString(undefined, { maximumFractionDigits: 0 });
}

export function PurchaseOrdersPanel() {
  const [poNumber, setPoNumber] = useState("");
  const [item, setItem] = useState("");
  const [loc, setLoc] = useState("");
  const [supplier, setSupplier] = useState("");
  const [status, setStatus] = useState("");
  const [page, setPage] = useState(0);
  const pageSize = 50;

  const { data: summary } = useQuery({
    queryKey: ["purchase-orders", "summary"],
    queryFn: fetchPOSummary,
    staleTime: STALE.FIVE_MIN,
  });

  const { data: aging } = useQuery({
    queryKey: ["purchase-orders", "aging"],
    queryFn: fetchPOAging,
    staleTime: STALE.FIVE_MIN,
  });

  const { data, isLoading } = useQuery({
    queryKey: ["purchase-orders", "rows", poNumber, item, loc, supplier, status, page],
    queryFn: () =>
      fetchPORows({
        po_number: poNumber || undefined,
        item: item || undefined,
        loc: loc || undefined,
        supplier: supplier || undefined,
        status: status || undefined,
        limit: pageSize,
        offset: page * pageSize,
      }),
    staleTime: STALE.ONE_MIN,
  });

  return (
    <div className="space-y-4">
      {/* KPI summary */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <KpiCard className={PANEL_KPI} label="Total PO Lines" value={(summary?.total_lines ?? 0).toLocaleString()} />
        <KpiCard className={PANEL_KPI} label="Open Lines" value={(summary?.open_lines ?? 0).toLocaleString()} colorClass="text-blue-600" />
        <KpiCard className={PANEL_KPI} label="Closed Lines" value={(summary?.closed_lines ?? 0).toLocaleString()} />
        <KpiCard className={PANEL_KPI} label="Open Value" value={formatCurrency(summary?.open_value ?? 0)} colorClass="text-blue-600" />
        <KpiCard className={PANEL_KPI} label="Total Value" value={formatCurrency(summary?.total_value ?? 0)} />
      </div>

      {/* Aging buckets */}
      {aging && aging.buckets.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {aging.buckets.map((b) => (
            <div key={b.age_bucket} className="rounded-lg border p-3 text-center">
              <div className="text-xs text-muted-foreground">{b.age_bucket} days</div>
              <div className="text-lg font-semibold">{b.line_count.toLocaleString()}</div>
              <div className="text-xs text-muted-foreground">{formatCurrency(b.total_value)}</div>
            </div>
          ))}
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-wrap gap-2">
        <input
          className="border rounded px-2 py-1 text-sm w-32"
          placeholder="PO #..."
          value={poNumber}
          onChange={(e) => { setPoNumber(e.target.value); setPage(0); }}
        />
        <input
          className="border rounded px-2 py-1 text-sm w-32"
          placeholder="Item..."
          value={item}
          onChange={(e) => { setItem(e.target.value); setPage(0); }}
        />
        <input
          className="border rounded px-2 py-1 text-sm w-32"
          placeholder="Location..."
          value={loc}
          onChange={(e) => { setLoc(e.target.value); setPage(0); }}
        />
        <input
          className="border rounded px-2 py-1 text-sm w-32"
          placeholder="Supplier..."
          value={supplier}
          onChange={(e) => { setSupplier(e.target.value); setPage(0); }}
        />
        <select
          className="border rounded px-2 py-1 text-sm"
          value={status}
          onChange={(e) => { setStatus(e.target.value); setPage(0); }}
        >
          <option value="">All Status</option>
          <option value="open">Open</option>
          <option value="closed">Closed</option>
        </select>
      </div>

      {/* Table */}
      {!isLoading && (!data || data.rows.length === 0) ? (
        <EmptyState
          icon={FileText}
          title="No purchase order data"
          description="Purchase orders track the full PO lifecycle including open and closed orders, delivery dates, and costs."
          steps={[
            { label: "Normalize PO data", command: "make normalize-purchase-order" },
            { label: "Load via data pipeline", command: "make load-purchase-order" },
          ]}
        />
      ) : (
        <>
          <div className="text-sm text-muted-foreground">
            {(data?.total ?? 0).toLocaleString()} rows
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left text-muted-foreground">
                  <th className="py-2 px-2">PO #</th>
                  <th className="py-2 px-2">Item</th>
                  <th className="py-2 px-2">Loc</th>
                  <th className="py-2 px-2">Supplier</th>
                  <th className="py-2 px-2">Status</th>
                  <th className="py-2 px-2 text-right">Qty</th>
                  <th className="py-2 px-2 text-right">Value</th>
                  <th className="py-2 px-2">Delivery</th>
                  <th className="py-2 px-2 text-right">LT (days)</th>
                </tr>
              </thead>
              <tbody>
                {data?.rows.map((r) => (
                  <tr key={r.po_ck} className="border-b hover:bg-muted/20">
                    <td className="py-1.5 px-2 font-mono">{r.po_number}</td>
                    <td className="py-1.5 px-2 font-mono">{r.item_id}</td>
                    <td className="py-1.5 px-2">{r.loc}</td>
                    <td className="py-1.5 px-2 truncate max-w-[150px]">{r.supplier_name ?? r.supplier_id}</td>
                    <td className="py-1.5 px-2">
                      <span className={`px-1.5 py-0.5 rounded text-xs ${r.is_closed ? "bg-gray-100" : "bg-blue-100 text-blue-700"}`}>
                        {r.is_closed ? "Closed" : "Open"}
                      </span>
                    </td>
                    <td className="py-1.5 px-2 text-right">{r.ordered_qty?.toLocaleString() ?? "-"}</td>
                    <td className="py-1.5 px-2 text-right">{formatCurrency(r.gross_value)}</td>
                    <td className="py-1.5 px-2">{r.delivery_date ?? "-"}</td>
                    <td className="py-1.5 px-2 text-right">{r.lead_time_actual ?? "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex gap-2 justify-end">
            <button
              className="px-3 py-1 text-sm border rounded disabled:opacity-40"
              disabled={page === 0}
              onClick={() => setPage((p) => p - 1)}
            >
              Prev
            </button>
            <button
              className="px-3 py-1 text-sm border rounded disabled:opacity-40"
              disabled={(data?.rows.length ?? 0) < pageSize}
              onClick={() => setPage((p) => p + 1)}
            >
              Next
            </button>
          </div>
        </>
      )}
    </div>
  );
}
