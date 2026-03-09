/**
 * F1.2 — Forward Inventory Projection Panel
 *
 * Shows day-by-day projected on-hand qty for 3 scenarios:
 *   - no_order (red)
 *   - with_open_po (green)
 *   - with_planned_orders (blue)
 * With safety stock reference line and key dates table.
 */

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ReferenceLine, ResponsiveContainer,
} from "recharts";
import { AlertTriangle, CheckCircle, RefreshCw } from "lucide-react";
import { projectionKeys, fetchProjection, fetchProjectionAtRisk, refreshProjection, fetchPlanningDate, queryKeys, STALE } from "@/api/queries";

const HORIZONS = [30, 60, 90];

export function ProjectionPanel() {
  const [itemNo, setItemNo] = useState("");
  const [loc, setLoc] = useState("");
  const [horizonDays, setHorizonDays] = useState(90);
  const [activeItem, setActiveItem] = useState<{ item_no: string; loc: string } | null>(null);
  const qc = useQueryClient();

  // Planning date
  const { data: planningDateInfo } = useQuery({
    queryKey: queryKeys.planningDate(),
    queryFn: fetchPlanningDate,
    staleTime: STALE.TEN_MIN,
  });

  // At-risk list
  const atRisk = useQuery({
    queryKey: projectionKeys.atRisk(30),
    queryFn: () => fetchProjectionAtRisk({ horizon_days: 30 }),
    staleTime: 5 * 60_000,
  });

  // DFU projection (only when user has submitted)
  const proj = useQuery({
    queryKey: projectionKeys.dfu({ item_no: activeItem?.item_no, loc: activeItem?.loc, horizon_days: horizonDays }),
    queryFn: () => fetchProjection({ item_no: activeItem!.item_no, loc: activeItem!.loc, horizon_days: horizonDays }),
    enabled: !!activeItem,
    staleTime: 5 * 60_000,
  });

  const refreshMut = useMutation({
    mutationFn: () => refreshProjection({ item_no: activeItem!.item_no, loc: activeItem!.loc, horizon_days: horizonDays }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["projection"] });
    },
  });

  const handleSearch = () => {
    if (itemNo.trim() && loc.trim()) {
      setActiveItem({ item_no: itemNo.trim(), loc: loc.trim() });
    }
  };

  const data = proj.data;

  const stockoutDateNoOrder = data?.key_dates?.no_order?.stockout_date;
  const stockoutDateWithPO = data?.key_dates?.with_open_po?.stockout_date;
  const daysNoOrder = data?.key_dates?.no_order?.days_until_stockout;

  return (
    <div className="space-y-4">
      {/* At-risk banner */}
      {atRisk.data && atRisk.data.total > 0 && (
        <div className="rounded-md border border-orange-300 bg-orange-50 dark:bg-orange-950/30 p-3 flex items-center gap-3">
          <AlertTriangle className="h-4 w-4 text-orange-500 shrink-0" />
          <span className="text-sm font-medium text-orange-800 dark:text-orange-300">
            {atRisk.data.total} DFU{atRisk.data.total !== 1 ? "s" : ""} at stockout risk within 30 days
          </span>
          <div className="ml-auto flex gap-2">
            {atRisk.data.items.slice(0, 3).map(item => (
              <button
                key={`${item.item_no}/${item.loc}`}
                onClick={() => {
                  setItemNo(item.item_no);
                  setLoc(item.loc);
                  setActiveItem({ item_no: item.item_no, loc: item.loc });
                }}
                className="text-xs px-2 py-0.5 rounded bg-orange-100 dark:bg-orange-900/50 hover:bg-orange-200 dark:hover:bg-orange-900 text-orange-800 dark:text-orange-200"
              >
                {item.item_no}/{item.loc} ({item.days_until_stockout}d)
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Item selector */}
      <div className="flex flex-wrap gap-2 items-end">
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-muted-foreground">Item No</label>
          <input
            className="h-8 rounded border px-2 text-sm bg-background w-36"
            placeholder="e.g. 100320"
            value={itemNo}
            onChange={e => setItemNo(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleSearch()}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-muted-foreground">Location</label>
          <input
            className="h-8 rounded border px-2 text-sm bg-background w-36"
            placeholder="e.g. 1401-BULK"
            value={loc}
            onChange={e => setLoc(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleSearch()}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-muted-foreground">Horizon</label>
          <select
            className="h-8 rounded border px-2 text-sm bg-background"
            value={horizonDays}
            onChange={e => setHorizonDays(Number(e.target.value))}
          >
            {HORIZONS.map(h => (
              <option key={h} value={h}>{h} days</option>
            ))}
          </select>
        </div>
        <button
          onClick={handleSearch}
          className="h-8 px-4 rounded bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90"
        >
          Project
        </button>
        {activeItem && (
          <button
            onClick={() => refreshMut.mutate()}
            disabled={refreshMut.isPending}
            className="h-8 px-3 rounded border text-sm flex items-center gap-1 hover:bg-muted"
          >
            <RefreshCw className={`h-3 w-3 ${refreshMut.isPending ? "animate-spin" : ""}`} />
            Refresh
          </button>
        )}
      </div>

      {proj.isError && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          No projection data found. Run <code className="font-mono">make projection-compute-dfu ITEM={itemNo} LOC={loc}</code> first,
          or click Refresh to compute now.
        </div>
      )}

      {data && (
        <>
          {/* Alert strip */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {stockoutDateNoOrder && (
              <div className="flex items-center gap-2 rounded-md border border-red-300 bg-red-50 dark:bg-red-950/30 p-3 text-sm">
                <AlertTriangle className="h-4 w-4 text-red-500 shrink-0" />
                <span className="text-red-800 dark:text-red-300">
                  <b>STOCKOUT RISK:</b> No-order scenario stocks out in{" "}
                  <b>{daysNoOrder} days</b> ({stockoutDateNoOrder})
                </span>
              </div>
            )}
            {!stockoutDateWithPO && stockoutDateNoOrder && (
              <div className="flex items-center gap-2 rounded-md border border-green-300 bg-green-50 dark:bg-green-950/30 p-3 text-sm">
                <CheckCircle className="h-4 w-4 text-green-500 shrink-0" />
                <span className="text-green-800 dark:text-green-300">
                  With confirmed PO(s): safe through projection horizon
                </span>
              </div>
            )}
          </div>

          {/* Meta chips */}
          <div className="flex gap-2 flex-wrap text-xs">
            <span className="rounded bg-muted px-2 py-1">
              On-hand: <b>{data.current_qty_on_hand.toLocaleString()} units</b>
            </span>
            <span className="rounded bg-muted px-2 py-1">
              Safety Stock: <b>{data.safety_stock.toLocaleString()} units</b>
            </span>
            <span className="rounded bg-muted px-2 py-1">
              Forecast: <b>
                {data.forecast_source === "production_forecast"
                  ? `Plan ${data.plan_version}`
                  : data.forecast_source === "champion_forecast"
                  ? "Champion Model"
                  : "Historical Avg"}
              </b>
            </span>
            {data.forecast_source === "champion_forecast" && (
              <span className="rounded bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 px-2 py-1">
                Using champion algorithm forecast as demand rate
              </span>
            )}
            {data.forecast_source === "fallback_avg" && (
              <span className="rounded bg-amber-100 dark:bg-amber-900/30 text-amber-800 dark:text-amber-300 px-2 py-1">
                ⚠ No forecast found — using 3-month historical average
              </span>
            )}
            {!data.open_po_data_available && (
              <span className="rounded bg-amber-100 dark:bg-amber-900/30 text-amber-800 dark:text-amber-300 px-2 py-1">
                ⚠ No open PO data — run 'make po-load'
              </span>
            )}
          </div>

          {/* Projection chart */}
          <div className="rounded-lg border bg-card p-4">
            <h4 className="font-medium text-sm mb-3">Inventory Projection — Next {horizonDays} Days</h4>
            <ResponsiveContainer width="100%" height={280}>
              <ComposedChart data={data.projection} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.4} />
                <XAxis
                  dataKey="projection_date"
                  tick={{ fontSize: 11 }}
                  tickFormatter={v => v?.slice(5)}  // MM-DD
                  interval={Math.floor(data.projection.length / 8)}
                />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip
                  formatter={(v: number, name: string) => [
                    typeof v === "number" ? v.toFixed(1) : v,
                    name,
                  ]}
                />
                <Legend />
                <ReferenceLine y={data.safety_stock} stroke="#f59e0b" strokeDasharray="6 3" label={{ value: "SS", position: "left", fontSize: 11 }} />
                {planningDateInfo?.is_frozen && (
                  <ReferenceLine
                    x={planningDateInfo.planning_date}
                    stroke="#f59e0b"
                    strokeWidth={1.5}
                    label={{ value: "Plan Date", position: "insideTopLeft", fontSize: 10, fill: "#f59e0b" }}
                  />
                )}
                {stockoutDateNoOrder && (
                  <ReferenceLine x={stockoutDateNoOrder} stroke="#ef4444" strokeDasharray="4 2" label={{ value: "Stockout", position: "top", fontSize: 10 }} />
                )}
                <Bar dataKey="receipts_expected" name="Receipts" fill="#86efac" opacity={0.7} yAxisId={0} />
                <Line dataKey="no_order_qty" name="No Order" stroke="#ef4444" dot={false} strokeWidth={2} />
                <Line dataKey="with_open_po_qty" name="With Open PO" stroke="#22c55e" dot={false} strokeWidth={2} />
                <Line dataKey="with_planned_orders_qty" name="With Planned" stroke="#3b82f6" dot={false} strokeWidth={2} strokeDasharray="5 3" />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          {/* Key dates table */}
          <div className="rounded-lg border bg-card overflow-x-auto">
            <table className="text-xs w-full">
              <thead className="bg-muted/40">
                <tr>
                  <th className="p-2 text-left font-medium">Key Date</th>
                  <th className="p-2 text-center font-medium">No Order</th>
                  <th className="p-2 text-center font-medium">With Open PO</th>
                  <th className="p-2 text-center font-medium">With Planned</th>
                </tr>
              </thead>
              <tbody>
                {(["reorder_trigger_date", "stockout_date", "days_until_stockout", "excess_date"] as const).map(key => (
                  <tr key={key} className="border-t">
                    <td className="p-2 text-muted-foreground capitalize">{key.replace(/_/g, " ")}</td>
                    {(["no_order", "with_open_po", "with_planned_orders"] as const).map(sce => {
                      const val = data.key_dates[sce]?.[key];
                      const isStockout = key === "stockout_date" && val;
                      return (
                        <td key={sce} className={`p-2 text-center ${isStockout ? "text-red-500 font-medium" : ""}`}>
                          {val ?? "—"}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Day-by-day table (sample rows) */}
          <div className="rounded-lg border bg-card overflow-x-auto">
            <div className="p-3 border-b">
              <h4 className="font-medium text-sm">Day-by-Day Detail (every 7th day)</h4>
            </div>
            <table className="text-xs w-full">
              <thead className="bg-muted/40">
                <tr>
                  <th className="p-2 text-left font-medium">Date</th>
                  <th className="p-2 text-right font-medium">No Order Qty</th>
                  <th className="p-2 text-right font-medium">With PO Qty</th>
                  <th className="p-2 text-right font-medium">Demand/Day</th>
                  <th className="p-2 text-right font-medium">Receipts</th>
                </tr>
              </thead>
              <tbody>
                {data.projection.filter((_, i) => i % 7 === 0).map(row => {
                  const noOrderStockout = row.no_order_stockout_risk;
                  return (
                    <tr key={row.projection_date} className={`border-t ${noOrderStockout ? "bg-red-50 dark:bg-red-950/20" : ""}`}>
                      <td className="p-2 font-mono">{row.projection_date}</td>
                      <td className={`p-2 text-right ${noOrderStockout ? "text-red-500 font-medium" : ""}`}>
                        {(row.no_order_qty ?? 0).toFixed(1)}
                        {noOrderStockout ? " ⚠" : ""}
                      </td>
                      <td className="p-2 text-right">{(row.with_open_po_qty ?? 0).toFixed(1)}</td>
                      <td className="p-2 text-right">{row.daily_demand_rate.toFixed(1)}</td>
                      <td className="p-2 text-right">{row.receipts_expected > 0 ? <span className="text-green-600 font-medium">+{row.receipts_expected.toFixed(0)}</span> : "0"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}

      {!activeItem && !atRisk.isLoading && (
        <div className="rounded-lg border border-dashed p-12 text-center text-muted-foreground text-sm">
          Enter an item number and location above to view forward inventory projection.
        </div>
      )}
    </div>
  );
}
