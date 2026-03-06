import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import {
  queryKeys,
  fetchEoqSummary,
  fetchEoqDetail,
  fetchEoqSensitivity,
  STALE,
  type EoqDetailRow,
} from "@/api/queries";

const PAGE = 50;

function fmt(n: number | null | undefined, decimals = 1): string {
  if (n == null) return "—";
  return Number(n).toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function fmtInt(n: number | null | undefined): string {
  if (n == null) return "—";
  return Math.round(Number(n)).toLocaleString();
}

export function EoqPanel() {
  const [abcFilter, setAbcFilter] = useState("");
  const [itemFilter, setItemFilter] = useState("");
  const [locFilter, setLocFilter] = useState("");
  const [eoqOffset, setEoqOffset] = useState(0);
  const [sensitivityItem, setSensitivityItem] = useState("");
  const [sensitivityLoc, setSensitivityLoc] = useState("");

  const { data: eoqSummary, isLoading: summaryLoading } = useQuery({
    queryKey: queryKeys.eoqSummary({ abc_vol: abcFilter }),
    queryFn: () => fetchEoqSummary({ abc_vol: abcFilter || undefined }),
    staleTime: STALE.FIVE_MIN,
  });

  const { data: eoqDetail, isLoading: detailLoading } = useQuery({
    queryKey: queryKeys.eoqDetail({
      item: itemFilter,
      loc: locFilter,
      abc_vol: abcFilter,
      offset: eoqOffset,
    }),
    queryFn: () =>
      fetchEoqDetail({
        item: itemFilter || undefined,
        loc: locFilter || undefined,
        abc_vol: abcFilter || undefined,
        limit: PAGE,
        offset: eoqOffset,
      }),
    staleTime: STALE.FIVE_MIN,
  });

  const { data: sensitivity, isLoading: sensLoading } = useQuery({
    queryKey: queryKeys.eoqSensitivity({
      item: sensitivityItem,
      loc: sensitivityLoc,
    }),
    queryFn: () =>
      fetchEoqSensitivity({
        item: sensitivityItem || undefined,
        loc: sensitivityLoc || undefined,
      }),
    staleTime: STALE.TEN_MIN,
  });

  const totalPages = eoqDetail ? Math.ceil(eoqDetail.total / PAGE) : 0;
  const currentPage = Math.floor(eoqOffset / PAGE) + 1;

  return (
    <div className="flex flex-col gap-6">
      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="rounded-lg border bg-card p-4">
          <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide">
            Total Cycle Stock
          </p>
          <p className="text-2xl font-bold mt-1 text-foreground">
            {summaryLoading ? "…" : fmtInt(eoqSummary?.total_cycle_stock)}
          </p>
          <p className="text-xs text-muted-foreground mt-1">units (avg EOQ/2)</p>
        </div>
        <div className="rounded-lg border bg-card p-4">
          <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide">
            Avg EOQ Size
          </p>
          <p className="text-2xl font-bold mt-1 text-foreground">
            {summaryLoading ? "…" : fmt(eoqSummary?.avg_effective_eoq, 0)}
          </p>
          <p className="text-xs text-muted-foreground mt-1">units per order</p>
        </div>
        <div className="rounded-lg border bg-card p-4">
          <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide">
            Avg Order Frequency
          </p>
          <p className="text-2xl font-bold mt-1 text-foreground">
            {summaryLoading ? "…" : fmt(eoqSummary?.avg_order_frequency, 1)}
          </p>
          <p className="text-xs text-muted-foreground mt-1">orders / year</p>
        </div>
        <div className="rounded-lg border bg-card p-4">
          <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide">
            Total Annual Cost
          </p>
          <p className="text-2xl font-bold mt-1 text-foreground">
            {summaryLoading
              ? "…"
              : eoqSummary?.total_annual_cost != null
              ? `$${fmtInt(eoqSummary.total_annual_cost)}`
              : "—"}
          </p>
          <p className="text-xs text-muted-foreground mt-1">holding + ordering</p>
        </div>
      </div>

      {/* By-ABC breakdown */}
      {eoqSummary && eoqSummary.by_abc.length > 0 && (
        <div className="rounded-lg border bg-card p-4">
          <h3 className="text-sm font-semibold text-foreground mb-3">By ABC Class</h3>
          <div className="overflow-x-auto">
            <table className="text-xs w-full">
              <thead>
                <tr className="border-b text-muted-foreground">
                  <th className="text-left py-1 pr-4">Class</th>
                  <th className="text-right py-1 pr-4">DFUs</th>
                  <th className="text-right py-1 pr-4">Avg EOQ</th>
                  <th className="text-right py-1 pr-4">Cycle Stock</th>
                  <th className="text-right py-1">Annual Cost</th>
                </tr>
              </thead>
              <tbody>
                {eoqSummary.by_abc.map((row) => (
                  <tr key={row.abc_vol} className="border-b last:border-0">
                    <td className="py-1 pr-4 font-medium">{row.abc_vol}</td>
                    <td className="text-right py-1 pr-4">{row.count.toLocaleString()}</td>
                    <td className="text-right py-1 pr-4">{fmt(row.avg_eoq, 0)}</td>
                    <td className="text-right py-1 pr-4">{fmtInt(row.total_cycle_stock)}</td>
                    <td className="text-right py-1">
                      {row.total_annual_cost != null
                        ? `$${fmtInt(row.total_annual_cost)}`
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Sensitivity Chart */}
      <div className="rounded-lg border bg-card p-4">
        <h3 className="text-sm font-semibold text-foreground mb-3">EOQ Sensitivity</h3>
        <p className="text-xs text-muted-foreground mb-3">
          How EOQ and total annual cost change as ordering cost varies.
          Optionally enter an item + location to use that DFU's demand.
        </p>
        <div className="flex gap-2 mb-4">
          <input
            className="h-8 rounded border border-input bg-background px-2 text-xs w-32"
            placeholder="Item No (optional)"
            value={sensitivityItem}
            onChange={(e) => setSensitivityItem(e.target.value)}
          />
          <input
            className="h-8 rounded border border-input bg-background px-2 text-xs w-32"
            placeholder="Location (optional)"
            value={sensitivityLoc}
            onChange={(e) => setSensitivityLoc(e.target.value)}
          />
        </div>
        {sensLoading ? (
          <div className="text-xs text-muted-foreground">Loading…</div>
        ) : sensitivity && sensitivity.curve.length > 0 ? (
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={sensitivity.curve} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis
                  dataKey="ordering_cost"
                  tickFormatter={(v) => `$${v}`}
                  tick={{ fontSize: 10 }}
                />
                <YAxis yAxisId="left" tick={{ fontSize: 10 }} />
                <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 10 }} />
                <Tooltip
                  formatter={(value: number, name: string) => [
                    name === "Total Cost" ? `$${Number(value).toFixed(2)}` : Number(value).toFixed(1),
                    name,
                  ]}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="effective_eoq"
                  name="Effective EOQ"
                  stroke="hsl(220, 70%, 55%)"
                  dot={false}
                  strokeWidth={2}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="total_annual_cost"
                  name="Total Cost"
                  stroke="hsl(10, 70%, 55%)"
                  dot={false}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="text-xs text-muted-foreground">No data available.</div>
        )}
      </div>

      {/* Detail Table */}
      <div className="rounded-lg border bg-card p-4">
        <h3 className="text-sm font-semibold text-foreground mb-3">EOQ Detail</h3>

        {/* Filters */}
        <div className="flex flex-wrap gap-2 mb-3">
          <input
            className="h-8 rounded border border-input bg-background px-2 text-xs w-36"
            placeholder="Filter by item…"
            value={itemFilter}
            onChange={(e) => { setItemFilter(e.target.value); setEoqOffset(0); }}
          />
          <input
            className="h-8 rounded border border-input bg-background px-2 text-xs w-36"
            placeholder="Filter by location…"
            value={locFilter}
            onChange={(e) => { setLocFilter(e.target.value); setEoqOffset(0); }}
          />
          <select
            className="h-8 rounded border border-input bg-background px-2 text-xs"
            value={abcFilter}
            onChange={(e) => { setAbcFilter(e.target.value); setEoqOffset(0); }}
          >
            <option value="">All ABC classes</option>
            <option value="A">A</option>
            <option value="B">B</option>
            <option value="C">C</option>
          </select>
        </div>

        {detailLoading ? (
          <div className="text-xs text-muted-foreground">Loading…</div>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="text-xs w-full">
                <thead>
                  <tr className="border-b text-muted-foreground text-right">
                    <th className="text-left py-1 pr-3">Item</th>
                    <th className="text-left py-1 pr-3">Loc</th>
                    <th className="py-1 pr-3">ABC</th>
                    <th className="py-1 pr-3">EOQ</th>
                    <th className="py-1 pr-3">Eff. EOQ</th>
                    <th className="py-1 pr-3">Cycle Stock</th>
                    <th className="py-1 pr-3">Orders/Yr</th>
                    <th className="py-1">Annual Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {(eoqDetail?.rows ?? []).map((r: EoqDetailRow, i) => (
                    <tr key={`${r.item_no}-${r.loc}-${i}`} className="border-b last:border-0 hover:bg-muted/40">
                      <td className="py-1 pr-3 font-mono text-xs">{r.item_no}</td>
                      <td className="py-1 pr-3 text-muted-foreground">{r.loc}</td>
                      <td className="py-1 pr-3 text-center">{r.abc_vol ?? "—"}</td>
                      <td className="py-1 pr-3 text-right">{fmt(r.eoq, 1)}</td>
                      <td className="py-1 pr-3 text-right font-medium">{fmt(r.effective_eoq, 1)}</td>
                      <td className="py-1 pr-3 text-right">{fmt(r.eoq_cycle_stock, 1)}</td>
                      <td className="py-1 pr-3 text-right">{fmt(r.order_frequency, 2)}</td>
                      <td className="py-1 text-right">
                        {r.total_annual_cost != null ? `$${fmt(r.total_annual_cost, 2)}` : "—"}
                      </td>
                    </tr>
                  ))}
                  {(eoqDetail?.rows ?? []).length === 0 && (
                    <tr>
                      <td colSpan={8} className="py-4 text-center text-muted-foreground">
                        No data.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center gap-3 mt-3 text-xs text-muted-foreground">
                <button
                  className="rounded border px-2 py-1 hover:bg-muted disabled:opacity-40"
                  disabled={currentPage <= 1}
                  onClick={() => setEoqOffset((p) => Math.max(0, p - PAGE))}
                >
                  Prev
                </button>
                <span>
                  Page {currentPage} / {totalPages} ({eoqDetail?.total.toLocaleString()} DFUs)
                </span>
                <button
                  className="rounded border px-2 py-1 hover:bg-muted disabled:opacity-40"
                  disabled={currentPage >= totalPages}
                  onClick={() => setEoqOffset((p) => p + PAGE)}
                >
                  Next
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
