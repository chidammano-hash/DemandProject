import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  LineChart,
  Line,
  ResponsiveContainer,
} from "recharts";
import { RefreshCw } from "lucide-react";
import { DataFreshnessBanner } from "@/components/DataFreshnessBanner";
import { RecommendedActionCard } from "@/components/RecommendedActionCard";
import { KpiCard } from "@/components/KpiCard";
import { EmptyState } from "@/components/EmptyState";
import { TableSkeleton } from "@/components/Skeleton";
import { useChartColors } from "@/hooks/useChartColors";
import { formatNumber, formatCompactNumber } from "@/lib/formatters";
import {
  replenishmentKeys,
  fetchReplenishmentSummary,
  fetchReplenishmentDetail,
  fetchReplenishmentComparison,
  fetchReplenishmentSku,
  type ReplenishmentDetailRow,
} from "@/api/queries";

const PAGE_SIZE = 50;
const PANEL_KPI = "rounded-lg bg-muted/30 p-3";

export function ReplenishmentPlanPanel() {
  const [planVersion, setPlanVersion] = useState("");
  const [policyType, setPolicyType] = useState("");
  const [abcVol, setAbcVol] = useState("");
  const [page, setPage] = useState(0);
  const [selectedSku, setSelectedSku] = useState<{ item_id: string; loc: string } | null>(null);

  const { chartColors } = useChartColors();

  const summaryQ = useQuery({
    queryKey: replenishmentKeys.summary(
      planVersion || undefined,
      policyType || undefined,
      abcVol || undefined
    ),
    queryFn: () =>
      fetchReplenishmentSummary({
        plan_version: planVersion || undefined,
        policy_type: policyType || undefined,
        abc_vol: abcVol || undefined,
      }),
    staleTime: 5 * 60 * 1000,
  });

  const detailQ = useQuery({
    queryKey: replenishmentKeys.detail({ planVersion, policyType, abcVol, page }),
    queryFn: () =>
      fetchReplenishmentDetail({
        plan_version: planVersion || undefined,
        policy_type: policyType || undefined,
        abc_vol: abcVol || undefined,
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
      }),
    staleTime: 5 * 60 * 1000,
  });

  const comparisonQ = useQuery({
    queryKey: replenishmentKeys.comparison(
      planVersion || undefined,
      abcVol || undefined,
      policyType || undefined
    ),
    queryFn: () =>
      fetchReplenishmentComparison({
        plan_version: planVersion || undefined,
        abc_vol: abcVol || undefined,
        policy_type: policyType || undefined,
      }),
    staleTime: 10 * 60 * 1000,
  });

  const skuQ = useQuery({
    queryKey: replenishmentKeys.sku(
      selectedSku?.item_id ?? "",
      selectedSku?.loc ?? "",
      planVersion || undefined
    ),
    queryFn: () =>
      fetchReplenishmentSku({
        item_id: selectedSku!.item_id,
        loc: selectedSku!.loc,
        plan_version: planVersion || undefined,
      }),
    enabled: selectedSku != null,
    staleTime: 5 * 60 * 1000,
  });

  const summary = summaryQ.data;
  const detail = detailQ.data;
  const comparison = comparisonQ.data;
  const sku = skuQ.data;

  const totalPages = detail ? Math.ceil(detail.total / PAGE_SIZE) : 0;

  return (
    <div className="space-y-4">
      <DataFreshnessBanner
        lastRefreshed={summary?.computed_at}
        source="Replenishment Plan"
        staleSec={86400}
      />

      {/* Info banner */}
      <div className="text-xs text-muted-foreground bg-muted/20 border rounded px-3 py-2 mb-3">
        Replenishment plan: Safety buffer, optimal order size, and reorder trigger targets for the current plan version. <strong className="text-foreground">At Risk = YES</strong> means current inventory is below the safety buffer — prioritize ordering.
      </div>

      {/* Filter controls */}
      <div className="flex flex-wrap gap-3">
        <label className="flex flex-col gap-1">
          <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Plan Version
          </span>
          <input
            className="h-8 w-32 rounded-md border border-input bg-background px-2 text-sm"
            value={planVersion}
            onChange={(e) => setPlanVersion(e.target.value)}
            placeholder="latest"
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Policy Type
          </span>
          <select
            className="h-8 w-44 rounded-md border border-input bg-background px-2 text-sm"
            value={policyType}
            onChange={(e) => {
              setPolicyType(e.target.value);
              setPage(0);
            }}
          >
            <option value="">All</option>
            <option value="continuous_rop">Continuous ROP</option>
            <option value="periodic_review">Periodic Review</option>
            <option value="min_max">Min/Max</option>
            <option value="manual">Manual</option>
          </select>
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            ABC Class
          </span>
          <select
            className="h-8 w-24 rounded-md border border-input bg-background px-2 text-sm"
            value={abcVol}
            onChange={(e) => {
              setAbcVol(e.target.value);
              setPage(0);
            }}
          >
            <option value="">All</option>
            <option value="A">A</option>
            <option value="B">B</option>
            <option value="C">C</option>
          </select>
        </label>
      </div>

      {/* KPI cards */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <KpiCard
          className={PANEL_KPI}
          label="Total SKUs"
          value={
            summaryQ.isLoading
              ? "..."
              : summary?.total_skus != null
              ? summary.total_skus.toLocaleString()
              : "—"
          }
        />
        <KpiCard
          className={PANEL_KPI}
          label="Avg Safety Buffer"
          value={summaryQ.isLoading ? "..." : formatNumber(summary?.avg_ss)}
          tooltip={{
            title: "Average units held as safety stock per item",
            description: "Higher means more protection but more capital tied up.",
          }}
        />
        <KpiCard
          className={PANEL_KPI}
          label="Optimal Order Size"
          value={summaryQ.isLoading ? "..." : formatNumber(summary?.avg_eoq)}
          tooltip={{
            title: "Economic order quantity balancing ordering cost vs holding cost",
            description: "The order size that minimizes total inventory management cost.",
          }}
        />
        <KpiCard
          className={PANEL_KPI}
          label="At Stockout Risk"
          value={
            summaryQ.isLoading
              ? "..."
              : summary?.below_ss_count != null
              ? `${summary.below_ss_count.toLocaleString()} (${summary.below_ss_pct?.toFixed(1)}%)`
              : "—"
          }
          colorClass={
            (summary?.below_ss_count ?? 0) > 0 ? "text-red-600" : undefined
          }
          tooltip={{
            title: "Items where current inventory is below the recommended safety buffer",
            description: "These items may run out before the next delivery arrives.",
          }}
        />
      </div>

      {/* Recommended actions based on current data */}
      {(summary?.below_ss_count ?? 0) > 0 && (
        <RecommendedActionCard
          severity="high"
          title={`${summary!.below_ss_count} items at stockout risk in the forward plan`}
          action="Review the replenishment plan and approve suggested order quantities"
        />
      )}

      {/* Comparison chart: Forecast SS vs Historical SS by ABC */}
      {(comparison?.by_abc?.length ?? 0) > 0 && comparison && (
        <div className="rounded-lg border bg-card p-4">
          <h3 className="mb-3 text-sm font-semibold text-foreground">
            Safety Buffer: Forecast-Based vs Historical by ABC Class
          </h3>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={comparison.by_abc}
                margin={{ top: 4, right: 16, left: 8, bottom: 4 }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke={chartColors.grid}
                />
                <XAxis dataKey="abc_vol" tick={{ fill: chartColors.axis }} />
                <YAxis
                  tickFormatter={formatCompactNumber}
                  tick={{ fill: chartColors.axis }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: chartColors.tooltip_bg,
                    borderColor: chartColors.tooltip_border,
                  }}
                  formatter={(v: number, name: string) => [
                    formatNumber(v),
                    name,
                  ]}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar
                  dataKey="avg_forecast_ss"
                  name="Forecast Buffer"
                  fill="#7c3aed"
                />
                <Bar
                  dataKey="avg_historical_ss"
                  name="Historical Buffer"
                  fill="#94a3b8"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Detail table */}
      <div className="rounded-lg border bg-card p-4">
        <h3 className="mb-3 text-sm font-semibold text-foreground">
          Replenishment Plan Detail
        </h3>

        {detailQ.isLoading ? (
          <TableSkeleton rows={8} cols={10} />
        ) : (detail?.rows ?? []).length === 0 ? (
          <EmptyState
            icon={RefreshCw}
            title="No forward replenishment plan"
            description="The replenishment plan uses ML production forecasts and CI bands to compute forward-looking safety stock, EOQ, and reorder points for each DFU across the next 12 months. It compares forward SS to historical SS to highlight where targets are shifting."
            steps={[
              { label: "Generate production forecasts", command: "make forecast-generate" },
              { label: "Apply schema (first time only)", command: "make replen-plan-schema" },
              { label: "Compute forward replenishment plan", command: "make replen-plan-compute" },
            ]}
          />
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b text-left text-muted-foreground">
                    <th className="pb-2 pr-3">Item</th>
                    <th className="pb-2 pr-3">Loc</th>
                    <th className="pb-2 pr-3">ABC</th>
                    <th className="pb-2 pr-3" title="Inventory review and ordering strategy for this item-location">Policy</th>
                    <th className="pb-2 pr-3 text-right">Fcst Qty</th>
                    <th className="pb-2 pr-3 text-right" title="Safety Buffer: units reserved to prevent stockouts (simulation-based)">Safety Buffer</th>
                    <th className="pb-2 pr-3 text-right" title="Historical buffer based on past demand variability">Hist. Buffer</th>
                    <th className="pb-2 pr-3 text-right" title="Buffer change vs last period — positive means increased buffer">Buffer Change</th>
                    <th className="pb-2 pr-3 text-right" title="Optimal Order Size: the cost-optimal quantity to order each time">Order Size</th>
                    <th className="pb-2 pr-3 text-right" title="Reorder Trigger: place an order when stock drops to this level">Reorder Trigger</th>
                    <th className="pb-2 text-right" title="YES = current inventory is below the safety buffer — action needed">At Risk</th>
                  </tr>
                </thead>
                <tbody>
                  {(detail?.rows ?? []).map(
                    (row: ReplenishmentDetailRow, i: number) => (
                        <tr
                          key={`${row.item_id}-${row.loc}-${i}`}
                          className={`cursor-pointer border-b last:border-0 hover:bg-muted/50 ${
                            selectedSku?.item_id === row.item_id &&
                            selectedSku.loc === row.loc
                              ? "bg-muted"
                              : ""
                          }`}
                          onClick={() =>
                            setSelectedSku({ item_id: row.item_id, loc: row.loc })
                          }
                        >
                          <td className="py-1 pr-3 font-mono">{row.item_id}</td>
                          <td className="py-1 pr-3">{row.loc}</td>
                          <td className="py-1 pr-3">{row.abc_vol ?? "—"}</td>
                          <td className="py-1 pr-3">{row.policy_type?.replace(/_/g, " ").replace(/\b\w/g, (c: string) => c.toUpperCase()) ?? "—"}</td>
                          <td className="py-1 pr-3 text-right">
                            {formatNumber(row.forecast_qty)}
                          </td>
                          <td className="py-1 pr-3 text-right">
                            {formatNumber(row.ss_combined)}
                          </td>
                          <td className="py-1 pr-3 text-right">
                            {formatNumber(row.historical_ss)}
                          </td>
                          <td
                            className={`py-1 pr-3 text-right ${
                              (row.ss_delta ?? 0) > 0
                                ? "text-amber-600"
                                : (row.ss_delta ?? 0) < 0
                                ? "text-green-600"
                                : ""
                            }`}
                          >
                            {row.ss_delta != null
                              ? (row.ss_delta > 0 ? "+" : "") +
                                formatNumber(row.ss_delta)
                              : "—"}
                          </td>
                          <td className="py-1 pr-3 text-right">
                            {formatNumber(row.eoq)}
                          </td>
                          <td className="py-1 pr-3 text-right">
                            {formatNumber(row.reorder_point)}
                          </td>
                          <td className="py-1 text-right">
                            {row.is_below_ss ? (
                              <span className="rounded bg-red-100 px-1 text-red-600 font-semibold">
                                YES
                              </span>
                            ) : (
                              <span className="text-muted-foreground">—</span>
                            )}
                          </td>
                        </tr>
                      )
                    )
                  }
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="mt-3 flex items-center gap-3 text-xs text-muted-foreground">
                <button
                  className="rounded border px-2 py-1 hover:bg-muted disabled:opacity-40"
                  disabled={page === 0}
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                >
                  Prev
                </button>
                <span>
                  Page {page + 1} / {totalPages} (
                  {detail?.total.toLocaleString()} rows)
                </span>
                <button
                  className="rounded border px-2 py-1 hover:bg-muted disabled:opacity-40"
                  disabled={page + 1 >= totalPages}
                  onClick={() => setPage((p) => p + 1)}
                >
                  Next
                </button>
              </div>
            )}
          </>
        )}
      </div>

      {/* SKU drill-down chart */}
      {selectedSku && (
        <div className="rounded-lg border bg-card p-4">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-foreground">
              SKU Drill-Down: {selectedSku.item_id} @ {selectedSku.loc}
            </h3>
            <button
              className="text-xs text-muted-foreground hover:text-foreground"
              onClick={() => setSelectedSku(null)}
            >
              ✕ close
            </button>
          </div>

          {skuQ.isLoading ? (
            <p className="text-xs text-muted-foreground">Loading…</p>
          ) : (sku?.series?.length ?? 0) > 0 && sku ? (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={sku.series}
                  margin={{ top: 4, right: 16, left: 8, bottom: 4 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke={chartColors.grid}
                  />
                  <XAxis
                    dataKey="plan_month"
                    tick={{ fill: chartColors.axis, fontSize: 10 }}
                  />
                  <YAxis
                    tickFormatter={formatCompactNumber}
                    tick={{ fill: chartColors.axis, fontSize: 10 }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: chartColors.tooltip_bg,
                      borderColor: chartColors.tooltip_border,
                    }}
                    formatter={(v: number, name: string) => [
                      formatNumber(v),
                      name,
                    ]}
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line
                    type="monotone"
                    dataKey="forecast_qty"
                    name="Forecast"
                    stroke="#2563eb"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="ss_combined"
                    name="Safety Buffer (Forward)"
                    stroke="#7c3aed"
                    strokeWidth={2}
                    strokeDasharray="5 3"
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="historical_ss"
                    name="Safety Buffer (Historical)"
                    stroke="#94a3b8"
                    strokeWidth={1.5}
                    strokeDasharray="3 2"
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="reorder_point"
                    name="Reorder Trigger"
                    stroke="#f97316"
                    strokeWidth={1.5}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">
              No forward plan data for this DFU.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
