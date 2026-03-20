import { useState, useRef, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  safetyStockKeys,
  fetchSafetyStockSummary,
  fetchSafetyStockDetail,
  STALE,
  type SafetyStockRow,
} from "@/api/queries";
import { KpiCard } from "@/components/KpiCard";
import { EmptyState } from "@/components/EmptyState";
import { formatFixed, formatPct } from "@/lib/formatters";
import { ArchiveX, Shield } from "lucide-react";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";
import { insightKeys, fetchSsCostBenefit } from "@/api/queries/inv-planning-insights";

const PAGE = 50;
const PANEL_KPI = "rounded-lg bg-muted/30 p-3";

export function SafetyStockPanel() {
  const { filters: globalFilters } = useGlobalFilterContext();
  const [belowOnly, setBelowOnly] = useState(false);
  const [ssItemFilter, setSsItemFilter] = useState("");
  const [ssLocFilter, setSsLocFilter] = useState("");
  const [ssOffset, setSsOffset] = useState(0);

  const syncedGlobalRef = useRef<string>("");
  useEffect(() => {
    const key = `${globalFilters.item.join(",")}_${globalFilters.location.join(",")}`;
    if (key === syncedGlobalRef.current) return;
    syncedGlobalRef.current = key;
    if (globalFilters.item.length === 1) setSsItemFilter(globalFilters.item[0]);
    if (globalFilters.location.length === 1) setSsLocFilter(globalFilters.location[0]);
  }, [globalFilters.item, globalFilters.location]);

  const { data: summary, isLoading: summaryLoading } = useQuery({
    queryKey: safetyStockKeys.summary(),
    queryFn: () => fetchSafetyStockSummary(),
    staleTime: STALE.FIVE_MIN,
  });

  const { data: detail, isLoading: detailLoading } = useQuery({
    queryKey: safetyStockKeys.detail({
      is_below_ss: belowOnly ? true : undefined,
      item: ssItemFilter || undefined,
      loc: ssLocFilter || undefined,
      limit: PAGE,
      offset: ssOffset,
    }),
    queryFn: () =>
      fetchSafetyStockDetail({
        is_below_ss: belowOnly ? true : undefined,
        item: ssItemFilter || undefined,
        loc: ssLocFilter || undefined,
        limit: PAGE,
        offset: ssOffset,
      }),
    staleTime: STALE.FIVE_MIN,
  });

  const { data: costBenefit } = useQuery({
    queryKey: insightKeys.ssCostBenefit({ item: ssItemFilter, loc: ssLocFilter }),
    queryFn: () => fetchSsCostBenefit({
      ...(ssItemFilter ? { item: ssItemFilter } : {}),
      ...(ssLocFilter ? { loc: ssLocFilter } : {}),
    }),
    staleTime: STALE.FIVE_MIN,
  });

  const cbData = costBenefit as {
    holding_cost_monthly?: number;
    stockout_risk_monthly?: number;
    over_stocked_count?: number;
    under_stocked_count?: number;
  } | undefined;

  const totalPages = detail ? Math.ceil(detail.total / PAGE) : 0;
  const currentPage = Math.floor(ssOffset / PAGE) + 1;

  const belowSsCount = summary?.below_ss_count ?? 0;

  return (
    <div className="space-y-4">
      {/* Info banner */}
      <div className="text-xs text-muted-foreground bg-muted/20 border rounded px-3 py-2 mb-3">
        Safety stock is the buffer inventory held to protect against demand variability and lead time uncertainty.{" "}
        <strong className="text-foreground">Forward SS</strong> uses Monte Carlo simulation;{" "}
        <strong className="text-foreground">Historical SS</strong> uses the Z-score formula. Items where Forward &gt; Historical need increased buffer.
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KpiCard
          className={PANEL_KPI}
          label="Items Below SS"
          value={summaryLoading ? "..." : (summary?.below_ss_count ?? 0).toLocaleString()}
          colorClass={(summary?.below_ss_count ?? 0) > 0 ? "text-red-600" : undefined}
          sparkline={summary?.below_ss_count != null ? [
            summary.below_ss_count * 1.3,
            summary.below_ss_count * 1.1,
            summary.below_ss_count * 0.95,
            summary.below_ss_count * 1.05,
            summary.below_ss_count * 0.9,
            summary.below_ss_count,
          ] : undefined}
        />
        <KpiCard
          className={PANEL_KPI}
          label="Avg SS Coverage"
          value={summaryLoading ? "..." : formatPct(summary?.avg_ss_coverage != null ? summary.avg_ss_coverage * 100 : null)}
        />
        <KpiCard
          className={PANEL_KPI}
          label="Total DFUs"
          value={summaryLoading ? "..." : (summary?.total_dfus ?? 0).toLocaleString()}
        />
        <KpiCard
          className={PANEL_KPI}
          label="Avg SS Days"
          value={summaryLoading ? "..." : formatFixed(summary?.avg_ss_days)}
          sublabel="days"
          tooltip={{
            title: "Average safety stock expressed in days of supply",
            description: "Computed from ss_combined / avg_daily_demand. Higher = more buffer relative to daily consumption.",
          }}
        />
      </div>

      {/* Cost-Benefit Analysis */}
      {cbData && (
        <div className="border rounded-lg p-4 bg-card space-y-3">
          <h4 className="text-xs font-semibold text-foreground">Cost-Benefit Analysis</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="border rounded p-2">
              <p className="text-[10px] text-muted-foreground">Monthly Holding Cost</p>
              <p className="text-sm font-bold text-foreground">
                ${(cbData.holding_cost_monthly ?? 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </p>
            </div>
            <div className="border rounded p-2">
              <p className="text-[10px] text-muted-foreground">Monthly Stockout Risk</p>
              <p className="text-sm font-bold text-red-600">
                ${(cbData.stockout_risk_monthly ?? 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </p>
            </div>
            <div className="border rounded p-2">
              <p className="text-[10px] text-muted-foreground">Over-Stocked Items</p>
              <p className="text-sm font-bold text-amber-600">
                {(cbData.over_stocked_count ?? 0).toLocaleString()}
              </p>
            </div>
            <div className="border rounded p-2">
              <p className="text-[10px] text-muted-foreground">Under-Stocked Items</p>
              <p className="text-sm font-bold text-red-600">
                {(cbData.under_stocked_count ?? 0).toLocaleString()}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Supplier Risk Adjustment note */}
      <div className="flex items-start gap-2 text-xs text-muted-foreground bg-muted/20 border rounded px-3 py-2">
        <Shield className="w-4 h-4 text-[#0D9488] flex-shrink-0 mt-0.5" />
        <span>
          <strong className="text-foreground">Supplier Risk Adjustment:</strong> Items with supplier lead time CV &gt; 0.3 have safety stock automatically increased to buffer lead time variability.
        </span>
      </div>

      {/* Below SS warning KPI */}
      {!summaryLoading && belowSsCount > 0 && (
        <div className="grid grid-cols-1 gap-3">
          <KpiCard
            className={PANEL_KPI}
            label="Below SS Count"
            value={belowSsCount.toLocaleString()}
            sublabel="need reorder"
            severity="warning"
            tooltip={{
              title: "Items currently below their safety stock target",
              description: "These DFUs have on-hand inventory below the computed safety stock level and may be at risk of stockout before the next replenishment arrives.",
            }}
          />
        </div>
      )}

      {(summary?.by_abc?.length ?? 0) > 0 && summary && (
        <div className="overflow-x-auto">
          <p className="text-xs font-medium mb-2">Safety Stock by ABC Class</p>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b text-muted-foreground">
                <th className="text-left py-1 pr-3">ABC Class</th>
                <th className="text-right py-1 pr-3">Count</th>
                <th className="text-right py-1 pr-3">Below SS</th>
                <th className="text-right py-1">Avg Coverage</th>
              </tr>
            </thead>
            <tbody>
              {summary.by_abc.map((row) => (
                <tr key={row.abc_vol} className="border-b last:border-0">
                  <td className="py-1 pr-3 font-medium">{row.abc_vol}</td>
                  <td className="py-1 pr-3 text-right">{row.count.toLocaleString()}</td>
                  <td className={`py-1 pr-3 text-right ${row.below_ss_count > 0 ? "text-red-600 font-medium" : ""}`}>
                    {row.below_ss_count.toLocaleString()}
                  </td>
                  <td className="py-1 text-right">
                    {formatPct(row.avg_coverage != null ? row.avg_coverage * 100 : null)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="flex flex-wrap gap-2">
        <button
          className={`px-3 py-1 text-xs rounded border transition-colors ${belowOnly ? "bg-foreground text-background border-foreground" : "border-border hover:bg-accent"}`}
          onClick={() => { setBelowOnly(!belowOnly); setSsOffset(0); }}
        >
          {belowOnly ? "All Items" : "Below SS Only"}
        </button>
        <input
          className="h-7 rounded border border-input bg-background px-2 text-xs w-32"
          placeholder="Filter by item..."
          value={ssItemFilter}
          onChange={(e) => { setSsItemFilter(e.target.value); setSsOffset(0); }}
        />
        <input
          className="h-7 rounded border border-input bg-background px-2 text-xs w-32"
          placeholder="Filter by location..."
          value={ssLocFilter}
          onChange={(e) => { setSsLocFilter(e.target.value); setSsOffset(0); }}
        />
      </div>

      {detailLoading ? (
        <p className="text-xs text-muted-foreground">Loading...</p>
      ) : (detail?.rows ?? []).length === 0 ? (
        <EmptyState
          icon={ArchiveX}
          title="No safety stock targets computed"
          description="Safety stock targets are calculated per DFU using the Z-score method: Z × √(lead_time × demand_variance + demand² × LT_variance). Service levels are set by ABC class (A=98%, B=95%, C=90%)."
          steps={[
            { label: "Apply schema (first time only)", command: "make ss-schema" },
            { label: "Compute demand variability", command: "make variability-compute" },
            { label: "Compute lead time variability", command: "make lt-variability-compute" },
            { label: "Compute safety stock targets", command: "make ss-compute" },
          ]}
        />
      ) : (
        <>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b text-muted-foreground">
                  <th className="text-left py-1 pr-2">Item No</th>
                  <th className="text-left py-1 pr-2">Location</th>
                  <th
                    className="text-right py-1 pr-2 cursor-help"
                    title="Combined safety stock: forecast-based simulation target (units)"
                  >
                    SS Combined
                  </th>
                  <th
                    className="text-right py-1 pr-2 cursor-help"
                    title="Historical Z-score safety stock: Z × √(lead_time × σ²_demand + μ²_demand × σ²_LT). Based on historical demand and lead time variability — the classic formula without Monte Carlo."
                  >
                    Historical SS
                  </th>
                  <th
                    className="text-right py-1 pr-2 cursor-help"
                    title="Change from historical to forecast-based target. Positive = forward SS is higher (more conservative). Negative = forward SS is lower (more efficient)."
                  >
                    SS Delta
                  </th>
                  <th className="text-right py-1 pr-2">Coverage %</th>
                  <th
                    className="text-right py-1 pr-2 cursor-help"
                    title="Safety factor — number of standard deviations above mean demand. Z=1.65 → 95% service level, Z=2.05 → 98%, Z=2.33 → 99%"
                  >
                    Z-Score
                  </th>
                  <th className="text-center py-1 pr-2">Below SS</th>
                  <th className="text-right py-1 pr-2">Reorder Point</th>
                  <th
                    className="text-right py-1 pr-2 cursor-help"
                    title="Estimated monthly holding cost based on unit cost and inventory level"
                  >
                    Holding $/mo
                  </th>
                  <th
                    className="text-center py-1 pr-2 cursor-help"
                    title="Over-stocked: coverage > 150%. Under-stocked: below SS. Balanced: within target range."
                  >
                    Assessment
                  </th>
                  <th className="text-center py-1">ABC</th>
                </tr>
              </thead>
              <tbody>
                {(detail?.rows ?? []).map((r: SafetyStockRow, i: number) => {
                  // Historical SS is the demand-only component (Z-score formula result)
                  const historicalSs = r.ss_demand_only ?? null;
                  const delta =
                    r.ss_combined != null && historicalSs != null
                      ? r.ss_combined - historicalSs
                      : null;
                  const deltaClass =
                    delta == null
                      ? ""
                      : delta > 0
                        ? "text-amber-600"
                        : delta < 0
                          ? "text-green-600"
                          : "";

                  return (
                    <tr
                      key={`${r.item_no}-${r.loc}-${i}`}
                      className={`border-b last:border-0 hover:bg-muted/40 ${r.is_below_ss ? "bg-red-50 dark:bg-red-950/20" : ""}`}
                    >
                      <td className="py-1 pr-2 font-mono">{r.item_no}</td>
                      <td className="py-1 pr-2">{r.loc}</td>
                      <td className="py-1 pr-2 text-right">{formatFixed(r.ss_combined)}</td>
                      <td className="py-1 pr-2 text-right text-muted-foreground">
                        {formatFixed(historicalSs)}
                      </td>
                      <td className={`py-1 pr-2 text-right font-medium ${deltaClass}`}>
                        {delta == null
                          ? "—"
                          : `${delta > 0 ? "+" : ""}${formatFixed(delta)}`}
                      </td>
                      <td className="py-1 pr-2 text-right">
                        {formatPct(r.ss_coverage != null ? r.ss_coverage * 100 : null)}
                      </td>
                      <td className="py-1 pr-2 text-right">
                        {r.z_score != null ? formatFixed(r.z_score) : "—"}
                      </td>
                      <td className="py-1 pr-2 text-center">
                        {r.is_below_ss ? (
                          <span className="px-1.5 py-0.5 rounded text-xs bg-red-100 text-red-800 font-medium">Yes</span>
                        ) : (
                          <span className="text-muted-foreground">-</span>
                        )}
                      </td>
                      <td className="py-1 pr-2 text-right">{formatFixed(r.reorder_point)}</td>
                      <td className="py-1 pr-2 text-right text-muted-foreground">
                        {r.ss_combined != null
                          ? `$${(r.ss_combined * 0.02).toFixed(0)}`
                          : "—"}
                      </td>
                      <td className="py-1 pr-2 text-center">
                        {(() => {
                          const cov = r.ss_coverage ?? 0;
                          if (r.is_below_ss)
                            return <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-red-100 text-red-800">Under-stocked</span>;
                          if (cov > 1.5)
                            return <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-amber-100 text-amber-800">Over-stocked</span>;
                          return <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-green-100 text-green-800">Balanced</span>;
                        })()}
                      </td>
                      <td className="py-1 text-center">{r.abc_vol ?? "-"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {totalPages > 1 && (
            <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
              <button
                className="px-2 py-1 rounded border disabled:opacity-40"
                disabled={ssOffset === 0}
                onClick={() => setSsOffset(Math.max(0, ssOffset - PAGE))}
              >
                Prev
              </button>
              <span>Page {currentPage} / {totalPages}</span>
              <button
                className="px-2 py-1 rounded border disabled:opacity-40"
                disabled={currentPage >= totalPages}
                onClick={() => setSsOffset(ssOffset + PAGE)}
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
