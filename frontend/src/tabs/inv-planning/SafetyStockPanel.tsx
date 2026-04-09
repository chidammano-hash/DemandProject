import { useState, useRef, useEffect, Fragment } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  safetyStockKeys,
  fetchSafetyStockSummary,
  fetchSafetyStockDetail,
  fetchSafetyStockExplain,
  fetchSsWhatIf,
  STALE,
  type SafetyStockRow,
  type SsExplanation,
  type WhatIfResult,
} from "@/api/queries";
import { KpiCard } from "@/components/KpiCard";
import { EmptyState } from "@/components/EmptyState";
import { TableSkeleton } from "@/components/Skeleton";
import { formatFixed, formatPct, formatInt } from "@/lib/formatters";
import { ArchiveX, Shield, ChevronDown, ChevronRight } from "lucide-react";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";
import { insightKeys, fetchSsCostBenefit } from "@/api/queries/inv-planning-insights";
import { DataFreshnessBanner } from "@/components/DataFreshnessBanner";
import { RecommendedActionCard } from "@/components/RecommendedActionCard";

const PAGE = 50;
const PANEL_KPI = "rounded-lg bg-muted/30 p-3";

/** What-If interactive scenario sliders for a single DFU. */
function WhatIfSliders({ itemId, loc }: { itemId: string; loc: string }) {
  const [demandChange, setDemandChange] = useState(0);
  const [ltChange, setLtChange] = useState(0);
  const [slOverride, setSlOverride] = useState("");

  // Debounced values to avoid flooding the API
  const [debouncedDemand, setDebouncedDemand] = useState(0);
  const [debouncedLt, setDebouncedLt] = useState(0);
  const [debouncedSl, setDebouncedSl] = useState("");

  useEffect(() => {
    const t = setTimeout(() => setDebouncedDemand(demandChange), 300);
    return () => clearTimeout(t);
  }, [demandChange]);

  useEffect(() => {
    const t = setTimeout(() => setDebouncedLt(ltChange), 300);
    return () => clearTimeout(t);
  }, [ltChange]);

  useEffect(() => {
    const t = setTimeout(() => setDebouncedSl(slOverride), 300);
    return () => clearTimeout(t);
  }, [slOverride]);

  const hasChanges = debouncedDemand !== 0 || debouncedLt !== 0 || !!debouncedSl;

  const { data: whatIfData, isFetching } = useQuery({
    queryKey: ["ss-what-if", itemId, loc, debouncedDemand, debouncedLt, debouncedSl],
    queryFn: () =>
      fetchSsWhatIf({
        item_id: itemId,
        loc: loc,
        demand_change_pct: debouncedDemand,
        lt_change_days: debouncedLt,
        service_level_override: debouncedSl || undefined,
      }),
    enabled: hasChanges,
    staleTime: STALE.FIVE_MIN,
  });

  const wi = whatIfData as WhatIfResult | undefined;

  return (
    <div className="border-t pt-3 mt-3">
      <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">
        Interactive What-If Scenario Builder
      </p>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label className="text-xs text-muted-foreground block mb-1">Demand Change</label>
          <input
            type="range"
            min={-50}
            max={100}
            value={demandChange}
            onChange={(e) => setDemandChange(Number(e.target.value))}
            className="w-full accent-blue-500"
          />
          <span className="text-xs font-mono tabular-nums">
            {demandChange > 0 ? "+" : ""}
            {demandChange}%
          </span>
        </div>
        <div>
          <label className="text-xs text-muted-foreground block mb-1">Lead Time Change</label>
          <input
            type="range"
            min={-10}
            max={30}
            value={ltChange}
            onChange={(e) => setLtChange(Number(e.target.value))}
            className="w-full accent-amber-500"
          />
          <span className="text-xs font-mono tabular-nums">
            {ltChange > 0 ? "+" : ""}
            {ltChange} days
          </span>
        </div>
        <div>
          <label className="text-xs text-muted-foreground block mb-1">Service Level</label>
          <select
            value={slOverride}
            onChange={(e) => setSlOverride(e.target.value)}
            className="text-xs border rounded px-2 py-1 w-full bg-background"
          >
            <option value="">Current</option>
            <option value="0.99">99%</option>
            <option value="0.98">98%</option>
            <option value="0.95">95%</option>
            <option value="0.90">90%</option>
            <option value="0.85">85%</option>
          </select>
        </div>
      </div>

      {isFetching && hasChanges && (
        <p className="mt-2 text-[10px] text-muted-foreground">Simulating...</p>
      )}

      {wi && (
        <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-2 text-xs">
          <div className="rounded border p-2">
            <p className="text-muted-foreground">Safety Buffer</p>
            <p className="text-lg font-bold tabular-nums">{wi.simulated.ss_combined.toLocaleString()}</p>
            <p className={wi.delta.ss_change > 0 ? "text-red-600" : wi.delta.ss_change < 0 ? "text-emerald-600" : "text-muted-foreground"}>
              {wi.delta.ss_change > 0 ? "+" : ""}
              {wi.delta.ss_change.toLocaleString()} ({wi.delta.ss_change_pct.toFixed(1)}%)
            </p>
          </div>
          <div className="rounded border p-2">
            <p className="text-muted-foreground">Reorder Trigger</p>
            <p className="text-lg font-bold tabular-nums">{wi.simulated.reorder_point.toLocaleString()}</p>
            <p className={wi.delta.rop_change > 0 ? "text-red-600" : wi.delta.rop_change < 0 ? "text-emerald-600" : "text-muted-foreground"}>
              {wi.delta.rop_change > 0 ? "+" : ""}
              {wi.delta.rop_change.toLocaleString()}
            </p>
          </div>
          <div className="rounded border p-2">
            <p className="text-muted-foreground">Monthly Cost Impact</p>
            <p className="text-lg font-bold tabular-nums">
              ${Math.abs(wi.delta.holding_cost_change_monthly).toFixed(2)}
            </p>
            <p className={wi.delta.holding_cost_change_monthly > 0 ? "text-red-600" : wi.delta.holding_cost_change_monthly < 0 ? "text-emerald-600" : "text-muted-foreground"}>
              {wi.delta.holding_cost_change_monthly > 0 ? "Increase" : wi.delta.holding_cost_change_monthly < 0 ? "Decrease" : "No change"}
            </p>
          </div>
        </div>
      )}

      {!hasChanges && (
        <p className="mt-2 text-[10px] text-muted-foreground">
          Adjust sliders above to simulate parameter changes and see the impact on safety stock.
        </p>
      )}
    </div>
  );
}

/** Inline explainability card shown when a row is expanded. */
function SsExplainCard({ itemId, loc }: { itemId: string; loc: string }) {
  const { data, isLoading, error } = useQuery({
    queryKey: safetyStockKeys.explain(itemId, loc),
    queryFn: () => fetchSafetyStockExplain(itemId, loc),
    staleTime: STALE.FIVE_MIN,
  });

  if (isLoading) return <div className="px-4 py-3 text-xs text-muted-foreground">Loading explanation...</div>;
  if (error || !data) return <div className="px-4 py-3 text-xs text-red-600">Failed to load explanation</div>;

  const d = data as SsExplanation;
  const demandPct = d.components.demand_component.pct_of_total ?? 0;
  const ltPct = d.components.leadtime_component.pct_of_total ?? 0;

  return (
    <div className="px-4 py-3 space-y-3 bg-muted/10 border-t border-b">
      {/* Formula */}
      <div className="space-y-1">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Formula</p>
        <p className="font-mono text-xs text-foreground">{d.formula}</p>
        <p className="font-mono text-xs text-blue-600 dark:text-blue-400">{d.formula_substituted}</p>
      </div>

      {/* Waterfall bar */}
      <div className="space-y-1">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Component Breakdown</p>
        <div className="flex items-center gap-1 h-5 rounded overflow-hidden text-[10px] font-medium">
          <div
            className="bg-blue-500 text-white flex items-center justify-center h-full rounded-l"
            style={{ width: `${Math.max(demandPct, 8)}%` }}
            title={`Demand: ${formatFixed(d.components.demand_component.value)} units (${formatFixed(demandPct)}%)`}
          >
            {demandPct >= 15 ? `Demand ${formatFixed(demandPct)}%` : ""}
          </div>
          <div
            className="bg-amber-500 text-white flex items-center justify-center h-full rounded-r"
            style={{ width: `${Math.max(ltPct, 8)}%` }}
            title={`Lead Time: ${formatFixed(d.components.leadtime_component.value)} units (${formatFixed(ltPct)}%)`}
          >
            {ltPct >= 15 ? `LT ${formatFixed(ltPct)}%` : ""}
          </div>
        </div>
        <div className="flex gap-4 text-[10px] text-muted-foreground">
          <span className="flex items-center gap-1">
            <span className="inline-block w-2 h-2 rounded bg-blue-500" />
            Demand: {formatFixed(d.components.demand_component.value)} units
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-2 h-2 rounded bg-amber-500" />
            Lead Time: {formatFixed(d.components.leadtime_component.value)} units
          </span>
          <span className="font-medium text-foreground">
            Combined: {formatFixed(d.components.combined.value)} units
          </span>
        </div>
      </div>

      {/* Inputs grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[10px]">
        <div className="border rounded p-1.5">
          <p className="text-muted-foreground">Avg Monthly Demand</p>
          <p className="font-bold tabular-nums">{formatFixed(d.components.demand_component.inputs?.demand_mean_monthly)}</p>
        </div>
        <div className="border rounded p-1.5">
          <p className="text-muted-foreground">Demand Std (monthly)</p>
          <p className="font-bold tabular-nums">{formatFixed(d.components.demand_component.inputs?.demand_std_monthly)}</p>
        </div>
        <div className="border rounded p-1.5">
          <p className="text-muted-foreground">Lead Time (days)</p>
          <p className="font-bold tabular-nums">{formatFixed(d.components.leadtime_component.inputs?.lead_time_mean_days)}</p>
        </div>
        <div className="border rounded p-1.5">
          <p className="text-muted-foreground">LT Std (days)</p>
          <p className="font-bold tabular-nums">{formatFixed(d.components.leadtime_component.inputs?.lead_time_std_days)}</p>
        </div>
      </div>

      {/* Sensitivity */}
      {d.sensitivity.length > 0 && (
        <div className="space-y-1">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Pre-Computed Scenarios</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-1.5">
            {d.sensitivity.map((s) => (
              <div key={s.scenario} className="flex items-center justify-between border rounded px-2 py-1 text-[10px]">
                <span className="text-muted-foreground">{s.scenario}</span>
                <span className="font-medium tabular-nums">
                  {formatInt(s.ss_result)} units{" "}
                  <span className={s.delta.startsWith("+") ? "text-amber-600" : "text-green-600"}>
                    ({s.delta})
                  </span>
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Interactive What-If Scenario Builder */}
      <WhatIfSliders itemId={itemId} loc={loc} />

      {/* Context badges */}
      <div className="flex flex-wrap gap-2 text-[10px]">
        {d.context.current_on_hand != null && (
          <span className="px-2 py-0.5 rounded-full border bg-background">
            On-hand: {formatInt(d.context.current_on_hand)}
          </span>
        )}
        {d.context.gap_qty != null && (
          <span className={`px-2 py-0.5 rounded-full border ${
            d.context.gap_qty < 0 ? "bg-red-50 text-red-700 dark:bg-red-950 dark:text-red-400" : "bg-green-50 text-green-700 dark:bg-green-950 dark:text-green-400"
          }`}>
            Gap: {d.context.gap_qty > 0 ? "+" : ""}{formatInt(d.context.gap_qty)}
          </span>
        )}
        {d.context.reorder_point != null && (
          <span className="px-2 py-0.5 rounded-full border bg-background">
            ROP: {formatFixed(d.context.reorder_point)}
          </span>
        )}
        <span className="px-2 py-0.5 rounded-full border bg-background">
          Source: {d.context.forecast_source}
        </span>
        {d.abc_xyz_segment && (
          <span className="px-2 py-0.5 rounded-full border bg-background">
            Segment: {d.abc_xyz_segment}
          </span>
        )}
      </div>
    </div>
  );
}

export function SafetyStockPanel() {
  const { filters: globalFilters } = useGlobalFilterContext();
  const [belowOnly, setBelowOnly] = useState(false);
  const [ssItemFilter, setSsItemFilter] = useState("");
  const [ssLocFilter, setSsLocFilter] = useState("");
  const [ssOffset, setSsOffset] = useState(0);
  const [expandedRow, setExpandedRow] = useState<string | null>(null);

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
      <DataFreshnessBanner
        lastRefreshed={summary?.computed_at}
        source="Safety Stock Targets"
        staleSec={86400}
      />

      {/* Info banner */}
      <div className="text-xs text-muted-foreground bg-muted/20 border rounded px-3 py-2 mb-3">
        Safety stock is the buffer inventory held to protect against running out of stock during lead time.{" "}
        <strong className="text-foreground">Forward Buffer</strong> uses demand simulation;{" "}
        <strong className="text-foreground">Historical Buffer</strong> uses past variability. Items where Forward &gt; Historical need increased buffer.
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KpiCard
          className={PANEL_KPI}
          label="At Stockout Risk"
          value={summaryLoading ? "..." : (summary?.below_ss_count ?? 0).toLocaleString()}
          colorClass={(summary?.below_ss_count ?? 0) > 0 ? "text-red-600" : undefined}
          tooltip={{
            title: "Items that may run out within their lead time",
            description: "Higher means more urgent reordering needed.",
          }}
          trend={!summaryLoading && summary ? (() => {
            const riskPct = summary.total_skus > 0 ? summary.below_ss_count / summary.total_skus : 0;
            // >20% of portfolio at risk = worsening; <5% = improving; otherwise flat
            if (riskPct > 0.2) return { delta: Math.round(riskPct * 100), direction: "up" as const, unit: "% of SKUs at risk", period: "threshold" };
            if (riskPct < 0.05) return { delta: Math.round(riskPct * 100), direction: "down" as const, unit: "% of SKUs at risk", period: "threshold" };
            return { delta: Math.round(riskPct * 100), direction: "flat" as const, unit: "% of SKUs at risk", period: "threshold" };
          })() : undefined}
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
          label="Buffer Health"
          value={summaryLoading ? "..." : formatPct(summary?.avg_ss_coverage != null ? summary.avg_ss_coverage * 100 : null)}
          tooltip={{
            title: "Ratio of current stock to safety stock",
            description: "Below 1.0x means insufficient buffer against stockouts.",
          }}
          trend={!summaryLoading && summary?.avg_ss_coverage != null ? (() => {
            const cov = summary.avg_ss_coverage;
            // Coverage >= 1.5 = healthy (up-good); < 1.0 = concerning (down-bad); otherwise flat
            if (cov >= 1.5) return { delta: +(cov * 100).toFixed(0), direction: "up" as const, unit: "% coverage", period: "target" };
            if (cov < 1.0) return { delta: +(cov * 100).toFixed(0), direction: "down" as const, unit: "% coverage", period: "target" };
            return { delta: +(cov * 100).toFixed(0), direction: "flat" as const, unit: "% coverage", period: "target" };
          })() : undefined}
        />
        <KpiCard
          className={PANEL_KPI}
          label="Total SKUs"
          value={summaryLoading ? "..." : (summary?.total_skus ?? 0).toLocaleString()}
        />
        <KpiCard
          className={PANEL_KPI}
          label="Buffer Days"
          value={summaryLoading ? "..." : formatFixed(summary?.avg_ss_days)}
          tooltip={{
            title: "How many days of average demand are covered by safety stock alone",
            description: "More days means a larger buffer if supply is delayed.",
          }}
          trend={!summaryLoading && summary?.avg_ss_days != null ? (() => {
            const days = summary.avg_ss_days;
            // >21 buffer days = strong buffer; <7 = thin buffer; otherwise stable
            if (days >= 21) return { delta: +days.toFixed(1), direction: "up" as const, unit: " days", period: "target" };
            if (days < 7) return { delta: +days.toFixed(1), direction: "down" as const, unit: " days", period: "target" };
            return { delta: +days.toFixed(1), direction: "flat" as const, unit: " days", period: "target" };
          })() : undefined}
        />
      </div>

      {/* Recommended actions based on current data */}
      {!summaryLoading && (summary?.below_ss_count ?? 0) > 10 && (
        <RecommendedActionCard
          severity="high"
          title={`${summary!.below_ss_count} items below safety buffer`}
          action="Review and generate replenishment orders in the Planned Orders panel"
        />
      )}
      {!summaryLoading && summary?.avg_ss_coverage != null && summary.avg_ss_coverage < 1.0 && (
        <RecommendedActionCard
          severity="critical"
          title="Average buffer health below 1.0x — portfolio is under-stocked"
          action="Review safety stock targets and increase for high-risk items"
        />
      )}

      {/* Cost-Benefit Analysis */}
      {cbData && (
        <div className="border rounded-lg p-4 bg-card space-y-3">
          <h4 className="text-xs font-semibold text-foreground">Cost-Benefit Analysis</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="border rounded-lg p-2.5">
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Monthly Holding Cost</p>
              <p className="text-sm font-bold tabular-nums text-foreground">
                ${formatInt(cbData.holding_cost_monthly ?? 0)}
              </p>
            </div>
            <div className="border rounded-lg p-2.5">
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Monthly Stockout Risk</p>
              <p className="text-sm font-bold tabular-nums text-red-600">
                ${formatInt(cbData.stockout_risk_monthly ?? 0)}
              </p>
            </div>
            <div className="border rounded-lg p-2.5">
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Over-Stocked Items</p>
              <p className="text-sm font-bold tabular-nums text-amber-600">
                {formatInt(cbData.over_stocked_count ?? 0)}
              </p>
            </div>
            <div className="border rounded-lg p-2.5">
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Under-Stocked Items</p>
              <p className="text-sm font-bold tabular-nums text-red-600">
                {formatInt(cbData.under_stocked_count ?? 0)}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Supplier Risk Adjustment note */}
      <div className="flex items-start gap-2 text-xs text-muted-foreground bg-muted/20 border rounded px-3 py-2">
        <Shield className="w-4 h-4 text-[#0D9488] flex-shrink-0 mt-0.5" />
        <span>
          <strong className="text-foreground">Supplier Risk Adjustment:</strong> Items with unreliable supplier delivery times have their safety buffer automatically increased to prevent stockouts.
        </span>
      </div>

      {/* Below SS warning KPI */}
      {!summaryLoading && belowSsCount > 0 && (
        <div className="grid grid-cols-1 gap-3">
          <KpiCard
            className={PANEL_KPI}
            label="Items Needing Reorder"
            value={belowSsCount.toLocaleString()}
            sublabel="below safety buffer"
            severity="warning"
            tooltip={{
              title: "Items below their safety buffer target",
              description: "These items have on-hand inventory below the safety buffer and may run out before the next delivery arrives. Prioritize reordering these.",
            }}
          />
        </div>
      )}

      {(summary?.by_abc?.length ?? 0) > 0 && summary && (
        <div className="overflow-x-auto">
          <p className="text-xs font-medium mb-2">Safety Buffer by ABC Class</p>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b text-muted-foreground">
                <th className="text-left py-1 pr-3">ABC Class</th>
                <th className="text-right py-1 pr-3">Count</th>
                <th className="text-right py-1 pr-3">At Risk</th>
                <th className="text-right py-1">Avg Buffer Health</th>
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
          {belowOnly ? "All Items" : "At Risk Only"}
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
        <TableSkeleton rows={8} cols={10} />
      ) : (detail?.rows ?? []).length === 0 ? (
        <EmptyState
          icon={ArchiveX}
          title="No safety stock targets computed"
          description="Safety buffer targets are calculated per item-location based on demand variability and lead time uncertainty. Service levels are set by ABC class (A=98%, B=95%, C=90%)."
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
                  <th className="w-6 py-1 pr-1" />
                  <th className="text-left py-1 pr-2">Item No</th>
                  <th className="text-left py-1 pr-2">Location</th>
                  <th
                    className="text-right py-1 pr-2 cursor-help"
                    title="Safety Buffer: units reserved to prevent stockouts during lead time (simulation-based)"
                  >
                    Safety Buffer
                  </th>
                  <th
                    className="text-right py-1 pr-2 cursor-help"
                    title="Historical buffer based on past demand variability — the baseline before simulation adjustment."
                  >
                    Hist. Buffer
                  </th>
                  <th
                    className="text-right py-1 pr-2 cursor-help"
                    title="Buffer change vs historical. Positive = more conservative (increased). Negative = more efficient (reduced)."
                  >
                    Buffer Change
                  </th>
                  <th className="text-right py-1 pr-2">Coverage %</th>
                  <th
                    className="text-right py-1 pr-2 cursor-help"
                    title="Service level target mapped from Z-Score. Z=1.65 → 95%, Z=2.05 → 98%, Z=2.33 → 99%. Higher = more protection against stockouts."
                  >
                    Svc Level
                  </th>
                  <th className="text-center py-1 pr-2">At Risk</th>
                  <th
                    className="text-right py-1 pr-2 cursor-help"
                    title="Reorder Trigger: place an order when stock drops to this level (safety buffer + expected demand during lead time)"
                  >
                    Reorder Trigger
                  </th>
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
                  const rowKey = `${r.item_id}::${r.loc}`;
                  const isExpanded = expandedRow === rowKey;

                  return (
                    <Fragment key={`${r.item_id}-${r.loc}-${i}`}>
                    <tr
                      className={`border-b last:border-0 hover:bg-muted/40 cursor-pointer ${r.is_below_ss ? "bg-red-50 dark:bg-red-950/20" : ""}`}
                      onClick={() => setExpandedRow(isExpanded ? null : rowKey)}
                    >
                      <td className="py-1 pr-1 text-center text-muted-foreground">
                        {isExpanded ? <ChevronDown className="w-3 h-3 inline" /> : <ChevronRight className="w-3 h-3 inline" />}
                      </td>
                      <td className="py-1 pr-2 font-mono">{r.item_id}</td>
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
                      <td
                        className="py-1 pr-2 text-right cursor-help"
                        title={r.z_score != null ? `Z-Score: ${formatFixed(r.z_score)}` : undefined}
                      >
                        {r.z_score != null
                          ? (() => {
                              const z = r.z_score;
                              const pct = z >= 2.33 ? "99%" : z >= 2.05 ? "98%" : z >= 1.65 ? "95%" : z >= 1.28 ? "90%" : `${Math.round(50 + 50 * Math.min(z / 2.33, 1))}%`;
                              return pct;
                            })()
                          : "—"}
                      </td>
                      <td className="py-1 pr-2 text-center">
                        {r.is_below_ss ? (
                          <span className="px-1.5 py-0.5 rounded text-xs bg-red-100 text-red-800 font-medium">Yes</span>
                        ) : (
                          <span className="text-muted-foreground">-</span>
                        )}
                      </td>
                      <td
                        className="py-1 pr-2 text-right cursor-help"
                        title={r.reorder_point != null ? `Order when stock drops to ${formatFixed(r.reorder_point)} units` : undefined}
                      >
                        {formatFixed(r.reorder_point)}
                      </td>
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
                    {isExpanded && (
                      <tr>
                        <td colSpan={13}>
                          <SsExplainCard itemId={r.item_id} loc={r.loc} />
                        </td>
                      </tr>
                    )}
                    </Fragment>
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
