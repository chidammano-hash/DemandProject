import { useState, useRef, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  demandSignalsKeys,
  fetchDemandSignalsSummary,
  fetchDemandSignals,
  STALE,
  type DemandSignalRow,
} from "@/api/queries";

import { KpiCard } from "@/components/KpiCard";
import { EmptyState } from "@/components/EmptyState";
import { formatInt } from "@/lib/formatters";
import { Radio } from "lucide-react";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";

const PAGE = 50;
const PANEL_KPI = "rounded-lg bg-muted/30 p-3";

export function DemandSignalsPanel() {
  const { filters: globalFilters } = useGlobalFilterContext();
  const [signalTypeFilter, setSignalTypeFilter] = useState("");
  const [alertPriorityFilter, setAlertPriorityFilter] = useState("");
  const [dsItemFilter, setDsItemFilter] = useState("");
  const [dsLocFilter, setDsLocFilter] = useState("");
  const [dsOffset, setDsOffset] = useState(0);

  const syncedGlobalRef = useRef<string>("");
  useEffect(() => {
    const key = `${globalFilters.item.join(",")}_${globalFilters.location.join(",")}`;
    if (key === syncedGlobalRef.current) return;
    syncedGlobalRef.current = key;
    if (globalFilters.item.length === 1) setDsItemFilter(globalFilters.item[0]);
    if (globalFilters.location.length === 1) setDsLocFilter(globalFilters.location[0]);
  }, [globalFilters.item, globalFilters.location]);

  const { data: summary } = useQuery({
    queryKey: demandSignalsKeys.summary(),
    queryFn: () => fetchDemandSignalsSummary(),
    staleTime: STALE.ONE_MIN,
  });

  const { data: signals, isLoading } = useQuery({
    queryKey: demandSignalsKeys.list({
      signal_type: signalTypeFilter || undefined,
      alert_priority: alertPriorityFilter || undefined,
      item: dsItemFilter || undefined,
      loc: dsLocFilter || undefined,
      limit: PAGE,
      offset: dsOffset,
    }),
    queryFn: () =>
      fetchDemandSignals({
        signal_type: signalTypeFilter || undefined,
        alert_priority: alertPriorityFilter || undefined,
        item: dsItemFilter || undefined,
        loc: dsLocFilter || undefined,
        limit: PAGE,
        offset: dsOffset,
      }),
    staleTime: STALE.ONE_MIN,
  });

  const totalPages = signals ? Math.ceil(signals.total / PAGE) : 0;
  const currentPage = Math.floor(dsOffset / PAGE) + 1;

  const SIGNAL_TYPE_COLORS: Record<string, string> = {
    above_plan: "bg-green-100 text-green-800",
    below_plan: "bg-red-100 text-red-800",
    on_plan: "bg-blue-100 text-blue-800",
  };

  const PRIORITY_COLORS: Record<string, string> = {
    urgent: "bg-red-100 text-red-800",
    watch: "bg-amber-100 text-amber-800",
    normal: "bg-neutral-100 text-neutral-600",
  };

  const SIGNAL_TYPE_LABELS: Record<string, string> = {
    demand_acceleration: "Demand Acceleration",
    demand_deceleration: "Demand Deceleration",
    above_plan: "Above Plan",
    below_plan: "Below Plan",
    spike: "Demand Spike",
    low_velocity: "Low Velocity",
    on_plan: "On Plan",
  };

  const ROW_BG: Record<string, string> = {
    urgent: "bg-red-50 dark:bg-red-950/20",
    watch: "bg-yellow-50 dark:bg-yellow-950/20",
    normal: "",
  };

  return (
    <div className="space-y-4">
      {/* Info banner */}
      <div className="text-xs text-muted-foreground bg-muted/20 border rounded px-3 py-2">
        <strong className="text-foreground">Demand Signals</strong> are short-horizon alerts derived from current sales velocity. They flag items where actual demand is tracking significantly above or below the statistical forecast, enabling proactive replenishment adjustments.
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KpiCard className={PANEL_KPI} label="Above Plan" value={(summary?.above_plan_count ?? 0).toLocaleString()} colorClass="text-green-600" />
        <KpiCard className={PANEL_KPI} label="Below Plan" value={(summary?.below_plan_count ?? 0).toLocaleString()} colorClass="text-red-600" />
        <KpiCard
          className={PANEL_KPI}
          label="Urgent Alerts"
          value={(summary?.urgent_count ?? 0).toLocaleString()}
          colorClass={(summary?.urgent_count ?? 0) > 0 ? "text-red-600" : undefined}
        />
        <KpiCard
          className={PANEL_KPI}
          label="Projected Stockouts"
          value={(summary?.projected_stockouts ?? 0).toLocaleString()}
          colorClass={(summary?.projected_stockouts ?? 0) > 0 ? "text-orange-600" : undefined}
        />
      </div>

      <div className="flex flex-wrap gap-2">
        {["", "above_plan", "below_plan", "on_plan"].map((t) => (
          <button
            key={t || "all"}
            className={`px-2 py-0.5 text-xs rounded-full border transition-colors ${
              signalTypeFilter === t
                ? "bg-foreground text-background border-foreground"
                : "border-border hover:bg-accent"
            }`}
            onClick={() => { setSignalTypeFilter(t); setDsOffset(0); }}
          >
            {t ? (SIGNAL_TYPE_LABELS[t] ?? t.replace(/_/g, " ").replace(/\b\w/g, (c: string) => c.toUpperCase())) : "All Types"}
          </button>
        ))}
        {["", "urgent", "watch"].map((p) => (
          <button
            key={p || "all-priority"}
            className={`px-2 py-0.5 text-xs rounded-full border transition-colors ${
              alertPriorityFilter === p
                ? "bg-foreground text-background border-foreground"
                : "border-border hover:bg-accent"
            }`}
            onClick={() => { setAlertPriorityFilter(p); setDsOffset(0); }}
          >
            {p ? p.charAt(0).toUpperCase() + p.slice(1) : "All Priority"}
          </button>
        ))}
        <input
          className="h-7 rounded border border-input bg-background px-2 text-xs w-28"
          placeholder="Filter by item..."
          value={dsItemFilter}
          onChange={(e) => { setDsItemFilter(e.target.value); setDsOffset(0); }}
        />
        <input
          className="h-7 rounded border border-input bg-background px-2 text-xs w-28"
          placeholder="Filter by location..."
          value={dsLocFilter}
          onChange={(e) => { setDsLocFilter(e.target.value); setDsOffset(0); }}
        />
      </div>

      {isLoading ? (
        <p className="text-xs text-muted-foreground">Loading...</p>
      ) : (signals?.rows ?? []).length === 0 ? (
        <EmptyState
          icon={Radio}
          title="No demand signals computed"
          description="Demand signals compare the most recent 2-week sales velocity against the monthly forecast to flag items running Above Plan, Below Plan, or On Plan. Urgent signals indicate projected stockout within the lead time window."
          steps={[
            { label: "Apply schema (first time only)", command: "make demand-signals-schema" },
            { label: "Compute short-horizon demand signals", command: "make demand-signals-compute" },
          ]}
        />
      ) : (
        <>
          {/* Confidence legend */}
          <div className="text-xs text-muted-foreground p-2 rounded bg-muted/30 border mb-3">
            <span className="font-medium text-foreground">Signal Confidence: </span>
            <span className="text-green-600">&#9679; High ≥ 0.8</span>
            {" · "}
            <span className="text-amber-600 ml-1">&#9679; Medium 0.5–0.79</span>
            {" · "}
            <span className="text-red-600 ml-1">&#9679; Low &lt; 0.5</span>
            <span className="ml-2 text-muted-foreground">— Based on data quality and historical signal accuracy</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b text-muted-foreground">
                  <th className="text-left py-1 pr-2">Item No</th>
                  <th className="text-left py-1 pr-2">Location</th>
                  <th className="text-center py-1 pr-2">Signal Type</th>
                  <th
                    className="text-right py-1 pr-2 cursor-help"
                    title="Magnitude of the demand signal relative to the statistical forecast (e.g. +15% above plan)"
                  >
                    Demand vs Fcst %
                  </th>
                  <th
                    className="text-center py-1 pr-2 cursor-help"
                    title="Signal reliability score. Urgent = high-confidence deviation requiring immediate action; Watch = moderate signal to monitor; Normal = within acceptable range."
                  >
                    Alert Priority
                  </th>
                  <th className="text-right py-1 pr-2">On Hand</th>
                  <th className="text-center py-1">Below SS</th>
                </tr>
              </thead>
              <tbody>
                {(signals?.rows ?? []).map((r: DemandSignalRow, i: number) => (
                    <tr
                      key={`${r.item_no}-${r.loc}-${i}`}
                      className={`border-b last:border-0 hover:bg-muted/40 ${ROW_BG[r.alert_priority] ?? ""}`}
                    >
                      <td className="py-1 pr-2 font-mono">{r.item_no}</td>
                      <td className="py-1 pr-2">{r.loc}</td>
                      <td className="py-1 pr-2 text-center">
                        <span
                          className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                            SIGNAL_TYPE_COLORS[r.signal_type] ?? "bg-neutral-100 text-neutral-600"
                          }`}
                        >
                          {SIGNAL_TYPE_LABELS[r.signal_type] ?? r.signal_type.replace(/_/g, " ")}
                        </span>
                      </td>
                      <td className="py-1 pr-2 text-right">
                        {r.demand_vs_forecast_pct != null ? `${r.demand_vs_forecast_pct.toFixed(1)}%` : "-"}
                      </td>
                      <td className="py-1 pr-2 text-center">
                        <span
                          className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                            PRIORITY_COLORS[r.alert_priority] ?? ""
                          }`}
                        >
                          {r.alert_priority}
                        </span>
                      </td>
                      <td className="py-1 pr-2 text-right">{formatInt(r.current_on_hand)}</td>
                      <td className="py-1 text-center">
                        {r.is_below_ss ? (
                          <span className="px-1 py-0.5 rounded text-xs bg-red-100 text-red-800">Yes</span>
                        ) : (
                          <span className="text-muted-foreground">-</span>
                        )}
                      </td>
                    </tr>
                  ))
                }
              </tbody>
            </table>
          </div>
          {totalPages > 1 && (
            <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
              <button
                className="px-2 py-1 rounded border disabled:opacity-40"
                disabled={dsOffset === 0}
                onClick={() => setDsOffset(Math.max(0, dsOffset - PAGE))}
              >
                Prev
              </button>
              <span>Page {currentPage} / {totalPages}</span>
              <button
                className="px-2 py-1 rounded border disabled:opacity-40"
                disabled={currentPage >= totalPages}
                onClick={() => setDsOffset(dsOffset + PAGE)}
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
