import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  demandSignalsKeys,
  fetchDemandSignalsSummary,
  fetchDemandSignals,
  STALE,
  type DemandSignalRow,
} from "@/api/queries";

const PAGE = 50;

function fmt(n: number | null | undefined, decimals = 1): string {
  if (n == null) return "—";
  return Number(n).toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

export function DemandSignalsPanel() {
  const [signalTypeFilter, setSignalTypeFilter] = useState("");
  const [alertPriorityFilter, setAlertPriorityFilter] = useState("");
  const [dsItemFilter, setDsItemFilter] = useState("");
  const [dsLocFilter, setDsLocFilter] = useState("");
  const [dsOffset, setDsOffset] = useState(0);

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

  const ROW_BG: Record<string, string> = {
    urgent: "bg-red-50 dark:bg-red-950/20",
    watch: "bg-yellow-50 dark:bg-yellow-950/20",
    normal: "",
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Above Plan</p>
          <p className="text-xl font-bold text-green-600">{(summary?.above_plan_count ?? 0).toLocaleString()}</p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Below Plan</p>
          <p className="text-xl font-bold text-red-600">{(summary?.below_plan_count ?? 0).toLocaleString()}</p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Urgent Alerts</p>
          <p className={`text-xl font-bold ${(summary?.urgent_count ?? 0) > 0 ? "text-red-600" : "text-foreground"}`}>
            {(summary?.urgent_count ?? 0).toLocaleString()}
          </p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Projected Stockouts</p>
          <p className={`text-xl font-bold ${(summary?.projected_stockouts ?? 0) > 0 ? "text-orange-600" : "text-foreground"}`}>
            {(summary?.projected_stockouts ?? 0).toLocaleString()}
          </p>
        </div>
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
            {t ? t.replace(/_/g, " ").replace(/\b\w/g, (c: string) => c.toUpperCase()) : "All Types"}
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
      ) : (
        <>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b text-muted-foreground">
                  <th className="text-left py-1 pr-2">Item No</th>
                  <th className="text-left py-1 pr-2">Location</th>
                  <th className="text-center py-1 pr-2">Signal Type</th>
                  <th className="text-right py-1 pr-2">Demand vs Fcst %</th>
                  <th className="text-center py-1 pr-2">Alert Priority</th>
                  <th className="text-right py-1 pr-2">On Hand</th>
                  <th className="text-center py-1">Below SS</th>
                </tr>
              </thead>
              <tbody>
                {(signals?.rows ?? []).length === 0 ? (
                  <tr>
                    <td colSpan={7} className="py-4 text-center text-muted-foreground">
                      No demand signals found.
                    </td>
                  </tr>
                ) : (
                  (signals?.rows ?? []).map((r: DemandSignalRow, i: number) => (
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
                          {r.signal_type.replace(/_/g, " ")}
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
                      <td className="py-1 pr-2 text-right">{fmt(r.current_on_hand, 0)}</td>
                      <td className="py-1 text-center">
                        {r.is_below_ss ? (
                          <span className="px-1 py-0.5 rounded text-xs bg-red-100 text-red-800">Yes</span>
                        ) : (
                          <span className="text-muted-foreground">-</span>
                        )}
                      </td>
                    </tr>
                  ))
                )}
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
