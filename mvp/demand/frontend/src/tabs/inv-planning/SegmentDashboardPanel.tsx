import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  insightKeys,
  fetchSegmentDashboard,
  STALE_INSIGHTS,
} from "@/api/queries";
import { KpiCard } from "@/components/KpiCard";
import { EmptyState } from "@/components/EmptyState";
import { formatFixed, formatPct, formatInt } from "@/lib/formatters";
import { getSeverityConfig } from "@/constants/severity";
import { LayoutGrid, ChevronDown } from "lucide-react";

const PANEL_KPI = "rounded-lg bg-muted/30 p-3";
const SEGMENTS = ["AX", "AY", "AZ", "BX", "BY", "BZ", "CX", "CY", "CZ"];

export function SegmentDashboardPanel() {
  const [segment, setSegment] = useState("AX");

  const { data, isLoading, error } = useQuery({
    queryKey: insightKeys.segmentDashboard(segment),
    queryFn: () => fetchSegmentDashboard(segment),
    staleTime: STALE_INSIGHTS.FIVE_MIN,
  });

  if (error) {
    return (
      <div className="text-xs text-red-600 p-4">
        Failed to load segment dashboard: {(error as Error).message}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="text-xs text-muted-foreground bg-muted/20 border rounded px-3 py-2">
        Deep-dive into a single ABC-XYZ segment. Select a segment to view KPIs, policy distribution,
        top exceptions, and recommended actions tailored to that segment profile.
      </div>

      {/* Segment selector */}
      <div className="flex items-center gap-2">
        <label className="text-xs text-muted-foreground font-medium">Segment:</label>
        <div className="relative">
          <select
            className="h-8 rounded border border-input bg-background px-3 pr-7 text-xs appearance-none cursor-pointer"
            value={segment}
            onChange={(e) => setSegment(e.target.value)}
          >
            {SEGMENTS.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
          <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground pointer-events-none" />
        </div>
        {/* Quick-select pills */}
        <div className="flex gap-1 ml-2">
          {SEGMENTS.map((s) => (
            <button
              key={s}
              className={`px-2 py-0.5 text-[10px] rounded border transition-colors ${
                s === segment
                  ? "bg-foreground text-background border-foreground"
                  : "border-border hover:bg-accent"
              }`}
              onClick={() => setSegment(s)}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <KpiCard
          className={PANEL_KPI}
          label="DFU Count"
          value={isLoading ? "..." : formatInt(data?.dfu_count)}
        />
        <KpiCard
          className={PANEL_KPI}
          label="Open Exceptions"
          value={isLoading ? "..." : formatInt(data?.open_exceptions)}
          colorClass={(data?.open_exceptions ?? 0) > 0 ? "text-orange-600" : undefined}
        />
        <KpiCard
          className={PANEL_KPI}
          label="Below SS"
          value={isLoading ? "..." : formatInt(data?.below_ss_count)}
          colorClass={(data?.below_ss_count ?? 0) > 0 ? "text-red-600" : undefined}
        />
        <KpiCard
          className={PANEL_KPI}
          label="Avg Fill Rate"
          value={isLoading ? "..." : formatPct(data?.avg_fill_rate != null ? data.avg_fill_rate * 100 : null)}
        />
        <KpiCard
          className={PANEL_KPI}
          label="Avg Health Score"
          value={isLoading ? "..." : formatFixed(data?.avg_health_score)}
        />
      </div>

      {isLoading ? (
        <p className="text-xs text-muted-foreground">Loading segment data...</p>
      ) : !data || data.dfu_count === 0 ? (
        <EmptyState
          icon={LayoutGrid}
          title={`No data for segment ${segment}`}
          description="This segment has no DFUs assigned. Ensure ABC-XYZ classification has been computed."
          steps={[
            { label: "Compute ABC-XYZ classification", command: "make abc-xyz-compute" },
          ]}
        />
      ) : (
        <>
          {/* Policy distribution */}
          {(data.policy_distribution?.length ?? 0) > 0 && (
            <div>
              <p className="text-xs font-medium mb-2">Policy Distribution</p>
              <div className="flex flex-wrap gap-2">
                {data.policy_distribution.map((p) => (
                  <div
                    key={p.policy_id}
                    className="flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs"
                  >
                    <span className="font-medium">{p.policy_id}</span>
                    <span className="text-muted-foreground">({p.count.toLocaleString()})</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Top exceptions */}
          {(data.top_exceptions?.length ?? 0) > 0 && (
            <div>
              <p className="text-xs font-medium mb-2">Top Exceptions</p>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b text-muted-foreground">
                      <th className="text-left py-1 pr-2">Item</th>
                      <th className="text-left py-1 pr-2">Location</th>
                      <th className="text-left py-1 pr-2">Type</th>
                      <th className="text-left py-1 pr-2">Severity</th>
                      <th className="text-left py-1">Detail</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.top_exceptions.map((exc, i) => {
                      const sevCfg = getSeverityConfig(exc.severity);
                      return (
                        <tr key={`${exc.item_no}-${exc.loc}-${i}`} className="border-b last:border-0 hover:bg-muted/40">
                          <td className="py-1 pr-2 font-mono">{exc.item_no}</td>
                          <td className="py-1 pr-2">{exc.loc}</td>
                          <td className="py-1 pr-2">{exc.exception_type}</td>
                          <td className="py-1 pr-2">
                            <span className={`text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded ${sevCfg.badge}`}>
                              {sevCfg.label}
                            </span>
                          </td>
                          <td className="py-1 text-muted-foreground">{exc.detail}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Recommended actions */}
          {(data.recommended_actions?.length ?? 0) > 0 && (
            <div>
              <p className="text-xs font-medium mb-2">Recommended Actions for {segment}</p>
              <ul className="space-y-1">
                {data.recommended_actions.map((action, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs text-muted-foreground">
                    <span className="inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-muted text-[10px] font-bold mt-0.5">
                      {i + 1}
                    </span>
                    {action}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </div>
  );
}
