import { useQuery } from "@tanstack/react-query";
import {
  queryKeys,
  fetchLtSummary,
  fetchLtProfile,
  STALE,
  type LtProfileRow,
} from "@/api/queries";

import { KpiCard } from "@/components/KpiCard";
import { formatFixed, formatPct } from "@/lib/formatters";

const PANEL_KPI = "rounded-lg bg-muted/30 p-3";

export function LeadTimePanel() {
  const { data: summary, isLoading } = useQuery({
    queryKey: queryKeys.ltSummary({}),
    queryFn: () => fetchLtSummary({}),
    staleTime: STALE.FIVE_MIN,
  });

  const { data: volatile } = useQuery({
    queryKey: queryKeys.ltProfile({ lt_variability_class: "volatile", limit: 10 }),
    queryFn: () => fetchLtProfile({ lt_variability_class: "volatile", limit: 10 }),
    staleTime: STALE.FIVE_MIN,
  });

  const classData = summary
    ? [
        { label: "Stable", count: summary.by_class.stable, color: "text-green-600" },
        { label: "Moderate", count: summary.by_class.moderate, color: "text-amber-600" },
        { label: "Volatile", count: summary.by_class.volatile, color: "text-red-600" },
      ]
    : [];

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-3">
        <KpiCard
          className={PANEL_KPI}
          label="Avg Lead Time"
          value={isLoading ? "..." : summary?.avg_lt_mean_days != null ? `${formatFixed(summary.avg_lt_mean_days)} days` : "-"}
        />
        <KpiCard
          className={PANEL_KPI}
          label="Volatile Suppliers"
          value={isLoading ? "..." : (summary?.by_class.volatile ?? 0).toLocaleString()}
          colorClass={(summary?.by_class.volatile ?? 0) > 0 ? "text-red-600" : undefined}
        />
        <KpiCard
          className={PANEL_KPI}
          label="Avg LT CV"
          value={isLoading ? "..." : formatPct(summary?.avg_lt_cv != null ? summary.avg_lt_cv * 100 : null)}
        />
      </div>

      {classData.length > 0 && (
        <div className="flex gap-4 text-xs">
          {classData.map((d) => (
            <div key={d.label} className="rounded-lg border bg-muted/30 px-4 py-2 text-center">
              <p className="text-muted-foreground">{d.label}</p>
              <p className={`text-lg font-bold ${d.color}`}>{d.count.toLocaleString()}</p>
            </div>
          ))}
        </div>
      )}

      {volatile && volatile.rows.length > 0 && (
        <div className="overflow-x-auto">
          <p className="text-xs font-medium mb-2">Top Volatile Lead Time Items</p>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b text-muted-foreground">
                <th className="text-left py-1 pr-2">Item No</th>
                <th className="text-left py-1 pr-2">Location</th>
                <th className="text-right py-1 pr-2">LT Mean (days)</th>
                <th className="text-right py-1 pr-2">LT Std</th>
                <th className="text-right py-1 pr-2">CV</th>
                <th className="text-center py-1">Class</th>
              </tr>
            </thead>
            <tbody>
              {volatile.rows.map((r: LtProfileRow, i: number) => (
                <tr key={`${r.item_no}-${r.loc}-${i}`} className="border-b last:border-0 hover:bg-muted/30">
                  <td className="py-1 pr-2 font-mono">{r.item_no}</td>
                  <td className="py-1 pr-2">{r.loc}</td>
                  <td className="py-1 pr-2 text-right">{r.lt_mean_days != null ? formatFixed(r.lt_mean_days) : "—"}</td>
                  <td className="py-1 pr-2 text-right">{r.lt_std_days != null ? formatFixed(r.lt_std_days) : "—"}</td>
                  <td className="py-1 pr-2 text-right">
                    {r.lt_cv != null ? formatPct(Number(r.lt_cv) * 100) : "—"}
                  </td>
                  <td className="py-1 text-center">
                    <span
                      className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                        r.lt_variability_class === "volatile"
                          ? "bg-red-100 text-red-800"
                          : r.lt_variability_class === "moderate"
                          ? "bg-amber-100 text-amber-800"
                          : "bg-green-100 text-green-800"
                      }`}
                    >
                      {r.lt_variability_class ?? "-"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
