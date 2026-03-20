import { useQuery } from "@tanstack/react-query";
import {
  queryKeys,
  fetchLtSummary,
  fetchLtProfile,
  STALE,
  type LtProfileRow,
} from "@/api/queries";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";

import { KpiCard } from "@/components/KpiCard";
import { EmptyState } from "@/components/EmptyState";
import { formatFixed, formatPct } from "@/lib/formatters";
import { Timer } from "lucide-react";

const PANEL_KPI = "rounded-lg bg-muted/30 p-3";

export function LeadTimePanel() {
  const { filters } = useGlobalFilterContext();
  const gf = {
    brand: filters.brand.length > 0 ? filters.brand.join(",") : undefined,
    category: filters.category.length > 0 ? filters.category.join(",") : undefined,
    market: filters.market.length > 0 ? filters.market.join(",") : undefined,
    item: filters.item.length === 1 ? filters.item[0] : undefined,
    location: filters.location.length === 1 ? filters.location[0] : undefined,
  };

  const { data: summary, isLoading } = useQuery({
    queryKey: queryKeys.ltSummary(gf),
    queryFn: () => fetchLtSummary({}),
    staleTime: STALE.FIVE_MIN,
  });

  const { data: volatile } = useQuery({
    queryKey: queryKeys.ltProfile({ ...gf, lt_variability_class: "volatile", limit: 10 }),
    queryFn: () => fetchLtProfile({ item: gf.item, location: gf.location, lt_variability_class: "volatile", limit: 10 }),
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
          sublabel="(lower = more reliable)"
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

      {!isLoading && (!volatile || volatile.rows.length === 0) && (
        <EmptyState
          icon={Timer}
          title="No lead time profiles computed"
          description="Lead time profiles measure delivery reliability per item-location: mean lead time, standard deviation, CV, and percentile bands (P25/P50/P75/P95). Stable (CV < 0.20), Moderate (CV 0.20–0.40), Volatile (CV > 0.40) — volatile suppliers are flagged for safety stock uplift."
          steps={[
            { label: "Load inventory snapshots (source data)", command: "make load-inventory" },
            { label: "Compute lead time variability profiles", command: "make lt-variability-compute" },
          ]}
        />
      )}

      {(volatile?.rows?.length ?? 0) > 0 && volatile && (
        <div className="overflow-x-auto">
          <div className="text-xs p-2 rounded bg-muted/30 border mb-3">
            <span className="font-medium text-foreground">Lead Time CV (Coefficient of Variation = std dev ÷ mean): </span>
            <span className="text-green-600">● Stable &lt; 0.20</span> ·{" "}
            <span className="text-amber-600 ml-1">● Moderate 0.20–0.40</span> ·{" "}
            <span className="text-red-600 ml-1">● Volatile &gt; 0.40</span>
            <span className="ml-2 text-muted-foreground">Lower = more predictable supplier delivery times.</span>
          </div>
          <p className="text-xs font-medium mb-2">Top Volatile Lead Time Items</p>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b text-muted-foreground">
                <th className="text-left py-1 pr-2">Item No</th>
                <th className="text-left py-1 pr-2">Location</th>
                <th className="text-right py-1 pr-2">LT Mean (days)</th>
                <th className="text-right py-1 pr-2">LT Std</th>
                <th className="text-right py-1 pr-2" title="Coefficient of Variation: standard deviation ÷ mean lead time. Lower = more reliable supplier.">CV</th>
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
