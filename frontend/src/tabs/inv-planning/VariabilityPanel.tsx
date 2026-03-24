import { useQuery } from "@tanstack/react-query";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  queryKeys,
  fetchVariabilitySummary,
  fetchVariabilityDetail,
  STALE,
  type VariabilityDetailRow,
} from "@/api/queries";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";

import { KpiCard } from "@/components/KpiCard";
import { EmptyState } from "@/components/EmptyState";
import { formatPct } from "@/lib/formatters";
import { BarChart2 } from "lucide-react";

const PANEL_KPI = "rounded-lg bg-muted/30 p-3";

export function VariabilityPanel() {
  const { filters } = useGlobalFilterContext();
  const gf = {
    brand: filters.brand.length > 0 ? filters.brand.join(",") : undefined,
    category: filters.category.length > 0 ? filters.category.join(",") : undefined,
    market: filters.market.length > 0 ? filters.market.join(",") : undefined,
    item: filters.item.length === 1 ? filters.item[0] : undefined,
    location: filters.location.length === 1 ? filters.location[0] : undefined,
  };

  const { data: summary, isLoading } = useQuery({
    queryKey: queryKeys.variabilitySummary(gf),
    queryFn: () => fetchVariabilitySummary({}),
    staleTime: STALE.FIVE_MIN,
  });

  const { data: volatile } = useQuery({
    queryKey: queryKeys.variabilityDetail({ ...gf, variability_class: "high", limit: 10 }),
    queryFn: () => fetchVariabilityDetail({ item: gf.item, location: gf.location, variability_class: "high", limit: 10 }),
    staleTime: STALE.FIVE_MIN,
  });

  const pieData = summary
    ? [
        { name: "Stable", value: summary.by_class.low, color: "#22c55e" },
        { name: "Moderate", value: summary.by_class.medium, color: "#f59e0b" },
        {
          name: "Volatile",
          value: (summary.by_class.high ?? 0) + (summary.by_class.lumpy ?? 0),
          color: "#ef4444",
        },
      ].filter((d) => d.value > 0)
    : [];

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-3">
        <KpiCard className={PANEL_KPI} label="Stable Items" value={isLoading ? "..." : (summary?.by_class.low ?? 0).toLocaleString()} colorClass="text-green-600" />
        <KpiCard
          className={PANEL_KPI}
          label="Volatile Items"
          value={isLoading ? "..." : ((summary?.by_class.high ?? 0) + (summary?.by_class.lumpy ?? 0)).toLocaleString()}
          colorClass={(summary?.by_class.high ?? 0) > 0 ? "text-red-600" : undefined}
        />
        <KpiCard
          className={PANEL_KPI}
          label="Avg CV"
          sublabel="(lower = more stable)"
          value={isLoading ? "..." : formatPct(summary?.avg_cv != null ? summary.avg_cv * 100 : null)}
        />
      </div>

      {!isLoading && (!summary || summary.total_skus === 0) && (
        <EmptyState
          icon={BarChart2}
          title="Demand variability not yet profiled"
          description="Demand variability profiles classify each DFU as Stable (CV < 0.30), Moderate (CV 0.30–0.80), Volatile (CV > 0.80), or Lumpy (intermittent, >30% zero months). These profiles feed safety stock and replenishment policy assignment."
          steps={[
            { label: "Run clustering (populates dim_sku variability_class)", command: "make cluster-all" },
            { label: "Alternatively compute variability only", command: "make variability-compute" },
          ]}
        />
      )}

      {pieData.length > 0 && (
        <div className="flex items-center gap-6">
          <div className="h-36 w-36 shrink-0">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={pieData} cx="50%" cy="50%" innerRadius={36} outerRadius={56} paddingAngle={2} dataKey="value">
                  {pieData.map((entry) => (
                    <Cell key={entry.name} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip formatter={(v: number) => [v.toLocaleString(), "Items"]} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex flex-col gap-1.5 text-xs">
            {pieData.map((d) => (
              <div key={d.name} className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-sm shrink-0" style={{ background: d.color }} />
                <span className="text-foreground">{d.name}</span>
                <span className="text-muted-foreground ml-auto pl-4">{d.value.toLocaleString()}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {volatile?.rows?.length > 0 && (
        <div className="overflow-x-auto">
          <div className="text-xs p-2 rounded bg-muted/30 border mb-3">
            <span className="font-medium text-foreground">Demand CV (Coefficient of Variation = std dev ÷ mean): </span>
            <span className="text-green-600">● Stable &lt; 0.30</span> ·{" "}
            <span className="text-amber-600 ml-1">● Moderate 0.30–0.80</span> ·{" "}
            <span className="text-red-600 ml-1">● Volatile &gt; 0.80</span> ·{" "}
            <span className="text-purple-600 ml-1">● Lumpy (intermittent, &gt;30% zero months)</span>
            <span className="ml-2 text-muted-foreground">Lower = more predictable demand. Higher = more safety stock needed.</span>
          </div>
          <p className="text-xs font-medium mb-2">Top Volatile Items</p>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b text-muted-foreground">
                <th className="text-left py-1 pr-2">Item No</th>
                <th className="text-left py-1 pr-2">Location</th>
                <th className="text-right py-1 pr-2" title="Coefficient of Variation: standard deviation ÷ mean demand. Lower = more predictable demand, less safety stock needed.">Demand CV</th>
                <th className="text-center py-1">Class</th>
              </tr>
            </thead>
            <tbody>
              {volatile.rows.map((r: VariabilityDetailRow, i: number) => (
                <tr key={`${r.item_id}-${r.loc}-${i}`} className="border-b last:border-0 hover:bg-muted/30">
                  <td className="py-1 pr-2 font-mono">{r.item_id}</td>
                  <td className="py-1 pr-2">{r.loc}</td>
                  <td className="py-1 pr-2 text-right">
                    {r.demand_cv != null ? (r.demand_cv * 100).toFixed(1) + "%" : "-"}
                  </td>
                  <td className="py-1 text-center">
                    <span
                      className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                        r.variability_class === "high" || r.variability_class === "lumpy"
                          ? "bg-red-100 text-red-800"
                          : r.variability_class === "medium"
                          ? "bg-amber-100 text-amber-800"
                          : "bg-green-100 text-green-800"
                      }`}
                    >
                      {r.variability_class ?? "-"}
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
