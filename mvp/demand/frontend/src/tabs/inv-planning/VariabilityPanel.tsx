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

function fmtPct(n: number | null | undefined): string {
  if (n == null) return "—";
  return `${Number(n).toFixed(1)}%`;
}

export function VariabilityPanel() {
  const { data: summary, isLoading } = useQuery({
    queryKey: queryKeys.variabilitySummary({}),
    queryFn: () => fetchVariabilitySummary({}),
    staleTime: STALE.FIVE_MIN,
  });

  const { data: volatile } = useQuery({
    queryKey: queryKeys.variabilityDetail({ variability_class: "high", limit: 10 }),
    queryFn: () => fetchVariabilityDetail({ variability_class: "high", limit: 10 }),
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
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Stable Items</p>
          <p className="text-xl font-bold text-green-600">
            {isLoading ? "..." : (summary?.by_class.low ?? 0).toLocaleString()}
          </p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Volatile Items</p>
          <p className={`text-xl font-bold ${(summary?.by_class.high ?? 0) > 0 ? "text-red-600" : "text-foreground"}`}>
            {isLoading ? "..." : ((summary?.by_class.high ?? 0) + (summary?.by_class.lumpy ?? 0)).toLocaleString()}
          </p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Avg CV</p>
          <p className="text-xl font-bold">
            {isLoading ? "..." : fmtPct(summary?.avg_cv != null ? summary.avg_cv * 100 : null)}
          </p>
        </div>
      </div>

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

      {volatile && volatile.rows.length > 0 && (
        <div className="overflow-x-auto">
          <p className="text-xs font-medium mb-2">Top Volatile Items</p>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b text-muted-foreground">
                <th className="text-left py-1 pr-2">Item No</th>
                <th className="text-left py-1 pr-2">Location</th>
                <th className="text-right py-1 pr-2">Demand CV</th>
                <th className="text-center py-1">Class</th>
              </tr>
            </thead>
            <tbody>
              {volatile.rows.map((r: VariabilityDetailRow, i: number) => (
                <tr key={`${r.item_no}-${r.loc}-${i}`} className="border-b last:border-0 hover:bg-muted/30">
                  <td className="py-1 pr-2 font-mono">{r.item_no}</td>
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
