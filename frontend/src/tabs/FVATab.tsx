/**
 * Forecast Value Added (FVA) & ROI Tracking Tab (Spec 08-07).
 *
 * Shows: FVA waterfall chart, intervention timeline, ROI KPI cards.
 */
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { fetchFVAWaterfall, fetchFVAROI, fetchFVAInterventions, fvaKeys, STALE_PLATFORM } from "@/api/queries";
import { KpiCard } from "@/components/KpiCard";
import { useThemeContext } from "@/context/ThemeContext";

const COLORS = { external: "#94a3b8", champion: "#3b82f6", ceiling: "#10b981", planner: "#8b5cf6" };

export default function FVATab() {
  const { theme } = useThemeContext();
  const [months, setMonths] = useState(12);

  const { data: waterfall } = useQuery({
    queryKey: fvaKeys.waterfall(months),
    queryFn: () => fetchFVAWaterfall(months),
    staleTime: STALE_PLATFORM,
  });

  const { data: roi } = useQuery({
    queryKey: fvaKeys.roi(months),
    queryFn: () => fetchFVAROI(months),
    staleTime: STALE_PLATFORM,
  });

  const { data: interventions } = useQuery({
    queryKey: fvaKeys.interventions,
    queryFn: () => fetchFVAInterventions(20),
    staleTime: STALE_PLATFORM,
  });

  const waterfallModels = waterfall?.waterfall?.models ?? [];
  const chartData = waterfallModels
    .filter((m: { accuracy_pct: number | null }) => m.accuracy_pct !== null)
    .map((m: { model_id: string; accuracy_pct: number }) => ({
      name: m.model_id,
      accuracy: Math.round(m.accuracy_pct * 100) / 100,
      fill: COLORS[m.model_id as keyof typeof COLORS] ?? "#64748b",
    }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-foreground">Forecast Value Added</h2>
          <p className="text-sm text-muted-foreground">Track the business impact of planning interventions</p>
        </div>
        <select
          value={months}
          onChange={(e) => setMonths(Number(e.target.value))}
          className="rounded border border-border bg-card px-3 py-1.5 text-sm"
        >
          {[3, 6, 12, 24].map((m) => (
            <option key={m} value={m}>{m} months</option>
          ))}
        </select>
      </div>

      {/* ROI KPI Cards */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <KpiCard label="Total Interventions" value={roi?.total_interventions ?? 0} />
        <KpiCard label="Measured" value={roi?.measured ?? 0} />
        <KpiCard label="Estimated Impact" value={`$${((roi?.total_estimated_impact ?? 0) / 1000).toFixed(0)}K`} />
        <KpiCard label="Actual Impact" value={`$${((roi?.total_actual_impact ?? 0) / 1000).toFixed(0)}K`} />
      </div>

      {/* FVA Waterfall Chart */}
      <div className="rounded-lg border border-border bg-card p-4">
        <h3 className="mb-4 text-sm font-medium text-foreground">Model Accuracy Waterfall</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="accuracy" radius={[4, 4, 0, 0]}>
                {chartData.map((entry: { fill: string }, i: number) => (
                  <Cell key={i} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recent Interventions */}
      <div className="rounded-lg border border-border bg-card p-4">
        <h3 className="mb-3 text-sm font-medium text-foreground">Recent Interventions</h3>
        <div className="space-y-2">
          {(interventions?.interventions ?? []).slice(0, 10).map((iv: {
            intervention_id: number; intervention_type: string; resource_type: string;
            resource_id: string; status: string; created_at: string | null;
            financial_impact_estimate: number | null;
          }) => (
            <div key={iv.intervention_id} className="flex items-center justify-between rounded-md border border-border/50 px-3 py-2 text-sm">
              <div className="flex items-center gap-3">
                <span className={`inline-block h-2 w-2 rounded-full ${iv.status === "measured" ? "bg-green-500" : "bg-amber-500"}`} />
                <span className="font-medium">{iv.intervention_type}</span>
                <span className="text-muted-foreground">{iv.resource_type} {iv.resource_id}</span>
              </div>
              <div className="flex items-center gap-3">
                {iv.financial_impact_estimate != null && (
                  <span className="text-xs text-muted-foreground">
                    ${(iv.financial_impact_estimate / 1000).toFixed(0)}K est.
                  </span>
                )}
                <span className="text-xs text-muted-foreground">{iv.created_at?.slice(0, 10)}</span>
              </div>
            </div>
          ))}
          {(!interventions?.interventions || interventions.interventions.length === 0) && (
            <p className="py-4 text-center text-sm text-muted-foreground">No interventions recorded yet</p>
          )}
        </div>
      </div>
    </div>
  );
}
