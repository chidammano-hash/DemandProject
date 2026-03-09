import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  simulationKeys,
  fetchSimulationResults,
  runSimulation,
  STALE,
  type SimulationResult,
} from "@/api/queries";

import { KpiCard } from "@/components/KpiCard";
import { formatFixed } from "@/lib/formatters";

const PANEL_KPI = "rounded-lg bg-muted/30 p-3";

export function SimulationPanel() {
  const queryClient = useQueryClient();
  const [simItemNo, setSimItemNo] = useState("");
  const [simLoc, setSimLoc] = useState("");
  const [simResult, setSimResult] = useState<SimulationResult | null>(null);

  const { data: recentRuns, isLoading: runsLoading } = useQuery({
    queryKey: simulationKeys.results({ limit: 10 }),
    queryFn: () => fetchSimulationResults({ limit: 10 }),
    staleTime: STALE.FIVE_MIN,
  });

  const runMutation = useMutation({
    mutationFn: (body: { item_no: string; loc: string }) => runSimulation(body),
    onSuccess: (result) => {
      setSimResult(result);
      queryClient.invalidateQueries({ queryKey: simulationKeys.results() });
    },
  });

  const activeResult = simResult ?? recentRuns?.rows?.[0] ?? null;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2 items-center">
        <input
          className="h-8 rounded border border-input bg-background px-2 text-xs w-36"
          placeholder="Item No"
          value={simItemNo}
          onChange={(e) => setSimItemNo(e.target.value)}
        />
        <input
          className="h-8 rounded border border-input bg-background px-2 text-xs w-36"
          placeholder="Location"
          value={simLoc}
          onChange={(e) => setSimLoc(e.target.value)}
        />
        <button
          className="h-8 px-4 text-xs rounded bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          disabled={runMutation.isPending || !simItemNo || !simLoc}
          onClick={() => runMutation.mutate({ item_no: simItemNo, loc: simLoc })}
        >
          {runMutation.isPending ? "Running..." : "Run Simulation"}
        </button>
      </div>

      {activeResult && (
        <div className="space-y-4">
          <div className="grid grid-cols-3 gap-3">
            <KpiCard className={PANEL_KPI} label="Recommended SS" value={formatFixed(activeResult.recommended_ss, 0)} />
            <KpiCard className={PANEL_KPI} label="Analytical SS" value={formatFixed(activeResult.analytical_ss, 0)} />
            <KpiCard
              className={PANEL_KPI}
              label="Difference %"
              value={activeResult.sim_vs_analytical_pct != null
                ? `${activeResult.sim_vs_analytical_pct > 0 ? "+" : ""}${activeResult.sim_vs_analytical_pct.toFixed(1)}%`
                : "-"}
              colorClass={(activeResult.sim_vs_analytical_pct ?? 0) > 0 ? "text-red-600" : "text-green-600"}
            />
          </div>

          {activeResult.results_by_ss_level && activeResult.results_by_ss_level.length > 0 && (
            <div>
              <p className="text-xs font-medium mb-2">Service Level Curve</p>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={activeResult.results_by_ss_level}
                    margin={{ top: 4, right: 16, left: 0, bottom: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="ss_qty" tick={{ fontSize: 10 }} />
                    <YAxis
                      domain={[0, 100]}
                      tickFormatter={(v: number) => `${v}%`}
                      tick={{ fontSize: 10 }}
                    />
                    <Tooltip formatter={(v: number) => [`${v.toFixed(1)}%`, "CSL"]} />
                    <Line
                      type="monotone"
                      dataKey="csl"
                      stroke="hsl(220, 70%, 55%)"
                      dot={false}
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      )}

      <div>
        <p className="text-xs font-medium mb-2">Recent Simulation Runs</p>
        {runsLoading ? (
          <p className="text-xs text-muted-foreground">Loading...</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b text-muted-foreground">
                  <th className="text-left py-1 pr-2">Item</th>
                  <th className="text-left py-1 pr-2">Location</th>
                  <th className="text-left py-1 pr-2">Date</th>
                  <th className="text-right py-1 pr-2">Sim SS</th>
                  <th className="text-right py-1 pr-2">Analytical SS</th>
                  <th className="text-right py-1 pr-2">Diff %</th>
                  <th className="text-right py-1">Duration (s)</th>
                </tr>
              </thead>
              <tbody>
                {(recentRuns?.rows ?? []).length === 0 ? (
                  <tr>
                    <td colSpan={7} className="py-4 text-center text-muted-foreground">
                      No simulation runs. Use the form above to run a simulation.
                    </td>
                  </tr>
                ) : (
                  (recentRuns?.rows ?? []).map((r: SimulationResult) => (
                    <tr key={r.sim_run_id} className="border-b last:border-0 hover:bg-muted/30">
                      <td className="py-1 pr-2 font-mono">{r.item_no}</td>
                      <td className="py-1 pr-2">{r.loc}</td>
                      <td className="py-1 pr-2">{r.simulation_date?.slice(0, 10) ?? "-"}</td>
                      <td className="py-1 pr-2 text-right">{formatFixed(r.recommended_ss, 0)}</td>
                      <td className="py-1 pr-2 text-right">{formatFixed(r.analytical_ss, 0)}</td>
                      <td
                        className={`py-1 pr-2 text-right ${
                          (r.sim_vs_analytical_pct ?? 0) > 5 ? "text-red-600" : "text-foreground"
                        }`}
                      >
                        {r.sim_vs_analytical_pct != null ? `${r.sim_vs_analytical_pct.toFixed(1)}%` : "-"}
                      </td>
                      <td className="py-1 text-right">{r.run_duration_secs?.toFixed(1) ?? "-"}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
