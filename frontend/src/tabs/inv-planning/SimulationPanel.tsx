import { useState, useRef, useEffect } from "react";
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
import { FlaskConical } from "lucide-react";
import {
  simulationKeys,
  fetchSimulationResults,
  runSimulation,
  STALE,
  type SimulationResult,
} from "@/api/queries";

import { KpiCard } from "@/components/KpiCard";
import { EmptyState } from "@/components/EmptyState";
import { formatFixed } from "@/lib/formatters";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";

const PANEL_KPI = "rounded-lg bg-muted/30 p-3";

export function SimulationPanel() {
  const { filters: globalFilters } = useGlobalFilterContext();
  const queryClient = useQueryClient();
  const [simItemNo, setSimItemNo] = useState("");
  const [simLoc, setSimLoc] = useState("");
  const [simResult, setSimResult] = useState<SimulationResult | null>(null);
  const [targetCsl, setTargetCsl] = useState(95);

  const syncedGlobalRef = useRef<string>("");
  useEffect(() => {
    const key = `${globalFilters.item.join(",")}_${globalFilters.location.join(",")}`;
    if (key === syncedGlobalRef.current) return;
    syncedGlobalRef.current = key;
    if (globalFilters.item.length === 1) setSimItemNo(globalFilters.item[0]);
    if (globalFilters.location.length === 1) setSimLoc(globalFilters.location[0]);
  }, [globalFilters.item, globalFilters.location]);

  const { data: recentRuns, isLoading: runsLoading } = useQuery({
    queryKey: simulationKeys.results({ limit: 10 }),
    queryFn: () => fetchSimulationResults({ limit: 10 }),
    staleTime: STALE.FIVE_MIN,
  });

  const runMutation = useMutation({
    mutationFn: (body: { item_no: string; loc: string; target_csl?: number }) => runSimulation(body),
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
          placeholder="e.g. 100320"
          value={simItemNo}
          onChange={(e) => setSimItemNo(e.target.value)}
        />
        <input
          className="h-8 rounded border border-input bg-background px-2 text-xs w-36"
          placeholder="e.g. 1401-BULK"
          value={simLoc}
          onChange={(e) => setSimLoc(e.target.value)}
        />
        <select
          className="h-8 rounded border border-input bg-background px-2 text-xs"
          value={targetCsl}
          onChange={(e) => setTargetCsl(Number(e.target.value))}
          title="Service Level: % of demand fulfilled without stockout. Higher = more safety stock."
        >
          <option value={85}>SL 85%</option>
          <option value={90}>SL 90%</option>
          <option value={95}>SL 95%</option>
          <option value={98}>SL 98%</option>
          <option value={99}>SL 99%</option>
        </select>
        <button
          className="h-8 px-4 text-xs rounded bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          disabled={runMutation.isPending || !simItemNo || !simLoc}
          onClick={() => runMutation.mutate({ item_no: simItemNo, loc: simLoc, target_csl: targetCsl / 100 })}
        >
          {runMutation.isPending ? "Simulating (~20s)..." : "Run Simulation"}
        </button>
      </div>

      {activeResult && (
        <div className="space-y-4">
          <div className="grid grid-cols-3 gap-3">
            <KpiCard className={PANEL_KPI} label="Safety Stock Target (units)" value={formatFixed(activeResult.recommended_ss, 0)} />
            <KpiCard className={PANEL_KPI} label="Z-Score Formula (units)" value={formatFixed(activeResult.analytical_ss, 0)} />
            <KpiCard
              className={PANEL_KPI}
              label="Assessment"
              value={(activeResult.sim_vs_analytical_pct ?? 0) > 5 ? "Over-stocked" : (activeResult.sim_vs_analytical_pct ?? 0) < -5 ? "Conservative" : "Aligned"}
              colorClass={(activeResult.sim_vs_analytical_pct ?? 0) > 0 ? "text-red-600" : "text-green-600"}
            />
          </div>

          {activeResult.results_by_ss_level && activeResult.results_by_ss_level.length > 0 && (
            <div>
              <p className="text-xs font-medium mb-2">Fill-Rate Trade-Off Curve</p>
              <p className="text-xs text-muted-foreground mt-1">Higher safety stock → higher fulfillment probability. Use to find the stock level for your target service level.</p>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={activeResult.results_by_ss_level}
                    margin={{ top: 4, right: 16, left: 0, bottom: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="ss_qty" tick={{ fontSize: 10 }} label={{ value: "Safety Stock (units)", position: "insideBottomRight", offset: -5, fontSize: 9, fill: "hsl(var(--muted-foreground))" }} />
                    <YAxis
                      domain={[0, 100]}
                      tickFormatter={(v: number) => `${v}%`}
                      tick={{ fontSize: 10 }}
                      label={{ value: "Fulfillment Probability (%)", angle: -90, position: "insideLeft", fontSize: 9, fill: "hsl(var(--muted-foreground))" }}
                    />
                    <Tooltip formatter={(v: number) => [`${v.toFixed(1)}%`, "Fulfillment Probability"]} />
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
        ) : (recentRuns?.rows ?? []).length === 0 && !runMutation.isPending ? (
          <EmptyState
            icon={FlaskConical}
            title="No simulation results yet"
            description="Monte Carlo simulation runs 10,000 random demand and lead-time scenarios from your historical data to compute probability-based safety stock targets. The trade-off curve shows how much stock you need to achieve each fill-rate level (e.g., 95% fulfillment). Compare to the Z-score formula to find over- or under-stocking."
            steps={[
              { label: "Enter item and location in the fields above", command: "Item No: e.g. 100320  |  Loc: e.g. 1401-BULK" },
              { label: "Click Run Simulation to start", command: "(no make command — runs interactively in the browser)" },
            ]}
          />
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
                {(recentRuns?.rows ?? []).map((r: SimulationResult) => (
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
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
