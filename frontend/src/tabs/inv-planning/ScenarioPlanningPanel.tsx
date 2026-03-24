/**
 * ScenarioPlanningPanel — F4.4 Supply Chain Scenario Planning
 * Shows disruption scenarios with financial impact assessment.
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Zap, TrendingDown, AlertTriangle, DollarSign } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { EmptyState } from "@/components/EmptyState";
import {
  scenarioKeys,
  fetchSupplyScenarios,
  fetchScenarioResults,
  STALE_EVO,
} from "@/api/queries/evolution";

const STATUS_COLORS: Record<string, string> = {
  completed: "bg-green-100 text-green-700",
  running: "bg-blue-100 text-blue-700",
  queued: "bg-amber-100 text-amber-700",
  failed: "bg-red-100 text-red-700",
  draft: "bg-gray-100 text-gray-600",
};

const DISRUPTION_LABELS: Record<string, string> = {
  supplier_delay: "Supplier Delay",
  capacity_constraint: "Capacity Constraint",
  demand_shock: "Demand Shock",
  transport_disruption: "Transport Disruption",
  quality_hold: "Quality Hold",
};

function fmtCurrency(v: number | null | undefined) {
  if (v == null) return "—";
  return `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

export function ScenarioPlanningPanel() {
  const [disruptionType, setDisruptionType] = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const { data, isLoading } = useQuery({
    queryKey: scenarioKeys.list({ disruption_type: disruptionType || undefined, status: statusFilter || undefined }),
    queryFn: () => fetchSupplyScenarios({ disruption_type: disruptionType || undefined, status: statusFilter || undefined }),
    staleTime: STALE_EVO.ONE_MIN,
  });

  const { data: results, isLoading: resultsLoading } = useQuery({
    queryKey: scenarioKeys.results(selectedId ?? ""),
    queryFn: () => fetchScenarioResults(selectedId!),
    enabled: !!selectedId,
    staleTime: STALE_EVO.FIVE_MIN,
  });

  const scenarios = data?.scenarios ?? [];
  const total = data?.total ?? 0;

  const completedCount = scenarios.filter((s) => s.status === "completed").length;
  const totalImpact = results?.total_impact ?? null;
  const totalStockoutDays = results?.total_stockout_days ?? null;

  return (
    <div className="space-y-6 p-4">
      {/* KPI row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: "Total Scenarios", value: total.toString(), icon: <Zap size={16} /> },
          { label: "Completed", value: completedCount.toString(), icon: <Zap size={16} /> },
          { label: "Total Stockout Days", value: selectedId ? (totalStockoutDays?.toFixed(1) ?? "—") : "Select scenario", icon: <TrendingDown size={16} />, warn: (totalStockoutDays ?? 0) > 0 },
          { label: "Total Impact", value: selectedId ? fmtCurrency(totalImpact) : "Select scenario", icon: <DollarSign size={16} />, warn: (totalImpact ?? 0) > 0 },
        ].map((c) => (
          <Card key={c.label} className={c.warn ? "border-amber-400" : ""}>
            <CardHeader className="pb-2 flex flex-row items-center justify-between">
              <CardTitle className="text-sm font-medium text-muted-foreground">{c.label}</CardTitle>
              <span className={c.warn ? "text-amber-500" : "text-muted-foreground"}>{c.icon}</span>
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold">{c.value}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3 items-end">
        <div>
          <label className="text-xs font-medium block mb-1">Disruption Type</label>
          <select className="border rounded px-2 py-1 text-sm" value={disruptionType} onChange={(e) => setDisruptionType(e.target.value)}>
            <option value="">All Types</option>
            {Object.entries(DISRUPTION_LABELS).map(([k, v]) => (
              <option key={k} value={k}>{v}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="text-xs font-medium block mb-1">Status</label>
          <select className="border rounded px-2 py-1 text-sm" value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
            <option value="">All Statuses</option>
            <option value="completed">Completed</option>
            <option value="running">Running</option>
            <option value="queued">Queued</option>
            <option value="draft">Draft</option>
          </select>
        </div>
      </div>

      {/* Scenarios list + results side by side */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Scenario list */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Scenarios ({total})</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            {isLoading ? (
              <p className="p-4 text-sm text-muted-foreground">Loading…</p>
            ) : !scenarios.length ? (
              <div className="p-6">
                <EmptyState
                  icon={Zap}
                  title="No supply chain scenarios"
                  description="Scenario planning models the financial impact of supply disruptions: supplier delays, port closures, demand shocks, or capacity constraints. Each scenario computes projected stockout days and revenue at risk per item-location."
                  steps={[
                    { label: "Compute safety stock and replenishment plan first", command: "make ss-compute && make replen-plan-compute" },
                    { label: "Create a scenario via the New Scenario button above", command: "(no CLI command — scenarios are created in the UI)" },
                  ]}
                />
              </div>
            ) : (
              <ul className="divide-y">
                {scenarios.map((s) => (
                  <li
                    key={s.scenario_id}
                    className={`p-3 cursor-pointer hover:bg-muted/40 transition-colors ${selectedId === s.scenario_id ? "bg-muted/60" : ""}`}
                    onClick={() => setSelectedId(s.scenario_id)}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-medium text-sm">{s.scenario_name}</span>
                      <span className={`text-xs px-1.5 py-0.5 rounded ${STATUS_COLORS[s.status] ?? "bg-gray-100 text-gray-600"}`}>
                        {s.status}
                      </span>
                    </div>
                    <div className="flex gap-3 text-xs text-muted-foreground">
                      <span>{DISRUPTION_LABELS[s.disruption_type] ?? s.disruption_type}</span>
                      <span>Impact: {s.impact_pct}%</span>
                      <span>Duration: {s.duration_weeks}w</span>
                    </div>
                    {(s.item_id || s.loc) && (
                      <p className="text-xs text-muted-foreground mt-0.5 font-mono">
                        {s.item_id ?? "All"} / {s.loc ?? "All"}
                      </p>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>

        {/* Results panel */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">
              {selectedId ? "Impact Results" : "Select a Scenario"}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {!selectedId ? (
              <p className="text-sm text-muted-foreground">Click a scenario to view results.</p>
            ) : resultsLoading ? (
              <p className="text-sm text-muted-foreground">Loading results…</p>
            ) : !results?.items?.length ? (
              <p className="text-sm text-muted-foreground">No results computed yet. Run the scenario first.</p>
            ) : (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div className="bg-red-50 p-3 rounded">
                    <p className="text-xs text-muted-foreground">Total Stockout Days</p>
                    <p className="text-xl font-bold text-red-600">{results.total_stockout_days.toFixed(1)}</p>
                  </div>
                  <div className="bg-amber-50 p-3 rounded">
                    <p className="text-xs text-muted-foreground">Total Financial Impact</p>
                    <p className="text-xl font-bold text-amber-600">{fmtCurrency(results.total_impact)}</p>
                  </div>
                </div>
                <div className="overflow-y-auto max-h-64">
                  <table className="w-full text-xs">
                    <thead className="bg-muted/50 text-xs uppercase text-muted-foreground sticky top-0">
                      <tr>
                        {["Item", "Loc", "Adj LT", "Supply Red.", "Stockout d", "Impact $"].map((h) => (
                          <th key={h} className="px-2 py-1.5 text-left">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {results.items.map((r, i) => (
                        <tr key={i} className={`border-t ${(r.stockout_days ?? 0) > 0 ? "bg-red-50" : ""}`}>
                          <td className="px-2 py-1 font-mono">{r.item_id}</td>
                          <td className="px-2 py-1">{r.loc}</td>
                          <td className="px-2 py-1">{r.adjusted_lt_days?.toFixed(1) ?? "—"}</td>
                          <td className="px-2 py-1 text-red-600">{r.supply_reduction?.toFixed(0) ?? "—"}</td>
                          <td className={`px-2 py-1 font-medium ${(r.stockout_days ?? 0) > 0 ? "text-red-600" : ""}`}>
                            {r.stockout_days?.toFixed(1) ?? "0"}
                          </td>
                          <td className="px-2 py-1">{fmtCurrency(r.total_impact)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
