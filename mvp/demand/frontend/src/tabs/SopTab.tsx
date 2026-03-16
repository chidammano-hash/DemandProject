/**
 * SopTab — F4.2 S&OP Cycle Automation
 * Shows active S&OP cycles, stage timeline, demand/supply gaps, and approved plan.
 */

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { CheckCircle, Clock, ChevronRight, AlertTriangle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  sopKeys,
  fetchSopCycles,
  fetchSopGaps,
  fetchApprovedPlan,
  STALE_EVO,
  type SopCycle,
  type SopGap,
} from "@/api/queries/evolution";

const STAGE_ORDER = ["demand_review", "supply_review", "pre_sop", "executive_sop", "approved", "closed"];

const STAGE_LABELS: Record<string, string> = {
  demand_review: "Demand Review",
  supply_review: "Supply Review",
  pre_sop: "Pre-S&OP",
  executive_sop: "Executive S&OP",
  approved: "Approved",
  closed: "Closed",
};

const SEVERITY_COLORS: Record<string, string> = {
  critical: "bg-red-100 text-red-700 border-red-200",
  high: "bg-orange-100 text-orange-700 border-orange-200",
  medium: "bg-amber-100 text-amber-700 border-amber-200",
  low: "bg-blue-100 text-blue-700 border-blue-200",
};

function StageTimeline({ currentStage }: { currentStage: string }) {
  const currentIdx = STAGE_ORDER.indexOf(currentStage);

  return (
    <div className="flex items-center gap-0">
      {STAGE_ORDER.map((stage, idx) => {
        const done = idx < currentIdx;
        const active = idx === currentIdx;
        return (
          <div key={stage} className="flex items-center">
            <div className={`flex flex-col items-center px-2 py-1 rounded ${active ? "bg-primary text-primary-foreground" : done ? "text-green-600" : "text-muted-foreground"}`}>
              <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold mb-1 ${active ? "bg-white text-primary" : done ? "bg-green-500 text-white" : "bg-muted"}`}>
                {done ? "✓" : idx + 1}
              </div>
              <span className="text-[10px] text-center leading-tight">{STAGE_LABELS[stage]}</span>
            </div>
            {idx < STAGE_ORDER.length - 1 && (
              <ChevronRight size={12} className={`mx-0.5 flex-shrink-0 ${done ? "text-green-500" : "text-muted-foreground/40"}`} />
            )}
          </div>
        );
      })}
    </div>
  );
}

function GapCard({ gap }: { gap: SopGap }) {
  return (
    <div className={`p-3 rounded border ${SEVERITY_COLORS[gap.severity] ?? "bg-gray-50 border-gray-200"}`}>
      <div className="flex items-start justify-between mb-1">
        <span className="font-medium text-sm">{gap.category}</span>
        <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${SEVERITY_COLORS[gap.severity]}`}>
          {gap.severity.toUpperCase()}
        </span>
      </div>
      <p className="text-xs text-muted-foreground mb-1">{gap.gap_type.replace(/_/g, " ")}</p>
      {gap.gap_qty && <p className="text-xs">Gap: {gap.gap_qty.toFixed(0)} units</p>}
      {gap.gap_value && <p className="text-xs">Value: ${gap.gap_value.toLocaleString(undefined, { maximumFractionDigits: 0 })}</p>}
      {gap.resolution_options && (
        <p className="text-xs mt-1 text-muted-foreground">{gap.resolution_options}</p>
      )}
      <span className={`text-xs mt-1 inline-block px-1.5 py-0.5 rounded ${gap.mitigation_status === "resolved" ? "bg-green-100 text-green-700" : "bg-gray-100 text-gray-600"}`}>
        {gap.mitigation_status}
      </span>
    </div>
  );
}

export default function SopTab() {
  const [selectedCycle, setSelectedCycle] = useState<SopCycle | null>(null);
  const [planMonth, setPlanMonth] = useState("");
  const [planItem, setPlanItem] = useState("");
  const [facilitatedBy, setFacilitatedBy] = useState("planner");
  const [approvedBy, setApprovedBy] = useState("");
  const [planVersion, setPlanVersion] = useState(() => `v${new Date().toISOString().slice(0, 10)}`);
  const queryClient = useQueryClient();

  const { data: cyclesData, isLoading: cyclesLoading } = useQuery({
    queryKey: sopKeys.cycles({}),
    queryFn: () => fetchSopCycles(),
    staleTime: STALE_EVO.ONE_MIN,
  });

  const { data: gapsData, isLoading: gapsLoading } = useQuery({
    queryKey: sopKeys.gaps(selectedCycle?.cycle_id ?? ""),
    queryFn: () => fetchSopGaps(selectedCycle!.cycle_id),
    enabled: !!selectedCycle,
    staleTime: STALE_EVO.FIVE_MIN,
  });

  const { data: planData, isLoading: planLoading } = useQuery({
    queryKey: sopKeys.approvedPlan({ plan_month: planMonth || undefined, item_no: planItem || undefined }),
    queryFn: () => fetchApprovedPlan({ plan_month: planMonth || undefined, item_no: planItem || undefined }),
    staleTime: STALE_EVO.FIVE_MIN,
  });

  const advanceMutation = useMutation({
    mutationFn: async (cycle_id: string) => {
      const res = await fetch(`/sop/cycles/${cycle_id}/advance`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ facilitated_by: facilitatedBy || "planner", notes: null }),
      });
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: sopKeys.cycles({}) });
    },
  });

  const cycles = cyclesData?.cycles ?? [];
  const gaps = gapsData?.gaps ?? [];
  const planRows = planData?.rows ?? [];

  const criticalGaps = gaps.filter((g) => g.severity === "critical").length;
  const totalGaps = gaps.length;

  return (
    <div className="p-4 space-y-6">
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-bold">S&OP Cycle Management</h1>
          <span className="text-sm text-muted-foreground">{cycles.length} active cycle{cycles.length !== 1 ? "s" : ""}</span>
        </div>
        <p className="text-sm text-muted-foreground max-w-3xl leading-relaxed">
          Manage the monthly <strong>Sales &amp; Operations Planning</strong> process. Each cycle progresses
          through 6 stages: <em>Demand Review</em> (validate statistical and judgmental forecasts),{" "}
          <em>Supply Review</em> (check capacity and material constraints),{" "}
          <em>Pre-S&amp;OP</em> (identify demand/supply gaps and propose mitigations),{" "}
          <em>Executive S&amp;OP</em> (leadership decision meeting), <em>Approved</em> (finalized
          consensus plan), and <em>Closed</em>. Select a cycle below to view its current stage, review
          gaps between demand and supply, advance it through stages, or approve the final plan.
        </p>
      </div>

      {/* Cycle list + detail */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Cycles list */}
        <div className="lg:col-span-1 space-y-2">
          <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">Cycles</h2>
          {cyclesLoading ? (
            <p className="text-sm text-muted-foreground">Loading…</p>
          ) : !cycles.length ? (
            <Card>
              <CardContent className="p-4 text-sm text-muted-foreground">
                No active S&OP cycles. Create one via the API or CLI.
              </CardContent>
            </Card>
          ) : (
            cycles.map((cycle) => (
              <Card
                key={cycle.cycle_id}
                className={`cursor-pointer transition-colors hover:bg-muted/30 ${selectedCycle?.cycle_id === cycle.cycle_id ? "ring-2 ring-primary" : ""}`}
                onClick={() => setSelectedCycle(cycle)}
              >
                <CardContent className="p-3">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-sm">{cycle.cycle_month?.slice(0, 7)}</span>
                    <span className={`text-xs px-1.5 py-0.5 rounded ${cycle.current_stage === "approved" ? "bg-green-100 text-green-700" : cycle.current_stage === "closed" ? "bg-gray-100 text-gray-600" : "bg-blue-100 text-blue-700"}`}>
                      {STAGE_LABELS[cycle.current_stage] ?? cycle.current_stage}
                    </span>
                  </div>
                  <StageTimeline currentStage={cycle.current_stage} />
                </CardContent>
              </Card>
            ))
          )}
        </div>

        {/* Cycle detail */}
        <div className="lg:col-span-2 space-y-4">
          {!selectedCycle ? (
            <Card>
              <CardContent className="p-6 text-center text-sm text-muted-foreground">
                Select a cycle to view details, gaps, and actions.
              </CardContent>
            </Card>
          ) : (
            <>
              {/* Stage actions */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base flex items-center gap-2">
                    <Clock size={16} />
                    Cycle {selectedCycle.cycle_month?.slice(0, 7)} — {STAGE_LABELS[selectedCycle.current_stage]}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto mb-4">
                    <StageTimeline currentStage={selectedCycle.current_stage} />
                  </div>
                  <div className="flex flex-wrap gap-2 mt-3 items-end">
                    <div className="flex flex-col gap-1">
                      <label className="text-xs text-muted-foreground">Facilitated by</label>
                      <input
                        className="border rounded px-2 py-1 text-sm w-36"
                        value={facilitatedBy}
                        onChange={(e) => setFacilitatedBy(e.target.value)}
                        placeholder="planner"
                      />
                    </div>
                    {selectedCycle.current_stage !== "closed" && (
                      <button
                        disabled={advanceMutation.isPending}
                        onClick={() => advanceMutation.mutate(selectedCycle.cycle_id)}
                        className="px-3 py-1.5 bg-primary text-primary-foreground rounded text-sm font-medium disabled:opacity-60"
                      >
                        {advanceMutation.isPending ? "Advancing…" : "Advance Stage"}
                      </button>
                    )}
                    {selectedCycle.current_stage === "executive_sop" && (
                      <>
                        <div className="flex flex-col gap-1">
                          <label className="text-xs text-muted-foreground">Approved by</label>
                          <input
                            className="border rounded px-2 py-1 text-sm w-36"
                            value={approvedBy}
                            onChange={(e) => setApprovedBy(e.target.value)}
                            placeholder="exec name"
                          />
                        </div>
                        <div className="flex flex-col gap-1">
                          <label className="text-xs text-muted-foreground">Plan version</label>
                          <input
                            className="border rounded px-2 py-1 text-sm w-32"
                            value={planVersion}
                            onChange={(e) => setPlanVersion(e.target.value)}
                          />
                        </div>
                        <button
                          disabled={!approvedBy || !planVersion}
                          onClick={async () => {
                            await fetch(`/sop/cycles/${selectedCycle.cycle_id}/approve`, {
                              method: "POST",
                              headers: { "Content-Type": "application/json" },
                              body: JSON.stringify({ approved_by: approvedBy, plan_version: planVersion }),
                            });
                            queryClient.invalidateQueries({ queryKey: sopKeys.cycles({}) });
                          }}
                          className="px-3 py-1.5 bg-green-600 text-white rounded text-sm font-medium disabled:opacity-50"
                        >
                          Approve Plan
                        </button>
                      </>
                    )}
                  </div>
                  {advanceMutation.isError && (
                    <p className="text-xs text-red-600 mt-2">{String(advanceMutation.error)}</p>
                  )}
                </CardContent>
              </Card>

              {/* Gaps */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base flex items-center gap-2">
                    <AlertTriangle size={16} />
                    Supply/Demand Gaps
                    {criticalGaps > 0 && (
                      <span className="ml-1 text-xs bg-red-500 text-white px-1.5 py-0.5 rounded-full">
                        {criticalGaps} critical
                      </span>
                    )}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {gapsLoading ? (
                    <p className="text-sm text-muted-foreground">Loading gaps…</p>
                  ) : !gaps.length ? (
                    <div className="flex items-center gap-2 text-green-600 text-sm">
                      <CheckCircle size={16} /> No gaps identified for this cycle.
                    </div>
                  ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {gaps.map((gap) => <GapCard key={gap.gap_id} gap={gap} />)}
                    </div>
                  )}
                </CardContent>
              </Card>
            </>
          )}
        </div>
      </div>

      {/* Approved plan */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-base flex items-center gap-2">
              <CheckCircle size={16} className="text-green-600" />
              Approved Plan
            </CardTitle>
            <div className="flex gap-2">
              <input
                type="month"
                value={planMonth}
                onChange={(e) => setPlanMonth(e.target.value)}
                className="border rounded px-2 py-1 text-sm"
              />
              <input
                className="border rounded px-2 py-1 text-sm w-36"
                value={planItem}
                onChange={(e) => setPlanItem(e.target.value)}
                placeholder="Item No"
              />
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {planLoading ? (
            <p className="p-4 text-sm text-muted-foreground">Loading…</p>
          ) : !planRows.length ? (
            <p className="p-4 text-sm text-muted-foreground">No approved plan rows found.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-muted/50 text-xs uppercase text-muted-foreground">
                  <tr>
                    {["Item", "Location", "Plan Month", "Approved Qty", "Approved By", "Approved At"].map((h) => (
                      <th key={h} className="px-3 py-2 text-left font-medium">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {planRows.map((r, i) => (
                    <tr key={i} className="border-t hover:bg-muted/30">
                      <td className="px-3 py-2 font-mono text-xs">{r.item_no}</td>
                      <td className="px-3 py-2 text-xs">{r.loc}</td>
                      <td className="px-3 py-2 text-xs">{r.plan_month}</td>
                      <td className="px-3 py-2 font-medium">{r.approved_qty.toLocaleString()}</td>
                      <td className="px-3 py-2 text-xs">{r.approved_by ?? "—"}</td>
                      <td className="px-3 py-2 text-xs">{r.approved_at?.slice(0, 10) ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
