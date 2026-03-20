import { useState } from "react";
import type { RefObject } from "react";
import type { ClusteringScenarioResult } from "@/api/queries";
import { formatNumber, formatCompactNumber } from "@/lib/formatters";
import { ScenarioCharts } from "@/components/ScenarioCharts";
import { ChevronDown, ChevronRight } from "lucide-react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

type PastScenarioJob = {
  job_id: string;
  job_label?: string | null;
  submitted_at: string;
  result: unknown;
};

type PastScenariosPanelProps = {
  pastScenarios: PastScenarioJob[] | undefined;
  scenarioResultRef: RefObject<HTMLDivElement>;
  onLoadResult: (result: ClusteringScenarioResult, label: string) => void;
};

export default function PastScenariosPanel({
  pastScenarios,
  scenarioResultRef,
  onLoadResult,
}: PastScenariosPanelProps) {
  const [expandedHistoryId, setExpandedHistoryId] = useState<string | null>(null);
  const [historyResult, setHistoryResult] = useState<ClusteringScenarioResult | null>(null);

  if (!pastScenarios || pastScenarios.length === 0) {
    return null;
  }

  return (
    <div className="space-y-2 mt-4">
      <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
        Past Scenarios ({pastScenarios.length})
      </p>
      <div className="space-y-1.5">
        {pastScenarios.map((pj) => {
          const jr = pj.result as Record<string, unknown> | null;
          const inner = jr ? ((jr.result ?? jr) as Record<string, unknown>) : null;
          const optK = inner?.optimal_k as number | undefined;
          const totalDfus = inner?.total_dfus as number | undefined;
          const runtimeSec = (jr?.runtime_seconds as number) ?? 0;
          const scenId = (jr?.scenario_id as string) || pj.job_id;
          const isExpanded = expandedHistoryId === pj.job_id;

          return (
            <div key={pj.job_id} className="rounded-lg border border-border/60">
              <button
                className="flex w-full items-center justify-between px-3 py-2 text-left hover:bg-muted/20 transition-colors rounded-lg"
                onClick={() => {
                  if (isExpanded) {
                    setExpandedHistoryId(null);
                    setHistoryResult(null);
                  } else {
                    setExpandedHistoryId(pj.job_id);
                    if (inner) {
                      setHistoryResult({
                        scenario_id: scenId,
                        status: "completed",
                        runtime_seconds: runtimeSec,
                        params: (jr?.params as Record<string, unknown>) || {},
                        result: inner as ClusteringScenarioResult["result"],
                      });
                    }
                  }
                }}
              >
                <div className="flex items-center gap-2">
                  {isExpanded ? (
                    <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
                  ) : (
                    <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
                  )}
                  <span className="text-sm font-medium">
                    {pj.job_label || "Scenario"}
                    {optK != null && <span className="text-muted-foreground font-normal"> &mdash; K={optK}</span>}
                    {totalDfus != null && <span className="text-muted-foreground font-normal">, {formatCompactNumber(totalDfus)} DFUs</span>}
                    {runtimeSec > 0 && <span className="text-muted-foreground font-normal">, {runtimeSec.toFixed(1)}s</span>}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-muted-foreground">
                    {new Date(pj.submitted_at).toLocaleString()}
                  </span>
                  {inner && (
                    <button
                      className="rounded-md px-2 py-0.5 text-[10px] font-medium bg-primary/10 text-primary hover:bg-primary/20 transition-colors"
                      onClick={(e) => {
                        e.stopPropagation();
                        const result: ClusteringScenarioResult = {
                          scenario_id: scenId,
                          status: "completed",
                          runtime_seconds: runtimeSec,
                          params: (jr?.params as Record<string, unknown>) || {},
                          result: inner as ClusteringScenarioResult["result"],
                        };
                        onLoadResult(result, pj.job_label?.replace("What-If Scenario ", "") || "H");
                        setTimeout(() => {
                          scenarioResultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
                        }, 100);
                      }}
                    >
                      Promote
                    </button>
                  )}
                </div>
              </button>
              {isExpanded && historyResult?.result && (
                <div className="px-3 pb-3 space-y-3">
                  <div className="max-h-[180px] overflow-y-auto rounded-md border border-input">
                    <Table>
                      <TableHeader>
                        <TableRow className="border-muted bg-muted/30">
                          <TableHead className="text-xs">Cluster</TableHead>
                          <TableHead className="text-xs text-right">DFUs</TableHead>
                          <TableHead className="text-xs text-right">%</TableHead>
                          <TableHead className="text-xs text-right">Avg demand</TableHead>
                          <TableHead className="text-xs text-right">CV</TableHead>
                          <TableHead className="text-xs text-right">Seasonality</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {historyResult.result.profiles.map((p) => (
                          <TableRow key={p.label}>
                            <TableCell className="text-xs font-medium">{p.label}</TableCell>
                            <TableCell className="text-right text-xs tabular-nums">{formatNumber(p.count)}</TableCell>
                            <TableCell className="text-right text-xs tabular-nums">{formatNumber(p.pct_of_total)}%</TableCell>
                            <TableCell className="text-right text-xs tabular-nums">{formatNumber(p.mean_demand)}</TableCell>
                            <TableCell className="text-right text-xs tabular-nums">{formatNumber(p.cv_demand)}</TableCell>
                            <TableCell className="text-right text-xs tabular-nums">{formatNumber(p.seasonality_strength)}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                  <ScenarioCharts result={historyResult.result} />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
