import type { RefObject } from "react";
import type { ClusteringScenarioResult } from "@/api/queries";
import { formatNumber } from "@/lib/formatters";
import { ScenarioCharts } from "@/components/ScenarioCharts";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

type ScenarioResultsPanelProps = {
  scenarioResult: ClusteringScenarioResult | null;
  scenarioLabel: string;
  scenarioResultRef: RefObject<HTMLDivElement>;
  showPromoteConfirm: boolean;
  onShowPromoteConfirm: () => void;
  onCancelPromote: () => void;
  onConfirmPromote: () => void;
};

export default function ScenarioResultsPanel({
  scenarioResult,
  scenarioLabel,
  scenarioResultRef,
  showPromoteConfirm,
  onShowPromoteConfirm,
  onCancelPromote,
  onConfirmPromote,
}: ScenarioResultsPanelProps) {
  if (scenarioResult?.status !== "completed" || !scenarioResult.result) {
    return null;
  }

  return (
    <>
      <div ref={scenarioResultRef} className="space-y-3 rounded-lg border border-border p-4">
        <div className="flex items-center justify-between">
          <p className="text-sm font-semibold">
            Scenario {scenarioLabel} &mdash; K={scenarioResult.result.optimal_k},{" "}
            {scenarioResult.result.total_dfus} DFUs, {scenarioResult.runtime_seconds.toFixed(1)}s
          </p>
          <button
            className="rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground hover:bg-primary/90"
            onClick={onShowPromoteConfirm}
          >
            Promote Scenario {scenarioLabel}
          </button>
        </div>

        {/* Result profile table */}
        <div className="max-h-[240px] overflow-y-auto rounded-md border border-input">
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
              {scenarioResult.result.profiles.map((p) => (
                <TableRow key={p.label}>
                  <TableCell className="text-sm font-medium">{p.label}</TableCell>
                  <TableCell className="text-right text-sm tabular-nums">{formatNumber(p.count)}</TableCell>
                  <TableCell className="text-right text-sm tabular-nums">{formatNumber(p.pct_of_total)}%</TableCell>
                  <TableCell className="text-right text-sm tabular-nums">{formatNumber(p.mean_demand)}</TableCell>
                  <TableCell className="text-right text-sm tabular-nums">{formatNumber(p.cv_demand)}</TableCell>
                  <TableCell className="text-right text-sm tabular-nums">{formatNumber(p.seasonality_strength)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>

        {/* Charts */}
        <ScenarioCharts result={scenarioResult.result} />
      </div>

      {/* Promote confirmation dialog */}
      {showPromoteConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="w-full max-w-md rounded-lg border border-border bg-card p-6 shadow-xl">
            <p className="text-sm font-semibold">Promote Scenario {scenarioLabel} to Production?</p>
            <p className="mt-2 text-xs text-muted-foreground">
              This will update <code>dim_dfu.ml_cluster</code> with the new cluster assignments.
            </p>
            <div className="mt-4 flex justify-end gap-3">
              <button
                className="rounded-md border border-input bg-background px-4 py-2 text-sm hover:bg-muted/50"
                onClick={onCancelPromote}
              >
                Cancel
              </button>
              <button
                className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
                onClick={onConfirmPromote}
              >
                Confirm Promote
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
