import type { RefObject } from "react";
import type { ClusteringScenarioResult } from "@/api/queries";
import { formatNumber, formatClusterLabel } from "@/lib/formatters";
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
            {(scenarioResult.result.total_dfus ?? scenarioResult.result.total_skus)?.toLocaleString()} SKUs, {scenarioResult.runtime_seconds.toFixed(1)}s
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
                <TableHead className="text-xs text-right">SKUs</TableHead>
                <TableHead className="text-xs text-right">%</TableHead>
                <TableHead className="text-xs text-right">Avg demand</TableHead>
                <TableHead className="text-xs text-right">CV</TableHead>
                <TableHead className="text-xs text-right">Seasonality</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {scenarioResult.result.profiles.map((p) => (
                <TableRow key={p.label}>
                  <TableCell className="text-sm font-medium" title={p.label}>{formatClusterLabel(p.label)}</TableCell>
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
        <ScenarioCharts result={scenarioResult.result} pcaScatter={scenarioResult.result.pca_scatter} />
      </div>

      {/* Promote confirmation dialog */}
      {showPromoteConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="w-full max-w-lg rounded-lg border border-border bg-card p-6 shadow-xl">
            <p className="text-base font-semibold">Promote Scenario {scenarioLabel} to Production?</p>

            {/* Warning box */}
            <div className="mt-3 rounded-md border border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-950/30 p-3">
              <p className="text-sm font-medium text-amber-800 dark:text-amber-300">
                ⚠ This action affects downstream models
              </p>
              <ul className="mt-1.5 space-y-1 text-xs text-amber-700 dark:text-amber-400">
                <li>• <strong>Cluster assignments</strong> in <code className="text-[10px] bg-amber-100 dark:bg-amber-900/50 rounded px-0.5">dim_sku.ml_cluster</code> will be overwritten for all SKUs</li>
                <li>• <strong>Backtesting</strong> — all per-cluster models (LGBM, CatBoost, XGBoost) will train on the new cluster boundaries. Previous backtest results may no longer be comparable.</li>
                <li>• <strong>Production forecasts</strong> — champion model selection and forecast generation will use the new clusters. Forecast values will shift.</li>
                <li>• <strong>Inventory planning</strong> — safety stock, EOQ, and exception detection rely on cluster-based demand patterns</li>
              </ul>
              <p className="mt-2 text-[11px] text-amber-600 dark:text-amber-500">
                Recommendation: Re-run backtests after promoting to validate model accuracy with the new clusters.
              </p>
            </div>

            <p className="mt-3 text-xs text-muted-foreground">
              K={scenarioResult.result.optimal_k} clusters, {(scenarioResult.result.total_dfus ?? scenarioResult.result.total_skus)?.toLocaleString()} SKUs will be reassigned.
            </p>

            <div className="mt-4 flex justify-end gap-3">
              <button
                className="rounded-md border border-input bg-background px-4 py-2 text-sm hover:bg-muted/50"
                onClick={onCancelPromote}
              >
                Cancel
              </button>
              <button
                className="rounded-md bg-amber-600 hover:bg-amber-700 px-4 py-2 text-sm font-medium text-white"
                onClick={onConfirmPromote}
              >
                Promote to Production
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
