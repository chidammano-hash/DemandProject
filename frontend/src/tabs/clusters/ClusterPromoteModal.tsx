/**
 * ClusterPromoteModal -- Warning modal for promoting cluster assignments.
 *
 * Shows experiment details, downstream impact warnings, and a tip about
 * re-running algorithm experiments with the new clusters.
 */
import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  Crown,
  Loader2,
  X,
  Check,
  Info,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  clusterExperimentKeys,
  promoteClusterExperiment,
  type ClusterExperiment,
} from "@/api/queries";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ClusterPromoteModalProps {
  experiment: ClusterExperiment;
  open: boolean;
  onClose: () => void;
  onPromoted: () => void;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WARNINGS = [
  "All SKU cluster assignments in dim_sku.ml_cluster will be overwritten",
  "Existing backtest results were trained on the old clusters",
  "Per-cluster forecast accuracy metrics may shift",
  "Inventory planning safety stock calculations use cluster segments",
  "Champion selection models were calibrated on previous cluster assignments",
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ClusterPromoteModal({
  experiment,
  open,
  onClose,
  onPromoted,
}: ClusterPromoteModalProps) {
  const queryClient = useQueryClient();
  const [confirmed, setConfirmed] = useState(false);

  const promoteMut = useMutation({
    mutationFn: () => promoteClusterExperiment(experiment.experiment_id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: clusterExperimentKeys.all });
      onPromoted();
    },
  });

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-sm">
      <div className="w-full max-w-lg rounded-xl border bg-card shadow-xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between border-b px-5 py-4 shrink-0">
          <div className="flex items-center gap-2">
            <Crown className="h-5 w-5 text-amber-500" />
            <div>
              <p className="text-sm font-semibold text-foreground">
                Promote Cluster Experiment
              </p>
              <p className="text-xs text-muted-foreground">
                #{experiment.experiment_id}: {experiment.label}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-md p-1 text-muted-foreground hover:text-foreground"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
          {/* Experiment summary */}
          <div className="grid grid-cols-3 gap-3">
            <div className="rounded-md border border-border/60 px-3 py-2 text-center">
              <p className="text-[10px] text-muted-foreground">K</p>
              <p className="text-lg font-bold tabular-nums">
                {experiment.optimal_k ?? "--"}
              </p>
            </div>
            <div className="rounded-md border border-border/60 px-3 py-2 text-center">
              <p className="text-[10px] text-muted-foreground">Silhouette</p>
              <p className="text-lg font-bold tabular-nums">
                {experiment.silhouette_score != null
                  ? experiment.silhouette_score.toFixed(4)
                  : "--"}
              </p>
            </div>
            <div className="rounded-md border border-border/60 px-3 py-2 text-center">
              <p className="text-[10px] text-muted-foreground">DFUs</p>
              <p className="text-lg font-bold tabular-nums">
                {experiment.total_dfus?.toLocaleString() ?? "--"}
              </p>
            </div>
          </div>

          {/* Warnings */}
          <div className="space-y-1.5">
            <p className="text-xs font-semibold text-foreground flex items-center gap-1.5">
              <AlertTriangle className="h-3.5 w-3.5 text-amber-500" />
              Downstream Impacts
            </p>
            <ul className="space-y-1">
              {WARNINGS.map((w, i) => (
                <li
                  key={i}
                  className="flex items-start gap-1.5 text-xs text-muted-foreground"
                >
                  <AlertTriangle className="h-3 w-3 mt-0.5 text-amber-500 shrink-0" />
                  <span>{w}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Tip */}
          <div className="rounded-md border border-blue-200 bg-blue-50 dark:border-blue-900/50 dark:bg-blue-950/30 px-3 py-2 flex items-start gap-2">
            <Info className="h-3.5 w-3.5 text-blue-500 mt-0.5 shrink-0" />
            <p className="text-xs text-blue-700 dark:text-blue-300">
              <strong>Tip:</strong> After promoting, re-run your algorithm tuning
              experiments with the new clusters to see how the segmentation change
              affects forecast accuracy.
            </p>
          </div>

          {/* Confirmation checkbox */}
          <label className="flex items-start gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={confirmed}
              onChange={(e) => setConfirmed(e.target.checked)}
              className="mt-0.5 rounded border-border"
            />
            <span className="text-[11px] text-muted-foreground">
              I understand this will overwrite production cluster assignments for
              all {experiment.total_dfus?.toLocaleString() ?? "N"} DFUs
            </span>
          </label>

          {/* Error */}
          {promoteMut.isError && (
            <div className="rounded-md border border-red-200 bg-red-50 px-2 py-1.5 text-[11px] text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-400 flex items-center gap-1">
              <AlertTriangle className="h-3 w-3 shrink-0" />
              {(promoteMut.error as Error).message}
            </div>
          )}

          {/* Success */}
          {promoteMut.isSuccess && (
            <div className="rounded-md border border-emerald-200 bg-emerald-50 px-2 py-1.5 text-[11px] text-emerald-700 dark:border-emerald-900/50 dark:bg-emerald-950/30 dark:text-emerald-400 flex items-center gap-1">
              <Check className="h-3 w-3 shrink-0" />
              Cluster assignments promoted to production successfully.
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-2 border-t px-5 py-3 shrink-0">
          <Button variant="ghost" size="sm" onClick={onClose}>
            {promoteMut.isSuccess ? "Done" : "Cancel"}
          </Button>
          {!promoteMut.isSuccess && (
            <Button
              size="sm"
              onClick={() => promoteMut.mutate()}
              disabled={!confirmed || promoteMut.isPending}
              className={cn(
                "gap-1.5",
                confirmed
                  ? "bg-amber-600 hover:bg-amber-700 text-white"
                  : "opacity-60",
              )}
            >
              {promoteMut.isPending ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Crown className="h-3.5 w-3.5" />
              )}
              Confirm Promote
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
