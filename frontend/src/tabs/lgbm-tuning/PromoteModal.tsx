/**
 * PromoteModal — confirmation dialog before promoting a tuning run to production.
 * Shows the run details, params that will be written, and asks for confirmation.
 */
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Loader2, X, Crown } from "lucide-react";
import { formatPct, formatFixed } from "@/lib/formatters";
import type { TuningRun } from "@/api/queries";

export function PromoteModal({
  run,
  onConfirm,
  onCancel,
  isPending,
}: {
  run: TuningRun;
  onConfirm: () => void;
  onCancel: () => void;
  isPending: boolean;
}) {
  const params = run.params ?? {};
  const paramKeys = Object.keys(params).sort();

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-sm">
      <div className="w-full max-w-lg rounded-xl border bg-card shadow-xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b px-5 py-4">
          <div className="flex items-center gap-2">
            <Crown className="h-4 w-4 text-amber-500" />
            <div>
              <p className="text-sm font-semibold text-foreground">Promote to Production</p>
              <p className="text-xs text-muted-foreground">
                Run #{run.run_id} — {run.run_label}
              </p>
            </div>
          </div>
          <button
            onClick={onCancel}
            className="rounded-md p-1 text-muted-foreground hover:text-foreground"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Body */}
        <div className="space-y-4 px-5 py-4">
          {/* KPIs */}
          <div className="flex flex-wrap gap-2">
            <Badge className="bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300">
              Accuracy: {formatPct(run.accuracy_pct)}
            </Badge>
            <Badge variant="outline">
              WAPE: {formatFixed(run.wape, 2)}
            </Badge>
            <Badge variant="outline">
              Bias: {formatFixed(run.bias, 2)}
            </Badge>
          </div>

          {/* What this will do */}
          <div>
            <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              This will:
            </p>
            <ul className="space-y-1 text-sm text-foreground list-disc pl-4">
              <li>Write this run's LGBM hyperparameters to <code className="text-xs bg-muted px-1 py-0.5 rounded">algorithm_config.yaml</code></li>
              <li>Mark this run as the promoted production configuration</li>
              <li>Clear any previously promoted run</li>
            </ul>
          </div>

          {/* Params preview */}
          {paramKeys.length > 0 && (
            <div>
              <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Parameters to write
              </p>
              <div className="rounded-md border bg-muted/30 p-3 max-h-48 overflow-y-auto">
                <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                  {paramKeys.map((key) => (
                    <div key={key} className="flex justify-between">
                      <span className="text-muted-foreground font-mono">{key}</span>
                      <span className="font-mono font-medium">{String(params[key])}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="flex justify-end gap-2 border-t px-5 py-3">
          <Button variant="ghost" size="sm" onClick={onCancel} disabled={isPending}>
            Cancel
          </Button>
          <Button
            size="sm"
            onClick={onConfirm}
            disabled={isPending}
            className="gap-1.5 bg-amber-600 hover:bg-amber-700 text-white"
          >
            {isPending && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
            Promote to Production
          </Button>
        </div>
      </div>
    </div>
  );
}
