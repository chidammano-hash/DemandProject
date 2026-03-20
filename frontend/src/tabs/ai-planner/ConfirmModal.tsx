/**
 * ConfirmModal — confirmation dialog before Accept/Resolve actions.
 * Shows recommendation, key metrics, and causal chain.
 */
import { formatCurrency as fmtCurrency } from "@/lib/formatters";
import { Button } from "@/components/ui/button";
import { Loader2, X } from "lucide-react";
import type { ConfirmActionState } from "./aiPlannerShared";
import { CausalChainCard } from "./CausalChainCard";

export function ConfirmModal({
  action,
  onConfirm,
  onCancel,
  isPending,
}: {
  action: ConfirmActionState;
  onConfirm: () => void;
  onCancel: () => void;
  isPending: boolean;
}) {
  const { insight, label, verb } = action;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-sm">
      <div className="w-full max-w-md rounded-xl border bg-card shadow-xl">
        {/* Modal header */}
        <div className="flex items-center justify-between border-b px-5 py-4">
          <div>
            <p className="text-sm font-semibold text-foreground">{label}</p>
            <p className="text-xs text-muted-foreground">
              {insight.item_no} @ {insight.loc}
            </p>
          </div>
          <button
            onClick={onCancel}
            className="rounded-md p-1 text-muted-foreground hover:text-foreground"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Modal body */}
        <div className="space-y-4 px-5 py-4">
          {/* What this will do */}
          <div>
            <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              AI Recommendation
            </p>
            <p className="rounded-md bg-muted/50 p-3 text-sm leading-relaxed text-foreground">
              {insight.recommendation}
            </p>
          </div>

          {/* Key metrics */}
          <div className="flex flex-wrap gap-2">
            {insight.dos != null && (
              <div className="rounded-md bg-muted px-3 py-1.5 text-xs">
                <span className="text-muted-foreground">Current DOS: </span>
                <strong>{insight.dos.toFixed(0)}d</strong>
                {insight.total_lt_days && (
                  <span className="text-muted-foreground"> (LT {insight.total_lt_days}d)</span>
                )}
              </div>
            )}
            {insight.financial_impact_estimate != null && (
              <div className="rounded-md bg-amber-50 px-3 py-1.5 text-xs dark:bg-amber-950/30">
                <span className="text-muted-foreground">At risk: </span>
                <strong className="text-amber-700 dark:text-amber-400">
                  {fmtCurrency(insight.financial_impact_estimate)}
                </strong>
              </div>
            )}
          </div>

          {/* Causal chain */}
          <CausalChainCard insight={insight} />
        </div>

        {/* Actions */}
        <div className="flex justify-end gap-2 border-t px-5 py-3">
          <Button variant="ghost" size="sm" onClick={onCancel} disabled={isPending}>
            Cancel
          </Button>
          <Button size="sm" onClick={onConfirm} disabled={isPending} className="gap-1.5">
            {isPending && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
            {verb}
          </Button>
        </div>
      </div>
    </div>
  );
}
