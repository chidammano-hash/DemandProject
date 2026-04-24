/**
 * ConfirmModal — confirmation dialog before Accept/Resolve actions.
 * Shows recommendation, key metrics, and causal chain.
 *
 * Gen-4 UX: uses Radix Dialog for focus trap, aria-modal, focus restore.
 */
import { formatCurrency as fmtCurrency } from "@/lib/formatters";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Loader2 } from "lucide-react";
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
    <Dialog open onOpenChange={(o) => { if (!o) onCancel(); }}>
      <DialogContent size="md">
        <DialogHeader>
          <DialogTitle>{label}</DialogTitle>
          <DialogDescription>
            {insight.item_id} @ {insight.loc}
          </DialogDescription>
        </DialogHeader>

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

        <DialogFooter>
          <Button variant="ghost" size="sm" onClick={onCancel} disabled={isPending}>
            Cancel
          </Button>
          <Button size="sm" onClick={onConfirm} disabled={isPending} className="gap-1.5">
            {isPending && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
            {verb}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
