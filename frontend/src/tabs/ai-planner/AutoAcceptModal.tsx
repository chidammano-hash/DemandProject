/**
 * AutoAcceptModal — bulk auto-accept workflow with preview/execute flow.
 *
 * Gen-4 UX: uses Radix Dialog for focus trap, aria-modal, focus restore.
 */
import { useState } from "react";
import type { InsightSeverity } from "@/types/ai-planner";
import type { AutoAcceptResponse } from "@/api/queries/ai-planner";
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
import { SEVERITY_THRESHOLD_LABELS } from "./aiPlannerShared";

export function AutoAcceptModal({
  onConfirm,
  onCancel,
  isPending,
  result,
}: {
  onConfirm: (minSeverity: InsightSeverity, dryRun: boolean) => void;
  onCancel: () => void;
  isPending: boolean;
  result: AutoAcceptResponse | null;
}) {
  const [minSeverity, setMinSeverity] = useState<InsightSeverity>("high");

  // After dry-run succeeds, parent sets result; split into preview vs execute result
  const isDryRunResult = result?.dry_run === true;
  const isExecuteResult = result?.dry_run === false;

  function handlePreview() {
    onConfirm(minSeverity, true);
  }

  return (
    <Dialog open onOpenChange={(o) => { if (!o) onCancel(); }}>
      <DialogContent size="sm">
        <DialogHeader>
          <DialogTitle>Auto-Accept Rules</DialogTitle>
          <DialogDescription>
            Bulk-accept open insights by severity threshold
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 px-5 py-4">
          {isExecuteResult ? (
            /* Final result state */
            <div className="rounded-lg bg-green-50 p-4 text-center dark:bg-green-950/30">
              <p className="text-2xl font-bold text-green-700 dark:text-green-400">{result!.accepted}</p>
              <p className="mt-1 text-sm text-muted-foreground">
                insights auto-accepted and logged to outcome tracker
              </p>
            </div>
          ) : isDryRunResult && result!.accepted > 0 ? (
            /* Confirmation step — PL-008 */
            <div className="space-y-3">
              <div className="rounded-lg bg-amber-50 p-4 text-center dark:bg-amber-950/30">
                <p className="text-2xl font-bold text-amber-700 dark:text-amber-400">{result!.accepted}</p>
                <p className="mt-1 text-sm text-muted-foreground">
                  open {SEVERITY_THRESHOLD_LABELS[minSeverity].toLowerCase()} insights found
                </p>
              </div>
              <div className="rounded-md border border-amber-200 bg-amber-50/50 px-3 py-2 text-xs text-amber-800 dark:border-amber-800 dark:bg-amber-950/20 dark:text-amber-300">
                This will permanently accept {result!.accepted} insight{result!.accepted !== 1 ? "s" : ""} and write outcome records. <strong>This cannot be undone.</strong>
              </div>
            </div>
          ) : isDryRunResult && result!.accepted === 0 ? (
            <div className="rounded-lg bg-muted/50 p-4 text-center">
              <p className="text-2xl font-bold text-foreground">0</p>
              <p className="mt-1 text-sm text-muted-foreground">
                no matching open insights found for this threshold
              </p>
            </div>
          ) : (
            /* Config state */
            <>
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-muted-foreground">
                  Accept insights at severity
                </label>
                <select
                  value={minSeverity}
                  onChange={(e) => setMinSeverity(e.target.value as InsightSeverity)}
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus:outline-none"
                >
                  {(["critical", "high", "medium", "low"] as InsightSeverity[]).map((s) => (
                    <option key={s} value={s}>{SEVERITY_THRESHOLD_LABELS[s]}</option>
                  ))}
                </select>
              </div>
              <p className="text-xs text-muted-foreground">
                All open insights at or above this severity will be marked <strong>Accepted</strong> and written
                to the outcome tracker for 30-day follow-up measurement.
              </p>
            </>
          )}
        </div>

        <DialogFooter>
          <Button variant="ghost" size="sm" onClick={onCancel}>
            {isExecuteResult ? "Close" : "Cancel"}
          </Button>
          {/* Config -> Preview */}
          {!result && (
            <Button variant="outline" size="sm" onClick={handlePreview} disabled={isPending}>
              {isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : "Preview"}
            </Button>
          )}
          {/* Preview with matches -> Confirm execute */}
          {isDryRunResult && result!.accepted > 0 && (
            <Button
              size="sm"
              onClick={() => onConfirm(minSeverity, false)}
              disabled={isPending}
              className="gap-1.5 bg-amber-600 hover:bg-amber-700"
            >
              {isPending && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
              Confirm — Accept {result!.accepted}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
