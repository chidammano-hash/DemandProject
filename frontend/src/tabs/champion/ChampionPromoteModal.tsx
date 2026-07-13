/** Confirmation dialog for governed assignment from a selected experiment. */
import { Crown, ShieldCheck } from "lucide-react";

import { type ChampionExperiment } from "@/api/queries";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface Props {
  experiment: ChampionExperiment;
  open: boolean;
  onClose: () => void;
  onAssign: (experimentId: number) => void;
  isPending: boolean;
}

export function ChampionPromoteModal({ experiment, open, onClose, onAssign, isPending }: Props) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="w-full max-w-lg rounded-lg border bg-background shadow-lg">
        {/* Header */}
        <div className="flex items-center gap-2 border-b px-6 py-4">
          <Crown className="h-5 w-5 text-amber-500" />
          <h2 className="text-lg font-semibold">Select &amp; Assign Champion</h2>
        </div>

        {/* Body */}
        <div className="px-6 py-4 space-y-4">
          {/* Summary */}
          <Card>
            <CardContent className="py-3 space-y-1 text-sm">
              <div>
                <span className="text-muted-foreground">Experiment:</span>{" "}
                <span className="font-medium">{experiment.label}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Strategy:</span>{" "}
                <span className="font-mono">{experiment.strategy}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Accuracy:</span>{" "}
                <span className="font-medium">
                  {experiment.champion_accuracy?.toFixed(2) ?? "--"}%
                </span>
              </div>
              <div>
                <span className="text-muted-foreground">Gap to Ceiling:</span>{" "}
                <span className="font-medium">{experiment.gap_bps?.toFixed(0) ?? "--"} bps</span>
              </div>
            </CardContent>
          </Card>

          <div className="flex items-start gap-3 rounded-md border border-blue-300 bg-blue-50 p-3 text-blue-950 dark:border-blue-800 dark:bg-blue-950/30 dark:text-blue-100">
            <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0" />
            <div className="space-y-1 text-sm">
              <p className="font-medium">Assignment is governed and atomic.</p>
              <p className="text-xs">
                This strategy will be re-evaluated against the current governed five-model
                backtests. Its resulting single-model or ensemble composition will be copied into
                historical champion rows. The active champion changes only after every row and
                checksum passes validation.
              </p>
            </div>
          </div>
          <p className="text-xs text-muted-foreground">
            When the assignment job completes, continue to Forecast to build a release candidate
            from the newly assigned champion.
          </p>
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-2 border-t px-6 py-3">
          <Button variant="outline" onClick={onClose} disabled={isPending}>
            Cancel
          </Button>
          <Button onClick={() => onAssign(experiment.experiment_id)} disabled={isPending}>
            {isPending ? "Queuing…" : "Assign Champion"}
          </Button>
        </div>
      </div>
    </div>
  );
}
