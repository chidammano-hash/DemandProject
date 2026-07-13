/** Read-only compatibility dialog for the retired manual promotion workflow. */
import { Crown, ShieldAlert } from "lucide-react";

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
}

export function ChampionPromoteModal({ experiment, open, onClose }: Props) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="w-full max-w-lg rounded-lg border bg-background shadow-lg">
        {/* Header */}
        <div className="flex items-center gap-2 border-b px-6 py-4">
          <Crown className="h-5 w-5 text-amber-500" />
          <h2 className="text-lg font-semibold">Manual Champion Promotion Retired</h2>
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
                <span className="font-medium">
                  {experiment.gap_bps?.toFixed(0) ?? "--"} bps
                </span>
              </div>
            </CardContent>
          </Card>

          <div className="flex items-start gap-3 rounded-md border border-amber-300 bg-amber-50 p-3 text-amber-950 dark:border-amber-800 dark:bg-amber-950/30 dark:text-amber-100">
            <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" />
            <div className="space-y-1 text-sm">
              <p className="font-medium">This experiment is analysis-only.</p>
              <p className="text-xs">
                Use the named champion-refresh pipeline to validate the current five-model
                lineage and atomically promote a governed champion. Manual config writes
                and results-load jobs are disabled.
              </p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end border-t px-6 py-3">
          <Button variant="outline" onClick={onClose}>
            Close
          </Button>
        </div>
      </div>
    </div>
  );
}
