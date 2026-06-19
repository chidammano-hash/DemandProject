/**
 * TrainingStatusIndicator -- compact badge showing per-model training readiness.
 */
import { CheckCircle2, XCircle, AlertTriangle } from "lucide-react";
import { timeAgo } from "@/components/shared-tuning-utils";

export function TrainingStatusIndicator({
  trained,
  trainingMode,
  trainedAt,
  needsTraining,
}: {
  trained: boolean;
  trainingMode: string | null;
  trainedAt: string | null;
  needsTraining: boolean;
}) {
  if (!needsTraining) {
    return (
      <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
        <CheckCircle2 className="h-3 w-3 text-blue-500" />
        No training needed
      </span>
    );
  }

  if (!trained) {
    return (
      <span className="inline-flex items-center gap-1 text-xs text-red-600 dark:text-red-400">
        <XCircle className="h-3 w-3" />
        Not Trained
      </span>
    );
  }

  if (trainingMode === "production") {
    return (
      <span className="inline-flex items-center gap-1 text-xs text-emerald-600 dark:text-emerald-400">
        <CheckCircle2 className="h-3 w-3" />
        Production Ready
        {trainedAt && (
          <span className="text-muted-foreground ml-1">({timeAgo(trainedAt)})</span>
        )}
      </span>
    );
  }

  return (
    <span className="inline-flex items-center gap-1 text-xs text-amber-600 dark:text-amber-400">
      <AlertTriangle className="h-3 w-3" />
      Backtest Only
      {trainedAt && (
        <span className="text-muted-foreground ml-1">({timeAgo(trainedAt)})</span>
      )}
    </span>
  );
}
