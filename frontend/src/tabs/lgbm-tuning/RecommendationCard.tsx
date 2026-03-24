import { useState } from "react";
import { Check, X, Loader2, Beaker } from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import type { TuningRecommendation } from "@/api/queries";
import { RunStatusCard } from "./RunStatusCard";

interface RecommendationCardProps {
  recommendation: TuningRecommendation;
  messageId: number;
  sessionId: string;
  onConfirm: (messageId: number) => void;
  onReject: (messageId: number) => void;
  isConfirming?: boolean;
  isConfirmed?: boolean;
  /** Run ID spawned from this recommendation (enables inline status tracking) */
  confirmedRunId?: number;
}

export function RecommendationCard({
  recommendation,
  messageId,
  sessionId,
  onConfirm,
  onReject,
  isConfirming,
  isConfirmed,
  confirmedRunId,
}: RecommendationCardProps) {
  const [rejected, setRejected] = useState(false);

  const riskColor =
    recommendation.risk_assessment.toLowerCase().includes("low")
      ? "text-emerald-600 dark:text-emerald-400"
      : recommendation.risk_assessment.toLowerCase().includes("high")
        ? "text-red-600 dark:text-red-400"
        : "text-amber-600 dark:text-amber-400";

  return (
    <div className="border border-primary/25 rounded-lg bg-primary/5 p-4 my-2">
      <div className="flex items-center gap-2 mb-3">
        <Beaker className="h-4 w-4 text-primary" />
        <span className="text-sm font-medium text-primary">
          Recommended Experiment
        </span>
        <Badge variant="outline" className="text-xs ml-auto">
          {recommendation.strategy_label}
        </Badge>
      </div>

      <p className="text-sm text-foreground/80 mb-3">
        {recommendation.description}
      </p>

      {/* Parameter overrides table */}
      <div className="bg-muted/60 rounded p-3 mb-3">
        <div className="text-xs text-muted-foreground uppercase tracking-wider mb-2">
          Parameter Changes
        </div>
        <div className="grid grid-cols-2 gap-1">
          {Object.entries(recommendation.overrides).map(([key, value]) => (
            <div key={key} className="flex justify-between text-sm">
              <span className="text-muted-foreground">{key}</span>
              <span className="text-amber-600 dark:text-amber-400 font-mono">{String(value)}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Impact and risk */}
      <div className="flex gap-4 text-xs mb-3">
        <div>
          <span className="text-muted-foreground">Expected: </span>
          <span className="text-foreground/80">{recommendation.expected_impact}</span>
        </div>
        <div>
          <span className="text-muted-foreground">Risk: </span>
          <span className={cn(riskColor)}>{recommendation.risk_assessment}</span>
        </div>
      </div>

      {/* Action buttons */}
      {!isConfirmed && !rejected && (
        <div className="flex gap-2">
          <button
            onClick={() => onConfirm(messageId)}
            disabled={isConfirming}
            className={cn(
              "flex items-center gap-1.5 px-3 py-1.5 rounded text-sm font-medium",
              "bg-emerald-600 hover:bg-emerald-500 text-white dark:bg-emerald-600 dark:hover:bg-emerald-500",
              "disabled:opacity-50 disabled:cursor-not-allowed",
            )}
          >
            {isConfirming ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Check className="h-3.5 w-3.5" />
            )}
            {isConfirming ? "Starting..." : "Confirm & Run"}
          </button>
          <button
            onClick={() => {
              setRejected(true);
              onReject(messageId);
            }}
            disabled={isConfirming}
            className={cn(
              "flex items-center gap-1.5 px-3 py-1.5 rounded text-sm",
              "border border-border hover:bg-muted text-muted-foreground",
              "disabled:opacity-50",
            )}
          >
            <X className="h-3.5 w-3.5" />
            Reject
          </button>
        </div>
      )}

      {isConfirmed && confirmedRunId && (
        <RunStatusCard
          sessionId={sessionId}
          runId={confirmedRunId}
          messageType="run_started"
        />
      )}
      {isConfirmed && !confirmedRunId && (
        <Badge className="bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300">
          Confirmed — run started
        </Badge>
      )}
      {rejected && (
        <Badge variant="outline" className="text-muted-foreground">
          Rejected
        </Badge>
      )}
    </div>
  );
}
