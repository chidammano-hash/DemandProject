/**
 * InsightCard — individual AI insight card with severity badge, causal chain,
 * metrics row, AI reasoning, and action buttons (Accept/Resolve/Snooze).
 */
import { useState } from "react";
import type { AiInsight } from "@/types/ai-planner";
import { cn } from "@/lib/utils";
import { formatCurrency as fmtCurrency } from "@/lib/formatters";
import { Button } from "@/components/ui/button";
import {
  AlertTriangle,
  AlertCircle,
  Info,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import {
  SEVERITY_STYLES,
  INSIGHT_TYPE_ICONS,
  INSIGHT_TYPE_LABELS,
  INSIGHT_TYPE_EXPLAINERS,
  deriveConfidence,
  CONFIDENCE_STYLES,
} from "./aiPlannerShared";
import { CausalChainCard } from "./CausalChainCard";

// ---------------------------------------------------------------------------
// SeverityIcon
// ---------------------------------------------------------------------------
function SeverityIcon({ severity }: { severity: AiInsight["severity"] }) {
  if (severity === "critical") return <AlertCircle className="h-4 w-4 text-red-500" />;
  if (severity === "high") return <AlertTriangle className="h-4 w-4 text-orange-500" />;
  if (severity === "medium") return <Info className="h-4 w-4 text-yellow-500" />;
  return <CheckCircle2 className="h-4 w-4 text-gray-400" />;
}

// ---------------------------------------------------------------------------
// ConfidenceBadge
// ---------------------------------------------------------------------------
function ConfidenceBadge({ insight }: { insight: AiInsight }) {
  const tier = deriveConfidence(insight);
  const style = CONFIDENCE_STYLES[tier];
  return (
    <span
      title={style.tooltip}
      className={cn("rounded-full px-2 py-0.5 text-xs font-medium cursor-help", style.className)}
    >
      {style.label} confidence
    </span>
  );
}

// ---------------------------------------------------------------------------
// InsightCard
// ---------------------------------------------------------------------------
export function InsightCard({
  insight,
  selected,
  onSelect,
  onAcknowledge,
  onResolve,
  onSnooze,
}: {
  insight: AiInsight;
  selected?: boolean;
  onSelect?: (id: number) => void;
  onAcknowledge: (insight: AiInsight) => void;
  onResolve: (insight: AiInsight) => void;
  onSnooze: (insight: AiInsight, days: number) => void;
}) {
  const [expanded, setExpanded] = useState(true);
  const [showSnoozePicker, setShowSnoozePicker] = useState(false);
  const [snoozeReason, setSnoozeReason] = useState("");
  const [snoozeDays, setSnoozeDays] = useState<number | null>(null);
  const s = SEVERITY_STYLES[insight.severity];
  const TypeIcon = INSIGHT_TYPE_ICONS[insight.insight_type];

  return (
    <div
      className={cn(
        "rounded-lg border border-l-4 bg-card p-4 shadow-sm transition-all",
        s.border,
        insight.status === "resolved" && "opacity-60",
        selected && "ring-2 ring-primary/30",
      )}
    >
      {/* Header row */}
      <div className="flex items-start gap-3">
        {/* Bulk select checkbox */}
        {onSelect && insight.status === "open" && (
          <input
            type="checkbox"
            checked={selected ?? false}
            onChange={() => onSelect(insight.insight_id)}
            className="mt-1 h-3.5 w-3.5 flex-shrink-0 cursor-pointer accent-primary"
            onClick={(e) => e.stopPropagation()}
            aria-label={`Select insight for ${insight.item_id}`}
          />
        )}
        <div className="mt-0.5 flex-shrink-0">
          <SeverityIcon severity={insight.severity} />
        </div>

        <div className="min-w-0 flex-1">
          {/* Badges row */}
          <div className="mb-1 flex flex-wrap items-center gap-2">
            <span className={cn("rounded-full px-2 py-0.5 text-xs font-semibold uppercase", s.badge)}>
              {insight.severity}
            </span>
            <span className="flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">
              <TypeIcon className="h-3 w-3" />
              {INSIGHT_TYPE_LABELS[insight.insight_type]}
            </span>
            {insight.status !== "open" && (
              <span className="rounded-full bg-muted px-2 py-0.5 text-xs capitalize text-muted-foreground">
                {insight.status}
              </span>
            )}
            <ConfidenceBadge insight={insight} />
          </div>

          {/* DFU identity */}
          <div className="mb-1 text-sm font-semibold text-foreground">
            {insight.item_id} @ {insight.loc}
            {insight.abc_vol && (
              <span className="ml-2 text-xs font-normal text-muted-foreground">
                ABC: {insight.abc_vol}
              </span>
            )}
            {insight.cluster_assignment && (
              <span className="ml-1 text-xs font-normal text-muted-foreground">
                • {insight.cluster_assignment}
              </span>
            )}
          </div>

          {/* Summary — the 1-sentence signal */}
          <p className="mb-2 text-sm font-medium text-foreground">{insight.summary}</p>

          {/* Type explainer — contextual background */}
          <p className="mb-3 rounded bg-muted/40 px-3 py-2 text-xs leading-relaxed text-muted-foreground">
            <strong>Why this matters:</strong> {INSIGHT_TYPE_EXPLAINERS[insight.insight_type]}
          </p>

          {/* Visual causal chain */}
          <CausalChainCard insight={insight} />

          {/* Recommendation */}
          <p className="mt-3 mb-3 border-l-2 border-muted pl-3 text-sm text-muted-foreground">
            <span className="font-semibold text-foreground">Action: </span>
            {insight.recommendation}
          </p>

          {/* Metrics row — contextual labels */}
          <div className="mb-3 flex flex-wrap gap-3 text-xs">
            {insight.dos != null && (
              <span
                className={cn(
                  "rounded px-2 py-1",
                  insight.dos < 14 ? "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300" : "bg-muted",
                )}
                title="Days of Supply — how many days current inventory will last at the current sales rate"
              >
                Days of Supply: <strong>{insight.dos.toFixed(0)}d</strong>
                {insight.total_lt_days != null && (
                  <span className="ml-1 text-muted-foreground">(lead time: {insight.total_lt_days}d)</span>
                )}
              </span>
            )}
            {insight.champion_wape != null && (
              <span
                className={cn(
                  "rounded px-2 py-1",
                  insight.champion_wape > 0.35 ? "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300" : "bg-muted",
                )}
                title="Weighted Absolute Percentage Error — lower is better. >35% is high, >50% is critical."
              >
                Forecast Error (WAPE): <strong>{(insight.champion_wape * 100).toFixed(1)}%</strong>
              </span>
            )}
            {insight.forecast_bias_pct != null && (
              <span
                className={cn(
                  "rounded px-2 py-1",
                  Math.abs(insight.forecast_bias_pct) > 0.2 ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300" : "bg-muted",
                )}
                title="Forecast bias: positive means over-forecasting, negative means under-forecasting"
              >
                Bias: <strong>{insight.forecast_bias_pct > 0 ? "+" : ""}{(insight.forecast_bias_pct * 100).toFixed(1)}%</strong>
                <span className="ml-1 text-muted-foreground">({insight.forecast_bias_pct > 0 ? "over-forecast" : "under-forecast"})</span>
              </span>
            )}
            {insight.current_policy_id && (
              <span className="rounded bg-muted px-2 py-1" title="Current replenishment policy assigned to this item-location">
                Policy: <strong>{insight.current_policy_id}</strong>
              </span>
            )}
            {insight.financial_impact_estimate != null && (
              <span
                className="rounded bg-amber-100 px-2 py-1 font-semibold text-amber-800 dark:bg-amber-900/30 dark:text-amber-300"
                title="Estimated financial exposure if no action is taken — includes lost revenue, carrying cost, or obsolescence risk"
              >
                {fmtCurrency(insight.financial_impact_estimate)} at risk
              </span>
            )}
          </div>

          {/* Detailed AI reasoning — expanded by default for full transparency */}
          {insight.reasoning && (
            <div className="mb-3">
              <button
                onClick={() => setExpanded(!expanded)}
                className="flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground"
              >
                {expanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                AI Reasoning Chain
              </button>
              {expanded && (
                <div className="mt-2 rounded border border-muted bg-muted/30 p-3 text-xs leading-relaxed text-muted-foreground">
                  <p className="mb-1 text-[10px] font-bold uppercase tracking-wider text-muted-foreground/70">
                    Step-by-step analysis from data signals to recommendation
                  </p>
                  <p className="whitespace-pre-line">{insight.reasoning}</p>
                </div>
              )}
            </div>
          )}

          {/* Action buttons — open modal, not direct mutations */}
          {insight.status === "open" && (
            <div className="flex gap-2 flex-wrap">
              <Button
                size="sm"
                variant="outline"
                onClick={() => onAcknowledge(insight)}
              >
                Accept
              </Button>
              <Button
                size="sm"
                variant="ghost"
                className="text-muted-foreground"
                onClick={() => onResolve(insight)}
              >
                Resolve
              </Button>
              {/* Snooze dialog (PL-012 enhanced with reason + date) */}
              {showSnoozePicker ? (
                <div className="w-full rounded-md border bg-muted/30 p-2 space-y-2">
                  <div className="flex items-center gap-1">
                    <span className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">Snooze until</span>
                    <button onClick={() => { setShowSnoozePicker(false); setSnoozeReason(""); setSnoozeDays(null); }} className="ml-auto text-muted-foreground hover:text-foreground px-1 text-xs">&#10005;</button>
                  </div>
                  <div className="flex items-center gap-1">
                    {[1, 3, 7, 14].map((d) => (
                      <button
                        key={d}
                        onClick={() => setSnoozeDays(snoozeDays === d ? null : d)}
                        className={cn(
                          "rounded border px-2 py-0.5 text-xs font-medium transition-colors",
                          snoozeDays === d
                            ? "border-primary bg-primary/10 text-primary"
                            : "border-input bg-background hover:bg-muted text-muted-foreground",
                        )}
                      >
                        {d}d
                      </button>
                    ))}
                  </div>
                  <input
                    type="text"
                    placeholder="Reason (optional) — e.g. confirmed with supplier"
                    value={snoozeReason}
                    onChange={(e) => setSnoozeReason(e.target.value)}
                    className="w-full rounded border border-input bg-background px-2 py-1 text-xs placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
                  />
                  <div className="flex justify-end gap-1">
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-6 text-xs"
                      onClick={() => { setShowSnoozePicker(false); setSnoozeReason(""); setSnoozeDays(null); }}
                    >
                      Cancel
                    </Button>
                    <Button
                      size="sm"
                      className="h-6 text-xs"
                      disabled={snoozeDays == null}
                      onClick={() => {
                        if (snoozeDays != null) {
                          onSnooze(insight, snoozeDays);
                          setShowSnoozePicker(false);
                          setSnoozeReason("");
                          setSnoozeDays(null);
                        }
                      }}
                    >
                      Confirm snooze
                    </Button>
                  </div>
                </div>
              ) : (
                <Button
                  size="sm"
                  variant="ghost"
                  className="text-muted-foreground"
                  title="Snooze this insight"
                  onClick={() => setShowSnoozePicker(true)}
                >
                  Snooze
                </Button>
              )}
            </div>
          )}
          {insight.status === "acknowledged" && (
            <Button
              size="sm"
              variant="ghost"
              className="text-muted-foreground"
              onClick={() => onResolve(insight)}
            >
              Mark Resolved
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
