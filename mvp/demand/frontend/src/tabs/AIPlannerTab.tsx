/**
 * IPAIfeature1 — AI Planning Agent
 *
 * Exception work-queue: structured, ranked, actionable insights generated
 * by an AI agent that reads across all data layers and traces causal chains.
 * NOT a chatbot — a proactive scan-and-triage system for planners.
 */
import { useState, useEffect, useRef, useCallback } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  queryKeys,
  fetchAiInsights,
  fetchAiMemos,
  triggerPortfolioScan,
  updateInsightStatus,
  triggerAutoAccept,
  snoozeInsight,
  STALE,
} from "@/api/queries";
import type { AutoAcceptResponse } from "@/api/queries/ai-planner";
import type { AiInsight, InsightSeverity, InsightStatus, InsightType } from "@/types/ai-planner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { formatCurrency as fmtCurrency } from "@/lib/formatters";
import {
  AlertTriangle,
  AlertCircle,
  Info,
  CheckCircle2,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  ChevronRight,
  Loader2,
  Brain,
  TrendingDown,
  Package,
  BarChart3,
  Zap,
  Target,
  DollarSign,
  X,
  Copy,
  Check,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const SEVERITY_ORDER: Record<InsightSeverity, number> = {
  critical: 0,
  high: 1,
  medium: 2,
  low: 3,
};

const SEVERITY_STYLES: Record<InsightSeverity, { badge: string; border: string; dot: string }> = {
  critical: {
    badge: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",
    border: "border-l-red-500",
    dot: "bg-red-500",
  },
  high: {
    badge: "bg-orange-100 text-orange-800 dark:bg-orange-900/40 dark:text-orange-300",
    border: "border-l-orange-500",
    dot: "bg-orange-500",
  },
  medium: {
    badge: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300",
    border: "border-l-yellow-500",
    dot: "bg-yellow-500",
  },
  low: {
    badge: "bg-gray-100 text-gray-700 dark:bg-gray-700/40 dark:text-gray-300",
    border: "border-l-gray-400",
    dot: "bg-gray-400",
  },
};

const INSIGHT_TYPE_ICONS: Record<InsightType, React.FC<{ className?: string }>> = {
  stockout_risk: ({ className }) => <Package className={className} />,
  excess_inventory: ({ className }) => <TrendingDown className={className} />,
  forecast_bias: ({ className }) => <BarChart3 className={className} />,
  policy_gap: ({ className }) => <Target className={className} />,
  champion_degradation: ({ className }) => <Zap className={className} />,
};

const INSIGHT_TYPE_LABELS: Record<InsightType, string> = {
  stockout_risk: "Stockout Risk",
  excess_inventory: "Excess Inventory",
  forecast_bias: "Forecast Bias",
  policy_gap: "Policy Gap",
  champion_degradation: "Model Degradation",
};

// ---------------------------------------------------------------------------
// AI confidence tier — derived from insight metrics
// ---------------------------------------------------------------------------
type ConfidenceTier = "high" | "medium" | "low";

function deriveConfidence(insight: AiInsight): ConfidenceTier {
  const wape = insight.champion_wape ? insight.champion_wape * 100 : null;
  const bias = insight.forecast_bias_pct ? Math.abs(insight.forecast_bias_pct) : null;
  const hasFinancial = insight.financial_impact_estimate != null && insight.financial_impact_estimate > 0;

  const strongSignal = (wape != null && wape > 50) || (bias != null && bias > 50);
  const moderateSignal = (wape != null && wape > 35) || (bias != null && bias > 20);

  if (strongSignal && hasFinancial) return "high";
  if (moderateSignal || strongSignal) return "medium";
  return "low";
}

const CONFIDENCE_STYLES: Record<ConfidenceTier, { label: string; className: string; tooltip: string }> = {
  high:   {
    label: "HIGH",
    className: "bg-teal-100 text-teal-800 dark:bg-teal-900/40 dark:text-teal-300",
    tooltip: "High confidence — forecast WAPE < 35% with financial impact confirmed. Act on this insight with high certainty.",
  },
  medium: {
    label: "MED",
    className: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300",
    tooltip: "Medium confidence — moderate forecast bias or variability detected. Review the AI reasoning before acting.",
  },
  low:    {
    label: "LOW",
    className: "bg-gray-100 text-gray-600 dark:bg-gray-700/40 dark:text-gray-400",
    tooltip: "Low confidence — limited signal strength or no financial impact data. Treat as an early warning, not a directive.",
  },
};

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
// CausalChainCard — visual [Forecast]→[Inventory]→[Policy]→[Financial] chain
// ---------------------------------------------------------------------------
interface CausalLink {
  layer: "forecast" | "inventory" | "policy" | "financial";
  signal: string;
  impact: string;
  isAlert: boolean;
}

const CAUSAL_LAYER_LABELS: Record<CausalLink["layer"], string> = {
  forecast: "Forecast",
  inventory: "Inventory",
  policy: "Policy",
  financial: "Impact",
};

const CAUSAL_LAYER_ICONS: Record<CausalLink["layer"], React.FC<{ className?: string }>> = {
  forecast:  ({ className }) => <BarChart3 className={className} />,
  inventory: ({ className }) => <Package className={className} />,
  policy:    ({ className }) => <Target className={className} />,
  financial: ({ className }) => <DollarSign className={className} />,
};

function buildCausalChain(insight: AiInsight): CausalLink[] {
  const chain: CausalLink[] = [];

  // Forecast layer
  if (insight.champion_wape != null || insight.forecast_bias_pct != null) {
    const wapePct = insight.champion_wape != null
      ? `WAPE ${(insight.champion_wape * 100).toFixed(1)}%` : null;
    const biasPct = insight.forecast_bias_pct != null
      ? `Bias ${insight.forecast_bias_pct > 0 ? "+" : ""}${(insight.forecast_bias_pct * 100).toFixed(1)}%`
      : null;
    const signal = [wapePct, biasPct].filter(Boolean).join(" · ") || "–";
    const isAlert =
      (insight.champion_wape ?? 0) > 0.35 || Math.abs(insight.forecast_bias_pct ?? 0) > 0.20;
    chain.push({
      layer: "forecast",
      signal,
      impact: insight.insight_type === "champion_degradation" ? "Model degrading"
            : insight.insight_type === "forecast_bias" ? "Systematic deviation"
            : "Inaccurate signal",
      isAlert,
    });
  }

  // Inventory layer
  if (insight.dos != null) {
    const lt = insight.total_lt_days;
    const signal = `DOS ${insight.dos.toFixed(0)}d${lt ? ` (LT ${lt}d)` : ""}`;
    const isBelow = lt != null && insight.dos < lt;
    const isExcess = insight.dos > 180;
    chain.push({
      layer: "inventory",
      signal,
      impact: isBelow ? "Below lead time" : isExcess ? "Excess stock" : "Tight coverage",
      isAlert: isBelow || isExcess,
    });
  }

  // Policy layer
  if (insight.current_policy_id || insight.insight_type === "policy_gap") {
    chain.push({
      layer: "policy",
      signal: insight.current_policy_id ?? "No policy",
      impact: insight.insight_type === "policy_gap" ? "Mismatch detected"
            : !insight.current_policy_id ? "Unassigned"
            : "Active policy",
      isAlert: insight.insight_type === "policy_gap" || !insight.current_policy_id,
    });
  }

  // Financial layer — always shown
  chain.push({
    layer: "financial",
    signal: insight.financial_impact_estimate != null
      ? fmtCurrency(insight.financial_impact_estimate)
      : "Est. pending",
    impact: insight.insight_type === "stockout_risk" ? "Lost sales"
          : insight.insight_type === "excess_inventory" ? "Capital locked"
          : "At risk",
    isAlert: (insight.financial_impact_estimate ?? 0) > 1000,
  });

  return chain;
}

function CausalChainCard({ insight }: { insight: AiInsight }) {
  const chain = buildCausalChain(insight);
  if (chain.length < 2) return null;

  return (
    <div className="mt-3 rounded-lg border bg-muted/20 p-3">
      <p className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
        Causal Chain
      </p>
      <div className="flex items-stretch gap-0 overflow-x-auto">
        {chain.map((link, idx) => {
          const LayerIcon = CAUSAL_LAYER_ICONS[link.layer];
          return (
            <div key={link.layer} className="flex items-center">
              <div
                className={cn(
                  "flex min-w-[90px] flex-col items-center gap-0.5 rounded-md px-2 py-2 text-center",
                  link.isAlert
                    ? "bg-red-50 dark:bg-red-950/30"
                    : "bg-muted/40",
                )}
              >
                <LayerIcon
                  className={cn(
                    "h-3.5 w-3.5",
                    link.isAlert ? "text-red-500" : "text-muted-foreground",
                  )}
                />
                <span className="text-[9px] font-semibold uppercase tracking-wide text-muted-foreground">
                  {CAUSAL_LAYER_LABELS[link.layer]}
                </span>
                <span
                  className={cn(
                    "text-[11px] font-bold leading-tight",
                    link.isAlert ? "text-red-700 dark:text-red-400" : "text-foreground",
                  )}
                >
                  {link.signal}
                </span>
                <span className="text-[9px] leading-tight text-muted-foreground">
                  {link.impact}
                </span>
              </div>
              {idx < chain.length - 1 && (
                <ChevronRight className="mx-0.5 h-3.5 w-3.5 flex-shrink-0 text-muted-foreground/40" />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// SeverityIcon
// ---------------------------------------------------------------------------
function SeverityIcon({ severity }: { severity: InsightSeverity }) {
  if (severity === "critical") return <AlertCircle className="h-4 w-4 text-red-500" />;
  if (severity === "high") return <AlertTriangle className="h-4 w-4 text-orange-500" />;
  if (severity === "medium") return <Info className="h-4 w-4 text-yellow-500" />;
  return <CheckCircle2 className="h-4 w-4 text-gray-400" />;
}

// ---------------------------------------------------------------------------
// Confirm Modal
// ---------------------------------------------------------------------------
interface ConfirmActionState {
  insight: AiInsight;
  status: InsightStatus;
  label: string;
  verb: string;
}

function ConfirmModal({
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

// ---------------------------------------------------------------------------
// AutoAcceptModal
// ---------------------------------------------------------------------------
const SEVERITY_THRESHOLD_LABELS: Record<InsightSeverity, string> = {
  critical: "Critical only",
  high: "High and above",
  medium: "Medium and above",
  low: "All severities",
};

function AutoAcceptModal({
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
  // PL-008: track preview result before final confirm
  const [previewResult, setPreviewResult] = useState<AutoAcceptResponse | null>(null);

  // After dry-run succeeds, parent sets result; split into preview vs execute result
  const isDryRunResult = result?.dry_run === true;
  const isExecuteResult = result?.dry_run === false;

  function handlePreview() {
    setPreviewResult(null);
    onConfirm(minSeverity, true);
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-sm">
      <div className="w-full max-w-sm rounded-xl border bg-card shadow-xl">
        <div className="flex items-center justify-between border-b px-5 py-4">
          <div>
            <p className="text-sm font-semibold">Auto-Accept Rules</p>
            <p className="text-xs text-muted-foreground">
              Bulk-accept open insights by severity threshold
            </p>
          </div>
          <button onClick={onCancel} className="rounded-md p-1 text-muted-foreground hover:text-foreground">
            <X className="h-4 w-4" />
          </button>
        </div>

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
                ⚠️ This will permanently accept {result!.accepted} insight{result!.accepted !== 1 ? "s" : ""} and write outcome records. <strong>This cannot be undone.</strong>
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

        <div className="flex justify-end gap-2 border-t px-5 py-3">
          <Button variant="ghost" size="sm" onClick={onCancel}>
            {isExecuteResult ? "Close" : "Cancel"}
          </Button>
          {/* Config → Preview */}
          {!result && (
            <Button variant="outline" size="sm" onClick={handlePreview} disabled={isPending}>
              {isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : "Preview"}
            </Button>
          )}
          {/* Preview with matches → Confirm execute */}
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
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// InsightCard
// ---------------------------------------------------------------------------
function InsightCard({
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
  const [expanded, setExpanded] = useState(false);
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
            aria-label={`Select insight for ${insight.item_no}`}
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
            {insight.item_no} @ {insight.loc}
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

          {/* Visual causal chain — replaces text block */}
          <CausalChainCard insight={insight} />

          {/* Recommendation */}
          <p className="mt-3 mb-3 border-l-2 border-muted pl-3 text-sm text-muted-foreground">
            <span className="font-semibold text-foreground">Action: </span>
            {insight.recommendation}
          </p>

          {/* Metrics row */}
          <div className="mb-3 flex flex-wrap gap-3 text-xs">
            {insight.dos != null && (
              <span className="rounded bg-muted px-2 py-1">
                DOS: <strong>{insight.dos.toFixed(0)}d</strong>
              </span>
            )}
            {insight.champion_wape != null && (
              <span className="rounded bg-muted px-2 py-1">
                WAPE: <strong>{(insight.champion_wape * 100).toFixed(1)}%</strong>
              </span>
            )}
            {insight.current_policy_id && (
              <span className="rounded bg-muted px-2 py-1">
                Policy: <strong>{insight.current_policy_id}</strong>
              </span>
            )}
            {insight.financial_impact_estimate != null && (
              <span className="rounded bg-amber-100 px-2 py-1 font-semibold text-amber-800 dark:bg-amber-900/30 dark:text-amber-300">
                {fmtCurrency(insight.financial_impact_estimate)} at risk
              </span>
            )}
          </div>

          {/* Detailed reasoning toggle (secondary — behind disclosure) */}
          {insight.reasoning && (
            <div className="mb-3">
              <button
                onClick={() => setExpanded(!expanded)}
                className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
              >
                {expanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                Full AI Reasoning
              </button>
              {expanded && (
                <p className="mt-2 rounded bg-muted/50 p-3 text-xs leading-relaxed text-muted-foreground">
                  {insight.reasoning}
                </p>
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
                    <button onClick={() => { setShowSnoozePicker(false); setSnoozeReason(""); setSnoozeDays(null); }} className="ml-auto text-muted-foreground hover:text-foreground px-1 text-xs">✕</button>
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

// ---------------------------------------------------------------------------
// Main tab component
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// BulkActionBar — sticky bar shown when ≥1 insight selected
// ---------------------------------------------------------------------------
function BulkActionBar({
  count,
  onAcknowledgeAll,
  onClear,
  isPending,
}: {
  count: number;
  onAcknowledgeAll: () => void;
  onClear: () => void;
  isPending: boolean;
}) {
  return (
    <div className="fixed bottom-6 left-1/2 z-40 -translate-x-1/2 flex items-center gap-3 rounded-full border bg-card px-5 py-2.5 shadow-lg">
      <span className="text-sm font-medium">{count} selected</span>
      <div className="h-4 w-px bg-border" />
      <Button size="sm" onClick={onAcknowledgeAll} disabled={isPending} className="rounded-full gap-1.5">
        {isPending && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
        Accept all
      </Button>
      <Button size="sm" variant="ghost" onClick={onClear} className="rounded-full text-muted-foreground">
        <X className="h-3.5 w-3.5" />
      </Button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// CopyButton — copies text to clipboard with transient "Copied!" state
// ---------------------------------------------------------------------------
function CopyButton({ text, label = "Copy" }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  function handleCopy() {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }
  return (
    <button
      onClick={handleCopy}
      className="flex items-center gap-1 rounded border border-input bg-background px-2 py-1 text-xs hover:bg-muted"
      title="Copy memo to clipboard"
    >
      {copied ? <Check className="h-3 w-3 text-green-600" /> : <Copy className="h-3 w-3" />}
      {copied ? "Copied!" : label}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Main tab component
// ---------------------------------------------------------------------------
export default function AIPlannerTab() {
  const qc = useQueryClient();

  // ── Persistent URL filter state ──────────────────────────────────────────
  // Read initial values from URL params so filters survive page refresh
  const getUrlParam = (key: string, fallback: string) => {
    try { return new URLSearchParams(window.location.search).get(key) ?? fallback; } catch { return fallback; }
  };
  const [severityFilter, setSeverityFilter] = useState(() => getUrlParam("ai_severity", "all"));
  const [statusFilter, setStatusFilter] = useState(() => getUrlParam("ai_status", "open"));
  const [typeFilter, setTypeFilter] = useState(() => getUrlParam("ai_type", "all"));

  // Sync filter changes to URL (pushState, doesn't reload page)
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (severityFilter === "all") params.delete("ai_severity"); else params.set("ai_severity", severityFilter);
    if (statusFilter === "open") params.delete("ai_status"); else params.set("ai_status", statusFilter);
    if (typeFilter === "all") params.delete("ai_type"); else params.set("ai_type", typeFilter);
    const newSearch = params.toString();
    const newUrl = `${window.location.pathname}${newSearch ? "?" + newSearch : ""}`;
    window.history.replaceState(null, "", newUrl);
  }, [severityFilter, statusFilter, typeFilter]);

  // ── Bulk selection state ──────────────────────────────────────────────────
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());

  const handleSelect = useCallback((id: number) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) { next.delete(id); } else { next.add(id); }
      return next;
    });
  }, []);

  const clearSelection = useCallback(() => setSelectedIds(new Set()), []);

  // Clear selection when filters change
  useEffect(() => { setSelectedIds(new Set()); }, [severityFilter, statusFilter, typeFilter]);

  // Confirm modal state
  const [confirmAction, setConfirmAction] = useState<ConfirmActionState | null>(null);

  // Auto-accept modal state
  const [showAutoAccept, setShowAutoAccept] = useState(false);
  const [autoAcceptResult, setAutoAcceptResult] = useState<AutoAcceptResponse | null>(null);

  const autoAcceptMutation = useMutation({
    mutationFn: triggerAutoAccept,
    onSuccess: (data) => {
      setAutoAcceptResult(data);
      if (!data.dry_run) {
        qc.invalidateQueries({ queryKey: queryKeys.aiInsights({}) });
      }
    },
  });

  const insightParams = {
    ...(severityFilter !== "all" && { severity: severityFilter as InsightSeverity }),
    ...(statusFilter !== "all" && { status: statusFilter as InsightStatus }),
    ...(typeFilter !== "all" && { insight_type: typeFilter as InsightType }),
    page_size: 50,
  };

  const insightsQ = useQuery({
    queryKey: queryKeys.aiInsights(insightParams),
    queryFn: () => fetchAiInsights(insightParams),
    staleTime: STALE.THIRTY_SEC,
    refetchInterval: 30_000,
  });

  const memosQ = useQuery({
    queryKey: queryKeys.aiMemos({ scope: "portfolio", limit: 5 }),
    queryFn: () => fetchAiMemos({ scope: "portfolio", limit: 5 }),
    staleTime: STALE.FIVE_MIN,
  });
  const [memoIndex, setMemoIndex] = useState(0); // PL-017: 0 = latest

  const [showScanSuccess, setShowScanSuccess] = useState(false);
  const [scanQueuedAt, setScanQueuedAt] = useState<Date | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // PL-007: Poll for new insights for up to 5 minutes after a scan is queued
  useEffect(() => {
    if (!scanQueuedAt) return;
    const POLL_INTERVAL = 30_000; // 30 s
    const TIMEOUT = 5 * 60_000;  // 5 min
    pollRef.current = setInterval(() => {
      qc.invalidateQueries({ queryKey: queryKeys.aiInsights({}) });
      qc.invalidateQueries({ queryKey: queryKeys.aiMemos({}) });
      if (Date.now() - scanQueuedAt.getTime() > TIMEOUT) {
        clearInterval(pollRef.current!);
        setScanQueuedAt(null);
      }
    }, POLL_INTERVAL);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [scanQueuedAt, qc]);

  const scanMutation = useMutation({
    mutationFn: triggerPortfolioScan,
    onSuccess: () => {
      setShowScanSuccess(true);
      setScanQueuedAt(new Date());
      setTimeout(() => setShowScanSuccess(false), 8000);
    },
  });

  const statusMutation = useMutation({
    mutationFn: ({ id, status }: { id: number; status: InsightStatus }) =>
      updateInsightStatus(id, status),
    onSuccess: () => {
      setConfirmAction(null);
      qc.invalidateQueries({ queryKey: queryKeys.aiInsights({}) });
    },
  });

  // Open confirm modal for acknowledge (Accept) action
  function handleAcknowledge(insight: AiInsight) {
    setConfirmAction({
      insight,
      status: "acknowledged",
      label: "Accept Recommendation",
      verb: "Accept",
    });
  }

  // Open confirm modal for resolve action
  function handleResolve(insight: AiInsight) {
    setConfirmAction({
      insight,
      status: "resolved",
      label: "Mark as Resolved",
      verb: "Resolve",
    });
  }

  const snoozeMutation = useMutation({
    mutationFn: ({ id, days }: { id: number; days: number }) => snoozeInsight(id, days),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.aiInsights({}) });
    },
  });

  function handleSnooze(insight: AiInsight, days: number) {
    snoozeMutation.mutate({ id: insight.insight_id, days });
  }

  // Bulk acknowledge — fires one mutation per selected insight sequentially
  const [bulkPending, setBulkPending] = useState(false);
  async function handleBulkAcknowledge() {
    if (selectedIds.size === 0) return;
    setBulkPending(true);
    try {
      for (const id of selectedIds) {
        await updateInsightStatus(id, "acknowledged");
      }
      qc.invalidateQueries({ queryKey: queryKeys.aiInsights({}) });
      setSelectedIds(new Set());
    } finally {
      setBulkPending(false);
    }
  }

  function executeConfirmedAction() {
    if (!confirmAction) return;
    statusMutation.mutate({ id: confirmAction.insight.insight_id, status: confirmAction.status });
  }

  const insights = insightsQ.data?.insights ?? [];
  const total = insightsQ.data?.total ?? 0;
  const allMemos = memosQ.data?.memos ?? [];
  const latestMemo = allMemos[memoIndex]; // PL-017: navigate history

  // Last-scan timestamp
  const lastScanAt = latestMemo?.created_at ?? (insights.length > 0 ? insights[0].created_at : null);
  const lastScanLabel = lastScanAt
    ? (() => {
        const diffMs = Date.now() - new Date(lastScanAt).getTime();
        const diffMin = Math.floor(diffMs / 60_000);
        const diffHr = Math.floor(diffMin / 60);
        const diffDay = Math.floor(diffHr / 24);
        if (diffMin < 2) return "just now";
        if (diffMin < 60) return `${diffMin} min ago`;
        if (diffHr < 24) return `${diffHr}h ago`;
        return `${diffDay}d ago`;
      })()
    : null;

  const openInsights = insights.filter((i) => i.status !== "resolved");
  const criticalCount = insights.filter((i) => i.severity === "critical").length;

  const sorted = [...insights].sort((a, b) => {
    const so = SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity];
    if (so !== 0) return so;
    return (b.financial_impact_estimate ?? 0) - (a.financial_impact_estimate ?? 0);
  });

  return (
    <>
      {/* Confirm modal */}
      {confirmAction && (
        <ConfirmModal
          action={confirmAction}
          onConfirm={executeConfirmedAction}
          onCancel={() => setConfirmAction(null)}
          isPending={statusMutation.isPending}
        />
      )}

      {/* Auto-accept modal */}
      {showAutoAccept && (
        <AutoAcceptModal
          onConfirm={(minSeverity, dryRun) =>
            autoAcceptMutation.mutate({ min_severity: minSeverity, insight_types: [], dry_run: dryRun })
          }
          onCancel={() => {
            setShowAutoAccept(false);
            setAutoAcceptResult(null);
            autoAcceptMutation.reset();
          }}
          isPending={autoAcceptMutation.isPending}
          result={autoAcceptResult}
        />
      )}

      <div className="space-y-6">
        {/* ---------------------------------------------------------------- */}
        {/* Header                                                           */}
        {/* ---------------------------------------------------------------- */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Brain className="h-6 w-6 text-violet-500" />
            <div>
              <h2 className="text-xl font-semibold">AI Planner</h2>
              <p className="text-sm text-muted-foreground">
                Proactive exception work-queue — AI-diagnosed planning issues ranked by financial impact
                {lastScanLabel && (
                  <span className="ml-2 text-xs text-muted-foreground/70">
                    · Last scan: {lastScanLabel}
                  </span>
                )}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              onClick={() => { setAutoAcceptResult(null); setShowAutoAccept(true); }}
              disabled={showAutoAccept}
              className="gap-2"
            >
              <Zap className="h-4 w-4" />
              Auto-Accept
            </Button>
            <Button
              onClick={() => scanMutation.mutate()}
              disabled={scanMutation.isPending}
              className="gap-2"
            >
              {scanMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Scanning portfolio…
                </>
              ) : (
                <>
                  <RefreshCw className="h-4 w-4" />
                  Generate Now
                </>
              )}
            </Button>
          </div>
        </div>

        {/* ---------------------------------------------------------------- */}
        {/* Distinction banner (PL-004)                                     */}
        {/* ---------------------------------------------------------------- */}
        <div className="rounded-md border border-violet-200 bg-violet-50 px-4 py-2 text-xs text-violet-700 dark:border-violet-800 dark:bg-violet-950/30 dark:text-violet-300">
          <strong>AI Planner</strong> shows ML-generated insights ranked by financial impact. For rule-based threshold alerts from replenishment policies, see the <strong>Exceptions</strong> tab.
        </div>

        {/* ---------------------------------------------------------------- */}
        {/* Portfolio Health Bar                                             */}
        {/* ---------------------------------------------------------------- */}
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {[
            {
              label: "Open Insights",
              value: statusFilter === "open" ? String(total) : String(openInsights.length),
              color: "text-foreground",
            },
            {
              label: "Critical",
              value: String(criticalCount),
              color: criticalCount > 0 ? "text-red-600 dark:text-red-400" : "text-foreground",
            },
            {
              label: "High Priority",
              value: String(insights.filter((i) => i.severity === "high").length),
              color: "text-orange-600 dark:text-orange-400",
            },
            {
              label: "Total Financial Risk",
              value: fmtCurrency(
                insights.reduce((s, i) => s + (i.financial_impact_estimate ?? 0), 0),
              ),
              color: "text-amber-600 dark:text-amber-400",
            },
          ].map((kpi) => (
            <Card key={kpi.label} className="py-3">
              <CardContent className="px-4">
                <p className="text-xs text-muted-foreground">{kpi.label}</p>
                <p className={cn("text-2xl font-bold", kpi.color)}>{kpi.value}</p>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Scan success/error banner (PL-007) */}
        {showScanSuccess && (
          <div className="rounded-md bg-green-50 px-4 py-2 text-sm text-green-800 dark:bg-green-900/20 dark:text-green-300 flex items-center gap-2">
            <Loader2 className="h-3.5 w-3.5 animate-spin flex-shrink-0" />
            Portfolio scan in progress — insights will refresh automatically. Track in the <strong>Jobs</strong> tab.
          </div>
        )}
        {!showScanSuccess && scanQueuedAt && (
          <div className="rounded-md border border-blue-200 bg-blue-50 px-4 py-2 text-sm text-blue-800 dark:border-blue-800 dark:bg-blue-950/30 dark:text-blue-300 flex items-center gap-2">
            <Loader2 className="h-3.5 w-3.5 animate-spin flex-shrink-0" />
            Background scan running — checking for new insights every 30s.
            <button className="ml-auto text-xs underline" onClick={() => { setScanQueuedAt(null); if (pollRef.current) clearInterval(pollRef.current); }}>Dismiss</button>
          </div>
        )}
        {scanMutation.isError && (
          <div className="rounded-md bg-red-50 px-4 py-2 text-sm text-red-800 dark:bg-red-900/20 dark:text-red-300">
            Scan failed: {(scanMutation.error as Error)?.message ?? "Unknown error"}
          </div>
        )}

        {/* ---------------------------------------------------------------- */}
        {/* Filter bar                                                       */}
        {/* ---------------------------------------------------------------- */}
        <div className="flex flex-wrap gap-3">
          <Select value={severityFilter} onValueChange={setSeverityFilter}>
            <SelectTrigger className="w-36">
              <SelectValue placeholder="Severity" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Severities</SelectItem>
              <SelectItem value="critical">Critical</SelectItem>
              <SelectItem value="high">High</SelectItem>
              <SelectItem value="medium">Medium</SelectItem>
              <SelectItem value="low">Low</SelectItem>
            </SelectContent>
          </Select>

          <Select value={statusFilter} onValueChange={setStatusFilter}>
            <SelectTrigger className="w-36">
              <SelectValue placeholder="Status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="open">Open</SelectItem>
              <SelectItem value="acknowledged">Acknowledged</SelectItem>
              <SelectItem value="resolved">Resolved</SelectItem>
              <SelectItem value="all">All Statuses</SelectItem>
            </SelectContent>
          </Select>

          <Select value={typeFilter} onValueChange={setTypeFilter}>
            <SelectTrigger className="w-44">
              <SelectValue placeholder="Insight Type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Types</SelectItem>
              <SelectItem value="stockout_risk">Stockout Risk</SelectItem>
              <SelectItem value="excess_inventory">Excess Inventory</SelectItem>
              <SelectItem value="forecast_bias">Forecast Bias</SelectItem>
              <SelectItem value="policy_gap">Policy Gap</SelectItem>
              <SelectItem value="champion_degradation">Model Degradation</SelectItem>
            </SelectContent>
          </Select>

          {insightsQ.isFetching && (
            <Loader2 className="h-5 w-5 animate-spin self-center text-muted-foreground" />
          )}
        </div>

        {/* ---------------------------------------------------------------- */}
        {/* Insight card list                                                */}
        {/* ---------------------------------------------------------------- */}
        {insightsQ.isLoading ? (
          <div className="flex items-center justify-center py-16 text-muted-foreground">
            <Loader2 className="mr-2 h-5 w-5 animate-spin" />
            Loading insights…
          </div>
        ) : insightsQ.isError ? (
          <div className="rounded-md border border-destructive/30 bg-destructive/10 p-6 text-center text-sm text-destructive">
            Failed to load insights: {(insightsQ.error as Error)?.message}
          </div>
        ) : sorted.length === 0 ? (
          <Card>
            <CardContent className="py-16 text-center">
              <Brain className="mx-auto mb-3 h-10 w-10 text-muted-foreground/40" />
              {statusFilter === "open" && severityFilter === "all" && typeFilter === "all" ? (
                <>
                  <p className="text-sm font-medium text-green-700 dark:text-green-400">
                    Portfolio looks healthy!
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    No open exceptions found.
                    {lastScanLabel && ` Last scan: ${lastScanLabel}.`}
                    {" "}Click "Generate Now" to run a fresh scan.
                  </p>
                </>
              ) : (
                <>
                  <p className="text-sm font-medium text-muted-foreground">No insights match your filters</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Adjust the severity, status, or type filters to see more results.
                  </p>
                </>
              )}
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-3">
            {sorted.map((insight) => (
              <InsightCard
                key={insight.insight_id}
                insight={insight}
                selected={selectedIds.has(insight.insight_id)}
                onSelect={handleSelect}
                onAcknowledge={handleAcknowledge}
                onResolve={handleResolve}
                onSnooze={handleSnooze}
              />
            ))}
            {total > sorted.length && (
              <p className="text-center text-xs text-muted-foreground">
                Showing {sorted.length} of {total} insights
              </p>
            )}
          </div>
        )}

        {/* ---------------------------------------------------------------- */}
        {/* Bulk action bar                                                  */}
        {/* ---------------------------------------------------------------- */}
        {selectedIds.size > 0 && (
          <BulkActionBar
            count={selectedIds.size}
            onAcknowledgeAll={handleBulkAcknowledge}
            onClear={clearSelection}
            isPending={bulkPending}
          />
        )}

        {/* ---------------------------------------------------------------- */}
        {/* Planning Memo panel                                              */}
        {/* ---------------------------------------------------------------- */}
        {latestMemo && (
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-base">Planning Memo</CardTitle>
                <div className="flex items-center gap-2">
                  <CopyButton text={latestMemo.narrative_text} label="Copy markdown" />
                  {latestMemo.model_version && (
                    <Badge variant="outline" className="text-xs">
                      {latestMemo.model_version}
                    </Badge>
                  )}
                  <span className="text-xs text-muted-foreground">
                    {new Date(latestMemo.period).toLocaleDateString("en-US", {
                      year: "numeric",
                      month: "long",
                    })}
                  </span>
                  {/* PL-017: History navigation */}
                  {allMemos.length > 1 && (
                    <div className="flex items-center gap-1 ml-1">
                      <button
                        onClick={() => setMemoIndex((i) => Math.min(i + 1, allMemos.length - 1))}
                        disabled={memoIndex >= allMemos.length - 1}
                        className="rounded p-0.5 text-muted-foreground hover:text-foreground disabled:opacity-30"
                        title="Older memo"
                      >
                        ‹
                      </button>
                      <span className="text-[10px] text-muted-foreground tabular-nums">
                        {memoIndex + 1}/{allMemos.length}
                      </span>
                      <button
                        onClick={() => setMemoIndex((i) => Math.max(i - 1, 0))}
                        disabled={memoIndex === 0}
                        className="rounded p-0.5 text-muted-foreground hover:text-foreground disabled:opacity-30"
                        title="Newer memo"
                      >
                        ›
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="prose prose-sm dark:prose-invert max-w-none">
                <pre className="whitespace-pre-wrap rounded bg-muted/50 p-4 text-xs leading-relaxed text-muted-foreground">
                  {latestMemo.narrative_text}
                </pre>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </>
  );
}
