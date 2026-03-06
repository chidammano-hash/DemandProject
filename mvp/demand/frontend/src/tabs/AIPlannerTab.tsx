/**
 * IPAIfeature1 — AI Planning Agent
 *
 * Exception work-queue: structured, ranked, actionable insights generated
 * by an AI agent that reads across all data layers and traces causal chains.
 * NOT a chatbot — a proactive scan-and-triage system for planners.
 */
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  queryKeys,
  fetchAiInsights,
  fetchAiMemos,
  triggerPortfolioScan,
  updateInsightStatus,
  STALE,
} from "@/api/queries";
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

const CONFIDENCE_STYLES: Record<ConfidenceTier, { label: string; className: string }> = {
  high:   { label: "HIGH", className: "bg-teal-100 text-teal-800 dark:bg-teal-900/40 dark:text-teal-300" },
  medium: { label: "MED",  className: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300" },
  low:    { label: "LOW",  className: "bg-gray-100 text-gray-600 dark:bg-gray-700/40 dark:text-gray-400" },
};

function ConfidenceBadge({ insight }: { insight: AiInsight }) {
  const tier = deriveConfidence(insight);
  const style = CONFIDENCE_STYLES[tier];
  return (
    <span className={cn("rounded-full px-2 py-0.5 text-xs font-medium", style.className)}>
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
// InsightCard
// ---------------------------------------------------------------------------
function InsightCard({
  insight,
  onAcknowledge,
  onResolve,
}: {
  insight: AiInsight;
  onAcknowledge: (insight: AiInsight) => void;
  onResolve: (insight: AiInsight) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const s = SEVERITY_STYLES[insight.severity];
  const TypeIcon = INSIGHT_TYPE_ICONS[insight.insight_type];

  return (
    <div
      className={cn(
        "rounded-lg border border-l-4 bg-card p-4 shadow-sm transition-all",
        s.border,
        insight.status === "resolved" && "opacity-60",
      )}
    >
      {/* Header row */}
      <div className="flex items-start gap-3">
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
            <div className="flex gap-2">
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
export default function AIPlannerTab() {
  const qc = useQueryClient();

  // Filter state
  const [severityFilter, setSeverityFilter] = useState<string>("all");
  const [statusFilter, setStatusFilter] = useState<string>("open");
  const [typeFilter, setTypeFilter] = useState<string>("all");

  // Confirm modal state
  const [confirmAction, setConfirmAction] = useState<ConfirmActionState | null>(null);

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
    queryKey: queryKeys.aiMemos({ scope: "portfolio", limit: 1 }),
    queryFn: () => fetchAiMemos({ scope: "portfolio", limit: 1 }),
    staleTime: STALE.FIVE_MIN,
  });

  const [showScanSuccess, setShowScanSuccess] = useState(false);

  const scanMutation = useMutation({
    mutationFn: triggerPortfolioScan,
    onSuccess: () => {
      setShowScanSuccess(true);
      setTimeout(() => setShowScanSuccess(false), 5000);
      setTimeout(() => {
        qc.invalidateQueries({ queryKey: queryKeys.aiInsights({}) });
        qc.invalidateQueries({ queryKey: queryKeys.aiMemos({}) });
      }, 5000);
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

  function executeConfirmedAction() {
    if (!confirmAction) return;
    statusMutation.mutate({ id: confirmAction.insight.insight_id, status: confirmAction.status });
  }

  const insights = insightsQ.data?.insights ?? [];
  const total = insightsQ.data?.total ?? 0;
  const latestMemo = memosQ.data?.memos?.[0];

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
      {/* Confirm modal (rendered outside main flow to avoid z-index issues) */}
      {confirmAction && (
        <ConfirmModal
          action={confirmAction}
          onConfirm={executeConfirmedAction}
          onCancel={() => setConfirmAction(null)}
          isPending={statusMutation.isPending}
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

        {/* Scan success/error banner */}
        {showScanSuccess && (
          <div className="rounded-md bg-green-50 px-4 py-2 text-sm text-green-800 dark:bg-green-900/20 dark:text-green-300">
            Portfolio scan queued — track progress in the <strong>Jobs</strong> tab. Insights will appear here when complete.
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
                onAcknowledge={handleAcknowledge}
                onResolve={handleResolve}
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
        {/* Planning Memo panel                                              */}
        {/* ---------------------------------------------------------------- */}
        {latestMemo && (
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-base">Latest Planning Memo</CardTitle>
                <div className="flex items-center gap-2">
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
