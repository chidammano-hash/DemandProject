/**
 * Command Center (Dashboard) — Monday morning work queue.
 * Replaces static metrics with ranked AI insights + AI digest.
 * Plan ref: UI_improvement_plan.md § 2.1
 */
import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  AlertTriangle,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  ArrowRight,
  Sparkles,
  Clock,
  RefreshCw,
  Package,
  BarChart3,
  Target,
  TrendingDown,
  Zap,
} from "lucide-react";

import { KpiCard } from "@/components/KpiCard";
import { AlertPanel } from "@/components/AlertPanel";
import { Skeleton } from "@/components/Skeleton";
import { useScenarioNotification } from "@/context/ScenarioNotificationContext";
import { useJobNotification } from "@/context/JobNotificationContext";
import { cn } from "@/lib/utils";
import { formatCurrency } from "@/lib/formatters";
import {
  queryKeys,
  STALE,
  fetchDashboardKpis,
  fetchDashboardAlerts,
  fetchAiInsights,
  fetchAiMemos,
} from "@/api/queries";
import type { DashboardFilterParams } from "@/api/queries";
import type { Alert } from "@/types/theme";
import type { AiInsight, InsightType } from "@/types/ai-planner";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function formatNumber(n: number | null): string {
  if (n == null) return "N/A";
  if (Math.abs(n) >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (Math.abs(n) >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toFixed(0);
}

function trendDirection(delta: number | null): "up" | "down" | "flat" {
  if (delta == null || delta === 0) return "flat";
  return delta > 0 ? "up" : "down";
}

// PL-013: Map alert types to destination tabs
const ALERT_TYPE_TAB: Partial<Record<string, string>> = {
  oos_risk:             "controlTower",
  bias_drift:           "accuracy",
  low_accuracy:         "accuracy",
  demand_spike:         "aiPlanner",
  allocation_shortage:  "invPlanning",
  scenario_complete:    "clusters",
  job_complete:         "jobs",
};

const INSIGHT_TYPE_ICONS: Record<InsightType, React.FC<{ className?: string }>> = {
  stockout_risk:         ({ className }) => <Package className={className} />,
  excess_inventory:      ({ className }) => <TrendingDown className={className} />,
  forecast_bias:         ({ className }) => <BarChart3 className={className} />,
  policy_gap:            ({ className }) => <Target className={className} />,
  champion_degradation:  ({ className }) => <Zap className={className} />,
};

const SEVERITY_STYLES = {
  critical: {
    dot:    "bg-red-500",
    border: "border-l-red-500 border-red-200 dark:border-red-800",
    badge:  "bg-red-100 text-red-700 dark:bg-red-950/50 dark:text-red-400",
    icon:   "text-red-500",
  },
  high: {
    dot:    "bg-amber-500",
    border: "border-l-amber-500 border-amber-200 dark:border-amber-800",
    badge:  "bg-amber-100 text-amber-700 dark:bg-amber-950/50 dark:text-amber-400",
    icon:   "text-amber-500",
  },
  medium: {
    dot:    "bg-yellow-500",
    border: "border-l-yellow-500 border-yellow-200 dark:border-yellow-800",
    badge:  "bg-yellow-100 text-yellow-700 dark:bg-yellow-950/50 dark:text-yellow-400",
    icon:   "text-yellow-500",
  },
  low: {
    dot:    "bg-slate-400",
    border: "border-l-slate-400 border-slate-200 dark:border-slate-700",
    badge:  "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400",
    icon:   "text-slate-400",
  },
};

// Mini causal chain: 3 inline nodes derived from insight fields
function MiniCausalChain({ insight }: { insight: AiInsight }) {
  const nodes: string[] = [];

  if (insight.champion_wape != null || insight.forecast_bias_pct != null) {
    const wape = insight.champion_wape != null ? `WAPE ${(insight.champion_wape * 100).toFixed(0)}%` : null;
    const bias = insight.forecast_bias_pct != null ? `Bias ${insight.forecast_bias_pct > 0 ? "+" : ""}${(insight.forecast_bias_pct * 100).toFixed(0)}%` : null;
    nodes.push([wape, bias].filter(Boolean).join(" · ") || "Forecast signal");
  }

  if (insight.dos != null) {
    nodes.push(`DOS ${insight.dos.toFixed(1)}d`);
  } else if (insight.total_lt_days != null) {
    nodes.push(`LT ${insight.total_lt_days}d`);
  }

  const outcomeLabel = insight.insight_type === "stockout_risk" ? "Stockout risk"
    : insight.insight_type === "excess_inventory" ? "Excess inventory"
    : insight.insight_type === "forecast_bias" ? "Forecast drift"
    : insight.insight_type === "policy_gap" ? "Policy mismatch"
    : "Model degraded";
  nodes.push(outcomeLabel);

  // Keep max 3 nodes
  const displayed = nodes.slice(0, 3);

  return (
    <div className="flex items-center gap-1 flex-wrap">
      {displayed.map((node, i) => (
        <span key={i} className="flex items-center gap-1">
          <span className="rounded bg-muted px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground">
            {node}
          </span>
          {i < displayed.length - 1 && (
            <ArrowRight className="h-2.5 w-2.5 text-muted-foreground/50 flex-shrink-0" />
          )}
        </span>
      ))}
    </div>
  );
}

// Single priority work queue item
function WorkQueueItem({
  insight,
  rank,
  onNavigate,
}: {
  insight: AiInsight;
  rank: number;
  onNavigate?: (tab: string) => void;
}) {
  const s = SEVERITY_STYLES[insight.severity];
  const TypeIcon = INSIGHT_TYPE_ICONS[insight.insight_type];
  const impact = insight.financial_impact_estimate;

  return (
    <div className={cn(
      "flex gap-3 rounded-md border border-l-4 bg-card px-3 py-2.5",
      s.border,
    )}>
      {/* Rank */}
      <span className="mt-0.5 flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full bg-muted text-[10px] font-bold text-muted-foreground">
        {rank}
      </span>

      {/* Body */}
      <div className="min-w-0 flex-1 space-y-1">
        {/* Header row */}
        <div className="flex items-start gap-2 flex-wrap">
          <span className={cn("rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide", s.badge)}>
            {insight.severity}
          </span>
          <span className="text-xs font-medium text-foreground">
            {insight.item_no} @ {insight.loc}
          </span>
          <TypeIcon className="h-3.5 w-3.5 text-muted-foreground flex-shrink-0 mt-0.5" />
          <span className="text-xs text-muted-foreground truncate">{insight.summary}</span>
        </div>

        {/* Mini causal chain */}
        <MiniCausalChain insight={insight} />

        {/* Footer: impact + actions */}
        <div className="flex items-center gap-2 pt-0.5 flex-wrap">
          {impact != null && impact > 0 && (
            <span className="text-[10px] font-semibold text-amber-600 dark:text-amber-400">
              Impact: {formatCurrency(impact)}
            </span>
          )}
          <div className="ml-auto flex items-center gap-1">
            <button
              onClick={() => onNavigate?.("aiPlanner")}
              className="rounded border border-input bg-background px-2 py-0.5 text-[10px] font-medium hover:bg-muted"
            >
              Review & Accept
            </button>
            <button
              onClick={() => onNavigate?.("aiPlanner")}
              className="rounded px-2 py-0.5 text-[10px] text-muted-foreground hover:text-foreground hover:bg-muted"
            >
              Snooze
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function DashboardTab({ onNavigate }: { onNavigate?: (tab: string) => void }) {
  const { filters } = useGlobalFilterContext();
  const { completedScenario, dismissNotification } = useScenarioNotification();
  const { recentCompletions, dismissCompletion } = useJobNotification();
  const [digestExpanded, setDigestExpanded] = useState(false);

  const filterParams: DashboardFilterParams = {
    brand: filters.brand,
    category: filters.category,
    market: filters.market,
    channel: filters.channel,
    item: filters.item,
    location: filters.location,
    time_grain: filters.timeGrain,
  };
  const fk = filterParams as unknown as Record<string, unknown>;

  // ── Data queries ───────────────────────────────────────────────────────────
  const kpisQuery = useQuery({
    queryKey: queryKeys.dashboardKpis({ ...fk, window: 3 }),
    queryFn: () => fetchDashboardKpis(3, filterParams),
    staleTime: STALE.FIVE_MIN,
  });

  const insightsQuery = useQuery({
    queryKey: queryKeys.aiInsights({ status: "open", page_size: 5 }),
    queryFn: () => fetchAiInsights({ status: "open", page_size: 5 }),
    staleTime: STALE.FIVE_MIN,
  });

  const memosQuery = useQuery({
    queryKey: queryKeys.aiMemos({ limit: 1 }),
    queryFn: () => fetchAiMemos({ limit: 1 }),
    staleTime: STALE.TEN_MIN,
  });

  const alertsQuery = useQuery({
    queryKey: queryKeys.dashboardAlerts(fk),
    queryFn: () => fetchDashboardAlerts(5, filterParams),
    staleTime: STALE.TWO_MIN,
  });

  // ── Derived values ─────────────────────────────────────────────────────────
  const allInsights: AiInsight[] = insightsQuery.data?.insights ?? [];
  const topInsights = allInsights
    .sort((a, b) => {
      const sev = { critical: 0, high: 1, medium: 2, low: 3 };
      return (sev[a.severity] ?? 4) - (sev[b.severity] ?? 4);
    })
    .slice(0, 3);

  const openCount = insightsQuery.data?.total ?? 0;
  const criticalCount = allInsights.filter((i) => i.severity === "critical").length;
  const totalAtRisk = allInsights.reduce((sum, i) => sum + (i.financial_impact_estimate ?? 0), 0);

  const kpis = kpisQuery.data;
  const dos = kpis?.weeks_of_supply != null ? kpis.weeks_of_supply * 7 : null;

  const latestMemo = memosQuery.data?.memos?.[0];
  const digestText = latestMemo?.narrative_text ?? "";
  // First 2 sentences for the collapsed digest preview
  const firstTwoSentences = digestText.split(/(?<=[.!?])\s+/).slice(0, 2).join(" ");

  const lastScanText = latestMemo?.created_at
    ? `AI scan: ${new Date(latestMemo.created_at).toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}`
    : null;

  // Merge scenario + job completion notifications into alerts
  const mergedAlerts: Alert[] = useMemo(() => {
    const base = alertsQuery.data?.alerts ?? [];
    const injected: Alert[] = [];
    if (completedScenario) {
      injected.push({
        id: `scenario-${completedScenario.id}`,
        type: "scenario_complete",
        severity: "low",
        title: `Scenario ${completedScenario.label} Complete`,
        detail: `What-If Scenario finished in ${completedScenario.runtimeSeconds.toFixed(0)}s.`,
        source_tab: "clusters",
      });
    }
    for (const job of recentCompletions) {
      injected.push({
        id: `job-${job.id}`,
        type: "job_complete",
        severity: job.status === "failed" ? "high" : "low",
        title: `${job.label} ${job.status === "completed" ? "Complete" : "Failed"}`,
        detail: job.status === "completed" ? `Finished in ${job.runtimeSeconds.toFixed(0)}s.` : "Check Jobs tab.",
        source_tab: "jobs",
      });
    }
    return [...injected, ...base];
  }, [alertsQuery.data, completedScenario, recentCompletions]);

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="animate-fade-in space-y-5">

      {/* ── Header: greeting + last scan ─────────────────────────────────── */}
      <div className="flex items-center justify-between">
        <div className="space-y-0.5">
          <h2 className="text-lg font-semibold">Command Center</h2>
          <p className="text-sm text-muted-foreground max-w-2xl">
            Your daily priority work queue. Review AI-generated insights ranked by financial impact,
            monitor portfolio health KPIs, and triage active alerts. Start here each morning to
            identify the highest-impact items that need your attention.
          </p>
          {lastScanText && (
            <p className="flex items-center gap-1 text-xs text-muted-foreground">
              <Clock className="h-3 w-3" />
              {lastScanText}
            </p>
          )}
        </div>
        <button
          onClick={() => onNavigate?.("aiPlanner")}
          className="flex items-center gap-1.5 rounded-md border border-input bg-background px-3 py-1.5 text-xs font-medium hover:bg-muted"
        >
          <Sparkles className="h-3.5 w-3.5 text-teal-500" />
          AI Planner
        </button>
      </div>

      {/* ── 4 AI-first KPI cards ─────────────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {insightsQuery.isLoading || kpisQuery.isLoading ? (
          Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-20 w-full rounded-md" />
          ))
        ) : (
          <>
            <KpiCard
              label="Open Insights"
              value={String(openCount)}
              severity={openCount > 5 ? "warning" : openCount === 0 ? "best" : "neutral"}
            />
            <KpiCard
              label="Critical"
              value={String(criticalCount)}
              severity={criticalCount > 0 ? "warning" : "best"}
            />
            <KpiCard
              label="Portfolio DOS"
              value={dos != null ? `${dos.toFixed(1)}d` : "N/A"}
              severity={dos != null && dos < 14 ? "warning" : dos != null && dos > 60 ? "best" : "neutral"}
              trend={kpis?.deltas?.weeks_of_supply != null
                ? { delta: kpis.deltas.weeks_of_supply * 7, direction: trendDirection(kpis.deltas.weeks_of_supply), unit: "d" }
                : undefined}
            />
            <KpiCard
              label="$ at Risk"
              value={totalAtRisk > 0 ? formatCurrency(totalAtRisk) : "$0"}
              severity={totalAtRisk > 50000 ? "warning" : totalAtRisk === 0 ? "best" : "neutral"}
            />
          </>
        )}
      </div>

      {/* ── Two-column row: Work Queue + Alerts ──────────────────────────── */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">

        {/* Priority work queue — top 3 ranked insights */}
        <div className="lg:col-span-2 space-y-2">
          <div className="flex items-center justify-between">
            <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Priority Work Queue
            </p>
            {openCount > 3 && (
              <button
                onClick={() => onNavigate?.("aiPlanner")}
                className="flex items-center gap-1 text-xs text-primary hover:underline"
              >
                +{openCount - 3} more <ArrowRight className="h-3 w-3" />
              </button>
            )}
          </div>

          {insightsQuery.isLoading ? (
            <div className="space-y-2">
              {Array.from({ length: 3 }).map((_, i) => (
                <Skeleton key={i} className="h-20 w-full rounded-md" />
              ))}
            </div>
          ) : topInsights.length > 0 ? (
            <div className="space-y-2">
              {topInsights.map((insight, idx) => (
                <WorkQueueItem
                  key={insight.insight_id}
                  insight={insight}
                  rank={idx + 1}
                  onNavigate={onNavigate}
                />
              ))}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center rounded-md border border-dashed border-border bg-muted/20 py-10 text-center">
              <Sparkles className="h-8 w-8 text-teal-500 mb-2" />
              <p className="text-sm font-medium text-teal-700 dark:text-teal-400">Portfolio looks healthy!</p>
              <p className="text-xs text-muted-foreground mt-1">
                {lastScanText ? `${lastScanText}. No open exceptions.` : "No open AI insights — all DFUs within thresholds."}
              </p>
            </div>
          )}
        </div>

        {/* Alerts panel */}
        <div className="space-y-2">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            System Alerts
          </p>
          {alertsQuery.isLoading ? (
            <Skeleton className="h-40 w-full" />
          ) : mergedAlerts.length > 0 ? (
            <AlertPanel
              alerts={mergedAlerts}
              onDismiss={(completedScenario || recentCompletions.length > 0) ? (id) => {
                if (id.startsWith("scenario-")) dismissNotification();
                if (id.startsWith("job-")) dismissCompletion(id.replace("job-", ""));
              } : undefined}
              onAlertClick={onNavigate ? (alert) => {
                const tab = alert.source_tab ?? ALERT_TYPE_TAB[alert.type];
                if (tab) onNavigate(tab);
              } : undefined}
            />
          ) : (
            <div className="rounded-md border border-dashed border-border bg-muted/20 p-4 text-center">
              <p className="text-xs text-muted-foreground">No active system alerts</p>
            </div>
          )}
        </div>
      </div>

      {/* ── AI Planning Digest ─────────────────────────────────────────────── */}
      {latestMemo && (
        <div className="rounded-md border bg-card">
          <button
            className="flex w-full items-center justify-between px-4 py-3 text-left"
            onClick={() => setDigestExpanded((v) => !v)}
          >
            <div className="flex items-center gap-2">
              <RefreshCw className="h-3.5 w-3.5 text-teal-500" />
              <span className="text-sm font-medium">AI Planning Digest</span>
              {latestMemo.model_version && (
                <span className="rounded-full border px-1.5 py-0.5 text-[10px] text-muted-foreground">
                  {latestMemo.model_version}
                </span>
              )}
            </div>
            {digestExpanded ? (
              <ChevronUp className="h-4 w-4 text-muted-foreground" />
            ) : (
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            )}
          </button>

          <div className={cn("px-4 pb-4 text-sm text-muted-foreground", !digestExpanded && "border-t")}>
            {digestExpanded ? (
              <pre className="mt-3 whitespace-pre-wrap font-sans text-xs leading-relaxed">
                {digestText}
              </pre>
            ) : (
              <p className="mt-2 line-clamp-2 text-xs leading-relaxed">
                {firstTwoSentences || digestText.slice(0, 200)}
                {digestText.length > 200 && "…"}
              </p>
            )}
            <button
              onClick={() => onNavigate?.("aiPlanner")}
              className="mt-2 flex items-center gap-1 text-xs text-primary hover:underline"
            >
              Read full memo in AI Planner <ArrowRight className="h-3 w-3" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
