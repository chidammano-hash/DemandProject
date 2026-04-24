/**
 * Command Center — Unified morning-triage screen for supply chain teams.
 *
 * Merges data from three sources into a single view:
 * - Control Tower KPIs (portfolio health, fill rate, open exceptions)
 * - AI Planner insights (ML-generated cross-dimensional analysis)
 * - Storyboard exceptions (rule-based threshold alerts)
 *
 * Layout: KPI Summary Bar -> Unified Exception Feed -> Trend Chart
 *
 * TODO (UX-2 Gen-4 Roadmap): Consolidate with ControlTowerTab.
 *
 * CommandCenterTab is the canonical triage overview wired into the sidebar
 * ("commandCenter"). ControlTowerTab is a 433-line legacy 5-zone dashboard
 * that was superseded by this screen but remains reachable via URL state
 * (`?tab=controlTower`). Merging them requires reconciling overlapping KPI
 * queries, migrating the trend-window chooser, and verifying no deep-link
 * consumers rely on the old key. Deferred pending UX review.
 */
import { useState, useMemo, useCallback } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import {
  controlTowerKeys,
  fetchControlTowerKpis,
  fetchControlTowerTrend,
  queryKeys,
  fetchAiInsights,
  updateInsightStatus,
  storyboardKeys,
  fetchSbExceptions,
  STALE,
} from "@/api/queries";
import type { AiInsight, InsightStatus } from "@/types/ai-planner";
import type { StoryboardException } from "@/types/storyboard";
import { Skeleton } from "@/components/Skeleton";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { formatCurrency } from "@/lib/formatters";
import { navigateToItem } from "@/lib/navigation";
import {
  Shield,
  AlertTriangle,
  TrendingUp,
  DollarSign,
  Brain,
  BookOpen,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Loader2,
  CheckCircle2,
  Activity,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export interface UnifiedException {
  id: string;
  source: "ai" | "rule";
  severity: "critical" | "high" | "medium" | "low";
  type: string;
  typeLabel: string;
  itemNo: string;
  location: string;
  summary: string;
  recommendation?: string;
  financialImpact?: number;
  createdAt: string;
  status: string;
  originalData: AiInsight | StoryboardException;
}

interface CommandCenterTabProps {
  onNavigate: (tab: string, params?: Record<string, string>) => void;
}

// ---------------------------------------------------------------------------
// Severity helpers
// ---------------------------------------------------------------------------
const SEVERITY_ORDER: Record<string, number> = {
  critical: 0,
  high: 1,
  medium: 2,
  low: 3,
};

const SEVERITY_BORDER: Record<string, string> = {
  critical: "border-l-red-500",
  high: "border-l-orange-500",
  medium: "border-l-yellow-500",
  low: "border-l-gray-400",
};

const SEVERITY_DOT: Record<string, string> = {
  critical: "bg-red-500",
  high: "bg-orange-500",
  medium: "bg-yellow-500",
  low: "bg-gray-400",
};

const SOURCE_BADGE: Record<string, { label: string; className: string }> = {
  ai: {
    label: "AI",
    className:
      "bg-teal-100 text-teal-800 dark:bg-teal-900/40 dark:text-teal-300",
  },
  rule: {
    label: "Rule",
    className:
      "bg-gray-100 text-gray-700 dark:bg-gray-700/40 dark:text-gray-300",
  },
};

// Map storyboard severity (0-1 float) to categorical
function mapRuleSeverity(
  score: number
): "critical" | "high" | "medium" | "low" {
  if (score >= 0.75) return "critical";
  if (score >= 0.5) return "high";
  if (score >= 0.25) return "medium";
  return "low";
}

// Exception type labels for storyboard
const EXCEPTION_TYPE_LABELS: Record<string, string> = {
  forecast_bias: "Forecast Bias",
  stockout_risk: "Stockout Risk",
  accuracy_drop: "Accuracy Drop",
  excess_risk: "Excess Risk",
  model_drift: "Model Drift",
  new_item: "New Item",
};

// AI insight type labels
const INSIGHT_TYPE_LABELS: Record<string, string> = {
  stockout_risk: "Stockout Risk",
  excess_inventory: "Excess Inventory",
  forecast_bias: "Forecast Bias",
  policy_gap: "Policy Gap",
  champion_degradation: "Model Degradation",
};

// ---------------------------------------------------------------------------
// Normalization: merge AI insights + storyboard exceptions
// ---------------------------------------------------------------------------
function normalizeAiInsight(insight: AiInsight): UnifiedException {
  return {
    id: `ai-${insight.insight_id}`,
    source: "ai",
    severity: insight.severity,
    type: insight.insight_type,
    typeLabel: INSIGHT_TYPE_LABELS[insight.insight_type] ?? insight.insight_type,
    itemNo: insight.item_id,
    location: insight.loc,
    summary: insight.summary,
    recommendation: insight.recommendation,
    financialImpact: insight.financial_impact_estimate ?? undefined,
    createdAt: insight.created_at,
    status: insight.status,
    originalData: insight,
  };
}

function normalizeStoryboardException(
  exc: StoryboardException
): UnifiedException {
  return {
    id: `rule-${exc.exception_id}`,
    source: "rule",
    severity: mapRuleSeverity(exc.severity),
    type: exc.exception_type,
    typeLabel:
      EXCEPTION_TYPE_LABELS[exc.exception_type] ?? exc.exception_type,
    itemNo: exc.item_id,
    location: exc.loc,
    summary: exc.headline ?? `${exc.exception_type} detected`,
    recommendation: undefined,
    financialImpact: exc.financial_impact ?? undefined,
    createdAt: exc.generated_at,
    status: exc.status,
    originalData: exc,
  };
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function CommandCenterTab({ onNavigate }: CommandCenterTabProps) {
  const qc = useQueryClient();

  // Filters
  const [severityFilter, setSeverityFilter] = useState<string>("all");
  const [sourceFilter, setSourceFilter] = useState<string>("all");
  const [statusFilter, setStatusFilter] = useState<string>("open");

  // Trend chart collapsible
  const [showTrend, setShowTrend] = useState(true);

  // ── Control Tower KPIs ──
  const kpisQ = useQuery({
    queryKey: controlTowerKeys.kpis(),
    queryFn: fetchControlTowerKpis,
    staleTime: STALE.FIVE_MIN,
  });

  // ── Control Tower Trend ──
  const trendQ = useQuery({
    queryKey: controlTowerKeys.trend(6),
    queryFn: () => fetchControlTowerTrend(6),
    staleTime: STALE.FIVE_MIN,
  });

  // ── AI Insights ──
  const aiQ = useQuery({
    queryKey: queryKeys.aiInsights({ status: statusFilter === "all" ? undefined : statusFilter }),
    queryFn: () =>
      fetchAiInsights({
        status: statusFilter === "all" ? undefined : (statusFilter as InsightStatus),
        page_size: 50,
      }),
    staleTime: STALE.THIRTY_SEC,
  });

  // ── Storyboard Exceptions ──
  const sbQ = useQuery({
    queryKey: storyboardKeys.list({ status: statusFilter === "all" ? "all" : statusFilter, limit: 50 }),
    queryFn: () =>
      fetchSbExceptions({
        status: statusFilter === "all" ? "all" : statusFilter,
        limit: 50,
      }),
    staleTime: STALE.THIRTY_SEC,
  });

  // ── Accept AI insight mutation ──
  const acceptMutation = useMutation({
    mutationFn: ({ id, status }: { id: number; status: "acknowledged" }) =>
      updateInsightStatus(id, status),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.aiInsights({}) });
    },
  });

  // ── Merge + sort exceptions ──
  const unified = useMemo<UnifiedException[]>(() => {
    const aiItems = (aiQ.data?.insights ?? []).map(normalizeAiInsight);
    const sbItems = (sbQ.data?.rows ?? []).map(normalizeStoryboardException);

    let all = [...aiItems, ...sbItems];

    // Apply severity filter
    if (severityFilter !== "all") {
      all = all.filter((e) => e.severity === severityFilter);
    }

    // Apply source filter
    if (sourceFilter !== "all") {
      all = all.filter((e) => e.source === sourceFilter);
    }

    // Sort: severity first, then created_at desc
    all.sort((a, b) => {
      const so = (SEVERITY_ORDER[a.severity] ?? 3) - (SEVERITY_ORDER[b.severity] ?? 3);
      if (so !== 0) return so;
      return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
    });

    return all;
  }, [aiQ.data, sbQ.data, severityFilter, sourceFilter]);

  // ── KPI values ──
  const h = kpisQ.data?.health;
  const ex = kpisQ.data?.exceptions;
  const fr = kpisQ.data?.fill_rate;

  const isLoading = kpisQ.isLoading || aiQ.isLoading || sbQ.isLoading;

  const handleAccept = useCallback(
    (item: UnifiedException) => {
      if (item.source === "ai") {
        const ai = item.originalData as AiInsight;
        acceptMutation.mutate({ id: ai.insight_id, status: "acknowledged" });
      }
    },
    [acceptMutation]
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-primary" />
            <h2 className="text-xl font-semibold">Command Center</h2>
          </div>
          <p className="text-sm text-muted-foreground max-w-3xl mt-1">
            Unified morning triage: portfolio health KPIs, AI-generated insights,
            and rule-based exceptions in a single view. Prioritized by severity
            and financial impact.
          </p>
        </div>
        {(aiQ.isFetching || sbQ.isFetching || kpisQ.isFetching) && (
          <Badge variant="outline" className="gap-1.5 text-muted-foreground animate-pulse">
            <Loader2 className="h-3 w-3 animate-spin" />
            Syncing
          </Badge>
        )}
      </div>

      {/* Section 1: KPI Summary Bar */}
      {isLoading && !kpisQ.data ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3" data-testid="kpi-skeletons">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="rounded-lg border bg-card p-4 space-y-3">
              <div className="flex items-center gap-2">
                <Skeleton className="h-4 w-4 rounded" />
                <Skeleton className="h-3 w-20" />
              </div>
              <Skeleton className="h-8 w-24" />
              <Skeleton className="h-2 w-16" />
            </div>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3" data-testid="kpi-cards">
          <KpiSummaryCard
            icon={Shield}
            label="Portfolio Health"
            value={
              h?.avg_health_score != null
                ? `${h.avg_health_score.toFixed(0)}/100`
                : "--"
            }
            color={
              h?.avg_health_score == null
                ? undefined
                : h.avg_health_score >= 80
                  ? "green"
                  : h.avg_health_score >= 60
                    ? "amber"
                    : "red"
            }
            progress={h?.avg_health_score != null ? h.avg_health_score / 100 : undefined}
          />
          <KpiSummaryCard
            icon={AlertTriangle}
            label="Open Exceptions"
            value={String(ex?.open_exceptions_total ?? 0)}
            badge={
              ex?.critical_exceptions
                ? `${ex.critical_exceptions} critical`
                : undefined
            }
            color={ex?.critical_exceptions ? "red" : undefined}
          />
          <KpiSummaryCard
            icon={TrendingUp}
            label="Fill Rate (3m)"
            value={
              fr?.portfolio_fill_rate_3m != null
                ? `${(fr.portfolio_fill_rate_3m * 100).toFixed(1)}%`
                : "--"
            }
            color={
              fr?.portfolio_fill_rate_3m == null
                ? undefined
                : fr.portfolio_fill_rate_3m >= 0.95
                  ? "green"
                  : fr.portfolio_fill_rate_3m >= 0.9
                    ? "amber"
                    : "red"
            }
            progress={fr?.portfolio_fill_rate_3m ?? undefined}
          />
          <KpiSummaryCard
            icon={DollarSign}
            label="Critical Items"
            value={String(ex?.critical_exceptions ?? 0)}
            color={ex?.critical_exceptions ? "red" : "green"}
          />
        </div>
      )}

      {/* Section 2: Unified Exception Feed */}
      <div className="space-y-3">
        {/* Filter toolbar */}
        <div className="flex flex-wrap items-center gap-2" data-testid="filter-toolbar">
          {/* Severity filter */}
          <div className="flex rounded-md border overflow-hidden text-xs">
            {["all", "critical", "high", "medium", "low"].map((sev) => (
              <button
                key={sev}
                onClick={() => setSeverityFilter(sev)}
                className={cn(
                  "px-3 py-1.5 font-medium transition-colors capitalize",
                  severityFilter === sev
                    ? "bg-primary text-primary-foreground"
                    : "hover:bg-muted"
                )}
              >
                {sev === "all" ? "All" : sev}
              </button>
            ))}
          </div>

          {/* Source filter */}
          <div className="flex rounded-md border overflow-hidden text-xs">
            {["all", "ai", "rule"].map((src) => (
              <button
                key={src}
                onClick={() => setSourceFilter(src)}
                className={cn(
                  "px-3 py-1.5 font-medium transition-colors",
                  sourceFilter === src
                    ? "bg-primary text-primary-foreground"
                    : "hover:bg-muted"
                )}
              >
                {src === "all" ? "All Sources" : src === "ai" ? "AI" : "Rules"}
              </button>
            ))}
          </div>

          {/* Status filter */}
          <div className="flex rounded-md border overflow-hidden text-xs">
            {["open", "acknowledged", "all"].map((st) => (
              <button
                key={st}
                onClick={() => setStatusFilter(st)}
                className={cn(
                  "px-3 py-1.5 font-medium transition-colors capitalize",
                  statusFilter === st
                    ? "bg-primary text-primary-foreground"
                    : "hover:bg-muted"
                )}
              >
                {st === "all" ? "All Statuses" : st === "acknowledged" ? "Accepted" : st}
              </button>
            ))}
          </div>

          {(aiQ.isFetching || sbQ.isFetching) && (
            <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
          )}
        </div>

        {/* Exception list */}
        {aiQ.isLoading && sbQ.isLoading ? (
          <div className="space-y-3" data-testid="feed-skeletons">
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="rounded-lg border border-l-4 bg-card p-4 space-y-3">
                <div className="flex items-center gap-2">
                  <Skeleton className="h-2.5 w-2.5 rounded-full" />
                  <Skeleton className="h-4 w-12 rounded-full" />
                  <Skeleton className="h-3 w-24" />
                </div>
                <Skeleton className="h-3 w-40" />
                <Skeleton className="h-4 w-full max-w-md" />
                <div className="flex gap-3">
                  <Skeleton className="h-3 w-16" />
                  <Skeleton className="h-3 w-20" />
                </div>
              </div>
            ))}
          </div>
        ) : unified.length === 0 ? (
          <div
            className="rounded-lg border bg-card p-12 text-center"
            data-testid="empty-state"
          >
            <CheckCircle2 className="mx-auto mb-3 h-10 w-10 text-green-500/50" />
            <p className="text-sm font-medium text-green-700 dark:text-green-400">
              Portfolio looks healthy!
            </p>
            <p className="mt-1 text-xs text-muted-foreground max-w-md mx-auto">
              No open exceptions matching your filters. All AI insights and
              rule-based threshold checks are clear.
            </p>
          </div>
        ) : (
          <div className="space-y-2" data-testid="exception-feed">
            {unified.map((item) => (
              <ExceptionFeedCard
                key={item.id}
                item={item}
                onAccept={handleAccept}
                onViewItem={() => navigateToItem(onNavigate, item.itemNo, item.location)}
                acceptPending={
                  acceptMutation.isPending &&
                  item.source === "ai" &&
                  acceptMutation.variables?.id ===
                    (item.originalData as AiInsight).insight_id
                }
              />
            ))}
          </div>
        )}
      </div>

      {/* Section 3: Trend Chart (collapsible) */}
      <div className="rounded-lg border bg-card shadow-sm">
        <button
          onClick={() => setShowTrend((v) => !v)}
          className="flex w-full items-center justify-between px-4 py-3 text-sm font-semibold hover:bg-muted/50 transition-colors rounded-t-lg"
          data-testid="trend-toggle"
        >
          <div className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
            <span>Portfolio Trend (6M)</span>
          </div>
          {showTrend ? (
            <ChevronUp className="h-4 w-4" />
          ) : (
            <ChevronDown className="h-4 w-4" />
          )}
        </button>
        {showTrend && (
          <div className="px-4 pb-4 border-t">
            {trendQ.isLoading ? (
              <div className="space-y-3 pt-4" data-testid="trend-skeleton">
                <Skeleton className="h-[200px] w-full rounded-md" />
              </div>
            ) : trendQ.data?.trend && trendQ.data.trend.length > 0 ? (
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={trendQ.data.trend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month_start" tick={{ fontSize: 10 }} />
                  <YAxis
                    yAxisId="left"
                    domain={[0, 100]}
                    tick={{ fontSize: 10 }}
                  />
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    domain={[0, 1]}
                    tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                    tick={{ fontSize: 10 }}
                  />
                  <Tooltip />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="avg_health_score"
                    stroke="#3b82f6"
                    dot={false}
                    strokeWidth={2}
                    name="Avg Health Score"
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="fill_rate"
                    stroke="#10b981"
                    dot={false}
                    strokeWidth={2}
                    name="Fill Rate"
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-xs text-muted-foreground py-6 text-center">
                No trend data available.
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// KPI Summary Card
// ---------------------------------------------------------------------------
function KpiSummaryCard({
  icon: Icon,
  label,
  value,
  badge,
  color,
  progress,
}: {
  icon: React.FC<{ className?: string }>;
  label: string;
  value: string;
  badge?: string;
  color?: "green" | "amber" | "red";
  progress?: number;
}) {
  const borderColor =
    color === "green"
      ? "border-l-green-500"
      : color === "amber"
        ? "border-l-amber-500"
        : color === "red"
          ? "border-l-red-500"
          : "border-l-border";

  const textColor =
    color === "green"
      ? "text-green-600 dark:text-green-400"
      : color === "amber"
        ? "text-amber-600 dark:text-amber-400"
        : color === "red"
          ? "text-red-600 dark:text-red-400"
          : "";

  const iconBg =
    color === "green"
      ? "bg-green-100 dark:bg-green-900/30"
      : color === "amber"
        ? "bg-amber-100 dark:bg-amber-900/30"
        : color === "red"
          ? "bg-red-100 dark:bg-red-900/30"
          : "bg-muted";

  const progressBg =
    color === "green"
      ? "bg-green-500"
      : color === "amber"
        ? "bg-amber-500"
        : color === "red"
          ? "bg-red-500"
          : "bg-primary";

  return (
    <div
      className={cn(
        "rounded-lg border border-l-4 bg-card p-4 shadow-sm transition-shadow hover:shadow-md",
        borderColor
      )}
    >
      <div className="flex items-center gap-2 mb-2">
        <div className={cn("rounded-md p-1.5", iconBg)}>
          <Icon className={cn("h-3.5 w-3.5", textColor || "text-muted-foreground")} />
        </div>
        <p className="text-xs font-medium text-muted-foreground">{label}</p>
      </div>
      <p className={cn("text-2xl font-bold tracking-tight", textColor)}>{value}</p>
      {badge && (
        <span className="inline-flex items-center gap-1 mt-1 text-[10px] font-semibold text-red-600 dark:text-red-400">
          <span className="h-1.5 w-1.5 rounded-full bg-red-500 animate-pulse" />
          {badge}
        </span>
      )}
      {progress != null && (
        <div className="mt-2 h-1.5 w-full rounded-full bg-muted overflow-hidden">
          <div
            className={cn("h-full rounded-full transition-all duration-500", progressBg)}
            style={{ width: `${Math.min(progress * 100, 100)}%` }}
          />
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Exception Feed Card
// ---------------------------------------------------------------------------
function ExceptionFeedCard({
  item,
  onAccept,
  onViewItem,
  acceptPending,
}: {
  item: UnifiedException;
  onAccept: (item: UnifiedException) => void;
  onViewItem: () => void;
  acceptPending?: boolean;
}) {
  const sourceBadge = SOURCE_BADGE[item.source];
  const SourceIcon = item.source === "ai" ? Brain : BookOpen;

  const severityBgColor: Record<string, string> = {
    critical: "bg-red-50 dark:bg-red-950/20",
    high: "bg-orange-50 dark:bg-orange-950/20",
    medium: "bg-yellow-50/50 dark:bg-yellow-950/10",
    low: "",
  };

  return (
    <div
      className={cn(
        "rounded-lg border border-l-4 bg-card p-4 shadow-sm transition-all hover:shadow-md",
        SEVERITY_BORDER[item.severity] ?? "border-l-gray-400",
        severityBgColor[item.severity] ?? ""
      )}
      data-testid="exception-card"
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0 space-y-2">
          {/* Top row: severity dot + source badge + type label */}
          <div className="flex items-center gap-2 flex-wrap">
            <span
              className={cn(
                "h-2.5 w-2.5 rounded-full flex-shrink-0 ring-2 ring-offset-1 ring-offset-card",
                SEVERITY_DOT[item.severity],
                item.severity === "critical" ? "ring-red-200 dark:ring-red-800" : "ring-transparent"
              )}
            />
            <span
              className={cn(
                "text-[10px] font-semibold px-2 py-0.5 rounded-full inline-flex items-center gap-1",
                sourceBadge.className
              )}
            >
              <SourceIcon className="h-2.5 w-2.5" />
              {sourceBadge.label}
            </span>
            <span className="text-xs font-medium text-foreground/70 bg-muted px-2 py-0.5 rounded">
              {item.typeLabel}
            </span>
            <span className={cn(
              "text-[10px] font-semibold uppercase tracking-wider px-1.5 py-0.5 rounded",
              item.severity === "critical" ? "text-red-700 dark:text-red-300 bg-red-100 dark:bg-red-900/40" :
              item.severity === "high" ? "text-orange-700 dark:text-orange-300 bg-orange-100 dark:bg-orange-900/40" :
              item.severity === "medium" ? "text-yellow-700 dark:text-yellow-300 bg-yellow-100 dark:bg-yellow-900/40" :
              "text-muted-foreground"
            )}>
              {item.severity}
            </span>
          </div>

          {/* Item/Location identity */}
          <p className="text-xs font-mono font-bold tracking-wide">
            {item.itemNo} @ {item.location}
          </p>

          {/* Summary */}
          <p className="text-sm leading-relaxed">{item.summary}</p>

          {/* Recommendation */}
          {item.recommendation && (
            <div className="flex items-start gap-1.5 text-xs text-muted-foreground bg-muted/50 rounded-md px-2.5 py-1.5">
              <CheckCircle2 className="h-3 w-3 mt-0.5 shrink-0 text-primary/60" />
              <span className="leading-snug">{item.recommendation}</span>
            </div>
          )}

          {/* Financial impact + timestamp */}
          <div className="flex items-center gap-3 flex-wrap pt-0.5">
            {item.financialImpact != null && item.financialImpact > 0 && (
              <span className="inline-flex items-center gap-1 text-xs font-semibold text-amber-700 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/20 px-2 py-0.5 rounded">
                <DollarSign className="h-3 w-3" />
                {formatCurrency(item.financialImpact)}
              </span>
            )}
            <span className="text-[10px] text-muted-foreground">
              {new Date(item.createdAt).toLocaleDateString()}
            </span>
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex flex-col gap-2 flex-shrink-0 pt-0.5">
          {item.source === "ai" && item.status === "open" && (
            <button
              onClick={() => onAccept(item)}
              disabled={acceptPending}
              className="inline-flex items-center justify-center gap-1 text-xs font-medium rounded-md border px-3 py-1.5 bg-green-50 border-green-200 text-green-700 hover:bg-green-100 hover:border-green-300 dark:bg-green-900/20 dark:border-green-800 dark:text-green-300 dark:hover:bg-green-900/40 transition-colors disabled:opacity-50"
            >
              {acceptPending ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <>
                  <CheckCircle2 className="h-3 w-3" />
                  Accept
                </>
              )}
            </button>
          )}
          <button
            onClick={onViewItem}
            className="inline-flex items-center justify-center gap-1 text-xs font-medium rounded-md border px-3 py-1.5 hover:bg-muted transition-colors"
          >
            View Item
            <ExternalLink className="h-3 w-3" />
          </button>
        </div>
      </div>
    </div>
  );
}
