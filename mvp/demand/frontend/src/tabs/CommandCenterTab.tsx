/**
 * Command Center — Unified morning-triage screen for supply chain teams.
 *
 * Merges data from three sources into a single view:
 * - Control Tower KPIs (portfolio health, fill rate, open exceptions)
 * - AI Planner insights (ML-generated cross-dimensional analysis)
 * - Storyboard exceptions (rule-based threshold alerts)
 *
 * Layout: KPI Summary Bar -> Unified Exception Feed -> Trend Chart
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
import type { AiInsight, InsightSeverity } from "@/types/ai-planner";
import type { StoryboardException } from "@/types/storyboard";
import { Skeleton } from "@/components/Skeleton";
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
    itemNo: insight.item_no,
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
    itemNo: exc.item_no,
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
        status: statusFilter === "all" ? undefined : (statusFilter as InsightSeverity),
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
      <div>
        <h2 className="text-xl font-semibold">Command Center</h2>
        <p className="text-sm text-muted-foreground max-w-3xl">
          Unified morning triage: portfolio health KPIs, AI-generated insights,
          and rule-based exceptions in a single view. Prioritized by severity
          and financial impact.
        </p>
      </div>

      {/* Section 1: KPI Summary Bar */}
      {isLoading && !kpisQ.data ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3" data-testid="kpi-skeletons">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-20 rounded-lg" />
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
              <Skeleton key={i} className="h-28 rounded-lg" />
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
      <div className="rounded-lg border bg-card">
        <button
          onClick={() => setShowTrend((v) => !v)}
          className="flex w-full items-center justify-between px-4 py-3 text-sm font-semibold hover:bg-muted/50 transition-colors"
          data-testid="trend-toggle"
        >
          <span>Portfolio Trend (6M)</span>
          {showTrend ? (
            <ChevronUp className="h-4 w-4" />
          ) : (
            <ChevronDown className="h-4 w-4" />
          )}
        </button>
        {showTrend && (
          <div className="px-4 pb-4">
            {trendQ.data?.trend && trendQ.data.trend.length > 0 ? (
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
              <p className="text-xs text-muted-foreground py-4">
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
}: {
  icon: React.FC<{ className?: string }>;
  label: string;
  value: string;
  badge?: string;
  color?: "green" | "amber" | "red";
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
      ? "text-green-600"
      : color === "amber"
        ? "text-amber-600"
        : color === "red"
          ? "text-red-600"
          : "";

  return (
    <div
      className={cn(
        "rounded-lg border border-l-4 bg-card p-3 shadow-sm",
        borderColor
      )}
    >
      <div className="flex items-center gap-1.5">
        <Icon className="h-3.5 w-3.5 text-muted-foreground" />
        <p className="text-xs text-muted-foreground">{label}</p>
      </div>
      <p className={cn("text-2xl font-bold", textColor)}>{value}</p>
      {badge && (
        <span className="text-[10px] font-medium text-red-600">{badge}</span>
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

  return (
    <div
      className={cn(
        "rounded-lg border border-l-4 bg-card p-3 shadow-sm transition-shadow hover:shadow-md",
        SEVERITY_BORDER[item.severity] ?? "border-l-gray-400"
      )}
      data-testid="exception-card"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0 space-y-1.5">
          {/* Top row: severity dot + source badge + type label */}
          <div className="flex items-center gap-2 flex-wrap">
            <span
              className={cn(
                "h-2 w-2 rounded-full flex-shrink-0",
                SEVERITY_DOT[item.severity]
              )}
            />
            <span
              className={cn(
                "text-[10px] font-semibold px-1.5 py-0.5 rounded-full",
                sourceBadge.className
              )}
            >
              <SourceIcon className="inline h-2.5 w-2.5 mr-0.5 -mt-px" />
              {sourceBadge.label}
            </span>
            <span className="text-xs text-muted-foreground font-medium">
              {item.typeLabel}
            </span>
            <span className="text-xs text-muted-foreground capitalize">
              {item.severity}
            </span>
          </div>

          {/* Item/Location identity */}
          <p className="text-xs font-mono font-bold">
            {item.itemNo} @ {item.location}
          </p>

          {/* Summary */}
          <p className="text-sm leading-snug">{item.summary}</p>

          {/* Recommendation */}
          {item.recommendation && (
            <p className="text-xs text-muted-foreground italic leading-snug">
              {item.recommendation}
            </p>
          )}

          {/* Financial impact + timestamp */}
          <div className="flex items-center gap-3 flex-wrap">
            {item.financialImpact != null && item.financialImpact > 0 && (
              <span className="inline-flex items-center gap-1 text-xs font-medium text-amber-700 dark:text-amber-400">
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
        <div className="flex flex-col gap-1.5 flex-shrink-0">
          {item.source === "ai" && item.status === "open" && (
            <button
              onClick={() => onAccept(item)}
              disabled={acceptPending}
              className="inline-flex items-center gap-1 text-xs font-medium rounded-md border px-2.5 py-1 hover:bg-green-50 hover:border-green-300 hover:text-green-800 dark:hover:bg-green-900/30 dark:hover:border-green-700 dark:hover:text-green-300 transition-colors disabled:opacity-50"
            >
              {acceptPending ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                "Accept"
              )}
            </button>
          )}
          <button
            onClick={onViewItem}
            className="inline-flex items-center gap-1 text-xs font-medium rounded-md border px-2.5 py-1 hover:bg-muted transition-colors"
          >
            View Item
            <ExternalLink className="h-3 w-3" />
          </button>
        </div>
      </div>
    </div>
  );
}
