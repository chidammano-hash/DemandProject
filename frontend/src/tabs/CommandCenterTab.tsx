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
 * CommandCenterTab is the canonical triage overview wired into the sidebar
 * ("commandCenter"). The legacy ControlTowerTab and AIPlannerTab screens were
 * superseded by this consolidated view and removed (U3.10); their old URL keys
 * (`?tab=controlTower`, `?tab=aiPlanner`) still resolve here via TAB_REDIRECTS
 * in useUrlState.ts so existing bookmarks keep working.
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
import { Skeleton } from "@/components/Skeleton";
import { ForecastReleaseGateCard } from "@/components/ForecastReleaseGateCard";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { formatCurrency, formatInt } from "@/lib/formatters";
import { navigateToItem } from "@/lib/navigation";
import { useChartColors } from "@/hooks/useChartColors";
import {
  Shield,
  AlertTriangle,
  TrendingUp,
  DollarSign,
  ChevronDown,
  ChevronUp,
  Loader2,
  CheckCircle2,
  Activity,
} from "lucide-react";
import {
  type UnifiedException,
  SEVERITY_ORDER,
  SEVERITY_BANDS,
  normalizeAiInsight,
  normalizeStoryboardException,
} from "./command-center/exceptions";
import { KpiSummaryCard } from "./command-center/KpiSummaryCard";
import { ExceptionFeedCard } from "./command-center/ExceptionFeedCard";

interface CommandCenterTabProps {
  onNavigate: (tab: string, params?: Record<string, string>) => void;
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function CommandCenterTab({ onNavigate }: CommandCenterTabProps) {
  const qc = useQueryClient();

  // Portfolio Trend series colors — read from the theme palette so the lines
  // adapt to Light/Soft/Dark instead of hardcoded hex (U1.2). trendColors[0]
  // (blue) keys the health-score line, trendColors[1] (teal) the fill-rate.
  const { trendColors } = useChartColors();

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
  // U7.10 — when a severity chip is active, push its numeric band to the server
  // so the (critical-first, 50-capped) feed reloads with rows that actually fall
  // in the selected band instead of an empty client-side filter result.
  const severityBand = severityFilter === "all" ? undefined : SEVERITY_BANDS[severityFilter];
  const sbQ = useQuery({
    queryKey: storyboardKeys.list({
      status: statusFilter === "all" ? "all" : statusFilter,
      severity: severityFilter,
      limit: 50,
    }),
    queryFn: () =>
      fetchSbExceptions({
        status: statusFilter === "all" ? "all" : statusFilter,
        severity_min: severityBand?.min,
        severity_max: severityBand?.max,
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
  // F2.1: when the control-tower MVs are unrefreshed the endpoint degrades to an
  // all-zero payload with a `warning`. Surface it so a "data unavailable" state
  // is never mistaken for a genuinely healthy portfolio.
  const kpisWarning = kpisQ.data?.warning;
  const kpisStale = Boolean(kpisWarning);

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

      {/* Planner release contract — quality, lineage, freshness, coverage, archive. */}
      <ForecastReleaseGateCard onNavigate={onNavigate} />

      {/* F2.1: degraded-data banner — the KPI MVs are unrefreshed, so the
          zeroed tiles are "data unavailable", not a healthy portfolio. */}
      {kpisStale && (
        <div
          data-testid="mv-stale-warning"
          role="alert"
          className="flex items-start gap-2 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-800 dark:border-amber-700 dark:bg-amber-950/40 dark:text-amber-300"
        >
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          <div>
            <p className="font-medium">Portfolio health data unavailable</p>
            <p className="mt-0.5 text-xs">
              Health KPIs are stale and showing zeros — they are not a sign of a
              healthy portfolio. Refresh the analytics views (run{" "}
              <code className="font-mono">make refresh-mvs-tiered</code>) to see
              live numbers.
            </p>
          </div>
        </div>
      )}

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
            value={ex?.open_exceptions_total != null ? formatInt(ex.open_exceptions_total) : "--"}
            badge={
              ex?.critical_exceptions
                ? `${formatInt(ex.critical_exceptions)} critical`
                : undefined
            }
            color={ex?.critical_exceptions ? "red" : undefined}
            caption="Replenishment exceptions only"
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
          {/* U2.3 — a distinct $ metric (proposed replenishment order value)
              instead of restating the "X critical" badge already on the Open
              Exceptions tile. DollarSign is correct here since this IS a currency
              value. */}
          <KpiSummaryCard
            icon={DollarSign}
            label="Order Value at Risk"
            value={
              ex?.recommended_order_value != null
                ? formatCurrency(ex.recommended_order_value)
                : "--"
            }
            color={ex?.recommended_order_value ? "amber" : undefined}
            caption="Proposed replenishment orders"
          />
        </div>
      )}

      {/* Section 2: Unified Exception Feed */}
      <div className="space-y-3">
        {/* Filter toolbar */}
        <div className="flex flex-wrap items-center gap-2" data-testid="filter-toolbar">
          {/* Severity filter — segmented single-select. role=group + aria-pressed
              let a screen reader announce the active chip and the mutual
              exclusivity (U2.4). */}
          <div role="group" aria-label="Severity filter" className="flex rounded-md border overflow-hidden text-xs">
            {["all", "critical", "high", "medium", "low"].map((sev) => (
              <button
                key={sev}
                aria-pressed={severityFilter === sev}
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
          <div role="group" aria-label="Source filter" className="flex rounded-md border overflow-hidden text-xs">
            {["all", "ai", "rule"].map((src) => (
              <button
                key={src}
                aria-pressed={sourceFilter === src}
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
          <div role="group" aria-label="Status filter" className="flex rounded-md border overflow-hidden text-xs">
            {["open", "acknowledged", "all"].map((st) => (
              <button
                key={st}
                aria-pressed={statusFilter === st}
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
          kpisStale ? (
            // F2.1: stale KPIs + an empty exception feed is "data unavailable",
            // NOT a healthy portfolio. Never show the green all-clear here.
            <div
              className="rounded-lg border bg-card p-12 text-center"
              data-testid="empty-state-stale"
            >
              <AlertTriangle className="mx-auto mb-3 h-10 w-10 text-amber-500/70" />
              <p className="text-sm font-medium text-amber-700 dark:text-amber-400">
                Exception data unavailable
              </p>
              <p className="mt-1 text-xs text-muted-foreground max-w-md mx-auto">
                Analytics views are stale, so this feed cannot be trusted to be
                empty. Refresh the data (run{" "}
                <code className="font-mono">make refresh-mvs-tiered</code>) to see
                open exceptions.
              </p>
            </div>
          ) : (
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
          )
        ) : (
          <>
            {/* U6.7 — the feed fetches at most 50 rows; surface that it is a
                slice of the full open-exception population (the same total the
                Open Exceptions tile shows), mirroring the Inv Planning feed. */}
            <p
              className="mb-2 text-xs text-muted-foreground"
              data-testid="feed-count-caption"
            >
              Showing top {formatInt(unified.length)} of{" "}
              {ex?.open_exceptions_total != null
                ? formatInt(ex.open_exceptions_total)
                : "—"}{" "}
              exceptions by severity. KPIs above reflect the full population.
            </p>
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
          </>
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
                    stroke={trendColors[0]}
                    dot={false}
                    strokeWidth={2}
                    name="Avg Health Score"
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="fill_rate"
                    stroke={trendColors[1]}
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
