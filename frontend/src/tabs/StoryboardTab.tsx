/**
 * Feature 40 — Exception Triage
 *
 * 3-zone command center for supply chain teams:
 * Zone 1: Summary KPI header + page description
 * Zone 2: Exception Queue (left, 40% width) — filterable, sortable exception list
 * Zone 3: Investigation Panel (right, 60% width) — detail view + actions
 *
 * Sub-components live in ./storyboard/:
 *   ExceptionCard, SbKpiCard, storyboardShared (constants + helpers)
 */
import { useState, useRef, useEffect, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type { StoryboardException } from "@/types/storyboard";
import {
  storyboardKeys as sbKeys,
  fetchSbSummary,
  fetchSbExceptions,
  fetchSbException,
  updateSbStatus,
  submitSbDecision,
} from "@/api/queries";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";
import { Skeleton } from "@/components/Skeleton";

import { ExceptionCard } from "./storyboard/ExceptionCard";
import { SbKpiCard } from "./storyboard/SbKpiCard";
import { InvestigationPanel } from "./storyboard/InvestigationPanel";
import {
  severityLabel,
  fmt,
  EXCEPTION_TYPES,
  STATUS_FILTERS,
  EXCEPTION_TYPE_LABELS,
  PAGE_SIZE,
} from "./storyboard/storyboardShared";

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function StoryboardTab() {
  const queryClient = useQueryClient();
  const { filters: globalFilters } = useGlobalFilterContext();

  const [selectedExceptionId, setSelectedExceptionId] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>("open");
  const [typeFilter, setTypeFilter] = useState<string>("all");
  const [itemFilter, setItemFilter] = useState<string>("");
  const [locFilter, setLocFilter] = useState<string>("");
  const [page, setPage] = useState<number>(0);

  const syncedGlobalRef = useRef<string>("");
  useEffect(() => {
    const key = `${globalFilters.item.join(",")}_${globalFilters.location.join(",")}`;
    if (key === syncedGlobalRef.current) return;
    syncedGlobalRef.current = key;
    if (globalFilters.item.length === 1) setItemFilter(globalFilters.item[0]);
    if (globalFilters.location.length === 1) setLocFilter(globalFilters.location[0]);
  }, [globalFilters.item, globalFilters.location]);

  // Panel action state
  const [decisionType, setDecisionType] = useState<string>("accept_exception");
  const [rationale, setRationale] = useState<string>("");
  const [actionSuccess, setActionSuccess] = useState<string | null>(null);

  // Summary query
  const { data: summary, isLoading: summaryLoading } = useQuery({
    queryKey: sbKeys.summary(),
    queryFn: fetchSbSummary,
    staleTime: 60_000,
  });

  // List query
  const listParams = {
    status: statusFilter,
    exception_type: typeFilter,
    item: itemFilter,
    loc: locFilter,
    limit: PAGE_SIZE,
    offset: page * PAGE_SIZE,
  };
  const { data: listData, isLoading: listLoading } = useQuery({
    queryKey: sbKeys.list(listParams),
    queryFn: () => fetchSbExceptions(listParams),
    staleTime: 30_000,
  });

  // Detail query
  const { data: detailData, isLoading: detailLoading } = useQuery({
    queryKey: sbKeys.detail(selectedExceptionId ?? ""),
    queryFn: () => fetchSbException(selectedExceptionId!),
    enabled: !!selectedExceptionId,
    staleTime: 30_000,
  });

  // Mutations
  const statusMutation = useMutation({
    mutationFn: ({ id, status }: { id: string; status: string }) =>
      updateSbStatus(id, status),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: sbKeys.summary() });
      queryClient.invalidateQueries({ queryKey: sbKeys.lists() });
      queryClient.invalidateQueries({ queryKey: sbKeys.detail(selectedExceptionId ?? "") });
      setActionSuccess("Status updated successfully");
      setTimeout(() => setActionSuccess(null), 3000);
    },
  });

  const decisionMutation = useMutation({
    mutationFn: ({
      id,
      type,
      rat,
    }: {
      id: string;
      type: string;
      rat: string;
    }) => submitSbDecision(id, type, rat),
    onSuccess: () => {
      setRationale("");
      queryClient.invalidateQueries({ queryKey: sbKeys.detail(selectedExceptionId ?? "") });
      setActionSuccess("Decision recorded");
      setTimeout(() => setActionSuccess(null), 3000);
    },
  });

  const topType =
    summary?.by_type && summary.by_type.length > 0
      ? summary.by_type[0].exception_type
      : "\u2014";

  const totalOpen = summary?.total_open ?? 0;
  const totalInvestigating = summary?.total_investigating ?? 0;

  // Severity distribution for mini bar
  const severityDist = useMemo(() => {
    if (!summary?.avg_severity) return null;
    const s = summary.avg_severity;
    return { label: severityLabel(s * 100 / 100), score: s };
  }, [summary?.avg_severity]);

  return (
    <div className="flex flex-col gap-4 p-4">
      {/* Header */}
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-semibold">Exception Triage</h2>
            {totalOpen > 0 && (
              <span className="inline-flex items-center gap-1 rounded-full bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300 text-xs font-medium px-2.5 py-0.5">
                <span className="h-1.5 w-1.5 rounded-full bg-red-500 animate-pulse" />
                {totalOpen} open
              </span>
            )}
          </div>
          <button
            className="text-xs rounded-md border px-3 py-1.5 hover:bg-muted transition-colors"
            onClick={() => {
              queryClient.invalidateQueries({ queryKey: sbKeys.summary() });
              queryClient.invalidateQueries({ queryKey: sbKeys.lists() });
            }}
          >
            Refresh
          </button>
        </div>
        <p className="text-sm text-muted-foreground max-w-3xl leading-relaxed">
          Review and resolve demand planning exceptions flagged by automated rule-based threshold
          checks. Each exception identifies an item-location pair that has breached a configured
          threshold (e.g. forecast bias &gt; 15%, stockout risk, accuracy degradation). Select an
          exception to investigate, then take action: override the forecast, escalate, or dismiss.
          For ML-generated insights ranked by financial impact, use the <strong>AI Planner</strong> tab.
        </p>
      </div>

      {/* ZONE 1: Summary KPI Strip */}
      {summaryLoading ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3" data-testid="summary-kpi-skeletons">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="rounded-lg border bg-card p-3.5 space-y-2">
              <Skeleton className="h-2.5 w-20" />
              <Skeleton className="h-6 w-14" />
              <Skeleton className="h-2 w-24" />
            </div>
          ))}
        </div>
      ) : (
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3" data-testid="summary-kpis">
        <SbKpiCard
          label="Open Exceptions"
          value={totalOpen}
          subtitle="Require triage"
          color="red"
        />
        <SbKpiCard
          label="Under Investigation"
          value={totalInvestigating}
          subtitle="In progress"
          color="amber"
        />
        <SbKpiCard
          label="Avg Severity"
          value={summary?.avg_severity != null ? fmt(summary.avg_severity, 2) : "\u2014"}
          subtitle={severityDist ? severityDist.label : undefined}
          color={
            summary?.avg_severity == null
              ? undefined
              : summary.avg_severity >= 0.7
              ? "red"
              : summary.avg_severity >= 0.4
              ? "amber"
              : "green"
          }
          severityBar={summary?.avg_severity ?? undefined}
        />
        <SbKpiCard
          label="Most Common Type"
          value={EXCEPTION_TYPE_LABELS[topType] ?? topType}
          subtitle={
            summary?.by_type?.[0]?.open_count != null
              ? `${summary.by_type[0].open_count} occurrences`
              : undefined
          }
        />
      </div>
      )}

      {/* ZONE 2 + ZONE 3: Split layout */}
      <div className="flex flex-col lg:flex-row gap-4 min-h-0">
        {/* ZONE 2: Exception Queue */}
        <div className="lg:w-[40%] flex flex-col gap-3">
          <div className="rounded-lg border bg-card shadow-sm flex flex-col">
            {/* Queue header */}
            <div className="px-4 pt-4 pb-3 border-b space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold">Exception Queue</h3>
                <span className="text-xs text-muted-foreground">
                  {listData?.total ?? 0} total
                </span>
              </div>

              {/* Status tabs */}
              <div className="flex border-b -mb-3">
                {STATUS_FILTERS.map((s) => (
                  <button
                    key={s}
                    onClick={() => {
                      setStatusFilter(s);
                      setPage(0);
                    }}
                    className={`px-3 py-2 text-xs font-medium border-b-2 transition-colors ${
                      statusFilter === s
                        ? "border-primary text-foreground"
                        : "border-transparent text-muted-foreground hover:text-foreground hover:border-muted-foreground/30"
                    }`}
                  >
                    {s === "all" ? "All" : s.charAt(0).toUpperCase() + s.slice(1)}
                    {s === "open" && totalOpen > 0 && (
                      <span className="ml-1.5 inline-flex items-center justify-center h-4 min-w-[16px] rounded-full bg-red-500 text-white text-[10px] font-bold px-1">
                        {totalOpen}
                      </span>
                    )}
                    {s === "investigating" && totalInvestigating > 0 && (
                      <span className="ml-1.5 inline-flex items-center justify-center h-4 min-w-[16px] rounded-full bg-yellow-500 text-white text-[10px] font-bold px-1">
                        {totalInvestigating}
                      </span>
                    )}
                  </button>
                ))}
              </div>
            </div>

            {/* Filters */}
            <div className="px-4 py-3 space-y-2 border-b">
              <div className="flex gap-2">
                <select
                  value={typeFilter}
                  onChange={(e) => {
                    setTypeFilter(e.target.value);
                    setPage(0);
                  }}
                  className="flex-1 h-8 text-xs rounded-md border border-input bg-background px-2"
                >
                  {EXCEPTION_TYPES.map((t) => (
                    <option key={t} value={t}>
                      {EXCEPTION_TYPE_LABELS[t] ?? t}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex gap-2">
                <input
                  type="text"
                  placeholder="Filter by item..."
                  value={itemFilter}
                  onChange={(e) => {
                    setItemFilter(e.target.value);
                    setPage(0);
                  }}
                  className="flex-1 h-8 text-xs rounded-md border border-input bg-background px-2.5"
                />
                <input
                  type="text"
                  placeholder="Filter by location..."
                  value={locFilter}
                  onChange={(e) => {
                    setLocFilter(e.target.value);
                    setPage(0);
                  }}
                  className="flex-1 h-8 text-xs rounded-md border border-input bg-background px-2.5"
                />
              </div>
            </div>

            {/* Exception list */}
            <div className="flex-1 overflow-y-auto max-h-[520px]">
              {listLoading && (
                <div className="divide-y" data-testid="exception-list-skeletons">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <div key={i} className="px-4 py-3 space-y-2">
                      <div className="flex items-center gap-2">
                        <Skeleton className="h-2 w-2 rounded-full" />
                        <Skeleton className="h-4 w-16 rounded" />
                        <Skeleton className="ml-auto h-3 w-12" />
                      </div>
                      <Skeleton className="h-3 w-32" />
                      <Skeleton className="h-3 w-full max-w-[200px]" />
                    </div>
                  ))}
                </div>
              )}
              {!listLoading && listData?.rows && listData.rows.length === 0 && (
                <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                  <div className="h-10 w-10 rounded-full bg-muted flex items-center justify-center mb-3">
                    <svg className="h-5 w-5 text-muted-foreground/50" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <p className="text-sm font-medium">No exceptions found</p>
                  <p className="text-xs mt-1 text-center max-w-[200px]">Try adjusting your filters or changing the status tab.</p>
                </div>
              )}
              <div className="divide-y">
                {listData?.rows?.map((exc: StoryboardException) => (
                  <ExceptionCard
                    key={exc.exception_id}
                    exception={exc}
                    isSelected={selectedExceptionId === exc.exception_id}
                    onSelect={() => {
                      setSelectedExceptionId(exc.exception_id);
                    }}
                  />
                ))}
              </div>
            </div>

            {/* Pagination */}
            {listData && listData.total > (page + 1) * PAGE_SIZE && (
              <div className="px-4 py-2.5 border-t">
                <button
                  className="w-full text-xs font-medium text-primary hover:text-primary/80 py-1"
                  onClick={() => setPage((p) => p + 1)}
                >
                  Show more ({listData.total - (page + 1) * PAGE_SIZE} remaining)
                </button>
              </div>
            )}
          </div>
        </div>

        {/* ZONE 3: Investigation Panel */}
        <InvestigationPanel
          selectedExceptionId={selectedExceptionId}
          detailData={detailData}
          detailLoading={detailLoading}
          actionSuccess={actionSuccess}
          decisionType={decisionType}
          setDecisionType={setDecisionType}
          rationale={rationale}
          setRationale={setRationale}
          statusMutation={statusMutation}
          decisionMutation={decisionMutation}
        />
      </div>
    </div>
  );
}
