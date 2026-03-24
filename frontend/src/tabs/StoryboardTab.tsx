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
import type {
  StoryboardException,
  PlannerDecision,
} from "@/types/storyboard";
import {
  storyboardKeys as sbKeys,
  fetchSbSummary,
  fetchSbExceptions,
  fetchSbException,
  updateSbStatus,
  submitSbDecision,
} from "@/api/queries";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";

import { ExceptionCard } from "./storyboard/ExceptionCard";
import { SbKpiCard } from "./storyboard/SbKpiCard";
import {
  severityLabel,
  severityColorClass,
  severityBg,
  fmt,
  fmtCurrency,
  daysAgo,
  EXCEPTION_TYPES,
  STATUS_FILTERS,
  EXCEPTION_TYPE_LABELS,
  EXCEPTION_TYPE_COLORS,
  DECISION_TYPES,
  DECISION_TYPE_COLORS,
  STATUS_COLORS,
  STATUS_DOT,
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
  const [newStatus, setNewStatus] = useState<string>("investigating");
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
      queryClient.invalidateQueries({ queryKey: ["sb-"] });
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
              queryClient.invalidateQueries({ queryKey: ["sb-"] });
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

      {summaryLoading && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <span className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
          Loading exception data...
        </div>
      )}

      {/* ZONE 1: Summary KPI Strip */}
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
            summary?.by_type?.[0]?.count != null
              ? `${summary.by_type[0].count} occurrences`
              : undefined
          }
        />
      </div>

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
                <div className="flex items-center gap-2 justify-center py-8 text-xs text-muted-foreground">
                  <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent" />
                  Loading...
                </div>
              )}
              {!listLoading && listData?.rows && listData.rows.length === 0 && (
                <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                  <p className="text-sm font-medium">No exceptions found</p>
                  <p className="text-xs mt-1">Try adjusting your filters or changing the status tab.</p>
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
                      setNewStatus("investigating");
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
        <div className="lg:w-[60%] flex flex-col gap-3">
          {!selectedExceptionId && (
            <div className="rounded-lg border bg-card shadow-sm p-8 flex flex-col items-center justify-center text-center min-h-[300px]">
              <div className="h-12 w-12 rounded-full bg-muted flex items-center justify-center mb-4">
                <svg className="h-6 w-6 text-muted-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
                </svg>
              </div>
              <p className="text-sm font-medium text-foreground">No exception selected</p>
              <p className="text-xs text-muted-foreground mt-1 max-w-xs">
                Click on an exception from the queue to view its details, supporting data, and take action.
              </p>
            </div>
          )}

          {selectedExceptionId && (
            <div className="rounded-lg border bg-card shadow-sm overflow-hidden">
              {detailLoading && (
                <div className="flex items-center gap-2 justify-center py-12 text-xs text-muted-foreground">
                  <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent" />
                  Loading exception details...
                </div>
              )}

              {detailData && (
                <>
                  {/* Success banner */}
                  {actionSuccess && (
                    <div className="bg-green-50 dark:bg-green-900/30 border-b border-green-200 dark:border-green-800 px-4 py-2 text-xs text-green-700 dark:text-green-300 font-medium">
                      {actionSuccess}
                    </div>
                  )}

                  {/* Detail header */}
                  <div className="px-4 pt-4 pb-3 border-b">
                    <div className="flex items-start justify-between">
                      <div className="space-y-1.5">
                        <div className="flex items-center gap-2">
                          <span
                            className={`text-xs px-2 py-0.5 rounded-full font-medium border ${
                              EXCEPTION_TYPE_COLORS[detailData.exception.exception_type] ?? ""
                            }`}
                          >
                            {EXCEPTION_TYPE_LABELS[detailData.exception.exception_type] ??
                              detailData.exception.exception_type}
                          </span>
                          <span
                            className={`flex items-center gap-1 text-xs px-2 py-0.5 rounded-full font-medium ${
                              STATUS_COLORS[detailData.exception.status] ?? ""
                            }`}
                          >
                            <span className={`h-1.5 w-1.5 rounded-full ${STATUS_DOT[detailData.exception.status] ?? ""}`} />
                            {detailData.exception.status}
                          </span>
                        </div>
                        <h3 className="text-base font-semibold">
                          {detailData.exception.item_id} @ {detailData.exception.loc}
                        </h3>
                        {detailData.exception.headline && (
                          <p className="text-sm text-muted-foreground leading-snug">
                            {detailData.exception.headline}
                          </p>
                        )}
                      </div>
                      {/* Severity badge */}
                      <div className="flex flex-col items-center gap-1 ml-4">
                        <div className={`text-2xl font-bold ${severityColorClass(detailData.exception.severity)}`}>
                          {Math.round(detailData.exception.severity * 100)}
                        </div>
                        <span className={`text-[10px] font-semibold uppercase tracking-wider ${severityColorClass(detailData.exception.severity)}`}>
                          {severityLabel(detailData.exception.severity)}
                        </span>
                        <div className="w-12 h-1.5 rounded-full bg-muted overflow-hidden">
                          <div
                            className={`h-full rounded-full ${severityBg(detailData.exception.severity)}`}
                            style={{ width: `${detailData.exception.severity * 100}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Key metrics strip */}
                  <div className="grid grid-cols-3 border-b divide-x">
                    <div className="px-4 py-3 text-center">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Financial Impact</p>
                      <p className="text-base font-bold mt-0.5">
                        {fmtCurrency(detailData.exception.financial_impact)}
                      </p>
                    </div>
                    <div className="px-4 py-3 text-center">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Age</p>
                      <p className="text-base font-bold mt-0.5">
                        {daysAgo(detailData.exception.generated_at)}
                      </p>
                    </div>
                    <div className="px-4 py-3 text-center">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Decisions</p>
                      <p className="text-base font-bold mt-0.5">
                        {detailData.decisions?.length ?? 0}
                      </p>
                    </div>
                  </div>

                  {/* Supporting data */}
                  {detailData.exception.supporting_data &&
                    Object.keys(detailData.exception.supporting_data).length > 0 && (
                      <div className="px-4 py-3 border-b">
                        <p className="text-xs font-semibold mb-2 uppercase tracking-wider text-muted-foreground">
                          Supporting Data
                        </p>
                        <div className="grid grid-cols-2 gap-x-6 gap-y-1.5">
                          {Object.entries(detailData.exception.supporting_data).map(
                            ([k, v]) => (
                              <div key={k} className="flex justify-between text-xs py-0.5">
                                <span className="text-muted-foreground capitalize">
                                  {k.replace(/_/g, " ")}
                                </span>
                                <span className="font-medium tabular-nums">
                                  {typeof v === "number"
                                    ? v.toLocaleString()
                                    : String(v)}
                                </span>
                              </div>
                            )
                          )}
                        </div>
                      </div>
                    )}

                  {/* Decision History */}
                  {detailData.decisions && detailData.decisions.length > 0 && (
                    <div className="px-4 py-3 border-b">
                      <p className="text-xs font-semibold mb-2 uppercase tracking-wider text-muted-foreground">
                        Decision History
                      </p>
                      <div className="space-y-2 max-h-40 overflow-y-auto">
                        {detailData.decisions.map((d: PlannerDecision, idx: number) => (
                          <div
                            key={d.decision_id}
                            className="flex items-start gap-3 text-xs"
                          >
                            <div className="flex flex-col items-center mt-1">
                              <div className="h-2 w-2 rounded-full bg-border" />
                              {idx < detailData.decisions.length - 1 && (
                                <div className="w-px h-6 bg-border mt-0.5" />
                              )}
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2 flex-wrap">
                                <span
                                  className={`px-2 py-0.5 rounded-full font-medium ${
                                    DECISION_TYPE_COLORS[d.decision_type] ?? ""
                                  }`}
                                >
                                  {d.decision_type.replace(/_/g, " ")}
                                </span>
                                <span className="text-muted-foreground">
                                  {d.decided_by} &middot; {new Date(d.decided_at).toLocaleDateString()}
                                </span>
                              </div>
                              {d.rationale && (
                                <p className="text-muted-foreground mt-0.5 leading-snug">{d.rationale}</p>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Action Panel */}
                  <div className="px-4 py-4 bg-muted/30 space-y-4">
                    <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      Take Action
                    </p>

                    <div className="flex flex-wrap gap-2">
                      <button
                        className="inline-flex items-center gap-1.5 text-xs font-medium rounded-md border px-3 py-1.5 hover:bg-yellow-50 hover:border-yellow-300 hover:text-yellow-800 dark:hover:bg-yellow-900/30 dark:hover:border-yellow-700 dark:hover:text-yellow-300 transition-colors disabled:opacity-50"
                        disabled={statusMutation.isPending || detailData.exception.status === "investigating"}
                        onClick={() => statusMutation.mutate({ id: selectedExceptionId, status: "investigating" })}
                      >
                        Start Investigation
                      </button>
                      <button
                        className="inline-flex items-center gap-1.5 text-xs font-medium rounded-md border px-3 py-1.5 hover:bg-green-50 hover:border-green-300 hover:text-green-800 dark:hover:bg-green-900/30 dark:hover:border-green-700 dark:hover:text-green-300 transition-colors disabled:opacity-50"
                        disabled={statusMutation.isPending || detailData.exception.status === "resolved"}
                        onClick={() => statusMutation.mutate({ id: selectedExceptionId, status: "resolved" })}
                      >
                        Mark Resolved
                      </button>
                      <button
                        className="inline-flex items-center gap-1.5 text-xs font-medium rounded-md border px-3 py-1.5 hover:bg-muted transition-colors disabled:opacity-50"
                        disabled={statusMutation.isPending || detailData.exception.status === "dismissed"}
                        onClick={() => statusMutation.mutate({ id: selectedExceptionId, status: "dismissed" })}
                      >
                        Dismiss
                      </button>
                    </div>

                    <div className="space-y-2">
                      <p className="text-xs text-muted-foreground">
                        Or record a formal decision with rationale:
                      </p>
                      <select
                        value={decisionType}
                        onChange={(e) => setDecisionType(e.target.value)}
                        className="w-full h-8 text-xs rounded-md border border-input bg-background px-2.5"
                      >
                        {DECISION_TYPES.map((dt) => (
                          <option key={dt.value} value={dt.value}>
                            {dt.label} {"\u2014"} {dt.desc}
                          </option>
                        ))}
                      </select>
                      <textarea
                        value={rationale}
                        onChange={(e) => setRationale(e.target.value)}
                        placeholder="Explain your reasoning (optional but recommended for audit trail)..."
                        rows={2}
                        className="w-full text-xs rounded-md border border-input bg-background px-2.5 py-2 resize-none"
                      />
                      <button
                        className="w-full h-8 text-xs font-medium rounded-md bg-primary text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
                        disabled={decisionMutation.isPending}
                        onClick={() => {
                          decisionMutation.mutate({
                            id: selectedExceptionId,
                            type: decisionType,
                            rat: rationale,
                          });
                        }}
                      >
                        {decisionMutation.isPending ? "Submitting..." : "Submit Decision"}
                      </button>
                    </div>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
