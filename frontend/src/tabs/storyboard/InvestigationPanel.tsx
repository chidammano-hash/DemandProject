/**
 * InvestigationPanel — ZONE 3 of the Exception Triage command center.
 *
 * Right-hand detail view (60% width): exception detail, supporting data,
 * decision history, and the action panel (status mutations + formal decision).
 * Extracted verbatim from StoryboardTab to keep the tab file under the
 * <600-line limit.
 */
import type { UseMutationResult } from "@tanstack/react-query";
import type {
  StoryboardException,
  PlannerDecision,
} from "@/types/storyboard";
import { Skeleton } from "@/components/Skeleton";
import { cn } from "@/lib/utils";
import { formatDate } from "@/lib/formatters";

import {
  severityLabel,
  severityColorClass,
  severityBg,
  fmtCurrency,
  daysAgo,
  EXCEPTION_TYPE_LABELS,
  EXCEPTION_TYPE_COLORS,
  DECISION_TYPES,
  DECISION_TYPE_COLORS,
  STATUS_COLORS,
  STATUS_DOT,
} from "./storyboardShared";

type SbDetail = { exception: StoryboardException; decisions: PlannerDecision[] };

export function InvestigationPanel({
  selectedExceptionId,
  detailData,
  detailLoading,
  actionSuccess,
  decisionType,
  setDecisionType,
  rationale,
  setRationale,
  statusMutation,
  decisionMutation,
}: {
  selectedExceptionId: string | null;
  detailData: SbDetail | undefined;
  detailLoading: boolean;
  actionSuccess: string | null;
  decisionType: string;
  setDecisionType: (value: string) => void;
  rationale: string;
  setRationale: (value: string) => void;
  statusMutation: UseMutationResult<
    void,
    Error,
    { id: string; status: string }
  >;
  decisionMutation: UseMutationResult<
    void,
    Error,
    { id: string; type: string; rat: string }
  >;
}) {
  return (
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
            <div className="p-4 space-y-4" data-testid="detail-skeleton">
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Skeleton className="h-5 w-20 rounded-full" />
                  <Skeleton className="h-5 w-16 rounded-full" />
                </div>
                <Skeleton className="h-5 w-40" />
                <Skeleton className="h-4 w-full max-w-sm" />
              </div>
              <div className="grid grid-cols-3 gap-3">
                {Array.from({ length: 3 }).map((_, i) => (
                  <div key={i} className="text-center space-y-1.5">
                    <Skeleton className="h-2 w-16 mx-auto" />
                    <Skeleton className="h-5 w-12 mx-auto" />
                  </div>
                ))}
              </div>
              <Skeleton className="h-20 w-full" />
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
                  <div className={cn(
                    "flex flex-col items-center gap-1 ml-4 px-3 py-2 rounded-lg",
                    detailData.exception.severity >= 0.75 ? "bg-red-50 dark:bg-red-950/30" :
                    detailData.exception.severity >= 0.5 ? "bg-orange-50 dark:bg-orange-950/30" :
                    detailData.exception.severity >= 0.25 ? "bg-yellow-50 dark:bg-yellow-950/30" :
                    "bg-green-50 dark:bg-green-950/30"
                  )}>
                    <div className={`text-2xl font-bold tabular-nums ${severityColorClass(detailData.exception.severity)}`}>
                      {Math.round(detailData.exception.severity * 100)}
                    </div>
                    <span className={`text-[10px] font-semibold uppercase tracking-wider ${severityColorClass(detailData.exception.severity)}`}>
                      {severityLabel(detailData.exception.severity)}
                    </span>
                    <div className="w-14 h-1.5 rounded-full bg-muted overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-300 ${severityBg(detailData.exception.severity)}`}
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
                              {d.decided_by} &middot; {formatDate(d.decided_at)}
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
              <div className="px-4 py-4 bg-muted/30 border-t space-y-4">
                <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  Take Action
                </p>

                <div className="flex flex-wrap gap-2">
                  <button
                    className={cn(
                      "inline-flex items-center gap-1.5 text-xs font-medium rounded-md border px-3 py-1.5 transition-colors disabled:opacity-50",
                      detailData.exception.status === "investigating"
                        ? "bg-yellow-100 border-yellow-300 text-yellow-800 dark:bg-yellow-900/40 dark:border-yellow-700 dark:text-yellow-300"
                        : "hover:bg-yellow-50 hover:border-yellow-300 hover:text-yellow-800 dark:hover:bg-yellow-900/30 dark:hover:border-yellow-700 dark:hover:text-yellow-300"
                    )}
                    disabled={statusMutation.isPending || detailData.exception.status === "investigating"}
                    onClick={() => statusMutation.mutate({ id: selectedExceptionId, status: "investigating" })}
                  >
                    {detailData.exception.status === "investigating" && (
                      <span className="h-1.5 w-1.5 rounded-full bg-yellow-500" />
                    )}
                    {statusMutation.isPending ? "Updating..." : "Start Investigation"}
                  </button>
                  <button
                    className={cn(
                      "inline-flex items-center gap-1.5 text-xs font-medium rounded-md border px-3 py-1.5 transition-colors disabled:opacity-50",
                      detailData.exception.status === "resolved"
                        ? "bg-green-100 border-green-300 text-green-800 dark:bg-green-900/40 dark:border-green-700 dark:text-green-300"
                        : "hover:bg-green-50 hover:border-green-300 hover:text-green-800 dark:hover:bg-green-900/30 dark:hover:border-green-700 dark:hover:text-green-300"
                    )}
                    disabled={statusMutation.isPending || detailData.exception.status === "resolved"}
                    onClick={() => statusMutation.mutate({ id: selectedExceptionId, status: "resolved" })}
                  >
                    {detailData.exception.status === "resolved" && (
                      <span className="h-1.5 w-1.5 rounded-full bg-green-500" />
                    )}
                    Mark Resolved
                  </button>
                  <button
                    className={cn(
                      "inline-flex items-center gap-1.5 text-xs font-medium rounded-md border px-3 py-1.5 transition-colors disabled:opacity-50",
                      detailData.exception.status === "dismissed"
                        ? "bg-gray-200 border-gray-300 text-gray-700 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-300"
                        : "hover:bg-muted"
                    )}
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
                        {dt.label} {"—"} {dt.desc}
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
  );
}
