/**
 * Feature 40 — Exception Triage
 *
 * 3-zone command center for demand planners:
 * Zone 1: Summary KPI header
 * Zone 2: Exception Queue (left, 40% width)
 * Zone 3: Investigation Panel (right, 60% width)
 */
import { useState, useRef, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type {
  StoryboardException,
  StoryboardSummary,
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

// ---------------------------------------------------------------------------
// Color maps
// ---------------------------------------------------------------------------
const EXCEPTION_TYPE_COLORS: Record<string, string> = {
  forecast_bias: "bg-blue-100 text-blue-800 border-blue-200",
  stockout_risk: "bg-red-100 text-red-800 border-red-200",
  accuracy_drop: "bg-orange-100 text-orange-800 border-orange-200",
  excess_risk: "bg-cyan-100 text-cyan-800 border-cyan-200",
  model_drift: "bg-yellow-100 text-yellow-800 border-yellow-200",
  new_item: "bg-green-100 text-green-800 border-green-200",
};

const DECISION_TYPE_COLORS: Record<string, string> = {
  override_forecast: "bg-blue-100 text-blue-800",
  accept_exception: "bg-green-100 text-green-800",
  escalate: "bg-red-100 text-red-800",
  dismiss: "bg-gray-100 text-gray-700",
  request_info: "bg-amber-100 text-amber-800",
};

const STATUS_COLORS: Record<string, string> = {
  open: "bg-red-100 text-red-800",
  investigating: "bg-yellow-100 text-yellow-800",
  resolved: "bg-green-100 text-green-800",
  dismissed: "bg-gray-100 text-gray-700",
};

function severityLabel(score: number): string {
  if (score >= 0.75) return "Critical";
  if (score >= 0.50) return "High";
  if (score >= 0.25) return "Medium";
  return "Low";
}

function getSeverityColor(severity: number): string {
  if (severity >= 0.7) return "text-red-600 bg-red-50 border-red-200";
  if (severity >= 0.4) return "text-orange-600 bg-orange-50 border-orange-200";
  return "text-yellow-600 bg-yellow-50 border-yellow-200";
}

function fmt(n: number | null | undefined, dec = 2): string {
  if (n == null) return "—";
  return Number(n).toFixed(dec);
}

function fmtCurrency(n: number | null | undefined): string {
  if (n == null) return "—";
  return `$${Number(n).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

function daysAgo(dateStr: string): string {
  const ms = Date.now() - new Date(dateStr).getTime();
  const days = Math.floor(ms / (1000 * 60 * 60 * 24));
  if (days === 0) return "today";
  if (days === 1) return "1 day ago";
  return `${days} days ago`;
}

const EXCEPTION_TYPES = [
  "all",
  "forecast_bias",
  "stockout_risk",
  "accuracy_drop",
  "excess_risk",
  "model_drift",
  "new_item",
];

const STATUS_FILTERS = ["all", "open", "investigating"];

const EXCEPTION_TYPE_LABELS: Record<string, string> = {
  all: "All Types",
  forecast_bias: "Forecast Bias",
  stockout_risk: "Stockout Risk",
  accuracy_drop: "Accuracy Drop",
  excess_risk: "Excess Risk",
  model_drift: "Model Drift",
  new_item: "New Item",
};

const DECISION_TYPES = [
  { value: "override_forecast", label: "Override Forecast" },
  { value: "accept_exception", label: "Accept Exception" },
  { value: "escalate", label: "Escalate" },
  { value: "dismiss", label: "Dismiss" },
  { value: "request_info", label: "Request Info" },
];

const PAGE_SIZE = 20;

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
    },
  });

  const topType =
    summary?.by_type && summary.by_type.length > 0
      ? summary.by_type[0].exception_type
      : "—";

  return (
    <div className="flex flex-col gap-4 p-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Exception Triage</h2>
        <p className="text-xs text-muted-foreground mt-0.5">Rule-based threshold alerts from replenishment policies. For ML-generated insights ranked by financial impact, see the <strong>AI Planner</strong> tab.</p>
        <button
          className="text-xs rounded border px-3 py-1 hover:bg-muted"
          onClick={() => {
            queryClient.invalidateQueries({ queryKey: ["sb-"] });
          }}
        >
          Refresh
        </button>
      </div>

      {summaryLoading && (
        <p className="text-sm text-muted-foreground">Loading storyboard data...</p>
      )}

      {/* ZONE 1: Summary KPI Strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3" data-testid="summary-kpis">
        <SbKpiCard
          label="Total Open"
          value={`${summary?.total_open ?? 0}`}
          color="red"
        />
        <SbKpiCard
          label="Investigating"
          value={`${summary?.total_investigating ?? 0}`}
          color="amber"
        />
        <SbKpiCard
          label="Avg Severity"
          value={summary?.avg_severity != null ? fmt(summary.avg_severity, 2) : "—"}
          color={
            summary?.avg_severity == null
              ? undefined
              : summary.avg_severity >= 0.7
              ? "red"
              : summary.avg_severity >= 0.4
              ? "amber"
              : "green"
          }
        />
        <SbKpiCard
          label="Top Exception Type"
          value={
            EXCEPTION_TYPE_LABELS[topType] ?? topType
          }
        />
      </div>

      {/* ZONE 2 + ZONE 3: Split layout */}
      <div className="flex flex-col lg:flex-row gap-4">
        {/* ZONE 2: Exception Queue (40% on large screens) */}
        <div className="lg:w-[40%] flex flex-col gap-3">
          <div className="rounded-lg border bg-card p-4 space-y-3">
            <h3 className="text-sm font-semibold">Exception Queue</h3>

            {/* Status filter pills */}
            <div className="flex flex-wrap gap-1.5">
              {STATUS_FILTERS.map((s) => (
                <button
                  key={s}
                  onClick={() => {
                    setStatusFilter(s);
                    setPage(0);
                  }}
                  className={`px-2.5 py-0.5 rounded-full text-xs font-medium border transition-colors ${
                    statusFilter === s
                      ? "bg-primary text-primary-foreground border-primary"
                      : "border-border hover:bg-muted"
                  }`}
                >
                  {s === "all" ? "All" : s.charAt(0).toUpperCase() + s.slice(1)}
                </button>
              ))}
            </div>

            {/* Type filter pills */}
            <div className="flex flex-wrap gap-1.5">
              {EXCEPTION_TYPES.map((t) => (
                <button
                  key={t}
                  onClick={() => {
                    setTypeFilter(t);
                    setPage(0);
                  }}
                  className={`px-2.5 py-0.5 rounded-full text-xs font-medium border transition-colors ${
                    typeFilter === t
                      ? "bg-primary text-primary-foreground border-primary"
                      : "border-border hover:bg-muted"
                  }`}
                >
                  {EXCEPTION_TYPE_LABELS[t] ?? t}
                </button>
              ))}
            </div>

            {/* Item/Loc search */}
            <div className="flex gap-2">
              <input
                type="text"
                placeholder="Item..."
                value={itemFilter}
                onChange={(e) => {
                  setItemFilter(e.target.value);
                  setPage(0);
                }}
                className="flex-1 text-xs rounded border px-2 py-1 bg-background"
              />
              <input
                type="text"
                placeholder="Location..."
                value={locFilter}
                onChange={(e) => {
                  setLocFilter(e.target.value);
                  setPage(0);
                }}
                className="flex-1 text-xs rounded border px-2 py-1 bg-background"
              />
            </div>

            {/* Exception list */}
            {listLoading && (
              <p className="text-xs text-muted-foreground">Loading exceptions...</p>
            )}
            {!listLoading && listData?.rows && listData.rows.length === 0 && (
              <p className="text-xs text-muted-foreground">No exceptions found.</p>
            )}
            <div className="space-y-2 max-h-[500px] overflow-y-auto">
              {listData?.rows?.map((exc: StoryboardException) => (
                <ExceptionCard
                  key={exc.exception_id}
                  exception={exc}
                  isSelected={selectedExceptionId === exc.exception_id}
                  onInvestigate={() => {
                    setSelectedExceptionId(exc.exception_id);
                    setNewStatus("investigating");
                  }}
                />
              ))}
            </div>

            {/* Pagination */}
            {listData && listData.total > (page + 1) * PAGE_SIZE && (
              <button
                className="w-full text-xs border rounded py-1.5 hover:bg-muted"
                onClick={() => setPage((p) => p + 1)}
              >
                Load more ({listData.total - (page + 1) * PAGE_SIZE} remaining)
              </button>
            )}
          </div>
        </div>

        {/* ZONE 3: Investigation Panel (60% on large screens) */}
        <div className="lg:w-[60%] flex flex-col gap-3">
          {!selectedExceptionId && (
            <div className="rounded-lg border bg-card p-8 flex items-center justify-center text-sm text-muted-foreground">
              Select an exception from the queue to investigate.
            </div>
          )}

          {selectedExceptionId && (
            <div className="rounded-lg border bg-card p-4 space-y-4">
              {detailLoading && (
                <p className="text-xs text-muted-foreground">Loading exception detail...</p>
              )}

              {detailData && (
                <>
                  {/* KPI Strip */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <div className="rounded border p-2 text-center">
                      <p className="text-[10px] text-muted-foreground">Severity</p>
                      <p
                        className={`text-lg font-bold ${
                          detailData.exception.severity >= 0.7
                            ? "text-red-600"
                            : detailData.exception.severity >= 0.4
                            ? "text-orange-600"
                            : "text-yellow-600"
                        }`}
                      >
                        {severityLabel(detailData.exception.severity)}
                      </p>
                    </div>
                    <div className="rounded border p-2 text-center">
                      <p className="text-[10px] text-muted-foreground">Financial Impact</p>
                      <p className="text-lg font-bold">
                        {fmtCurrency(detailData.exception.financial_impact)}
                      </p>
                    </div>
                    <div className="rounded border p-2 text-center">
                      <p className="text-[10px] text-muted-foreground">Status</p>
                      <span
                        className={`inline-block text-xs px-2 py-0.5 rounded-full font-medium ${
                          STATUS_COLORS[detailData.exception.status] ?? ""
                        }`}
                      >
                        {detailData.exception.status}
                      </span>
                    </div>
                    <div className="rounded border p-2 text-center">
                      <p className="text-[10px] text-muted-foreground">Days Open</p>
                      <p className="text-sm font-medium">
                        {daysAgo(detailData.exception.generated_at)}
                      </p>
                    </div>
                  </div>

                  {/* Exception Details */}
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <span
                        className={`text-xs px-2 py-0.5 rounded-full font-medium border ${
                          EXCEPTION_TYPE_COLORS[detailData.exception.exception_type] ?? ""
                        }`}
                      >
                        {EXCEPTION_TYPE_LABELS[detailData.exception.exception_type] ??
                          detailData.exception.exception_type}
                      </span>
                      <span className="text-sm font-medium">
                        {detailData.exception.item_no} @ {detailData.exception.loc}
                      </span>
                    </div>
                    {detailData.exception.headline && (
                      <p className="text-sm text-muted-foreground">
                        {detailData.exception.headline}
                      </p>
                    )}

                    {/* Supporting data key-value pairs */}
                    {detailData.exception.supporting_data &&
                      Object.keys(detailData.exception.supporting_data).length > 0 && (
                        <div className="rounded border bg-muted/30 p-3 space-y-1">
                          <p className="text-xs font-semibold mb-2">Supporting Data</p>
                          {Object.entries(detailData.exception.supporting_data).map(
                            ([k, v]) => (
                              <div key={k} className="flex justify-between text-xs">
                                <span className="text-muted-foreground capitalize">
                                  {k.replace(/_/g, " ")}
                                </span>
                                <span className="font-medium">
                                  {typeof v === "number"
                                    ? v.toLocaleString()
                                    : String(v)}
                                </span>
                              </div>
                            )
                          )}
                        </div>
                      )}
                  </div>

                  {/* Decision History */}
                  {detailData.decisions && detailData.decisions.length > 0 && (
                    <div className="space-y-2">
                      <p className="text-xs font-semibold">Decision History</p>
                      <div className="space-y-2 max-h-48 overflow-y-auto">
                        {detailData.decisions.map((d: PlannerDecision) => (
                          <div
                            key={d.decision_id}
                            className="rounded border p-2.5 text-xs space-y-1"
                          >
                            <div className="flex items-center gap-2">
                              <span
                                className={`px-2 py-0.5 rounded-full font-medium ${
                                  DECISION_TYPE_COLORS[d.decision_type] ?? ""
                                }`}
                              >
                                {d.decision_type.replace(/_/g, " ")}
                              </span>
                              <span className="text-muted-foreground ml-auto">
                                {d.decided_by} · {new Date(d.decided_at).toLocaleDateString()}
                              </span>
                            </div>
                            {d.rationale && (
                              <p className="text-muted-foreground">{d.rationale}</p>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Action Panel */}
                  <div className="border-t pt-4 space-y-4">
                    {/* Status update */}
                    <div className="space-y-2">
                      <p className="text-xs font-semibold">Update Status</p>
                      <div className="flex gap-2">
                        <select
                          value={newStatus}
                          onChange={(e) => setNewStatus(e.target.value)}
                          className="flex-1 text-xs rounded border px-2 py-1.5 bg-background"
                        >
                          <option value="investigating">Investigating</option>
                          <option value="resolved">Resolved</option>
                          <option value="dismissed">Dismissed</option>
                        </select>
                        <button
                          className="text-xs rounded border px-3 py-1.5 hover:bg-muted disabled:opacity-50"
                          disabled={statusMutation.isPending}
                          onClick={() => {
                            statusMutation.mutate({
                              id: selectedExceptionId,
                              status: newStatus,
                            });
                          }}
                        >
                          {statusMutation.isPending ? "Updating..." : "Update Status"}
                        </button>
                      </div>
                    </div>

                    {/* Submit decision */}
                    <div className="space-y-2">
                      <p className="text-xs font-semibold">Submit Decision</p>
                      <select
                        value={decisionType}
                        onChange={(e) => setDecisionType(e.target.value)}
                        className="w-full text-xs rounded border px-2 py-1.5 bg-background"
                      >
                        {DECISION_TYPES.map((dt) => (
                          <option key={dt.value} value={dt.value}>
                            {dt.label}
                          </option>
                        ))}
                      </select>
                      <textarea
                        value={rationale}
                        onChange={(e) => setRationale(e.target.value)}
                        placeholder="Rationale (optional)..."
                        rows={3}
                        className="w-full text-xs rounded border px-2 py-1.5 bg-background resize-none"
                      />
                      <button
                        className="w-full text-xs rounded border px-3 py-1.5 hover:bg-muted disabled:opacity-50"
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

// ---------------------------------------------------------------------------
// ExceptionCard helper
// ---------------------------------------------------------------------------
function ExceptionCard({
  exception,
  isSelected,
  onInvestigate,
}: {
  exception: StoryboardException;
  isSelected: boolean;
  onInvestigate: () => void;
}) {
  return (
    <div
      className={`rounded border p-2.5 text-xs space-y-1.5 cursor-pointer transition-colors ${
        isSelected ? "border-primary bg-primary/5" : "hover:bg-muted/50"
      }`}
    >
      {/* Top row: severity + type badge + status */}
      <div className="flex items-center gap-2">
        <span
          className={`px-1.5 py-0.5 rounded font-bold border text-[10px] ${getSeverityColor(
            exception.severity
          )}`}
        >
          {severityLabel(exception.severity)}
        </span>
        <span
          className={`px-1.5 py-0.5 rounded-full text-[10px] font-medium border ${
            EXCEPTION_TYPE_COLORS[exception.exception_type] ?? ""
          }`}
        >
          {EXCEPTION_TYPE_LABELS[exception.exception_type] ?? exception.exception_type}
        </span>
        <span
          className={`ml-auto px-1.5 py-0.5 rounded-full text-[10px] font-medium ${
            STATUS_COLORS[exception.status] ?? ""
          }`}
        >
          {exception.status}
        </span>
      </div>

      {/* Item + Loc */}
      <p className="font-medium">
        {exception.item_no} @ {exception.loc}
      </p>

      {/* Headline */}
      {exception.headline && (
        <p className="text-muted-foreground leading-tight">{exception.headline}</p>
      )}

      {/* Financial impact + date */}
      <div className="flex items-center justify-between">
        {exception.financial_impact != null ? (
          <span className="text-muted-foreground">
            Impact: {fmtCurrency(exception.financial_impact)}
          </span>
        ) : (
          <span />
        )}
        <span className="text-muted-foreground">{daysAgo(exception.generated_at)}</span>
      </div>

      {/* Investigate button */}
      <button
        className="w-full text-xs border rounded py-1 hover:bg-muted mt-1"
        onClick={onInvestigate}
      >
        Investigate
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// KPI Card helper
// ---------------------------------------------------------------------------
function SbKpiCard({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: "green" | "amber" | "red";
}) {
  const textColor =
    color === "green"
      ? "text-green-600"
      : color === "amber"
      ? "text-amber-600"
      : color === "red"
      ? "text-red-600"
      : "";

  return (
    <div className="rounded-lg border bg-card p-3">
      <p className="text-xs text-muted-foreground truncate">{label}</p>
      <p className={`text-xl font-bold truncate ${textColor}`}>{value}</p>
    </div>
  );
}
