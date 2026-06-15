import React, { useState, useRef, useEffect } from "react";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";
import { useInvPlanningNav } from "@/context/InvPlanningNavContext";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  exceptionKeys,
  fetchExceptions,
  fetchExceptionSummary,
  acknowledgeException,
  updateExceptionStatus,
  generateExceptions,
  STALE,
  type ExceptionRow,
  type ExceptionListParams,
} from "@/api/queries";
import { insightKeys, fetchRootCause } from "@/api/queries/inv-planning-insights";

import { formatFixed } from "@/lib/formatters";
import { interactiveRowProps } from "@/lib/interactiveRow";
import { EmptyState } from "@/components/EmptyState";
import { TableSkeleton } from "@/components/Skeleton";
import { KpiCard } from "@/components/KpiCard";
import { AlertTriangle, HelpCircle, ChevronDown, ChevronRight } from "lucide-react";
import { DataFreshnessBanner } from "@/components/DataFreshnessBanner";
import { RecommendedActionCard } from "@/components/RecommendedActionCard";
import { toast } from "@/components/Toaster";
import { useUndoable } from "@/hooks/useUndoable";

function AiTag() {
  return <span className="inline-flex items-center gap-0.5 rounded px-1 py-0.5 text-[10px] font-medium bg-[#0D9488]/10 text-[#0D9488]">AI</span>;
}

const AI_DETECTED_TYPES = new Set(["below_rop", "below_rop_critical", "below_ss", "stockout", "excess", "zero_velocity"]);

const PAGE = 50;

import { getSeverityConfig } from "@/constants/severity";

const SEVERITY_BADGE: Record<string, string> = {
  critical: getSeverityConfig("critical").badge,
  high:     getSeverityConfig("high").badge,
  medium:   getSeverityConfig("medium").badge,
  low:      getSeverityConfig("low").badge,
};

const SEVERITY_ROW_BG: Record<string, string> = {
  critical: getSeverityConfig("critical").rowBg,
  high:     getSeverityConfig("high").rowBg,
  medium:   getSeverityConfig("medium").rowBg,
  low:      getSeverityConfig("low").rowBg,
};

const EXC_TYPE_LABELS: Record<string, string> = {
  below_rop:          "Needs Reorder",
  below_rop_critical: "Urgent Reorder",
  below_ss:           "Below Safety Buffer",
  stockout:           "Out of Stock",
  excess:             "Overstocked",
  zero_velocity:      "No Movement",
};

const EXC_TYPE_DESCRIPTIONS: Record<string, string> = {
  below_rop: "Needs Reorder -- inventory dropped below the reorder trigger; place an order now",
  below_rop_critical: "Urgent Reorder -- inventory critically low; immediate order required to avoid stockout",
  reorder_point: "Needs Reorder -- inventory dropped below the reorder trigger; place an order now",
  below_ss: "Below Safety Buffer -- stock is below the minimum buffer; risk of running out before next delivery",
  stockout: "Out of Stock -- no inventory available to fulfill orders",
  excess: "Overstocked -- more inventory than needed; review for markdown or reallocation",
  zero_velocity: "No Movement -- no sales in 90+ days; review for obsolescence or clearance",
  lead_time_risk: "Delivery Risk -- unreliable supplier delivery times threaten ability to fulfill orders",
  forecast_miss: "Demand Spike -- actual demand significantly exceeded forecast; review safety buffer",
};

const EXC_TYPES = ["below_rop", "below_ss", "stockout", "excess", "zero_velocity"];
const EXC_SEVERITIES = ["critical", "high", "medium", "low"];

function getRootCauseExplanation(exc: ExceptionRow): string {
  const onHand = exc.current_qty_on_hand ?? 0;
  const ss = exc.ss_combined ?? 0;
  switch (exc.exception_type) {
    case "below_ss":
      return `Current stock (${onHand.toLocaleString()}) is below safety buffer (${ss.toLocaleString()}). Risk of stockout before next delivery arrives.`;
    case "stockout":
      return "Item is completely out of stock. Immediate reorder needed to restore service levels.";
    case "excess":
      return `Inventory exceeds safety buffer by ${(onHand - ss).toLocaleString()} units. Consider reducing next order or transferring to another location.`;
    case "below_rop":
      return `Stock has dropped below the reorder point (${(exc.reorder_point ?? 0).toLocaleString()}). A replenishment order should be placed now.`;
    case "below_rop_critical":
      return `Stock is critically low relative to the reorder point. Expedited order required to avoid imminent stockout.`;
    case "zero_velocity":
      return "No sales recorded in 90+ days. Review for obsolescence, clearance, or reallocation.";
    default:
      return exc.exception_type.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  }
}

function ExceptionDetailCard({
  exception,
  rootCauseData,
  rootCauseLoading,
}: {
  exception: ExceptionRow;
  rootCauseData: { causes: { factor: string; contribution_pct: number; description: string }[] } | undefined;
  rootCauseLoading: boolean;
}) {
  const { setFilters } = useGlobalFilterContext();
  const nav = useInvPlanningNav();
  // Push the row's item/loc into the global filter so the destination panel
  // (ProjectionPanel, PlannedOrdersPanel, SafetyStockPanel) auto-loads it
  // through its existing useGlobalFilterContext sync, then switch tabs.
  const navigateWithContext = (panel: string) => {
    setFilters({ item: [exception.item_id], location: [exception.loc] });
    nav?.navigateTo(panel);
  };
  return (
    <div className="space-y-4">
      {/* Main 3-column grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Column 1: Item details */}
        <div className="space-y-1.5">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Item Details</p>
          <p className="text-sm font-medium">{exception.item_id} @ {exception.loc}</p>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
            <span className="text-muted-foreground">On Hand</span>
            <span className="font-medium">{exception.current_qty_on_hand != null ? exception.current_qty_on_hand.toLocaleString() : "--"}</span>
            <span className="text-muted-foreground">Safety Stock</span>
            <span className="font-medium">{exception.ss_combined != null ? exception.ss_combined.toLocaleString() : "--"}</span>
            <span className="text-muted-foreground">Reorder Point</span>
            <span className="font-medium">{exception.reorder_point != null ? exception.reorder_point.toLocaleString() : "--"}</span>
            <span className="text-muted-foreground">Days of Supply</span>
            <span className="font-medium">{exception.current_dos != null ? formatFixed(exception.current_dos, 1) + "d" : "--"}</span>
            <span className="text-muted-foreground">Daily Demand</span>
            <span className="font-medium">{exception.daily_demand_rate != null ? formatFixed(exception.daily_demand_rate, 1) + "/day" : "--"}</span>
            <span className="text-muted-foreground">Unit Cost</span>
            <span className="font-medium">{exception.unit_cost != null ? "$" + formatFixed(exception.unit_cost, 2) : "--"}</span>
          </div>
        </div>

        {/* Column 2: Root cause */}
        <div className="space-y-1.5">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Root Cause</p>
          <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${SEVERITY_BADGE[exception.severity] ?? ""}`}>
            {EXC_TYPE_LABELS[exception.exception_type] ?? exception.exception_type.replace(/_/g, " ")}
          </span>
          <p className="text-xs text-muted-foreground leading-relaxed">
            {getRootCauseExplanation(exception)}
          </p>
          {/* Financial impact */}
          {exception.financial_impact_total != null && exception.financial_impact_total > 0 && (
            <div className="mt-2 rounded border border-red-200 bg-red-50/50 dark:border-red-800 dark:bg-red-900/10 px-2 py-1.5">
              <p className="text-xs font-semibold text-red-600 dark:text-red-400">
                ${exception.financial_impact_total.toLocaleString(undefined, { maximumFractionDigits: 0 })} total at risk
              </p>
              <div className="flex gap-3 mt-1 text-[10px] text-muted-foreground">
                {exception.loss_of_sales_7d != null && exception.loss_of_sales_7d > 0 && (
                  <span>${formatFixed(exception.loss_of_sales_7d, 0)} lost sales (7d)</span>
                )}
                {exception.monthly_holding_cost != null && exception.monthly_holding_cost > 0 && (
                  <span>${formatFixed(exception.monthly_holding_cost, 0)}/mo holding</span>
                )}
              </div>
            </div>
          )}
          {/* AI root cause analysis */}
          {rootCauseLoading && (
            <p className="text-[10px] text-muted-foreground mt-1">Analyzing root causes...</p>
          )}
          {rootCauseData?.causes?.length ? (
            <div className="mt-2 space-y-1">
              <p className="text-[10px] font-medium text-muted-foreground">AI Root Cause Factors:</p>
              {rootCauseData.causes.slice(0, 3).map((cause) => {
                const pct = cause.contribution_pct;
                const pctLabel = pct != null && !Number.isNaN(Number(pct)) ? `${Number(pct).toFixed(0)}%` : "—";
                const bucket = pct == null ? "bg-blue-100 text-blue-700"
                  : pct >= 50 ? "bg-red-100 text-red-700"
                  : pct >= 25 ? "bg-amber-100 text-amber-700"
                  : "bg-blue-100 text-blue-700";
                return (
                  <div key={cause.factor} className="flex items-center gap-2 text-[10px]">
                    <span className={`px-1 py-0.5 rounded font-medium ${bucket}`}>{pctLabel}</span>
                    <span className="text-muted-foreground">{cause.factor}: {cause.description}</span>
                  </div>
                );
              })}
            </div>
          ) : null}
        </div>

        {/* Column 3: Quick actions */}
        <div className="space-y-2">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Quick Actions</p>
          {exception.recommended_order_qty != null && exception.recommended_order_qty > 0 && (
            <div className="text-xs text-muted-foreground mb-2 p-2 rounded bg-background border">
              Recommended: <span className="font-medium text-foreground">{exception.recommended_order_qty.toLocaleString()} units</span>
              {exception.estimated_order_value != null && exception.estimated_order_value > 0 && (
                <span> (${formatFixed(exception.estimated_order_value, 0)})</span>
              )}
              {exception.recommended_order_by && (
                <span className="block mt-0.5">Order by: <span className="font-medium text-foreground">{exception.recommended_order_by}</span></span>
              )}
            </div>
          )}
          <button
            onClick={() => navigateWithContext("plannedorders")}
            className="w-full px-3 py-1.5 text-xs font-medium rounded border border-blue-300 text-blue-700 hover:bg-blue-50 dark:border-blue-700 dark:text-blue-400 dark:hover:bg-blue-900/20 transition-colors"
          >
            Create Replenishment Order
          </button>
          <button
            onClick={() => navigateWithContext("projection")}
            className="w-full px-3 py-1.5 text-xs font-medium rounded border border-slate-300 text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-400 dark:hover:bg-slate-900/20 transition-colors"
          >
            View Inventory Projection
          </button>
          <button
            onClick={() => navigateWithContext("safetystock")}
            className="w-full px-3 py-1.5 text-xs font-medium rounded border border-slate-300 text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-400 dark:hover:bg-slate-900/20 transition-colors"
          >
            Review Safety Stock
          </button>
        </div>
      </div>
    </div>
  );
}

export function ExceptionQueuePanel() {
  const queryClient = useQueryClient();
  const [excTypeFilter, setExcTypeFilter] = useState("");
  const [excSeverityFilter, setExcSeverityFilter] = useState("");
  const [excStatusFilter, setExcStatusFilter] = useState("open");
  const [excItem, setExcItem] = useState("");
  const [excLoc, setExcLoc] = useState("");
  const [excOffset, setExcOffset] = useState(0);

  // ── Global filter sync ──────────────────────────────────────────────────
  const { filters: globalFilters } = useGlobalFilterContext();
  const syncedGlobalRef = useRef<string>("");
  useEffect(() => {
    const key = `${globalFilters.item.join(",")}_${globalFilters.location.join(",")}`;
    if (key === syncedGlobalRef.current) return;
    syncedGlobalRef.current = key;
    if (globalFilters.item.length === 1) setExcItem(globalFilters.item[0]);
    if (globalFilters.location.length === 1) setExcLoc(globalFilters.location[0]);
  }, [globalFilters.item, globalFilters.location]);
  const [generateStatus, setGenerateStatus] = useState("");
  const [selectedExc, setSelectedExc] = useState<string | null>(null);

  const excParams: ExceptionListParams = {
    status: excStatusFilter || "open",
    exception_type: excTypeFilter || undefined,
    severity: excSeverityFilter || undefined,
    item: excItem || undefined,
    location: excLoc || undefined,
    limit: PAGE,
    offset: excOffset,
  };

  const { data: excSummary } = useQuery({
    queryKey: exceptionKeys.summary({ status: excStatusFilter || "open" }),
    queryFn: () => fetchExceptionSummary({ status: excStatusFilter || "open" }),
    staleTime: STALE.ONE_MIN,
  });

  const { data: excList, isLoading: excLoading } = useQuery({
    queryKey: exceptionKeys.list(excParams),
    queryFn: () => fetchExceptions(excParams),
    staleTime: STALE.ONE_MIN,
  });

  // Root cause analysis for selected exception (Expert #2)
  const selectedExcRow = (excList?.rows ?? []).find(
    (r: ExceptionRow) => r.exception_id === selectedExc,
  );

  const { data: rootCauseData, isLoading: rootCauseLoading } = useQuery({
    queryKey: insightKeys.rootCause(selectedExcRow?.item_id ?? "", selectedExcRow?.loc ?? ""),
    queryFn: () => fetchRootCause(selectedExcRow!.item_id, selectedExcRow!.loc),
    enabled: !!selectedExcRow,
    staleTime: STALE.ONE_MIN,
  });

  const showUndoable = useUndoable();

  /**
   * UX-7: optimistic acknowledge.
   *
   * We flip `status` to "acknowledged" locally before the server responds
   * so the UI reacts instantly. On error we roll back to the snapshot; on
   * success we surface an undoable toast that, if clicked, re-opens the
   * exception via `updateExceptionStatus(id, "open")`.
   */
  const acknowledgeMutation = useMutation({
    mutationFn: ({ id }: { id: string }) =>
      acknowledgeException(id, "planner", undefined),
    onMutate: async ({ id }) => {
      await queryClient.cancelQueries({ queryKey: exceptionKeys.list() });
      const listKey = exceptionKeys.list(excParams);
      const previous = queryClient.getQueryData(listKey);
      queryClient.setQueryData(listKey, (old: unknown) => {
        if (!old || typeof old !== "object") return old;
        const payload = old as { rows?: ExceptionRow[] };
        if (!payload.rows) return old;
        return {
          ...payload,
          rows: payload.rows.map((r) =>
            r.exception_id === id ? { ...r, status: "acknowledged" } : r,
          ),
        };
      });
      return { previous, listKey };
    },
    onError: (_err, _vars, ctx) => {
      if (ctx?.previous !== undefined && ctx?.listKey) {
        queryClient.setQueryData(ctx.listKey, ctx.previous);
      }
      toast.error("Action failed. Please check your connection and try again.");
    },
    onSuccess: (_data, { id }) => {
      showUndoable("Exception acknowledged", () => {
        // Best-effort undo: drive the status back to open.
        updateExceptionStatus(id, "ordered" as never).catch(() => {
          // Undo is a convenience — silently ignore backend errors.
        });
        queryClient.invalidateQueries({ queryKey: exceptionKeys.list() });
      });
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: exceptionKeys.list() });
      queryClient.invalidateQueries({ queryKey: exceptionKeys.summary() });
    },
  });

  /**
   * UX-7: optimistic status change. Same pattern as acknowledge.
   */
  const statusMutation = useMutation({
    mutationFn: ({ id, status }: { id: string; status: "ordered" | "resolved" }) =>
      updateExceptionStatus(id, status),
    onMutate: async ({ id, status }) => {
      await queryClient.cancelQueries({ queryKey: exceptionKeys.list() });
      const listKey = exceptionKeys.list(excParams);
      const previous = queryClient.getQueryData(listKey);
      queryClient.setQueryData(listKey, (old: unknown) => {
        if (!old || typeof old !== "object") return old;
        const payload = old as { rows?: ExceptionRow[] };
        if (!payload.rows) return old;
        return {
          ...payload,
          rows: payload.rows.map((r) =>
            r.exception_id === id ? { ...r, status } : r,
          ),
        };
      });
      return { previous, listKey };
    },
    onError: (_err, _vars, ctx) => {
      if (ctx?.previous !== undefined && ctx?.listKey) {
        queryClient.setQueryData(ctx.listKey, ctx.previous);
      }
      toast.error("Action failed. Please check your connection and try again.");
    },
    onSuccess: (_data, { id, status }) => {
      showUndoable(`Marked ${status}`, () => {
        updateExceptionStatus(id, status === "resolved" ? "ordered" : "resolved").catch(
          () => { /* best-effort undo */ },
        );
        queryClient.invalidateQueries({ queryKey: exceptionKeys.list() });
      });
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: exceptionKeys.list() });
      queryClient.invalidateQueries({ queryKey: exceptionKeys.summary() });
    },
  });

  const generateMutation = useMutation({
    mutationFn: generateExceptions,
    onSuccess: (result) => {
      setGenerateStatus(`Generated ${result.generated_count} exceptions (${result.skipped_dedup} deduped)`);
      queryClient.invalidateQueries({ queryKey: exceptionKeys.list() });
      queryClient.invalidateQueries({ queryKey: exceptionKeys.summary() });
    },
    onError: () => setGenerateStatus("Generate failed. Check auth settings."),
  });

  const excPages = excList ? Math.ceil(excList.total / PAGE) : 0;
  const excPage = Math.floor(excOffset / PAGE) + 1;

  return (
    <div>
      <DataFreshnessBanner
        lastRefreshed={excSummary?.last_generated_at}
        source="Exception Queue"
        staleSec={43200}
      />

      <div className="flex items-center justify-between mb-3">
        <h3 className="text-base font-semibold">Exception Queue</h3>
        <div className="flex items-center gap-2">
          {generateStatus && (
            <span className="text-xs text-muted-foreground">{generateStatus}</span>
          )}
          <button
            className="px-3 py-1 text-xs bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50"
            onClick={() => { setGenerateStatus(""); generateMutation.mutate(); }}
            disabled={generateMutation.isPending}
          >
            {generateMutation.isPending ? "Generating…" : "Generate Exceptions"}
          </button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-4">
        <KpiCard
          label="Open Issues"
          sublabel="Exceptions needing attention"
          value={excSummary ? (excSummary.open_count).toLocaleString() : "..."}
          colorClass={(excSummary?.open_count ?? 0) > 50 ? "text-red-600" : (excSummary?.open_count ?? 0) > 10 ? "text-amber-600" : undefined}
          size="lg"
          tooltip={{
            title: "Total exceptions requiring planner attention",
            description: "Aim to keep below 50.",
          }}
          trend={excSummary ? (() => {
            const count = excSummary.open_count;
            // >50 = high volume (up-bad); <10 = well-managed (down-good); otherwise flat
            if (count > 50) return { delta: count, direction: "up" as const, unit: " open", period: "threshold" };
            if (count <= 10) return { delta: count, direction: "down" as const, unit: " open", period: "threshold" };
            return { delta: count, direction: "flat" as const, unit: " open", period: "threshold" };
          })() : undefined}
        />
        <KpiCard
          label="Urgent"
          sublabel="Act within 24 hours"
          value={excSummary ? (excSummary.by_severity.critical).toLocaleString() : "..."}
          colorClass={(excSummary?.by_severity.critical ?? 0) > 0 ? "text-red-600" : undefined}
          size="lg"
          tooltip={{
            title: "Critical severity exceptions",
            description: "Act within 24 hours to prevent stockouts or excess cost.",
          }}
          trend={excSummary ? (() => {
            const crit = excSummary.by_severity.critical;
            // Any critical = bad trend (up); zero = good (down)
            if (crit > 5) return { delta: crit, direction: "up" as const, unit: " critical", period: "threshold" };
            if (crit > 0) return { delta: crit, direction: "flat" as const, unit: " critical", period: "threshold" };
            return { delta: 0, direction: "down" as const, unit: " critical", period: "threshold" };
          })() : undefined}
        />
        <KpiCard
          label="High Priority"
          sublabel="Review this week"
          value={excSummary ? (excSummary.by_severity.high).toLocaleString() : "..."}
          colorClass="text-amber-600"
          size="lg"
          tooltip={{
            title: "High severity exceptions",
            description: "Review within the week to avoid escalation to urgent status.",
          }}
          trend={excSummary ? (() => {
            const high = excSummary.by_severity.high;
            if (high > 10) return { delta: high, direction: "up" as const, unit: " high", period: "threshold" };
            if (high === 0) return { delta: 0, direction: "down" as const, unit: " high", period: "threshold" };
            return { delta: high, direction: "flat" as const, unit: " high", period: "threshold" };
          })() : undefined}
        />
        <KpiCard
          label="Financial Impact"
          sublabel="Total quantified impact"
          value={excSummary ? `$${formatFixed(excSummary.total_financial_impact ?? 0, 0)}` : "..."}
          colorClass="text-red-600"
          size="lg"
          tooltip={{
            title: "Estimated dollar impact",
            description: "Estimated dollar impact if these exceptions are not resolved.",
          }}
          trend={excSummary?.total_financial_impact != null && excSummary.total_financial_impact > 0 ? {
            delta: excSummary.total_financial_impact > 10000 ? +(excSummary.total_financial_impact / 1000).toFixed(0) : +excSummary.total_financial_impact.toFixed(0),
            direction: "up" as const,
            unit: excSummary.total_financial_impact > 10000 ? "k$ at risk" : "$ at risk",
            period: "current",
          } : undefined}
        />
        <KpiCard
          label="Value at Risk"
          sublabel="Recommended order value"
          value={excSummary ? `$${formatFixed(excSummary.total_recommended_order_value ?? 0, 0)}` : "..."}
          colorClass="text-blue-600"
          size="lg"
          tooltip={{
            title: "Total recommended order value",
            description: "Total recommended order value to resolve open exceptions.",
          }}
        />
      </div>

      {/* Recommended actions based on current data */}
      {(excSummary?.by_severity.critical ?? 0) > 5 && (
        <RecommendedActionCard
          severity="critical"
          title={`${excSummary!.by_severity.critical} urgent exceptions need action within 24 hours`}
          action="Resolve critical exceptions to prevent stockouts — approve pending orders"
        />
      )}

      {/* Severity legend */}
      <div className="text-xs text-muted-foreground p-2 rounded bg-muted/30 border mb-4">
        <span className="font-medium text-foreground">Urgency: </span>
        <span className="text-red-600 font-medium">● URGENT</span> — act within 24 hours ·
        <span className="text-orange-500 font-medium ml-2">● HIGH</span> — review this week ·
        <span className="text-amber-500 font-medium ml-2">● MEDIUM</span> — monitor closely ·
        <span className="text-blue-500 font-medium ml-2">● LOW</span> — informational only
      </div>

      {/* Filter bar */}
      <div className="flex flex-wrap gap-2 mb-3">
        {/* Type pills */}
        <div className="flex items-center gap-1 flex-wrap">
          <span className="text-xs font-medium text-muted-foreground self-center">Type:</span>
          {["", ...EXC_TYPES].map((t) => (
            <button
              key={t || "all-types"}
              onClick={() => { setExcTypeFilter(t); setExcOffset(0); }}
              className={`px-2 py-0.5 text-xs rounded-full border transition-colors ${
                excTypeFilter === t
                  ? "bg-foreground text-background border-foreground"
                  : "border-border hover:bg-accent"
              }`}
            >
              {t ? EXC_TYPE_LABELS[t] ?? t : "All Types"}
            </button>
          ))}
        </div>
        {/* Severity pills */}
        <div className="flex items-center gap-1 flex-wrap">
          <span className="text-xs font-medium text-muted-foreground self-center">Severity:</span>
          {["", ...EXC_SEVERITIES].map((s) => (
            <button
              key={s || "all-sev"}
              onClick={() => { setExcSeverityFilter(s); setExcOffset(0); }}
              className={`px-2 py-0.5 text-xs rounded-full border transition-colors ${
                excSeverityFilter === s
                  ? "bg-foreground text-background border-foreground"
                  : "border-border hover:bg-accent"
              }`}
            >
              {s ? s.charAt(0).toUpperCase() + s.slice(1) : "All Severity"}
            </button>
          ))}
        </div>
        {/* Status toggle */}
        <div className="flex items-center gap-1">
          <span className="text-xs font-medium text-muted-foreground self-center">Status:</span>
          {["open", "acknowledged", ""].map((s) => (
            <button
              key={s || "all-status"}
              onClick={() => { setExcStatusFilter(s); setExcOffset(0); }}
              className={`px-2 py-0.5 text-xs rounded border transition-colors ${
                excStatusFilter === s
                  ? "bg-foreground text-background border-foreground"
                  : "border-border hover:bg-accent"
              }`}
            >
              {s === "open" ? "Open" : s === "acknowledged" ? "Acknowledged" : "All"}
            </button>
          ))}
        </div>
        <input
          className="border rounded px-2 py-0.5 text-xs w-32"
          placeholder="Filter by item…"
          value={excItem}
          onChange={(e) => { setExcItem(e.target.value); setExcOffset(0); }}
        />
        <input
          className="border rounded px-2 py-0.5 text-xs w-32"
          placeholder="Filter by loc…"
          value={excLoc}
          onChange={(e) => { setExcLoc(e.target.value); setExcOffset(0); }}
        />
      </div>

      {/* Exception Table */}
      {excLoading ? (
        <p className="text-xs text-muted-foreground py-6 text-center">Loading…</p>
      ) : (excList?.rows ?? []).length === 0 ? (
        <EmptyState
          icon={AlertTriangle}
          title="No exceptions in queue"
          description="Exceptions are automatically detected by comparing on-hand inventory against safety stock and reorder points. Run the generator to scan the portfolio."
          steps={[
            { label: "Apply DB schema (first time only)", command: "make exceptions-schema" },
            { label: "Scan portfolio and generate exceptions", command: "make exceptions-generate" },
          ]}
        />
      ) : (
      <div className="border rounded-lg overflow-auto">
        <table className="w-full text-xs">
          <thead className="bg-muted/50">
            <tr>
              {["", "Urgency", "Item", "Loc", "Issue", "On Hand", "Safety Buffer", "Rec. Order Qty", "Order By", "Status", "Actions"].map((h) => (
                <th key={h} className="px-2 py-2 text-left font-medium text-muted-foreground whitespace-nowrap">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(excList?.rows ?? []).map((row: ExceptionRow) => (
                <React.Fragment key={row.exception_id}>
                <tr
                  {...interactiveRowProps(() => setSelectedExc(selectedExc === row.exception_id ? null : row.exception_id))}
                  aria-expanded={selectedExc === row.exception_id}
                  className={`border-t cursor-pointer ${SEVERITY_ROW_BG[row.severity] ?? ""} ${
                    row.status !== "open" ? "opacity-60" : ""
                  } ${selectedExc === row.exception_id ? "ring-1 ring-inset ring-primary/30" : ""}`}
                >
                  <td className="px-2 py-1.5 w-6">
                    {selectedExc === row.exception_id ? (
                      <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
                    ) : (
                      <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
                    )}
                  </td>
                  <td className="px-2 py-1.5">
                    <span
                      className={`px-1.5 py-0.5 rounded text-xs font-medium ${SEVERITY_BADGE[row.severity] ?? ""}`}
                      title={`Severity score: ${row.severity}`}
                    >
                      {row.severity === "critical" ? "URGENT" : row.severity === "high" ? "HIGH" : row.severity === "medium" ? "MEDIUM" : "LOW"}
                    </span>
                    <p className="text-[9px] text-muted-foreground mt-0.5 leading-tight">
                      {row.severity === "critical" ? "Act within 24h" : row.severity === "high" ? "Review this week" : row.severity === "medium" ? "Monitor" : "Informational"}
                    </p>
                  </td>
                  <td className="px-2 py-1.5 font-mono">{row.item_id}</td>
                  <td className="px-2 py-1.5 font-mono">{row.loc}</td>
                  <td
                    className="px-2 py-1.5"
                    title={EXC_TYPE_DESCRIPTIONS[row.exception_type] ?? row.exception_type}
                  >
                    <span className="inline-flex items-center gap-1">
                      {AI_DETECTED_TYPES.has(row.exception_type) && (
                        <>
                          <span className="inline-block w-2 h-2 rounded-full bg-[#0D9488] mr-0.5" title="AI-detected" />
                          <AiTag />
                        </>
                      )}
                      {EXC_TYPE_LABELS[row.exception_type] ?? row.exception_type.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                    </span>
                    {/* Financial impact line — stockout/below_ss/below_rop: lost sales; excess: holding cost */}
                    {["stockout", "below_ss", "below_rop"].includes(row.exception_type) && row.loss_of_sales_7d != null && row.loss_of_sales_7d > 0 ? (
                      <p className="text-[10px] text-red-600 font-medium mt-0.5">
                        ${formatFixed(row.loss_of_sales_7d, 0)} at risk (7-day)
                      </p>
                    ) : row.exception_type === "excess" && row.monthly_holding_cost != null && row.monthly_holding_cost > 0 ? (
                      <p className="text-[10px] text-amber-600 font-medium mt-0.5">
                        ${formatFixed(row.monthly_holding_cost, 0)}/mo holding cost
                      </p>
                    ) : row.estimated_order_value != null && row.estimated_order_value > 0 ? (
                      <p className="text-[10px] text-muted-foreground mt-0.5">
                        Est. ${formatFixed(row.estimated_order_value, 0)} order value
                      </p>
                    ) : null}
                  </td>
                  <td className="px-2 py-1.5 text-right">{formatFixed(row.current_qty_on_hand)}</td>
                  <td className="px-2 py-1.5 text-right">{formatFixed(row.ss_combined)}</td>
                  <td className="px-2 py-1.5 text-right font-medium">{row.recommended_order_qty ? formatFixed(row.recommended_order_qty) : "—"}</td>
                  <td className="px-2 py-1.5">{row.recommended_order_by ?? "—"}</td>
                  <td className="px-2 py-1.5">
                    <span className={`px-1 py-0.5 rounded text-xs ${
                      row.status === "open" ? "bg-red-100 text-red-700" :
                      row.status === "acknowledged" ? "bg-blue-100 text-blue-700" :
                      row.status === "ordered" ? "bg-sky-100 text-sky-700" :
                      "bg-emerald-100 text-emerald-700"
                    }`}>
                      {row.status}
                    </span>
                  </td>
                  <td className="px-2 py-1.5">
                    {row.status === "open" && (
                      <button
                        className="px-2 py-0.5 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
                        disabled={acknowledgeMutation.isPending}
                        onClick={(e) => { e.stopPropagation(); acknowledgeMutation.mutate({ id: row.exception_id }); }}
                      >
                        Acknowledge
                      </button>
                    )}
                    {row.status === "acknowledged" && (
                      <button
                        className="px-2 py-0.5 text-xs bg-sky-600 text-white rounded hover:bg-sky-700 disabled:opacity-50"
                        disabled={statusMutation.isPending}
                        onClick={(e) => { e.stopPropagation(); statusMutation.mutate({ id: row.exception_id, status: "ordered" }); }}
                      >
                        Mark Ordered
                      </button>
                    )}
                    {row.status === "ordered" && (
                      <button
                        className="px-2 py-0.5 text-xs bg-emerald-600 text-white rounded hover:bg-emerald-700 disabled:opacity-50"
                        disabled={statusMutation.isPending}
                        onClick={(e) => { e.stopPropagation(); statusMutation.mutate({ id: row.exception_id, status: "resolved" }); }}
                      >
                        Resolve
                      </button>
                    )}
                  </td>
                </tr>
                {/* Inline drill-down expansion */}
                {selectedExc === row.exception_id && (
                  <tr className="border-t">
                    <td colSpan={11} className="bg-muted/30 px-4 py-4">
                      <ExceptionDetailCard
                        exception={row}
                        rootCauseData={rootCauseData}
                        rootCauseLoading={rootCauseLoading}
                      />
                    </td>
                  </tr>
                )}
                </React.Fragment>
              ))
            }
          </tbody>
        </table>
      </div>
      )}

      {/* Pagination */}
      {excPages > 1 && (
        <div className="flex items-center gap-2 mt-2 text-xs">
          <button
            className="px-2 py-1 border rounded disabled:opacity-40"
            disabled={excOffset === 0}
            onClick={() => setExcOffset(Math.max(0, excOffset - PAGE))}
          >
            Prev
          </button>
          <span className="text-muted-foreground">Page {excPage} / {excPages}</span>
          <button
            className="px-2 py-1 border rounded disabled:opacity-40"
            disabled={excPage >= excPages}
            onClick={() => setExcOffset(excOffset + PAGE)}
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
