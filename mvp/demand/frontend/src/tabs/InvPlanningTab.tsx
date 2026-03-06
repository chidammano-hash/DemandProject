/**
 * IPfeature4 + IPfeature5 + IPfeature6 + IPfeature7 + IPfeature8–IPfeature14
 * EOQ & Cycle Stock + Replenishment Policy + Health Score + Exception Queue +
 * Fill Rate + ABC-XYZ + Supplier + Intramonth + Safety Stock + Variability +
 * Lead Time + Demand Signals + Simulation + Investment Plan
 *
 * Inventory Planning tab: Exception Queue, Portfolio Health panel, EOQ KPI cards,
 * sensitivity line chart, paginated EOQ detail table, and Policy Management panel.
 */
import { useState } from "react";
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
  PieChart,
  Pie,
  Cell,
} from "recharts";
import {
  queryKeys,
  fetchEoqSummary,
  fetchEoqDetail,
  fetchEoqSensitivity,
  fetchPolicies,
  fetchPolicyCompliance,
  assignPolicy,
  updatePolicy,
  healthKeys,
  fetchHealthSummary,
  fetchHealthDetail,
  fetchHealthHeatmap,
  exceptionKeys,
  fetchExceptions,
  fetchExceptionSummary,
  acknowledgeException,
  updateExceptionStatus,
  generateExceptions,
  fillRateKeys,
  fetchFillRateSummary,
  fetchFillRateTrend,
  abcXyzKeys,
  fetchAbcXyzMatrix,
  fetchAbcXyzSummary,
  fetchAbcXyzDetail,
  supplierKeys,
  fetchSupplierSummary,
  fetchSupplierDetail,
  intramonthKeys,
  fetchIntramonthSummary,
  fetchIntramonthDetail,
  safetyStockKeys,
  fetchSafetyStockSummary,
  fetchSafetyStockDetail,
  fetchVariabilitySummary,
  fetchVariabilityDetail,
  fetchLtSummary,
  fetchLtProfile,
  demandSignalsKeys,
  fetchDemandSignalsSummary,
  fetchDemandSignals,
  simulationKeys,
  fetchSimulationResults,
  runSimulation,
  investmentKeys,
  fetchInvestmentSummary,
  fetchInvestmentDetail,
  fetchInvestmentFrontier,
  runInvestmentPlan,
  STALE,
  type EoqDetailRow,
  type ReplenishmentPolicy,
  type HealthDetailRow,
  type ExceptionRow,
  type ExceptionListParams,
  type AbcXyzCell,
  type SupplierRow,
  type IntramonthStockoutRow,
  type SafetyStockRow,
  type VariabilityDetailRow,
  type LtProfileRow,
  type DemandSignalRow,
  type SimulationResult,
  type InvestmentRow,
  type FrontierPoint,
} from "@/api/queries";

const PAGE = 50;

const POLICY_TYPE_COLORS: Record<string, string> = {
  continuous_rop:  "bg-blue-100 text-blue-800",
  periodic_review: "bg-violet-100 text-violet-800",
  min_max:         "bg-emerald-100 text-emerald-800",
  manual:          "bg-amber-100 text-amber-800",
};

function fmt(n: number | null | undefined, decimals = 1): string {
  if (n == null) return "—";
  return Number(n).toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function fmtInt(n: number | null | undefined): string {
  if (n == null) return "—";
  return Math.round(Number(n)).toLocaleString();
}

function fmtPct(n: number | null | undefined): string {
  if (n == null) return "—";
  return `${Number(n).toFixed(1)}%`;
}

type EditPolicyState = {
  policy: ReplenishmentPolicy;
  service_level: string;
  review_cycle_days: string;
};

export function InvPlanningTab() {
  const queryClient = useQueryClient();
  const [abcFilter, setAbcFilter] = useState("");
  const [itemFilter, setItemFilter] = useState("");
  const [locFilter, setLocFilter] = useState("");
  const [eoqOffset, setEoqOffset] = useState(0);
  const [sensitivityItem, setSensitivityItem] = useState("");
  const [sensitivityLoc, setSensitivityLoc] = useState("");

  // Policy Management state
  const [editPolicy, setEditPolicy] = useState<EditPolicyState | null>(null);
  const [autoAssignStatus, setAutoAssignStatus] = useState<string | null>(null);

  // EOQ Summary
  const { data: eoqSummary, isLoading: summaryLoading } = useQuery({
    queryKey: queryKeys.eoqSummary({ abc_vol: abcFilter }),
    queryFn: () => fetchEoqSummary({ abc_vol: abcFilter || undefined }),
    staleTime: STALE.FIVE_MIN,
  });

  // EOQ Detail
  const { data: eoqDetail, isLoading: detailLoading } = useQuery({
    queryKey: queryKeys.eoqDetail({
      item: itemFilter,
      loc: locFilter,
      abc_vol: abcFilter,
      offset: eoqOffset,
    }),
    queryFn: () =>
      fetchEoqDetail({
        item: itemFilter || undefined,
        loc: locFilter || undefined,
        abc_vol: abcFilter || undefined,
        limit: PAGE,
        offset: eoqOffset,
      }),
    staleTime: STALE.FIVE_MIN,
  });

  // EOQ Sensitivity curve
  const { data: sensitivity, isLoading: sensLoading } = useQuery({
    queryKey: queryKeys.eoqSensitivity({
      item: sensitivityItem,
      loc: sensitivityLoc,
    }),
    queryFn: () =>
      fetchEoqSensitivity({
        item: sensitivityItem || undefined,
        loc: sensitivityLoc || undefined,
      }),
    staleTime: STALE.TEN_MIN,
  });

  // Policy Management
  const { data: policyList, isLoading: policyLoading } = useQuery({
    queryKey: queryKeys.policyList(),
    queryFn: fetchPolicies,
    staleTime: STALE.FIVE_MIN,
  });

  const { data: compliance, isLoading: complianceLoading, refetch: refetchCompliance } = useQuery({
    queryKey: queryKeys.policyCompliance(),
    queryFn: fetchPolicyCompliance,
    staleTime: STALE.FIVE_MIN,
  });

  const updatePolicyMutation = useMutation({
    mutationFn: ({ policyId, body }: { policyId: string; body: Parameters<typeof updatePolicy>[1] }) =>
      updatePolicy(policyId, body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.policyList() });
      queryClient.invalidateQueries({ queryKey: queryKeys.policyCompliance() });
      setEditPolicy(null);
    },
  });

  const autoAssignMutation = useMutation({
    mutationFn: async (policies: ReplenishmentPolicy[]) => {
      const results = await Promise.all(
        policies.map((p) => assignPolicy({ segment: p.segment ?? undefined, policy_id: p.policy_id }))
      );
      return results.reduce(
        (acc, r) => ({ assigned: acc.assigned + r.assigned_count, failed: acc.failed + r.failed_count }),
        { assigned: 0, failed: 0 },
      );
    },
    onSuccess: (result) => {
      setAutoAssignStatus(`Assigned ${result.assigned} DFUs${result.failed ? `, ${result.failed} failed` : ""}`);
      queryClient.invalidateQueries({ queryKey: queryKeys.policyCompliance() });
      refetchCompliance();
    },
    onError: () => {
      setAutoAssignStatus("Auto-assign failed. Check auth settings.");
    },
  });

  // Health Score
  const [healthTierFilter, setHealthTierFilter] = useState("");
  const [healthDetailOffset, setHealthDetailOffset] = useState(0);

  const { data: healthSummary, isLoading: healthSummaryLoading } = useQuery({
    queryKey: healthKeys.summary(),
    queryFn: () => fetchHealthSummary(),
    staleTime: STALE.FIVE_MIN,
  });

  const { data: healthDetail, isLoading: healthDetailLoading } = useQuery({
    queryKey: healthKeys.detail({ health_tier: healthTierFilter || undefined, limit: PAGE, offset: healthDetailOffset }),
    queryFn: () =>
      fetchHealthDetail({
        health_tier: healthTierFilter || undefined,
        limit: PAGE,
        offset: healthDetailOffset,
        sort_by: "health_score",
        sort_dir: "asc",
      }),
    staleTime: STALE.FIVE_MIN,
  });

  const { data: healthHeatmap } = useQuery({
    queryKey: healthKeys.heatmap("abc_vol", "variability_class"),
    queryFn: () => fetchHealthHeatmap("abc_vol", "variability_class"),
    staleTime: STALE.TEN_MIN,
  });

  // Exception Queue (IPfeature7)
  const [excTypeFilter, setExcTypeFilter] = useState("");
  const [excSeverityFilter, setExcSeverityFilter] = useState("");
  const [excStatusFilter, setExcStatusFilter] = useState("open");
  const [excItem, setExcItem] = useState("");
  const [excLoc, setExcLoc] = useState("");
  const [excOffset, setExcOffset] = useState(0);
  const [generateStatus, setGenerateStatus] = useState("");

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

  const { data: excList, isLoading: excLoading, refetch: refetchExc } = useQuery({
    queryKey: exceptionKeys.list(excParams),
    queryFn: () => fetchExceptions(excParams),
    staleTime: STALE.ONE_MIN,
  });

  const acknowledgeMutation = useMutation({
    mutationFn: ({ id }: { id: string }) =>
      acknowledgeException(id, "planner", undefined),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: exceptionKeys.list() });
      queryClient.invalidateQueries({ queryKey: exceptionKeys.summary() });
    },
  });

  const statusMutation = useMutation({
    mutationFn: ({ id, status }: { id: string; status: "ordered" | "resolved" }) =>
      updateExceptionStatus(id, status),
    onSuccess: () => {
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

  const SEVERITY_BADGE: Record<string, string> = {
    critical: "bg-red-100 text-red-800",
    high:     "bg-amber-100 text-amber-800",
    medium:   "bg-yellow-100 text-yellow-800",
    low:      "bg-neutral-100 text-neutral-600",
  };

  const SEVERITY_ROW_BG: Record<string, string> = {
    critical: "bg-red-50",
    high:     "bg-amber-50",
    medium:   "bg-yellow-50",
    low:      "",
  };

  const EXC_TYPE_LABELS: Record<string, string> = {
    below_rop:         "Below ROP",
    below_rop_critical: "Below ROP (Critical)",
    below_ss:          "Below SS",
    stockout:          "Stockout",
    excess:            "Excess",
    zero_velocity:     "Zero Velocity",
  };

  const EXC_TYPES = ["below_rop", "below_ss", "stockout", "excess", "zero_velocity"];
  const EXC_SEVERITIES = ["critical", "high", "medium", "low"];

  const totalPages = eoqDetail ? Math.ceil(eoqDetail.total / PAGE) : 0;
  const currentPage = Math.floor(eoqOffset / PAGE) + 1;

  const healthDetailPages = healthDetail ? Math.ceil(healthDetail.total / PAGE) : 0;
  const healthDetailPage = Math.floor(healthDetailOffset / PAGE) + 1;

  const TIER_COLORS: Record<string, string> = {
    healthy:  "#22c55e",
    monitor:  "#3b82f6",
    at_risk:  "#f59e0b",
    critical: "#ef4444",
  };

  const TIER_LABEL: Record<string, string> = {
    healthy:  "Healthy",
    monitor:  "Monitor",
    at_risk:  "At Risk",
    critical: "Critical",
  };

  const tierPieData = healthSummary
    ? (["healthy", "monitor", "at_risk", "critical"] as const)
        .map((t) => ({ name: TIER_LABEL[t], value: healthSummary.by_tier[t], tier: t }))
        .filter((d) => d.value > 0)
    : [];

  return (
    <div className="flex flex-col gap-6 p-4">
      {/* Header */}
      <div>
        <h2 className="text-xl font-semibold text-foreground">
          Inventory Planning — Health Score, EOQ &amp; Replenishment Policy
        </h2>
        <p className="text-sm text-muted-foreground mt-1">
          Portfolio health scoring, Economic Order Quantity targets, and replenishment policy assignments per item-location.
        </p>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Exception Queue (IPfeature7)                                        */}
      {/* ------------------------------------------------------------------ */}
      <div>
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
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
          {[
            {
              label: "Total Open",
              value: excSummary?.open_count ?? 0,
              color: (excSummary?.open_count ?? 0) > 50 ? "text-red-600" : (excSummary?.open_count ?? 0) > 10 ? "text-amber-600" : "text-foreground",
            },
            {
              label: "Critical",
              value: excSummary?.by_severity.critical ?? 0,
              color: (excSummary?.by_severity.critical ?? 0) > 0 ? "text-red-600" : "text-foreground",
            },
            {
              label: "High",
              value: excSummary?.by_severity.high ?? 0,
              color: "text-amber-600",
            },
            {
              label: "Rec. Order Value",
              value: `$${fmt(excSummary?.total_recommended_order_value ?? 0, 0)}`,
              color: "text-blue-600",
              isStr: true,
            },
          ].map(({ label, value, color, isStr }) => (
            <div key={label} className="border rounded-lg p-3 bg-card">
              <p className="text-xs text-muted-foreground">{label}</p>
              <p className={`text-2xl font-bold mt-1 ${color}`}>
                {isStr ? value : value.toLocaleString()}
              </p>
            </div>
          ))}
        </div>

        {/* Filter bar */}
        <div className="flex flex-wrap gap-2 mb-3">
          {/* Type pills */}
          <div className="flex gap-1 flex-wrap">
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
          <div className="flex gap-1 flex-wrap">
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
          <div className="flex gap-1">
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
        <div className="border rounded-lg overflow-auto">
          <table className="w-full text-xs">
            <thead className="bg-muted/50">
              <tr>
                {["Severity", "Item", "Loc", "Type", "Qty on Hand", "SS Target", "Rec. Order Qty", "Order By", "Status", "Actions"].map((h) => (
                  <th key={h} className="px-2 py-2 text-left font-medium text-muted-foreground whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {excLoading ? (
                <tr><td colSpan={10} className="px-3 py-4 text-center text-muted-foreground">Loading…</td></tr>
              ) : (excList?.rows ?? []).length === 0 ? (
                <tr><td colSpan={10} className="px-3 py-4 text-center text-muted-foreground">No exceptions found</td></tr>
              ) : (
                (excList?.rows ?? []).map((row: ExceptionRow) => (
                  <tr
                    key={row.exception_id}
                    className={`border-t ${SEVERITY_ROW_BG[row.severity] ?? ""} ${
                      row.status !== "open" ? "opacity-60" : ""
                    }`}
                  >
                    <td className="px-2 py-1.5">
                      <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${SEVERITY_BADGE[row.severity] ?? ""}`}>
                        {row.severity}
                      </span>
                    </td>
                    <td className="px-2 py-1.5 font-mono">{row.item_no}</td>
                    <td className="px-2 py-1.5 font-mono">{row.loc}</td>
                    <td className="px-2 py-1.5">{EXC_TYPE_LABELS[row.exception_type] ?? row.exception_type}</td>
                    <td className="px-2 py-1.5 text-right">{fmt(row.current_qty_on_hand)}</td>
                    <td className="px-2 py-1.5 text-right">{fmt(row.ss_combined)}</td>
                    <td className="px-2 py-1.5 text-right font-medium">{row.recommended_order_qty ? fmt(row.recommended_order_qty) : "—"}</td>
                    <td className="px-2 py-1.5">{row.recommended_order_by ?? "—"}</td>
                    <td className="px-2 py-1.5">
                      <span className={`px-1 py-0.5 rounded text-xs ${
                        row.status === "open" ? "bg-red-100 text-red-700" :
                        row.status === "acknowledged" ? "bg-blue-100 text-blue-700" :
                        row.status === "ordered" ? "bg-violet-100 text-violet-700" :
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
                          onClick={() => acknowledgeMutation.mutate({ id: row.exception_id })}
                        >
                          Acknowledge
                        </button>
                      )}
                      {row.status === "acknowledged" && (
                        <button
                          className="px-2 py-0.5 text-xs bg-violet-600 text-white rounded hover:bg-violet-700 disabled:opacity-50"
                          disabled={statusMutation.isPending}
                          onClick={() => statusMutation.mutate({ id: row.exception_id, status: "ordered" })}
                        >
                          Mark Ordered
                        </button>
                      )}
                      {row.status === "ordered" && (
                        <button
                          className="px-2 py-0.5 text-xs bg-emerald-600 text-white rounded hover:bg-emerald-700 disabled:opacity-50"
                          disabled={statusMutation.isPending}
                          onClick={() => statusMutation.mutate({ id: row.exception_id, status: "resolved" })}
                        >
                          Resolve
                        </button>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

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

      {/* ------------------------------------------------------------------ */}
      {/* Portfolio Health section (IPfeature6)                               */}
      {/* ------------------------------------------------------------------ */}
      <div>
        <h3 className="text-base font-semibold text-foreground mb-3">Portfolio Health Score</h3>

        {/* Health KPI cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
          {(["healthy", "monitor", "at_risk", "critical"] as const).map((tier) => {
            const count = healthSummary?.by_tier[tier] ?? 0;
            const total = healthSummary?.total_dfus ?? 0;
            const pct = total > 0 ? ((count / total) * 100).toFixed(1) : "0.0";
            const colors: Record<string, string> = {
              healthy:  "border-green-200 bg-green-50 dark:bg-green-950",
              monitor:  "border-blue-200 bg-blue-50 dark:bg-blue-950",
              at_risk:  "border-amber-200 bg-amber-50 dark:bg-amber-950",
              critical: "border-red-200 bg-red-50 dark:bg-red-950",
            };
            const textColors: Record<string, string> = {
              healthy:  "text-green-700 dark:text-green-300",
              monitor:  "text-blue-700 dark:text-blue-300",
              at_risk:  "text-amber-700 dark:text-amber-300",
              critical: "text-red-700 dark:text-red-300",
            };
            return (
              <button
                key={tier}
                className={`rounded-lg border p-3 text-left cursor-pointer transition-opacity ${colors[tier]} ${healthTierFilter === tier ? "ring-2 ring-offset-1 ring-current opacity-100" : "opacity-90 hover:opacity-100"}`}
                onClick={() => {
                  setHealthTierFilter(healthTierFilter === tier ? "" : tier);
                  setHealthDetailOffset(0);
                }}
              >
                <p className={`text-xs font-medium uppercase tracking-wide ${textColors[tier]}`}>
                  {TIER_LABEL[tier]}
                </p>
                <p className={`text-2xl font-bold mt-1 ${textColors[tier]}`}>
                  {healthSummaryLoading ? "…" : count.toLocaleString()}
                </p>
                <p className={`text-xs mt-0.5 ${textColors[tier]} opacity-80`}>{pct}% of portfolio</p>
              </button>
            );
          })}
        </div>

        {/* Health donut + component scores */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div className="rounded-lg border bg-card p-4">
            <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
              Health Distribution
            </h4>
            {healthSummaryLoading ? (
              <div className="text-xs text-muted-foreground">Loading…</div>
            ) : tierPieData.length > 0 ? (
              <div className="flex items-center gap-4">
                <div className="h-40 w-40 shrink-0">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={tierPieData}
                        cx="50%"
                        cy="50%"
                        innerRadius={40}
                        outerRadius={64}
                        paddingAngle={2}
                        dataKey="value"
                      >
                        {tierPieData.map((entry) => (
                          <Cell key={entry.tier} fill={TIER_COLORS[entry.tier]} />
                        ))}
                      </Pie>
                      <Tooltip formatter={(v: number) => [v.toLocaleString(), "DFUs"]} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="flex flex-col gap-1.5 text-xs">
                  {tierPieData.map((d) => (
                    <div key={d.tier} className="flex items-center gap-1.5">
                      <span
                        className="w-2.5 h-2.5 rounded-sm shrink-0"
                        style={{ background: TIER_COLORS[d.tier] }}
                      />
                      <span className="text-foreground">{d.name}</span>
                      <span className="text-muted-foreground ml-auto pl-4">{d.value.toLocaleString()}</span>
                    </div>
                  ))}
                  <div className="mt-1 pt-1 border-t text-muted-foreground">
                    Avg score: <span className="font-semibold text-foreground">{healthSummary?.avg_health_score?.toFixed(1) ?? "—"}</span>
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">No health score data. Run health-schema + health-refresh.</p>
            )}
          </div>

          <div className="rounded-lg border bg-card p-4">
            <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
              Score Components (avg pts / 25)
            </h4>
            {healthSummaryLoading ? (
              <div className="text-xs text-muted-foreground">Loading…</div>
            ) : healthSummary?.component_avgs ? (
              <div className="flex flex-col gap-2">
                {(
                  [
                    ["SS Coverage",       healthSummary.component_avgs.ss_coverage],
                    ["DOS Target",        healthSummary.component_avgs.dos_target],
                    ["Stockout Risk",     healthSummary.component_avgs.stockout_risk],
                    ["Forecast Accuracy", healthSummary.component_avgs.forecast_accuracy],
                  ] as [string, number | null][]
                ).map(([label, val]) => {
                  const pct = val != null ? Math.min(100, (val / 25) * 100) : 0;
                  return (
                    <div key={label}>
                      <div className="flex justify-between text-xs mb-0.5">
                        <span className="text-muted-foreground">{label}</span>
                        <span className="font-medium text-foreground">{val?.toFixed(1) ?? "—"}</span>
                      </div>
                      <div className="h-1.5 rounded-full bg-muted overflow-hidden">
                        <div
                          className="h-full rounded-full bg-blue-500"
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">No data.</p>
            )}
          </div>
        </div>

        {/* Heatmap: ABC × Variability */}
        {healthHeatmap && healthHeatmap.cells.length > 0 && (
          <div className="rounded-lg border bg-card p-4 mb-4">
            <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
              Avg Health Score by ABC Class × Variability
            </h4>
            <div className="overflow-x-auto">
              <table className="text-xs w-full">
                <thead>
                  <tr className="border-b text-muted-foreground">
                    <th className="text-left py-1 pr-3">Variability \ ABC</th>
                    {healthHeatmap.x_labels.map((x) => (
                      <th key={x} className="text-center py-1 px-2">{x}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {healthHeatmap.y_labels.map((y) => (
                    <tr key={y} className="border-b last:border-0">
                      <td className="py-1 pr-3 font-medium">{y}</td>
                      {healthHeatmap.x_labels.map((x) => {
                        const cell = healthHeatmap.cells.find((c) => c.x === x && c.y === y);
                        const score = cell?.avg_health_score;
                        const bg =
                          score == null ? ""
                          : score >= 80 ? "bg-green-100 text-green-800"
                          : score >= 60 ? "bg-blue-100 text-blue-800"
                          : score >= 40 ? "bg-amber-100 text-amber-800"
                          : "bg-red-100 text-red-800";
                        return (
                          <td key={x} className={`text-center py-1 px-2 rounded font-medium ${bg}`}>
                            {score != null ? score.toFixed(0) : "—"}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Health detail table */}
        <div className="rounded-lg border bg-card p-4">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-semibold text-foreground">
              Health Detail
              {healthTierFilter && (
                <span className={`ml-2 px-1.5 py-0.5 rounded text-xs font-medium ${
                  healthTierFilter === "healthy"  ? "bg-green-100 text-green-800" :
                  healthTierFilter === "monitor"  ? "bg-blue-100 text-blue-800" :
                  healthTierFilter === "at_risk"  ? "bg-amber-100 text-amber-800" :
                  "bg-red-100 text-red-800"
                }`}>
                  {TIER_LABEL[healthTierFilter]}
                </span>
              )}
            </h4>
            {healthTierFilter && (
              <button
                className="text-xs text-muted-foreground underline"
                onClick={() => { setHealthTierFilter(""); setHealthDetailOffset(0); }}
              >
                Clear filter
              </button>
            )}
          </div>
          {healthDetailLoading ? (
            <div className="text-xs text-muted-foreground">Loading…</div>
          ) : (
            <>
              <div className="overflow-x-auto">
                <table className="text-xs w-full">
                  <thead>
                    <tr className="border-b text-muted-foreground">
                      <th className="text-left py-1 pr-3">Item</th>
                      <th className="text-left py-1 pr-3">Location</th>
                      <th className="text-center py-1 pr-3">Score</th>
                      <th className="text-center py-1 pr-3">Tier</th>
                      <th className="text-right py-1 pr-3">SS Cov</th>
                      <th className="text-right py-1 pr-3">DOS Tgt</th>
                      <th className="text-right py-1 pr-3">Stockout</th>
                      <th className="text-right py-1">Fcst Acc</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(healthDetail?.rows ?? []).map((row: HealthDetailRow) => {
                      const tierBg: Record<string, string> = {
                        healthy:  "bg-green-100 text-green-800",
                        monitor:  "bg-blue-100 text-blue-800",
                        at_risk:  "bg-amber-100 text-amber-800",
                        critical: "bg-red-100 text-red-800",
                      };
                      return (
                        <tr key={`${row.item_no}-${row.loc}`} className="border-b last:border-0">
                          <td className="py-1 pr-3 font-mono">{row.item_no}</td>
                          <td className="py-1 pr-3">{row.loc}</td>
                          <td className="text-center py-1 pr-3 font-bold">{row.health_score}</td>
                          <td className="text-center py-1 pr-3">
                            <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${tierBg[row.health_tier] ?? ""}`}>
                              {TIER_LABEL[row.health_tier] ?? row.health_tier}
                            </span>
                          </td>
                          <td className="text-right py-1 pr-3">{row.score_ss_coverage}</td>
                          <td className="text-right py-1 pr-3">{row.score_dos_target}</td>
                          <td className="text-right py-1 pr-3">{row.score_stockout_risk}</td>
                          <td className="text-right py-1">{row.score_forecast_accuracy}</td>
                        </tr>
                      );
                    })}
                    {(healthDetail?.rows ?? []).length === 0 && (
                      <tr>
                        <td colSpan={8} className="text-center py-4 text-muted-foreground">
                          No records. Run <code>make health-schema health-refresh</code> to populate.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
              {healthDetailPages > 1 && (
                <div className="flex items-center gap-2 mt-3 text-xs">
                  <button
                    disabled={healthDetailOffset === 0}
                    onClick={() => setHealthDetailOffset(Math.max(0, healthDetailOffset - PAGE))}
                    className="px-2 py-1 rounded border disabled:opacity-40"
                  >
                    ‹ Prev
                  </button>
                  <span className="text-muted-foreground">
                    Page {healthDetailPage} of {healthDetailPages}
                    {healthDetail && ` · ${healthDetail.total.toLocaleString()} total`}
                  </span>
                  <button
                    disabled={healthDetailPage >= healthDetailPages}
                    onClick={() => setHealthDetailOffset(healthDetailOffset + PAGE)}
                    className="px-2 py-1 rounded border disabled:opacity-40"
                  >
                    Next ›
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="rounded-lg border bg-card p-4">
          <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide">
            Total Cycle Stock
          </p>
          <p className="text-2xl font-bold mt-1 text-foreground">
            {summaryLoading ? "…" : fmtInt(eoqSummary?.total_cycle_stock)}
          </p>
          <p className="text-xs text-muted-foreground mt-1">units (avg EOQ/2)</p>
        </div>
        <div className="rounded-lg border bg-card p-4">
          <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide">
            Avg EOQ Size
          </p>
          <p className="text-2xl font-bold mt-1 text-foreground">
            {summaryLoading ? "…" : fmt(eoqSummary?.avg_effective_eoq, 0)}
          </p>
          <p className="text-xs text-muted-foreground mt-1">units per order</p>
        </div>
        <div className="rounded-lg border bg-card p-4">
          <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide">
            Avg Order Frequency
          </p>
          <p className="text-2xl font-bold mt-1 text-foreground">
            {summaryLoading ? "…" : fmt(eoqSummary?.avg_order_frequency, 1)}
          </p>
          <p className="text-xs text-muted-foreground mt-1">orders / year</p>
        </div>
        <div className="rounded-lg border bg-card p-4">
          <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide">
            Total Annual Cost
          </p>
          <p className="text-2xl font-bold mt-1 text-foreground">
            {summaryLoading
              ? "…"
              : eoqSummary?.total_annual_cost != null
              ? `$${fmtInt(eoqSummary.total_annual_cost)}`
              : "—"}
          </p>
          <p className="text-xs text-muted-foreground mt-1">holding + ordering</p>
        </div>
      </div>

      {/* By-ABC breakdown */}
      {eoqSummary && eoqSummary.by_abc.length > 0 && (
        <div className="rounded-lg border bg-card p-4">
          <h3 className="text-sm font-semibold text-foreground mb-3">By ABC Class</h3>
          <div className="overflow-x-auto">
            <table className="text-xs w-full">
              <thead>
                <tr className="border-b text-muted-foreground">
                  <th className="text-left py-1 pr-4">Class</th>
                  <th className="text-right py-1 pr-4">DFUs</th>
                  <th className="text-right py-1 pr-4">Avg EOQ</th>
                  <th className="text-right py-1 pr-4">Cycle Stock</th>
                  <th className="text-right py-1">Annual Cost</th>
                </tr>
              </thead>
              <tbody>
                {eoqSummary.by_abc.map((row) => (
                  <tr key={row.abc_vol} className="border-b last:border-0">
                    <td className="py-1 pr-4 font-medium">{row.abc_vol}</td>
                    <td className="text-right py-1 pr-4">{row.count.toLocaleString()}</td>
                    <td className="text-right py-1 pr-4">{fmt(row.avg_eoq, 0)}</td>
                    <td className="text-right py-1 pr-4">{fmtInt(row.total_cycle_stock)}</td>
                    <td className="text-right py-1">
                      {row.total_annual_cost != null
                        ? `$${fmtInt(row.total_annual_cost)}`
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Sensitivity Chart */}
      <div className="rounded-lg border bg-card p-4">
        <h3 className="text-sm font-semibold text-foreground mb-3">EOQ Sensitivity</h3>
        <p className="text-xs text-muted-foreground mb-3">
          How EOQ and total annual cost change as ordering cost varies.
          Optionally enter an item + location to use that DFU's demand.
        </p>
        <div className="flex gap-2 mb-4">
          <input
            className="h-8 rounded border border-input bg-background px-2 text-xs w-32"
            placeholder="Item No (optional)"
            value={sensitivityItem}
            onChange={(e) => setSensitivityItem(e.target.value)}
          />
          <input
            className="h-8 rounded border border-input bg-background px-2 text-xs w-32"
            placeholder="Location (optional)"
            value={sensitivityLoc}
            onChange={(e) => setSensitivityLoc(e.target.value)}
          />
        </div>
        {sensLoading ? (
          <div className="text-xs text-muted-foreground">Loading…</div>
        ) : sensitivity && sensitivity.curve.length > 0 ? (
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={sensitivity.curve} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis
                  dataKey="ordering_cost"
                  tickFormatter={(v) => `$${v}`}
                  tick={{ fontSize: 10 }}
                />
                <YAxis yAxisId="left" tick={{ fontSize: 10 }} />
                <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 10 }} />
                <Tooltip
                  formatter={(value: number, name: string) => [
                    name === "Total Cost" ? `$${Number(value).toFixed(2)}` : Number(value).toFixed(1),
                    name,
                  ]}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="effective_eoq"
                  name="Effective EOQ"
                  stroke="hsl(220, 70%, 55%)"
                  dot={false}
                  strokeWidth={2}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="total_annual_cost"
                  name="Total Cost"
                  stroke="hsl(10, 70%, 55%)"
                  dot={false}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="text-xs text-muted-foreground">No data available.</div>
        )}
      </div>

      {/* Policy Management */}
      <div className="rounded-lg border bg-card p-4">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-semibold text-foreground">Policy Management</h3>
            <p className="text-xs text-muted-foreground mt-0.5">
              Replenishment policies by ABC class and demand variability.
            </p>
          </div>
          <div className="flex items-center gap-2">
            {autoAssignStatus && (
              <span className="text-xs text-muted-foreground">{autoAssignStatus}</span>
            )}
            <button
              className="h-7 rounded border border-input bg-background px-3 text-xs font-medium hover:bg-muted disabled:opacity-50"
              disabled={autoAssignMutation.isPending || policyLoading}
              onClick={() => {
                setAutoAssignStatus(null);
                autoAssignMutation.mutate(policyList?.policies ?? []);
              }}
            >
              {autoAssignMutation.isPending ? "Assigning…" : "Auto-assign All"}
            </button>
          </div>
        </div>

        {/* Compliance Gauge */}
        <div className="mb-5">
          {complianceLoading ? (
            <div className="text-xs text-muted-foreground">Loading compliance…</div>
          ) : compliance ? (
            <div className="flex items-center gap-6">
              {/* Ring gauge (SVG) */}
              <div className="relative flex-shrink-0">
                <svg width="80" height="80" viewBox="0 0 80 80">
                  <circle cx="40" cy="40" r="34" fill="none" stroke="hsl(var(--border))" strokeWidth="8" />
                  <circle
                    cx="40" cy="40" r="34"
                    fill="none"
                    stroke={compliance.assignment_pct >= 80 ? "#22c55e" : compliance.assignment_pct >= 50 ? "#f59e0b" : "#ef4444"}
                    strokeWidth="8"
                    strokeDasharray={`${(compliance.assignment_pct / 100) * 213.6} 213.6`}
                    strokeLinecap="round"
                    transform="rotate(-90 40 40)"
                  />
                  <text x="40" y="44" textAnchor="middle" fontSize="13" fontWeight="600" fill="currentColor">
                    {compliance.assignment_pct.toFixed(0)}%
                  </text>
                </svg>
              </div>
              <div className="text-sm text-foreground">
                <p className="font-medium">DFU Coverage</p>
                <p className="text-muted-foreground text-xs mt-0.5">
                  {compliance.assigned_count.toLocaleString()} of {compliance.total_dfus.toLocaleString()} DFUs assigned
                </p>
                {compliance.unassigned_count > 0 && (
                  <p className="text-xs text-amber-600 dark:text-amber-400 mt-0.5">
                    {compliance.unassigned_count.toLocaleString()} DFUs unassigned
                  </p>
                )}
              </div>
            </div>
          ) : null}
        </div>

        {/* Policy Cards */}
        {policyLoading ? (
          <div className="text-xs text-muted-foreground">Loading policies…</div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
            {(policyList?.policies ?? []).map((policy) => (
              <div
                key={policy.policy_id}
                className="rounded-md border bg-background p-3 flex flex-col gap-1.5"
              >
                <div className="flex items-start justify-between gap-1">
                  <span className="text-xs font-semibold text-foreground leading-tight">{policy.policy_name}</span>
                  <button
                    className="text-xs text-muted-foreground hover:text-foreground underline flex-shrink-0"
                    onClick={() =>
                      setEditPolicy({
                        policy,
                        service_level: policy.service_level != null ? String(policy.service_level) : "",
                        review_cycle_days: policy.review_cycle_days != null ? String(policy.review_cycle_days) : "",
                      })
                    }
                  >
                    Edit
                  </button>
                </div>
                <span
                  className={`self-start rounded px-1.5 py-0.5 text-xs font-medium ${POLICY_TYPE_COLORS[policy.policy_type] ?? "bg-gray-100 text-gray-700"}`}
                >
                  {policy.policy_type.replace(/_/g, " ")}
                </span>
                <div className="text-xs text-muted-foreground space-y-0.5">
                  {policy.segment && <p>Segment: <span className="font-medium text-foreground">{policy.segment}</span></p>}
                  {policy.service_level != null && (
                    <p>Service level: <span className="font-medium text-foreground">{(policy.service_level * 100).toFixed(0)}%</span></p>
                  )}
                  {policy.review_cycle_days != null && (
                    <p>Review cycle: <span className="font-medium text-foreground">{policy.review_cycle_days}d</span></p>
                  )}
                  <p>DFUs: <span className="font-medium text-foreground">{policy.dfu_count.toLocaleString()}</span></p>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Compliance Table (by policy) */}
        {compliance && Object.keys(compliance.by_policy).length > 0 && (
          <div>
            <h4 className="text-xs font-semibold text-foreground mb-2">Policy Compliance</h4>
            <div className="overflow-x-auto">
              <table className="text-xs w-full">
                <thead>
                  <tr className="border-b text-muted-foreground">
                    <th className="text-left py-1 pr-4">Policy</th>
                    <th className="text-left py-1 pr-4">Type</th>
                    <th className="text-right py-1 pr-4">DFUs</th>
                    <th className="text-right py-1 pr-4">Below SS%</th>
                    <th className="text-right py-1 pr-4">SS Coverage</th>
                    <th className="text-right py-1">Avg DOS</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(compliance.by_policy)
                    .sort(([, a], [, b]) => (b.below_ss_pct ?? -1) - (a.below_ss_pct ?? -1))
                    .map(([pid, bp]) => (
                      <tr key={pid} className="border-b last:border-0">
                        <td className="py-1 pr-4 font-medium">{bp.policy_name}</td>
                        <td className="py-1 pr-4">
                          <span className={`rounded px-1.5 py-0.5 text-xs ${POLICY_TYPE_COLORS[bp.policy_type] ?? ""}`}>
                            {bp.policy_type.replace(/_/g, " ")}
                          </span>
                        </td>
                        <td className="py-1 pr-4 text-right">{bp.dfu_count.toLocaleString()}</td>
                        <td className="py-1 pr-4 text-right">{fmtPct(bp.below_ss_pct)}</td>
                        <td className="py-1 pr-4 text-right">{fmtPct(bp.avg_ss_coverage)}</td>
                        <td className="py-1 text-right">{fmt(bp.avg_dos, 1)}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* Edit Policy Modal */}
      {editPolicy && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
          <div className="rounded-lg border bg-card p-6 w-80 shadow-xl">
            <h3 className="text-sm font-semibold text-foreground mb-4">Edit Policy</h3>
            <p className="text-xs text-muted-foreground mb-4">{editPolicy.policy.policy_name}</p>
            <div className="flex flex-col gap-3">
              <label className="flex flex-col gap-1">
                <span className="text-xs text-muted-foreground">Service Level (0–1)</span>
                <input
                  className="h-8 rounded border border-input bg-background px-2 text-xs"
                  type="number"
                  step="0.01"
                  min="0"
                  max="1"
                  value={editPolicy.service_level}
                  onChange={(e) => setEditPolicy((s) => s ? { ...s, service_level: e.target.value } : null)}
                />
              </label>
              {editPolicy.policy.policy_type === "periodic_review" && (
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-muted-foreground">Review Cycle (days)</span>
                  <input
                    className="h-8 rounded border border-input bg-background px-2 text-xs"
                    type="number"
                    min="1"
                    value={editPolicy.review_cycle_days}
                    onChange={(e) => setEditPolicy((s) => s ? { ...s, review_cycle_days: e.target.value } : null)}
                  />
                </label>
              )}
            </div>
            <div className="flex gap-2 mt-5">
              <button
                className="flex-1 h-8 rounded bg-primary text-primary-foreground text-xs font-medium hover:bg-primary/90 disabled:opacity-50"
                disabled={updatePolicyMutation.isPending}
                onClick={() => {
                  const body: Parameters<typeof updatePolicy>[1] = {};
                  const sl = parseFloat(editPolicy.service_level);
                  if (!isNaN(sl)) body.service_level = sl;
                  const rcd = parseInt(editPolicy.review_cycle_days, 10);
                  if (!isNaN(rcd)) body.review_cycle_days = rcd;
                  updatePolicyMutation.mutate({ policyId: editPolicy.policy.policy_id, body });
                }}
              >
                {updatePolicyMutation.isPending ? "Saving…" : "Save"}
              </button>
              <button
                className="flex-1 h-8 rounded border border-input bg-background text-xs hover:bg-muted"
                onClick={() => setEditPolicy(null)}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Detail Table */}
      <div className="rounded-lg border bg-card p-4">
        <h3 className="text-sm font-semibold text-foreground mb-3">EOQ Detail</h3>

        {/* Filters */}
        <div className="flex flex-wrap gap-2 mb-3">
          <input
            className="h-8 rounded border border-input bg-background px-2 text-xs w-36"
            placeholder="Filter by item…"
            value={itemFilter}
            onChange={(e) => { setItemFilter(e.target.value); setEoqOffset(0); }}
          />
          <input
            className="h-8 rounded border border-input bg-background px-2 text-xs w-36"
            placeholder="Filter by location…"
            value={locFilter}
            onChange={(e) => { setLocFilter(e.target.value); setEoqOffset(0); }}
          />
          <select
            className="h-8 rounded border border-input bg-background px-2 text-xs"
            value={abcFilter}
            onChange={(e) => { setAbcFilter(e.target.value); setEoqOffset(0); }}
          >
            <option value="">All ABC classes</option>
            <option value="A">A</option>
            <option value="B">B</option>
            <option value="C">C</option>
          </select>
        </div>

        {detailLoading ? (
          <div className="text-xs text-muted-foreground">Loading…</div>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="text-xs w-full">
                <thead>
                  <tr className="border-b text-muted-foreground text-right">
                    <th className="text-left py-1 pr-3">Item</th>
                    <th className="text-left py-1 pr-3">Loc</th>
                    <th className="py-1 pr-3">ABC</th>
                    <th className="py-1 pr-3">EOQ</th>
                    <th className="py-1 pr-3">Eff. EOQ</th>
                    <th className="py-1 pr-3">Cycle Stock</th>
                    <th className="py-1 pr-3">Orders/Yr</th>
                    <th className="py-1">Annual Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {(eoqDetail?.rows ?? []).map((r: EoqDetailRow, i) => (
                    <tr key={`${r.item_no}-${r.loc}-${i}`} className="border-b last:border-0 hover:bg-muted/40">
                      <td className="py-1 pr-3 font-mono text-xs">{r.item_no}</td>
                      <td className="py-1 pr-3 text-muted-foreground">{r.loc}</td>
                      <td className="py-1 pr-3 text-center">{r.abc_vol ?? "—"}</td>
                      <td className="py-1 pr-3 text-right">{fmt(r.eoq, 1)}</td>
                      <td className="py-1 pr-3 text-right font-medium">{fmt(r.effective_eoq, 1)}</td>
                      <td className="py-1 pr-3 text-right">{fmt(r.eoq_cycle_stock, 1)}</td>
                      <td className="py-1 pr-3 text-right">{fmt(r.order_frequency, 2)}</td>
                      <td className="py-1 text-right">
                        {r.total_annual_cost != null ? `$${fmt(r.total_annual_cost, 2)}` : "—"}
                      </td>
                    </tr>
                  ))}
                  {(eoqDetail?.rows ?? []).length === 0 && (
                    <tr>
                      <td colSpan={8} className="py-4 text-center text-muted-foreground">
                        No data.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center gap-3 mt-3 text-xs text-muted-foreground">
                <button
                  className="rounded border px-2 py-1 hover:bg-muted disabled:opacity-40"
                  disabled={currentPage <= 1}
                  onClick={() => setEoqOffset((p) => Math.max(0, p - PAGE))}
                >
                  Prev
                </button>
                <span>
                  Page {currentPage} / {totalPages} ({eoqDetail?.total.toLocaleString()} DFUs)
                </span>
                <button
                  className="rounded border px-2 py-1 hover:bg-muted disabled:opacity-40"
                  disabled={currentPage >= totalPages}
                  onClick={() => setEoqOffset((p) => p + PAGE)}
                >
                  Next
                </button>
              </div>
            )}
          </>
        )}
      </div>

      {/* ================================================================
          IPfeature8: Fill Rate Analytics Panel
          ================================================================ */}
      <div className="rounded-lg border bg-card p-4 space-y-4">
        <h3 className="font-semibold text-base">Fill Rate Analytics</h3>
        <FillRatePanel />
      </div>

      {/* ================================================================
          IPfeature11: ABC-XYZ Classification Panel
          ================================================================ */}
      <div className="rounded-lg border bg-card p-4 space-y-4">
        <h3 className="font-semibold text-base">ABC-XYZ Segmentation</h3>
        <AbcXyzPanel />
      </div>

      {/* ================================================================
          IPfeature12: Supplier Performance Panel
          ================================================================ */}
      <div className="rounded-lg border bg-card p-4 space-y-4">
        <h3 className="font-semibold text-base">Supplier Performance</h3>
        <SupplierPanel />
      </div>

      {/* ================================================================
          IPfeature14: Intra-Month Stockout Detection Panel
          ================================================================ */}
      <div className="rounded-lg border bg-card p-4 space-y-4">
        <h3 className="font-semibold text-base">Intra-Month Stockout Detection</h3>
        <IntramonthPanel />
      </div>

      {/* ================================================================
          IPfeature3: Safety Stock Panel
          ================================================================ */}
      <div className="rounded-lg border bg-card p-4 space-y-4">
        <h3 className="font-semibold text-base">Safety Stock</h3>
        <SafetyStockPanel />
      </div>

      {/* ================================================================
          IPfeature1: Demand Variability Panel
          ================================================================ */}
      <div className="rounded-lg border bg-card p-4 space-y-4">
        <h3 className="font-semibold text-base">Demand Variability</h3>
        <VariabilityPanel />
      </div>

      {/* ================================================================
          IPfeature2: Lead Time Panel
          ================================================================ */}
      <div className="rounded-lg border bg-card p-4 space-y-4">
        <h3 className="font-semibold text-base">Lead Time Analysis</h3>
        <LeadTimePanel />
      </div>

      {/* ================================================================
          IPfeature9: Demand Signals Panel
          ================================================================ */}
      <div className="rounded-lg border bg-card p-4 space-y-4">
        <h3 className="font-semibold text-base">Demand Signals</h3>
        <DemandSignalsPanel />
      </div>

      {/* ================================================================
          IPfeature10: Simulation Panel
          ================================================================ */}
      <div className="rounded-lg border bg-card p-4 space-y-4">
        <h3 className="font-semibold text-base">Safety Stock Simulation</h3>
        <SimulationPanel />
      </div>

      {/* ================================================================
          IPfeature13: Investment Plan Panel
          ================================================================ */}
      <div className="rounded-lg border bg-card p-4 space-y-4">
        <h3 className="font-semibold text-base">Investment Plan</h3>
        <InvestmentPanel />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// IPfeature8: Fill Rate Analytics sub-component
// ---------------------------------------------------------------------------
function FillRatePanel() {
  const { data: summary, isLoading } = useQuery({
    queryKey: fillRateKeys.summary(),
    queryFn: () => fetchFillRateSummary(),
    staleTime: STALE,
  });
  const { data: trendData } = useQuery({
    queryKey: fillRateKeys.trend(),
    queryFn: () => fetchFillRateTrend(),
    staleTime: STALE,
  });

  const pct = (n: number | null | undefined) =>
    n == null ? "—" : `${(Number(n) * 100).toFixed(1)}%`;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Portfolio Fill Rate</p>
          <p className="text-xl font-bold">{pct(summary?.portfolio_fill_rate)}</p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Total Ordered</p>
          <p className="text-xl font-bold">{Math.round(Number(summary?.total_ordered ?? 0)).toLocaleString()}</p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Total Shortage</p>
          <p className="text-xl font-bold text-red-600">{Math.round(Number(summary?.total_shortage_qty ?? 0)).toLocaleString()}</p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Partial Fulfillment Events</p>
          <p className="text-xl font-bold">{(summary?.partial_fulfillment_events ?? 0).toLocaleString()}</p>
        </div>
      </div>
      {isLoading && <p className="text-xs text-muted-foreground">Loading fill rate data...</p>}
      {trendData?.months && trendData.months.length > 0 && (
        <div>
          <p className="text-xs font-medium mb-2">Monthly Fill Rate Trend</p>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={trendData.months}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month_start" tick={{ fontSize: 10 }} />
              <YAxis tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} domain={[0, 1]} tick={{ fontSize: 10 }} />
              <Tooltip formatter={(v: number) => `${(v * 100).toFixed(1)}%`} />
              <Line type="monotone" dataKey="fill_rate" stroke="#3b82f6" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// IPfeature11: ABC-XYZ Classification sub-component
// ---------------------------------------------------------------------------
function AbcXyzPanel() {
  const { data: matrix } = useQuery({
    queryKey: abcXyzKeys.matrix(),
    queryFn: fetchAbcXyzMatrix,
    staleTime: STALE,
  });
  const { data: summary } = useQuery({
    queryKey: abcXyzKeys.summary(),
    queryFn: fetchAbcXyzSummary,
    staleTime: STALE,
  });

  const ABC = ["A", "B", "C"];
  const XYZ = ["X", "Y", "Z"];
  const cellMap = new Map((matrix?.cells ?? []).map((c: AbcXyzCell) => [c.segment, c]));

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-3">
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Total DFUs</p>
          <p className="text-xl font-bold">{Number(summary?.total_dfus ?? 0).toLocaleString()}</p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Classified</p>
          <p className="text-xl font-bold">{(matrix?.total_classified ?? 0).toLocaleString()}</p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Z-Class (High Variability)</p>
          <p className="text-xl font-bold text-amber-600">{Number(summary?.z_count ?? 0).toLocaleString()}</p>
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b">
              <th className="text-left py-1 pr-2">ABC╲XYZ</th>
              {XYZ.map(x => <th key={x} className="text-center py-1 px-2">{x}</th>)}
            </tr>
          </thead>
          <tbody>
            {ABC.map(a => (
              <tr key={a} className="border-b">
                <td className="py-1 pr-2 font-medium">{a}</td>
                {XYZ.map(x => {
                  const seg = a + x;
                  const cell = cellMap.get(seg);
                  return (
                    <td key={x} className="text-center py-1 px-2">
                      {cell ? (
                        <div className="rounded bg-blue-50 dark:bg-blue-900/20 px-2 py-1">
                          <span className="font-semibold">{cell.dfu_count}</span>
                          <br />
                          <span className="text-muted-foreground">SL {((cell.avg_service_level ?? 0) * 100).toFixed(0)}%</span>
                        </div>
                      ) : <span className="text-muted-foreground">—</span>}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// IPfeature12: Supplier Performance sub-component
// ---------------------------------------------------------------------------
function SupplierPanel() {
  const { data: summary } = useQuery({
    queryKey: supplierKeys.summary(),
    queryFn: fetchSupplierSummary,
    staleTime: STALE,
  });
  const { data: detail } = useQuery({
    queryKey: supplierKeys.detail({ limit: 10 }),
    queryFn: () => fetchSupplierDetail({ limit: 10, sort_by: "supplier_reliability_score", sort_dir: "asc" }),
    staleTime: STALE,
  });

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Total Suppliers</p>
          <p className="text-xl font-bold">{Number(summary?.total_suppliers ?? 0).toLocaleString()}</p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Avg Reliability Score</p>
          <p className="text-xl font-bold">{Number(summary?.avg_reliability_score ?? 0).toFixed(0)}/100</p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Avg Lead Time (days)</p>
          <p className="text-xl font-bold">{Number(summary?.avg_lead_time_days ?? 0).toFixed(1)}</p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Low Reliability (&lt;40)</p>
          <p className="text-xl font-bold text-red-600">{Number(summary?.low_reliability_count ?? 0).toLocaleString()}</p>
        </div>
      </div>
      {detail && detail.rows.length > 0 && (
        <div className="overflow-x-auto">
          <p className="text-xs font-medium mb-2">Suppliers by Reliability (lowest first)</p>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b">
                <th className="text-left py-1 pr-2">Supplier</th>
                <th className="text-right py-1 px-2">Score</th>
                <th className="text-right py-1 px-2">SKU-Locs</th>
                <th className="text-right py-1 px-2">Avg LT (d)</th>
                <th className="text-right py-1 px-2">LT CV</th>
                <th className="text-right py-1 px-2">% Stable</th>
              </tr>
            </thead>
            <tbody>
              {detail.rows.map((r: SupplierRow) => (
                <tr key={r.supplier_no} className="border-b hover:bg-muted/30">
                  <td className="py-1 pr-2 font-medium">{r.supplier_name ?? r.supplier_no}</td>
                  <td className={`text-right py-1 px-2 font-semibold ${(r.supplier_reliability_score ?? 100) < 40 ? "text-red-600" : (r.supplier_reliability_score ?? 100) < 70 ? "text-amber-600" : "text-green-600"}`}>
                    {r.supplier_reliability_score ?? "—"}
                  </td>
                  <td className="text-right py-1 px-2">{r.sku_loc_count?.toLocaleString() ?? "—"}</td>
                  <td className="text-right py-1 px-2">{r.avg_lt_mean_days?.toFixed(1) ?? "—"}</td>
                  <td className="text-right py-1 px-2">{r.avg_lt_cv?.toFixed(2) ?? "—"}</td>
                  <td className="text-right py-1 px-2">{r.pct_stable_lt != null ? `${(r.pct_stable_lt * 100).toFixed(0)}%` : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// IPfeature14: Intra-Month Stockout Detection sub-component
// ---------------------------------------------------------------------------
function IntramonthPanel() {
  const { data: summary } = useQuery({
    queryKey: intramonthKeys.summary(),
    queryFn: () => fetchIntramonthSummary(),
    staleTime: STALE,
  });
  const { data: detail } = useQuery({
    queryKey: intramonthKeys.detail({ limit: 10, had_stockout: true }),
    queryFn: () => fetchIntramonthDetail({ limit: 10, had_stockout: "true", sort_by: "stockout_day_rate", sort_dir: "desc" }),
    staleTime: STALE,
  });

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Items with Stockout</p>
          <p className="text-xl font-bold text-red-600">{Number(summary?.items_with_stockout ?? 0).toLocaleString()}</p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Extended Stockouts (7d+)</p>
          <p className="text-xl font-bold text-red-700">{Number(summary?.items_with_extended_stockout ?? 0).toLocaleString()}</p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Total Stockout Days</p>
          <p className="text-xl font-bold">{Number(summary?.total_stockout_days ?? 0).toLocaleString()}</p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Est. Lost Sales</p>
          <p className="text-xl font-bold">{Number(summary?.total_est_lost_sales ?? 0).toFixed(0)}</p>
        </div>
      </div>
      {detail && detail.rows.length > 0 && (
        <div className="overflow-x-auto">
          <p className="text-xs font-medium mb-2">Top Stockout Items (current period)</p>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b">
                <th className="text-left py-1 pr-2">Item</th>
                <th className="text-left py-1 px-2">Loc</th>
                <th className="text-right py-1 px-2">Stockout Days</th>
                <th className="text-right py-1 px-2">Day Rate</th>
                <th className="text-right py-1 px-2">Est. Lost Sales</th>
                <th className="text-center py-1 px-2">Extended?</th>
              </tr>
            </thead>
            <tbody>
              {detail.rows.map((r: IntramonthStockoutRow) => (
                <tr key={`${r.item_no}-${r.loc}-${r.month_start}`} className="border-b hover:bg-muted/30">
                  <td className="py-1 pr-2 font-medium">{r.item_no}</td>
                  <td className="py-1 px-2">{r.loc}</td>
                  <td className="text-right py-1 px-2 text-red-600 font-semibold">{r.stockout_days}</td>
                  <td className="text-right py-1 px-2">{r.stockout_day_rate != null ? `${(r.stockout_day_rate * 100).toFixed(0)}%` : "—"}</td>
                  <td className="text-right py-1 px-2">{r.est_lost_sales?.toFixed(0) ?? "—"}</td>
                  <td className="text-center py-1 px-2">{r.had_extended_stockout ? <span className="text-red-600 font-bold">Yes</span> : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// IPfeature3: Safety Stock sub-component
// ---------------------------------------------------------------------------
function SafetyStockPanel() {
  const [belowOnly, setBelowOnly] = useState(false);
  const [ssItemFilter, setSsItemFilter] = useState("");
  const [ssLocFilter, setSsLocFilter] = useState("");
  const [ssOffset, setSsOffset] = useState(0);

  const { data: summary, isLoading: summaryLoading } = useQuery({
    queryKey: safetyStockKeys.summary(),
    queryFn: () => fetchSafetyStockSummary(),
    staleTime: STALE.FIVE_MIN,
  });

  const { data: detail, isLoading: detailLoading } = useQuery({
    queryKey: safetyStockKeys.detail({
      is_below_ss: belowOnly ? true : undefined,
      item: ssItemFilter || undefined,
      loc: ssLocFilter || undefined,
      limit: PAGE,
      offset: ssOffset,
    }),
    queryFn: () =>
      fetchSafetyStockDetail({
        is_below_ss: belowOnly ? true : undefined,
        item: ssItemFilter || undefined,
        loc: ssLocFilter || undefined,
        limit: PAGE,
        offset: ssOffset,
      }),
    staleTime: STALE.FIVE_MIN,
  });

  const totalPages = detail ? Math.ceil(detail.total / PAGE) : 0;
  const currentPage = Math.floor(ssOffset / PAGE) + 1;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Items Below SS</p>
          <p className={`text-xl font-bold ${(summary?.below_ss_count ?? 0) > 0 ? "text-red-600" : "text-foreground"}`}>
            {summaryLoading ? "..." : (summary?.below_ss_count ?? 0).toLocaleString()}
          </p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Avg SS Coverage</p>
          <p className="text-xl font-bold">
            {summaryLoading ? "..." : fmtPct(summary?.avg_ss_coverage != null ? summary.avg_ss_coverage * 100 : null)}
          </p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Total DFUs</p>
          <p className="text-xl font-bold">{summaryLoading ? "..." : (summary?.total_dfus ?? 0).toLocaleString()}</p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Avg SS Days</p>
          <p className="text-xl font-bold">{summaryLoading ? "..." : fmt(summary?.avg_ss_days, 1)}</p>
        </div>
      </div>

      {summary && summary.by_abc.length > 0 && (
        <div className="overflow-x-auto">
          <p className="text-xs font-medium mb-2">Safety Stock by ABC Class</p>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b text-muted-foreground">
                <th className="text-left py-1 pr-3">ABC Class</th>
                <th className="text-right py-1 pr-3">Count</th>
                <th className="text-right py-1 pr-3">Below SS</th>
                <th className="text-right py-1">Avg Coverage</th>
              </tr>
            </thead>
            <tbody>
              {summary.by_abc.map((row) => (
                <tr key={row.abc_vol} className="border-b last:border-0">
                  <td className="py-1 pr-3 font-medium">{row.abc_vol}</td>
                  <td className="py-1 pr-3 text-right">{row.count.toLocaleString()}</td>
                  <td className={`py-1 pr-3 text-right ${row.below_ss_count > 0 ? "text-red-600 font-medium" : ""}`}>
                    {row.below_ss_count.toLocaleString()}
                  </td>
                  <td className="py-1 text-right">
                    {fmtPct(row.avg_coverage != null ? row.avg_coverage * 100 : null)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="flex flex-wrap gap-2">
        <button
          className={`px-3 py-1 text-xs rounded border transition-colors ${belowOnly ? "bg-foreground text-background border-foreground" : "border-border hover:bg-accent"}`}
          onClick={() => { setBelowOnly(!belowOnly); setSsOffset(0); }}
        >
          {belowOnly ? "All Items" : "Below SS Only"}
        </button>
        <input
          className="h-7 rounded border border-input bg-background px-2 text-xs w-32"
          placeholder="Filter by item..."
          value={ssItemFilter}
          onChange={(e) => { setSsItemFilter(e.target.value); setSsOffset(0); }}
        />
        <input
          className="h-7 rounded border border-input bg-background px-2 text-xs w-32"
          placeholder="Filter by location..."
          value={ssLocFilter}
          onChange={(e) => { setSsLocFilter(e.target.value); setSsOffset(0); }}
        />
      </div>

      {detailLoading ? (
        <p className="text-xs text-muted-foreground">Loading...</p>
      ) : (
        <>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b text-muted-foreground">
                  <th className="text-left py-1 pr-2">Item No</th>
                  <th className="text-left py-1 pr-2">Location</th>
                  <th className="text-right py-1 pr-2">SS (qty)</th>
                  <th className="text-right py-1 pr-2">Coverage %</th>
                  <th className="text-center py-1 pr-2">Below SS</th>
                  <th className="text-right py-1 pr-2">Reorder Point</th>
                  <th className="text-center py-1">ABC</th>
                </tr>
              </thead>
              <tbody>
                {(detail?.rows ?? []).length === 0 ? (
                  <tr>
                    <td colSpan={7} className="py-4 text-center text-muted-foreground">
                      No data. Run make health-schema health-refresh to populate.
                    </td>
                  </tr>
                ) : (
                  (detail?.rows ?? []).map((r: SafetyStockRow, i: number) => (
                    <tr
                      key={`${r.item_no}-${r.loc}-${i}`}
                      className={`border-b last:border-0 hover:bg-muted/40 ${r.is_below_ss ? "bg-red-50 dark:bg-red-950/20" : ""}`}
                    >
                      <td className="py-1 pr-2 font-mono">{r.item_no}</td>
                      <td className="py-1 pr-2">{r.loc}</td>
                      <td className="py-1 pr-2 text-right">{fmt(r.ss_combined, 1)}</td>
                      <td className="py-1 pr-2 text-right">
                        {fmtPct(r.ss_coverage != null ? r.ss_coverage * 100 : null)}
                      </td>
                      <td className="py-1 pr-2 text-center">
                        {r.is_below_ss ? (
                          <span className="px-1.5 py-0.5 rounded text-xs bg-red-100 text-red-800 font-medium">Yes</span>
                        ) : (
                          <span className="text-muted-foreground">-</span>
                        )}
                      </td>
                      <td className="py-1 pr-2 text-right">{fmt(r.reorder_point, 1)}</td>
                      <td className="py-1 text-center">{r.abc_vol ?? "-"}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
          {totalPages > 1 && (
            <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
              <button
                className="px-2 py-1 rounded border disabled:opacity-40"
                disabled={ssOffset === 0}
                onClick={() => setSsOffset(Math.max(0, ssOffset - PAGE))}
              >
                Prev
              </button>
              <span>Page {currentPage} / {totalPages}</span>
              <button
                className="px-2 py-1 rounded border disabled:opacity-40"
                disabled={currentPage >= totalPages}
                onClick={() => setSsOffset(ssOffset + PAGE)}
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// IPfeature1: Demand Variability sub-component
// ---------------------------------------------------------------------------
function VariabilityPanel() {
  const { data: summary, isLoading } = useQuery({
    queryKey: queryKeys.variabilitySummary({}),
    queryFn: () => fetchVariabilitySummary({}),
    staleTime: STALE.FIVE_MIN,
  });

  const { data: volatile } = useQuery({
    queryKey: queryKeys.variabilityDetail({ variability_class: "high", limit: 10 }),
    queryFn: () => fetchVariabilityDetail({ variability_class: "high", limit: 10 }),
    staleTime: STALE.FIVE_MIN,
  });

  const pieData = summary
    ? [
        { name: "Stable", value: summary.by_class.low, color: "#22c55e" },
        { name: "Moderate", value: summary.by_class.medium, color: "#f59e0b" },
        {
          name: "Volatile",
          value: (summary.by_class.high ?? 0) + (summary.by_class.lumpy ?? 0),
          color: "#ef4444",
        },
      ].filter((d) => d.value > 0)
    : [];

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-3">
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Stable Items</p>
          <p className="text-xl font-bold text-green-600">
            {isLoading ? "..." : (summary?.by_class.low ?? 0).toLocaleString()}
          </p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Volatile Items</p>
          <p className={`text-xl font-bold ${(summary?.by_class.high ?? 0) > 0 ? "text-red-600" : "text-foreground"}`}>
            {isLoading ? "..." : ((summary?.by_class.high ?? 0) + (summary?.by_class.lumpy ?? 0)).toLocaleString()}
          </p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Avg CV</p>
          <p className="text-xl font-bold">
            {isLoading ? "..." : fmtPct(summary?.avg_cv != null ? summary.avg_cv * 100 : null)}
          </p>
        </div>
      </div>

      {pieData.length > 0 && (
        <div className="flex items-center gap-6">
          <div className="h-36 w-36 shrink-0">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={pieData} cx="50%" cy="50%" innerRadius={36} outerRadius={56} paddingAngle={2} dataKey="value">
                  {pieData.map((entry) => (
                    <Cell key={entry.name} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip formatter={(v: number) => [v.toLocaleString(), "Items"]} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex flex-col gap-1.5 text-xs">
            {pieData.map((d) => (
              <div key={d.name} className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-sm shrink-0" style={{ background: d.color }} />
                <span className="text-foreground">{d.name}</span>
                <span className="text-muted-foreground ml-auto pl-4">{d.value.toLocaleString()}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {volatile && volatile.rows.length > 0 && (
        <div className="overflow-x-auto">
          <p className="text-xs font-medium mb-2">Top Volatile Items</p>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b text-muted-foreground">
                <th className="text-left py-1 pr-2">Item No</th>
                <th className="text-left py-1 pr-2">Location</th>
                <th className="text-right py-1 pr-2">Demand CV</th>
                <th className="text-center py-1">Class</th>
              </tr>
            </thead>
            <tbody>
              {volatile.rows.map((r: VariabilityDetailRow, i: number) => (
                <tr key={`${r.item_no}-${r.loc}-${i}`} className="border-b last:border-0 hover:bg-muted/30">
                  <td className="py-1 pr-2 font-mono">{r.item_no}</td>
                  <td className="py-1 pr-2">{r.loc}</td>
                  <td className="py-1 pr-2 text-right">
                    {r.demand_cv != null ? (r.demand_cv * 100).toFixed(1) + "%" : "-"}
                  </td>
                  <td className="py-1 text-center">
                    <span
                      className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                        r.variability_class === "high" || r.variability_class === "lumpy"
                          ? "bg-red-100 text-red-800"
                          : r.variability_class === "medium"
                          ? "bg-amber-100 text-amber-800"
                          : "bg-green-100 text-green-800"
                      }`}
                    >
                      {r.variability_class ?? "-"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// IPfeature2: Lead Time sub-component
// ---------------------------------------------------------------------------
function LeadTimePanel() {
  const { data: summary, isLoading } = useQuery({
    queryKey: queryKeys.ltSummary({}),
    queryFn: () => fetchLtSummary({}),
    staleTime: STALE.FIVE_MIN,
  });

  const { data: volatile } = useQuery({
    queryKey: queryKeys.ltProfile({ lt_variability_class: "volatile", limit: 10 }),
    queryFn: () => fetchLtProfile({ lt_variability_class: "volatile", limit: 10 }),
    staleTime: STALE.FIVE_MIN,
  });

  const classData = summary
    ? [
        { label: "Stable", count: summary.by_class.stable, color: "text-green-600" },
        { label: "Moderate", count: summary.by_class.moderate, color: "text-amber-600" },
        { label: "Volatile", count: summary.by_class.volatile, color: "text-red-600" },
      ]
    : [];

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-3">
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Avg Lead Time</p>
          <p className="text-xl font-bold">
            {isLoading ? "..." : summary?.avg_lt_mean_days != null ? `${summary.avg_lt_mean_days.toFixed(1)} days` : "-"}
          </p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Volatile Suppliers</p>
          <p className={`text-xl font-bold ${(summary?.by_class.volatile ?? 0) > 0 ? "text-red-600" : "text-foreground"}`}>
            {isLoading ? "..." : (summary?.by_class.volatile ?? 0).toLocaleString()}
          </p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Avg LT CV</p>
          <p className="text-xl font-bold">
            {isLoading ? "..." : fmtPct(summary?.avg_lt_cv != null ? summary.avg_lt_cv * 100 : null)}
          </p>
        </div>
      </div>

      {classData.length > 0 && (
        <div className="flex gap-4 text-xs">
          {classData.map((d) => (
            <div key={d.label} className="rounded-lg border bg-muted/30 px-4 py-2 text-center">
              <p className="text-muted-foreground">{d.label}</p>
              <p className={`text-lg font-bold ${d.color}`}>{d.count.toLocaleString()}</p>
            </div>
          ))}
        </div>
      )}

      {volatile && volatile.rows.length > 0 && (
        <div className="overflow-x-auto">
          <p className="text-xs font-medium mb-2">Top Volatile Lead Time Items</p>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b text-muted-foreground">
                <th className="text-left py-1 pr-2">Item No</th>
                <th className="text-left py-1 pr-2">Location</th>
                <th className="text-right py-1 pr-2">LT Mean (days)</th>
                <th className="text-right py-1 pr-2">LT Std</th>
                <th className="text-right py-1 pr-2">CV</th>
                <th className="text-center py-1">Class</th>
              </tr>
            </thead>
            <tbody>
              {volatile.rows.map((r: LtProfileRow, i: number) => (
                <tr key={`${r.item_no}-${r.loc}-${i}`} className="border-b last:border-0 hover:bg-muted/30">
                  <td className="py-1 pr-2 font-mono">{r.item_no}</td>
                  <td className="py-1 pr-2">{r.loc}</td>
                  <td className="py-1 pr-2 text-right">{r.lt_mean_days?.toFixed(1) ?? "-"}</td>
                  <td className="py-1 pr-2 text-right">{r.lt_std_days?.toFixed(1) ?? "-"}</td>
                  <td className="py-1 pr-2 text-right">
                    {r.lt_cv != null ? (r.lt_cv * 100).toFixed(1) + "%" : "-"}
                  </td>
                  <td className="py-1 text-center">
                    <span
                      className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                        r.lt_variability_class === "volatile"
                          ? "bg-red-100 text-red-800"
                          : r.lt_variability_class === "moderate"
                          ? "bg-amber-100 text-amber-800"
                          : "bg-green-100 text-green-800"
                      }`}
                    >
                      {r.lt_variability_class ?? "-"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// IPfeature9: Demand Signals sub-component
// ---------------------------------------------------------------------------
function DemandSignalsPanel() {
  const [signalTypeFilter, setSignalTypeFilter] = useState("");
  const [alertPriorityFilter, setAlertPriorityFilter] = useState("");
  const [dsItemFilter, setDsItemFilter] = useState("");
  const [dsLocFilter, setDsLocFilter] = useState("");
  const [dsOffset, setDsOffset] = useState(0);

  const { data: summary } = useQuery({
    queryKey: demandSignalsKeys.summary(),
    queryFn: () => fetchDemandSignalsSummary(),
    staleTime: STALE.ONE_MIN,
  });

  const { data: signals, isLoading } = useQuery({
    queryKey: demandSignalsKeys.list({
      signal_type: signalTypeFilter || undefined,
      alert_priority: alertPriorityFilter || undefined,
      item: dsItemFilter || undefined,
      loc: dsLocFilter || undefined,
      limit: PAGE,
      offset: dsOffset,
    }),
    queryFn: () =>
      fetchDemandSignals({
        signal_type: signalTypeFilter || undefined,
        alert_priority: alertPriorityFilter || undefined,
        item: dsItemFilter || undefined,
        loc: dsLocFilter || undefined,
        limit: PAGE,
        offset: dsOffset,
      }),
    staleTime: STALE.ONE_MIN,
  });

  const totalPages = signals ? Math.ceil(signals.total / PAGE) : 0;
  const currentPage = Math.floor(dsOffset / PAGE) + 1;

  const SIGNAL_TYPE_COLORS: Record<string, string> = {
    above_plan: "bg-green-100 text-green-800",
    below_plan: "bg-red-100 text-red-800",
    on_plan: "bg-blue-100 text-blue-800",
  };

  const PRIORITY_COLORS: Record<string, string> = {
    urgent: "bg-red-100 text-red-800",
    watch: "bg-amber-100 text-amber-800",
    normal: "bg-neutral-100 text-neutral-600",
  };

  const ROW_BG: Record<string, string> = {
    urgent: "bg-red-50 dark:bg-red-950/20",
    watch: "bg-yellow-50 dark:bg-yellow-950/20",
    normal: "",
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Above Plan</p>
          <p className="text-xl font-bold text-green-600">{(summary?.above_plan_count ?? 0).toLocaleString()}</p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Below Plan</p>
          <p className="text-xl font-bold text-red-600">{(summary?.below_plan_count ?? 0).toLocaleString()}</p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Urgent Alerts</p>
          <p className={`text-xl font-bold ${(summary?.urgent_count ?? 0) > 0 ? "text-red-600" : "text-foreground"}`}>
            {(summary?.urgent_count ?? 0).toLocaleString()}
          </p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Projected Stockouts</p>
          <p className={`text-xl font-bold ${(summary?.projected_stockouts ?? 0) > 0 ? "text-orange-600" : "text-foreground"}`}>
            {(summary?.projected_stockouts ?? 0).toLocaleString()}
          </p>
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        {["", "above_plan", "below_plan", "on_plan"].map((t) => (
          <button
            key={t || "all"}
            className={`px-2 py-0.5 text-xs rounded-full border transition-colors ${
              signalTypeFilter === t
                ? "bg-foreground text-background border-foreground"
                : "border-border hover:bg-accent"
            }`}
            onClick={() => { setSignalTypeFilter(t); setDsOffset(0); }}
          >
            {t ? t.replace(/_/g, " ").replace(/\b\w/g, (c: string) => c.toUpperCase()) : "All Types"}
          </button>
        ))}
        {["", "urgent", "watch"].map((p) => (
          <button
            key={p || "all-priority"}
            className={`px-2 py-0.5 text-xs rounded-full border transition-colors ${
              alertPriorityFilter === p
                ? "bg-foreground text-background border-foreground"
                : "border-border hover:bg-accent"
            }`}
            onClick={() => { setAlertPriorityFilter(p); setDsOffset(0); }}
          >
            {p ? p.charAt(0).toUpperCase() + p.slice(1) : "All Priority"}
          </button>
        ))}
        <input
          className="h-7 rounded border border-input bg-background px-2 text-xs w-28"
          placeholder="Filter by item..."
          value={dsItemFilter}
          onChange={(e) => { setDsItemFilter(e.target.value); setDsOffset(0); }}
        />
        <input
          className="h-7 rounded border border-input bg-background px-2 text-xs w-28"
          placeholder="Filter by location..."
          value={dsLocFilter}
          onChange={(e) => { setDsLocFilter(e.target.value); setDsOffset(0); }}
        />
      </div>

      {isLoading ? (
        <p className="text-xs text-muted-foreground">Loading...</p>
      ) : (
        <>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b text-muted-foreground">
                  <th className="text-left py-1 pr-2">Item No</th>
                  <th className="text-left py-1 pr-2">Location</th>
                  <th className="text-center py-1 pr-2">Signal Type</th>
                  <th className="text-right py-1 pr-2">Demand vs Fcst %</th>
                  <th className="text-center py-1 pr-2">Alert Priority</th>
                  <th className="text-right py-1 pr-2">On Hand</th>
                  <th className="text-center py-1">Below SS</th>
                </tr>
              </thead>
              <tbody>
                {(signals?.rows ?? []).length === 0 ? (
                  <tr>
                    <td colSpan={7} className="py-4 text-center text-muted-foreground">
                      No demand signals found.
                    </td>
                  </tr>
                ) : (
                  (signals?.rows ?? []).map((r: DemandSignalRow, i: number) => (
                    <tr
                      key={`${r.item_no}-${r.loc}-${i}`}
                      className={`border-b last:border-0 hover:bg-muted/40 ${ROW_BG[r.alert_priority] ?? ""}`}
                    >
                      <td className="py-1 pr-2 font-mono">{r.item_no}</td>
                      <td className="py-1 pr-2">{r.loc}</td>
                      <td className="py-1 pr-2 text-center">
                        <span
                          className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                            SIGNAL_TYPE_COLORS[r.signal_type] ?? "bg-neutral-100 text-neutral-600"
                          }`}
                        >
                          {r.signal_type.replace(/_/g, " ")}
                        </span>
                      </td>
                      <td className="py-1 pr-2 text-right">
                        {r.demand_vs_forecast_pct != null ? `${r.demand_vs_forecast_pct.toFixed(1)}%` : "-"}
                      </td>
                      <td className="py-1 pr-2 text-center">
                        <span
                          className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                            PRIORITY_COLORS[r.alert_priority] ?? ""
                          }`}
                        >
                          {r.alert_priority}
                        </span>
                      </td>
                      <td className="py-1 pr-2 text-right">{fmt(r.current_on_hand, 0)}</td>
                      <td className="py-1 text-center">
                        {r.is_below_ss ? (
                          <span className="px-1 py-0.5 rounded text-xs bg-red-100 text-red-800">Yes</span>
                        ) : (
                          <span className="text-muted-foreground">-</span>
                        )}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
          {totalPages > 1 && (
            <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
              <button
                className="px-2 py-1 rounded border disabled:opacity-40"
                disabled={dsOffset === 0}
                onClick={() => setDsOffset(Math.max(0, dsOffset - PAGE))}
              >
                Prev
              </button>
              <span>Page {currentPage} / {totalPages}</span>
              <button
                className="px-2 py-1 rounded border disabled:opacity-40"
                disabled={currentPage >= totalPages}
                onClick={() => setDsOffset(dsOffset + PAGE)}
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// IPfeature10: Safety Stock Simulation sub-component
// ---------------------------------------------------------------------------
function SimulationPanel() {
  const queryClient = useQueryClient();
  const [simItemNo, setSimItemNo] = useState("");
  const [simLoc, setSimLoc] = useState("");
  const [simResult, setSimResult] = useState<SimulationResult | null>(null);

  const { data: recentRuns, isLoading: runsLoading } = useQuery({
    queryKey: simulationKeys.results({ limit: 10 }),
    queryFn: () => fetchSimulationResults({ limit: 10 }),
    staleTime: STALE.FIVE_MIN,
  });

  const runMutation = useMutation({
    mutationFn: (body: { item_no: string; loc: string }) => runSimulation(body),
    onSuccess: (result) => {
      setSimResult(result);
      queryClient.invalidateQueries({ queryKey: simulationKeys.results() });
    },
  });

  const activeResult = simResult ?? recentRuns?.rows?.[0] ?? null;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2 items-center">
        <input
          className="h-8 rounded border border-input bg-background px-2 text-xs w-36"
          placeholder="Item No"
          value={simItemNo}
          onChange={(e) => setSimItemNo(e.target.value)}
        />
        <input
          className="h-8 rounded border border-input bg-background px-2 text-xs w-36"
          placeholder="Location"
          value={simLoc}
          onChange={(e) => setSimLoc(e.target.value)}
        />
        <button
          className="h-8 px-4 text-xs rounded bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          disabled={runMutation.isPending || !simItemNo || !simLoc}
          onClick={() => runMutation.mutate({ item_no: simItemNo, loc: simLoc })}
        >
          {runMutation.isPending ? "Running..." : "Run Simulation"}
        </button>
      </div>

      {activeResult && (
        <div className="space-y-4">
          <div className="grid grid-cols-3 gap-3">
            <div className="rounded-lg border bg-muted/30 p-3">
              <p className="text-xs text-muted-foreground">Recommended SS</p>
              <p className="text-xl font-bold">{fmt(activeResult.recommended_ss, 0)}</p>
            </div>
            <div className="rounded-lg border bg-muted/30 p-3">
              <p className="text-xs text-muted-foreground">Analytical SS</p>
              <p className="text-xl font-bold">{fmt(activeResult.analytical_ss, 0)}</p>
            </div>
            <div className="rounded-lg border bg-muted/30 p-3">
              <p className="text-xs text-muted-foreground">Difference %</p>
              <p
                className={`text-xl font-bold ${
                  (activeResult.sim_vs_analytical_pct ?? 0) > 0 ? "text-red-600" : "text-green-600"
                }`}
              >
                {activeResult.sim_vs_analytical_pct != null
                  ? `${activeResult.sim_vs_analytical_pct > 0 ? "+" : ""}${activeResult.sim_vs_analytical_pct.toFixed(1)}%`
                  : "-"}
              </p>
            </div>
          </div>

          {activeResult.results_by_ss_level && activeResult.results_by_ss_level.length > 0 && (
            <div>
              <p className="text-xs font-medium mb-2">Service Level Curve</p>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={activeResult.results_by_ss_level}
                    margin={{ top: 4, right: 16, left: 0, bottom: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="ss_qty" tick={{ fontSize: 10 }} />
                    <YAxis
                      domain={[0, 100]}
                      tickFormatter={(v: number) => `${v}%`}
                      tick={{ fontSize: 10 }}
                    />
                    <Tooltip formatter={(v: number) => [`${v.toFixed(1)}%`, "CSL"]} />
                    <Line
                      type="monotone"
                      dataKey="csl"
                      stroke="hsl(220, 70%, 55%)"
                      dot={false}
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      )}

      <div>
        <p className="text-xs font-medium mb-2">Recent Simulation Runs</p>
        {runsLoading ? (
          <p className="text-xs text-muted-foreground">Loading...</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b text-muted-foreground">
                  <th className="text-left py-1 pr-2">Item</th>
                  <th className="text-left py-1 pr-2">Location</th>
                  <th className="text-left py-1 pr-2">Date</th>
                  <th className="text-right py-1 pr-2">Sim SS</th>
                  <th className="text-right py-1 pr-2">Analytical SS</th>
                  <th className="text-right py-1 pr-2">Diff %</th>
                  <th className="text-right py-1">Duration (s)</th>
                </tr>
              </thead>
              <tbody>
                {(recentRuns?.rows ?? []).length === 0 ? (
                  <tr>
                    <td colSpan={7} className="py-4 text-center text-muted-foreground">
                      No simulation runs. Use the form above to run a simulation.
                    </td>
                  </tr>
                ) : (
                  (recentRuns?.rows ?? []).map((r: SimulationResult) => (
                    <tr key={r.sim_run_id} className="border-b last:border-0 hover:bg-muted/30">
                      <td className="py-1 pr-2 font-mono">{r.item_no}</td>
                      <td className="py-1 pr-2">{r.loc}</td>
                      <td className="py-1 pr-2">{r.simulation_date?.slice(0, 10) ?? "-"}</td>
                      <td className="py-1 pr-2 text-right">{fmt(r.recommended_ss, 0)}</td>
                      <td className="py-1 pr-2 text-right">{fmt(r.analytical_ss, 0)}</td>
                      <td
                        className={`py-1 pr-2 text-right ${
                          (r.sim_vs_analytical_pct ?? 0) > 5 ? "text-red-600" : "text-foreground"
                        }`}
                      >
                        {r.sim_vs_analytical_pct != null ? `${r.sim_vs_analytical_pct.toFixed(1)}%` : "-"}
                      </td>
                      <td className="py-1 text-right">{r.run_duration_secs?.toFixed(1) ?? "-"}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// IPfeature13: Investment Plan sub-component
// ---------------------------------------------------------------------------
function InvestmentPanel() {
  const queryClient = useQueryClient();
  const [invOffset, setInvOffset] = useState(0);
  const [runStatus, setRunStatus] = useState("");

  const { data: summary, isLoading: summaryLoading } = useQuery({
    queryKey: investmentKeys.summary(),
    queryFn: () => fetchInvestmentSummary(),
    staleTime: STALE.TEN_MIN,
  });

  const { data: detail, isLoading: detailLoading } = useQuery({
    queryKey: investmentKeys.detail({ limit: 20, offset: invOffset }),
    queryFn: () => fetchInvestmentDetail({ limit: 20, offset: invOffset }),
    staleTime: STALE.TEN_MIN,
  });

  const { data: frontier } = useQuery({
    queryKey: investmentKeys.frontier(),
    queryFn: () => fetchInvestmentFrontier(),
    staleTime: STALE.TEN_MIN,
  });

  const runPlanMutation = useMutation({
    mutationFn: () => runInvestmentPlan(),
    onSuccess: () => {
      setRunStatus("Plan computed successfully.");
      queryClient.invalidateQueries({ queryKey: investmentKeys.summary() });
      queryClient.invalidateQueries({ queryKey: investmentKeys.detail() });
      queryClient.invalidateQueries({ queryKey: investmentKeys.frontier() });
    },
    onError: () => setRunStatus("Failed to compute plan. Check auth settings."),
  });

  const totalPages = detail ? Math.ceil(detail.total / 20) : 0;
  const currentPage = Math.floor(invOffset / 20) + 1;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        {runStatus && <span className="text-xs text-muted-foreground">{runStatus}</span>}
        <button
          className="h-8 px-4 text-xs rounded bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          disabled={runPlanMutation.isPending}
          onClick={() => { setRunStatus(""); runPlanMutation.mutate(); }}
        >
          {runPlanMutation.isPending ? "Computing..." : "Run Plan"}
        </button>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Investment Gap</p>
          <p className="text-xl font-bold text-amber-600">
            {summaryLoading
              ? "..."
              : summary?.total_investment_gap != null
              ? `$${fmtInt(summary.total_investment_gap)}`
              : "-"}
          </p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Current Portfolio CSL</p>
          <p className="text-xl font-bold">
            {summaryLoading
              ? "..."
              : fmtPct(summary?.avg_current_csl != null ? summary.avg_current_csl * 100 : null)}
          </p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Target Portfolio CSL</p>
          <p className="text-xl font-bold text-green-600">
            {summaryLoading
              ? "..."
              : fmtPct(summary?.avg_recommended_csl != null ? summary.avg_recommended_csl * 100 : null)}
          </p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">DFUs Analyzed</p>
          <p className="text-xl font-bold">
            {summaryLoading ? "..." : (summary?.total_items ?? 0).toLocaleString()}
          </p>
        </div>
      </div>

      {frontier && frontier.length > 0 && (
        <div>
          <p className="text-xs font-medium mb-2">Efficient Frontier</p>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={frontier} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis
                  dataKey="cumulative_investment"
                  tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`}
                  tick={{ fontSize: 10 }}
                />
                <YAxis
                  dataKey="achievable_csl"
                  tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                  domain={[0, 1]}
                  tick={{ fontSize: 10 }}
                />
                <Tooltip
                  formatter={(v: number, name: string) =>
                    name === "achievable_csl"
                      ? [`${(v * 100).toFixed(1)}%`, "Achievable CSL"]
                      : [`$${v.toLocaleString()}`, name]
                  }
                />
                <Line
                  type="monotone"
                  dataKey="achievable_csl"
                  stroke="hsl(142, 70%, 45%)"
                  dot={false}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {detailLoading ? (
        <p className="text-xs text-muted-foreground">Loading...</p>
      ) : (
        <>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b text-muted-foreground">
                  <th className="text-right py-1 pr-2">Rank</th>
                  <th className="text-left py-1 pr-2">Item No</th>
                  <th className="text-left py-1 pr-2">Loc</th>
                  <th className="text-center py-1 pr-2">ABC</th>
                  <th className="text-right py-1 pr-2">Current CSL</th>
                  <th className="text-right py-1 pr-2">Target CSL</th>
                  <th className="text-right py-1 pr-2">Inv. Gap ($)</th>
                  <th className="text-right py-1">Marginal ROI</th>
                </tr>
              </thead>
              <tbody>
                {(detail?.rows ?? []).length === 0 ? (
                  <tr>
                    <td colSpan={8} className="py-4 text-center text-muted-foreground">
                      No data. Click Run Plan to compute the investment plan.
                    </td>
                  </tr>
                ) : (
                  (detail?.rows ?? []).map((r: InvestmentRow) => (
                    <tr key={`${r.investment_rank}`} className="border-b last:border-0 hover:bg-muted/30">
                      <td className="py-1 pr-2 text-right text-muted-foreground">{r.investment_rank}</td>
                      <td className="py-1 pr-2 font-mono">{r.item_no}</td>
                      <td className="py-1 pr-2">{r.loc}</td>
                      <td className="py-1 pr-2 text-center">{r.abc_vol ?? "-"}</td>
                      <td className="py-1 pr-2 text-right">
                        {r.current_csl != null ? `${(r.current_csl * 100).toFixed(1)}%` : "-"}
                      </td>
                      <td className="py-1 pr-2 text-right text-green-600">
                        {r.recommended_csl != null ? `${(r.recommended_csl * 100).toFixed(1)}%` : "-"}
                      </td>
                      <td className="py-1 pr-2 text-right text-amber-600">
                        {r.investment_increment != null ? `$${fmtInt(r.investment_increment)}` : "-"}
                      </td>
                      <td className="py-1 text-right">
                        {r.marginal_roi != null ? r.marginal_roi.toFixed(2) : "-"}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
          {totalPages > 1 && (
            <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
              <button
                className="px-2 py-1 rounded border disabled:opacity-40"
                disabled={invOffset === 0}
                onClick={() => setInvOffset(Math.max(0, invOffset - 20))}
              >
                Prev
              </button>
              <span>
                Page {currentPage} / {totalPages} - {detail?.total.toLocaleString()} items
              </span>
              <button
                className="px-2 py-1 rounded border disabled:opacity-40"
                disabled={currentPage >= totalPages}
                onClick={() => setInvOffset(invOffset + 20)}
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

