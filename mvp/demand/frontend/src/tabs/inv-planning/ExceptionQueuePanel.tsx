import { useState } from "react";
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

import { formatFixed } from "@/lib/formatters";
import { EmptyState } from "@/components/EmptyState";
import { AlertTriangle } from "lucide-react";

const PAGE = 50;

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
  below_rop:          "Below ROP",
  below_rop_critical: "Below ROP (Critical)",
  below_ss:           "Below SS",
  stockout:           "Stockout",
  excess:             "Excess",
  zero_velocity:      "Zero Velocity",
};

const EXC_TYPE_DESCRIPTIONS: Record<string, string> = {
  reorder_point: "Reorder Point — on-hand fell below ROP; order needed",
  below_ss: "Below Safety Stock — stock below minimum buffer target",
  excess: "Excess Inventory — stock exceeds target; review for disposal",
  zero_velocity: "Zero Velocity — no sales in 90+ days; review for obsolescence",
  lead_time_risk: "Lead Time Risk — supplier LT variability threatens service level",
  forecast_miss: "Forecast Miss — actual demand significantly exceeded forecast",
};

const EXC_TYPES = ["below_rop", "below_ss", "stockout", "excess", "zero_velocity"];
const EXC_SEVERITIES = ["critical", "high", "medium", "low"];

export function ExceptionQueuePanel() {
  const queryClient = useQueryClient();
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

  const { data: excList, isLoading: excLoading } = useQuery({
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
    onError: () => {
      alert("Action failed. Please check your connection and try again.");
    },
  });

  const statusMutation = useMutation({
    mutationFn: ({ id, status }: { id: string; status: "ordered" | "resolved" }) =>
      updateExceptionStatus(id, status),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: exceptionKeys.list() });
      queryClient.invalidateQueries({ queryKey: exceptionKeys.summary() });
    },
    onError: () => {
      alert("Action failed. Please check your connection and try again.");
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
            value: `$${formatFixed(excSummary?.total_recommended_order_value ?? 0, 0)}`,
            color: "text-blue-600",
            isStr: true,
          },
        ].map(({ label, value, color, isStr }) => (
          <div key={label} className="border rounded-lg p-3 bg-card">
            <p className="text-xs text-muted-foreground">{label}</p>
            <p className={`text-2xl font-bold mt-1 ${color}`}>
              {isStr ? value : (value as number).toLocaleString()}
            </p>
          </div>
        ))}
      </div>

      {/* Severity legend */}
      <div className="text-xs text-muted-foreground p-2 rounded bg-muted/30 border mb-4">
        <span className="font-medium text-foreground">Severity: </span>
        <span className="text-red-600 font-medium">● Critical</span> — immediate action required ·
        <span className="text-orange-500 font-medium ml-2">● High</span> — review within 24h ·
        <span className="text-amber-500 font-medium ml-2">● Medium</span> — review this week ·
        <span className="text-blue-500 font-medium ml-2">● Low</span> — informational
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
              {["Severity", "Item", "Loc", "Type", "Qty on Hand", "SS Target", "Rec. Order Qty", "Order By", "Status", "Actions"].map((h) => (
                <th key={h} className="px-2 py-2 text-left font-medium text-muted-foreground whitespace-nowrap">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(excList?.rows ?? []).map((row: ExceptionRow) => (
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
                  <td
                    className="px-2 py-1.5"
                    title={EXC_TYPE_DESCRIPTIONS[row.exception_type] ?? row.exception_type}
                  >
                    {row.exception_type.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
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
                        onClick={() => acknowledgeMutation.mutate({ id: row.exception_id })}
                      >
                        Acknowledge
                      </button>
                    )}
                    {row.status === "acknowledged" && (
                      <button
                        className="px-2 py-0.5 text-xs bg-sky-600 text-white rounded hover:bg-sky-700 disabled:opacity-50"
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
