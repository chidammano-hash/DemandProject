/**
 * EchelonPanel — F3.5 Multi-Echelon Safety Stock
 * Shows DC-level risk-pooled safety stock targets and cascade risk scores.
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Network, ShieldAlert, TrendingDown } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  echelonKeys,
  fetchEchelonTargets,
  fetchEchelonSummary,
  STALE_EVO,
} from "@/api/queries/evolution";

const PAGE = 50;

const SEVERITY_COLORS: Record<string, string> = {
  critical: "bg-red-100 text-red-700",
  high: "bg-orange-100 text-orange-700",
  medium: "bg-amber-100 text-amber-700",
  low: "bg-blue-100 text-blue-700",
  ok: "bg-green-100 text-green-700",
};

export function EchelonPanel() {
  const [itemNo, setItemNo] = useState("");
  const [severity, setSeverity] = useState("");
  const [page, setPage] = useState(1);

  const params = {
    item_no: itemNo || undefined,
    severity: severity || undefined,
    page,
    page_size: PAGE,
  };

  const { data: summary, isLoading: sumLoading } = useQuery({
    queryKey: echelonKeys.summary({}),
    queryFn: () => fetchEchelonSummary(),
    staleTime: STALE_EVO.FIVE_MIN,
  });

  const { data, isLoading } = useQuery({
    queryKey: echelonKeys.targets(params),
    queryFn: () => fetchEchelonTargets(params),
    staleTime: STALE_EVO.ONE_MIN,
  });

  const rows = data?.rows ?? [];
  const total = data?.total ?? 0;

  return (
    <div className="space-y-6 p-4">
      {/* KPI cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: "Network Nodes", value: sumLoading ? "…" : (summary?.total_nodes ?? 0).toLocaleString(), icon: <Network size={16} /> },
          { label: "Critical Risk", value: sumLoading ? "…" : (summary?.critical_count ?? 0).toLocaleString(), icon: <ShieldAlert size={16} />, warn: (summary?.critical_count ?? 0) > 0 },
          { label: "High Risk", value: sumLoading ? "…" : (summary?.high_count ?? 0).toLocaleString(), icon: <ShieldAlert size={16} />, warn: (summary?.high_count ?? 0) > 0 },
          { label: "Avg Coverage (days)", value: sumLoading ? "…" : summary?.avg_coverage_days != null ? summary.avg_coverage_days.toFixed(1) : "—", icon: <TrendingDown size={16} /> },
        ].map((c) => (
          <Card key={c.label} className={c.warn ? "border-red-400" : ""}>
            <CardHeader className="pb-2 flex flex-row items-center justify-between">
              <CardTitle className="text-sm font-medium text-muted-foreground">{c.label}</CardTitle>
              <span className={c.warn ? "text-red-500" : "text-muted-foreground"}>{c.icon}</span>
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold">{c.value}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Filter row */}
      <div className="flex flex-wrap gap-3 items-end">
        <div>
          <label className="text-xs font-medium block mb-1">Item No</label>
          <input
            className="border rounded px-2 py-1 text-sm w-40"
            value={itemNo}
            onChange={(e) => { setItemNo(e.target.value); setPage(1); }}
            placeholder="e.g. 100320"
          />
        </div>
        <div>
          <label className="text-xs font-medium block mb-1">Severity</label>
          <select
            className="border rounded px-2 py-1 text-sm"
            value={severity}
            onChange={(e) => { setSeverity(e.target.value); setPage(1); }}
          >
            <option value="">All</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
            <option value="ok">OK</option>
          </select>
        </div>
      </div>

      {/* Echelon targets table */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">
            Echelon Safety Stock Targets{" "}
            <span className="text-sm font-normal text-muted-foreground">({total.toLocaleString()} rows)</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {isLoading ? (
            <p className="p-4 text-sm text-muted-foreground">Loading…</p>
          ) : !rows.length ? (
            <p className="p-4 text-sm text-muted-foreground">No echelon targets found.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-muted/50 text-xs uppercase text-muted-foreground">
                  <tr>
                    {["Item", "Loc", "Node Type", "Pooled σ", "Echelon SS", "Echelon ROP", "Coverage (d)", "Risk Score", "Severity"].map((h) => (
                      <th key={h} className="px-3 py-2 text-left font-medium">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r, i) => (
                    <tr key={i} className="border-t hover:bg-muted/30 transition-colors">
                      <td className="px-3 py-2 font-mono text-xs">{r.item_no}</td>
                      <td className="px-3 py-2 text-xs">{r.loc}</td>
                      <td className="px-3 py-2 text-xs">{r.node_type}</td>
                      <td className="px-3 py-2 text-xs">{r.pooled_sigma?.toFixed(1) ?? "—"}</td>
                      <td className="px-3 py-2 text-xs">{r.echelon_ss?.toFixed(1) ?? "—"}</td>
                      <td className="px-3 py-2 text-xs">{r.echelon_rop?.toFixed(1) ?? "—"}</td>
                      <td className="px-3 py-2 text-xs">{r.downstream_coverage_days?.toFixed(1) ?? "—"}</td>
                      <td className="px-3 py-2 font-medium">{r.cascade_risk_score?.toFixed(1) ?? "—"}</td>
                      <td className="px-3 py-2">
                        {r.cascade_risk_severity ? (
                          <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${SEVERITY_COLORS[r.cascade_risk_severity] ?? "bg-gray-100 text-gray-600"}`}>
                            {r.cascade_risk_severity.toUpperCase()}
                          </span>
                        ) : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Pagination */}
      {total > PAGE && (
        <div className="flex gap-2 items-center justify-end text-sm">
          <button disabled={page <= 1} onClick={() => setPage(p => p - 1)} className="px-3 py-1 border rounded disabled:opacity-40">← Prev</button>
          <span>Page {page} / {Math.ceil(total / PAGE)}</span>
          <button disabled={page * PAGE >= total} onClick={() => setPage(p => p + 1)} className="px-3 py-1 border rounded disabled:opacity-40">Next →</button>
        </div>
      )}
    </div>
  );
}
