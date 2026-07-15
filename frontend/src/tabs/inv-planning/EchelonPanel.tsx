/**
 * EchelonPanel — F3.5 Multi-Echelon Safety Stock
 * Shows DC-level risk-pooled safety stock targets and cascade risk scores.
 */

import { useState, useRef, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Network, ShieldAlert, TrendingDown } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { EmptyState } from "@/components/EmptyState";
import {
  echelonKeys,
  fetchEchelonTargets,
  fetchEchelonSummary,
  STALE_EVO,
} from "@/api/queries/evolution";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";

const PAGE = 50;

const SEVERITY_COLORS: Record<string, string> = {
  critical: "bg-red-100 text-red-700",
  high: "bg-orange-100 text-orange-700",
  medium: "bg-amber-100 text-amber-700",
  low: "bg-blue-100 text-blue-700",
  ok: "bg-green-100 text-green-700",
};

export function EchelonPanel() {
  const { filters: globalFilters } = useGlobalFilterContext();
  const [itemNo, setItemNo] = useState("");
  const [severity, setSeverity] = useState("");
  const [page, setPage] = useState(1);

  const syncedGlobalRef = useRef<string>("");
  useEffect(() => {
    const key = `${globalFilters.item.join(",")}`;
    if (key === syncedGlobalRef.current) return;
    syncedGlobalRef.current = key;
    if (globalFilters.item.length === 1) setItemNo(globalFilters.item[0]);
  }, [globalFilters.item]);

  const params = {
    item_id: itemNo || undefined,
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
      {/* Concept explanation */}
      <div className="text-xs text-muted-foreground bg-muted/20 border rounded px-3 py-2 mb-3">
        <strong className="text-foreground">Multi-Echelon Safety Stock</strong> models your inventory network (DCs → warehouses → stores) as a connected system. By pooling demand uncertainty across echelons, total network safety stock can be reduced while maintaining the same service level.
      </div>

      {/* KPI cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: "Network Nodes", value: sumLoading ? "…" : (summary?.total_nodes ?? 0).toLocaleString(), icon: <Network size={16} /> },
          { label: "Critical Cascade Risk", sublabel: "immediate action needed", value: sumLoading ? "…" : (summary?.critical_count ?? 0).toLocaleString(), icon: <ShieldAlert size={16} />, warn: (summary?.critical_count ?? 0) > 0 },
          { label: "High Risk", value: sumLoading ? "…" : (summary?.high_count ?? 0).toLocaleString(), icon: <ShieldAlert size={16} />, warn: (summary?.high_count ?? 0) > 0 },
          { label: "Avg Echelon Coverage (days)", sublabel: "network-weighted", value: sumLoading ? "…" : summary?.avg_coverage_days != null ? summary.avg_coverage_days.toFixed(1) : "—", icon: <TrendingDown size={16} /> },
        ].map((c) => (
          <Card key={c.label} className={c.warn ? "border-red-400" : ""}>
            <CardHeader className="pb-2 flex flex-row items-center justify-between">
              <div>
                <CardTitle className="text-sm font-medium text-muted-foreground">{c.label}</CardTitle>
                {"sublabel" in c && c.sublabel && (
                  <p className="text-xs text-muted-foreground/70">{c.sublabel}</p>
                )}
              </div>
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
            <div className="p-6">
              <EmptyState
                icon={Network}
                title="No multi-echelon safety stock targets"
                description="Multi-echelon SS pools demand uncertainty across downstream nodes (stores/DCs) so that the network holds less total stock than the sum of independent SS targets. Risk is propagated upstream with cascade severity scoring."
                steps={[
                  { label: "Compute single-echelon SS first", command: "make ss-compute" },
                  { label: "Compute multi-echelon SS targets", command: "make echelon-ss-compute" },
                ]}
              />
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-muted/50 text-xs uppercase text-muted-foreground">
                  <tr>
                    {[
                      { label: "Item" },
                      { label: "Loc" },
                      { label: "Node Type" },
                      { label: "Pooled σ", title: "Pooled demand standard deviation: combined variability across all downstream nodes. Lower pooling benefit = more correlated demand." },
                      { label: "Echelon SS" },
                      { label: "ROP", title: "Reorder Point: place a replenishment order when inventory reaches this level" },
                      { label: "Coverage (d)", title: "Days of supply covered by the echelon safety stock (inventory ÷ daily demand)" },
                      { label: "Risk Score" },
                      { label: "Severity" },
                    ].map((h) => (
                      <th key={h.label} className="px-3 py-2 text-left font-medium" title={h.title}>{h.label}{h.title ? " ⓘ" : ""}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r, i) => (
                    <tr key={i} className="border-t hover:bg-muted/30 transition-colors">
                      <td className="px-3 py-2 font-mono text-xs">{r.item_id}</td>
                      <td className="px-3 py-2 text-xs">{r.loc}</td>
                      <td className="px-3 py-2 text-xs">{r.node_type}</td>
                      <td className="px-3 py-2 text-xs">{r.pooled_sigma?.toFixed(1) ?? "—"}</td>
                      <td className="px-3 py-2 text-xs">{r.echelon_ss?.toFixed(1) ?? "—"}</td>
                      <td className="px-3 py-2 text-xs">{r.echelon_rop?.toFixed(1) ?? "—"}</td>
                      <td className="px-3 py-2 text-xs">{r.downstream_coverage_days?.toFixed(1) ?? "—"}</td>
                      <td className="px-3 py-2 font-medium">{r.cascade_risk_score?.toFixed(1) ?? "—"}</td>
                      <td className="px-3 py-2">
                        {r.cascade_risk_severity ? (
                          <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${SEVERITY_COLORS[r.cascade_risk_severity] ?? "bg-muted text-muted-foreground"}`}>
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

      {/* Cascade risk legend */}
      <div className="text-xs text-muted-foreground p-2 rounded bg-muted/30 border mt-3">
        <span className="font-medium text-foreground">Cascade Risk Score (0–100): </span>
        <span className="text-green-600">● OK 0–20</span> ·{" "}
        <span className="text-blue-600 ml-1">● Low 20–40</span> ·{" "}
        <span className="text-amber-600 ml-1">● Medium 40–60</span> ·{" "}
        <span className="text-orange-600 ml-1">● High 60–80</span> ·{" "}
        <span className="text-red-600 ml-1">● Critical 80–100</span>
        <span className="ml-2 text-muted-foreground">— Higher score = upstream depletion risk propagates downstream</span>
      </div>

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
