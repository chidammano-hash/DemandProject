/**
 * BlendedDemandPanel — F3.4 Demand Sensing Integration
 * Shows blended forecast (sensing signal + statistical model) by DFU and week.
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Activity, AlertCircle, Radio } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  blendedKeys,
  fetchBlendedForecast,
  fetchBlendedSummary,
  STALE_EVO,
} from "@/api/queries/evolution";

const PAGE = 50;

export function BlendedDemandPanel() {
  const [itemNo, setItemNo] = useState("");
  const [loc, setLoc] = useState("");
  const [page, setPage] = useState(1);

  const params = {
    item_no: itemNo || undefined,
    loc: loc || undefined,
    page,
    page_size: PAGE,
  };

  const { data: summary, isLoading: sumLoading } = useQuery({
    queryKey: blendedKeys.summary({}),
    queryFn: () => fetchBlendedSummary(),
    staleTime: STALE_EVO.FIVE_MIN,
  });

  const { data, isLoading } = useQuery({
    queryKey: blendedKeys.list(params),
    queryFn: () => fetchBlendedForecast(params),
    staleTime: STALE_EVO.ONE_MIN,
  });

  const rows = data?.rows ?? [];
  const total = data?.total ?? 0;

  return (
    <div className="space-y-6 p-4">
      {/* KPI summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: "DFUs Active", value: sumLoading ? "…" : (summary?.total_dfus ?? 0).toLocaleString(), icon: <Activity size={16} /> },
          { label: "Weeks Computed", value: sumLoading ? "…" : (summary?.total_weeks ?? 0).toLocaleString(), icon: <Radio size={16} /> },
          { label: "Avg Alpha (Current)", value: sumLoading ? "…" : summary?.avg_alpha != null ? summary.avg_alpha.toFixed(3) : "—", icon: <Activity size={16} /> },
          { label: "Capped Outliers", value: sumLoading ? "…" : (summary?.capped_count ?? 0).toLocaleString(), icon: <AlertCircle size={16} />, warn: (summary?.capped_count ?? 0) > 0 },
        ].map((c) => (
          <Card key={c.label} className={c.warn ? "border-amber-400" : ""}>
            <CardHeader className="pb-2 flex flex-row items-center justify-between">
              <CardTitle className="text-sm font-medium text-muted-foreground">{c.label}</CardTitle>
              <span className={c.warn ? "text-amber-500" : "text-muted-foreground"}>{c.icon}</span>
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
          <label className="text-xs font-medium block mb-1">Location</label>
          <input
            className="border rounded px-2 py-1 text-sm w-40"
            value={loc}
            onChange={(e) => { setLoc(e.target.value); setPage(1); }}
            placeholder="e.g. 1401-BULK"
          />
        </div>
      </div>

      {/* Blended forecast table */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">
            Blended Weekly Demand Plan{" "}
            <span className="text-sm font-normal text-muted-foreground">({total.toLocaleString()} rows)</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {isLoading ? (
            <p className="p-4 text-sm text-muted-foreground">Loading…</p>
          ) : !rows.length ? (
            <p className="p-4 text-sm text-muted-foreground">No blended demand data found.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-muted/50 text-xs uppercase text-muted-foreground">
                  <tr>
                    {["Item", "Loc", "Week Start", "Alpha", "Sensing Qty", "Stat Fcst", "Blended Qty", "Spike Ratio", "Capped?"].map((h) => (
                      <th key={h} className="px-3 py-2 text-left font-medium">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r, i) => (
                    <tr key={i} className="border-t hover:bg-muted/30 transition-colors">
                      <td className="px-3 py-2 font-mono text-xs">{r.item_no}</td>
                      <td className="px-3 py-2 text-xs">{r.loc}</td>
                      <td className="px-3 py-2 text-xs">{r.week_start}</td>
                      <td className="px-3 py-2 text-xs">
                        <span
                          className={`inline-block px-1.5 py-0.5 rounded text-xs font-medium ${
                            r.alpha_weight > 0.7 ? "bg-blue-100 text-blue-700" :
                            r.alpha_weight > 0.3 ? "bg-cyan-100 text-cyan-700" :
                            "bg-gray-100 text-gray-600"
                          }`}
                        >
                          {r.alpha_weight.toFixed(2)}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-xs">{r.sensing_signal_qty.toFixed(1)}</td>
                      <td className="px-3 py-2 text-xs">{r.statistical_forecast_qty.toFixed(1)}</td>
                      <td className="px-3 py-2 font-medium">{r.blended_qty.toFixed(1)}</td>
                      <td className={`px-3 py-2 text-xs ${r.velocity_spike_ratio > 1.5 ? "text-amber-600 font-medium" : ""}`}>
                        {r.velocity_spike_ratio.toFixed(2)}×
                      </td>
                      <td className="px-3 py-2">
                        {r.is_outlier_capped ? (
                          <span className="text-xs bg-amber-100 text-amber-700 px-1.5 py-0.5 rounded">Yes</span>
                        ) : (
                          <span className="text-xs text-muted-foreground">No</span>
                        )}
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
