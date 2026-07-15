/**
 * BlendedDemandPanel — F3.4 Demand Sensing Integration
 * Shows blended forecast (sensing signal + statistical model) by DFU and week.
 */

import { useState, useRef, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Activity, AlertCircle, Layers, Radio } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { EmptyState } from "@/components/EmptyState";
import {
  blendedKeys,
  fetchBlendedForecast,
  fetchBlendedSummary,
  STALE_EVO,
} from "@/api/queries/evolution";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";

const PAGE = 50;

export function BlendedDemandPanel() {
  const { filters: globalFilters } = useGlobalFilterContext();
  const [itemNo, setItemNo] = useState("");
  const [loc, setLoc] = useState("");
  const [page, setPage] = useState(1);

  const syncedGlobalRef = useRef<string>("");
  useEffect(() => {
    const key = `${globalFilters.item.join(",")}_${globalFilters.location.join(",")}`;
    if (key === syncedGlobalRef.current) return;
    syncedGlobalRef.current = key;
    if (globalFilters.item.length === 1) setItemNo(globalFilters.item[0]);
    if (globalFilters.location.length === 1) setLoc(globalFilters.location[0]);
  }, [globalFilters.item, globalFilters.location]);

  const params = {
    item_id: itemNo || undefined,
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
      {/* Info banner */}
      <div className="text-xs text-muted-foreground bg-muted/20 border rounded px-3 py-2">
        <strong className="text-foreground">Blended Demand</strong> combines short-horizon sensing signals with statistical forecasts using a linearly decaying alpha weight:
        <code className="mx-1 font-mono bg-muted px-1 rounded">Blended = (α × Sensing) + ((1−α) × Forecast)</code>
        Week 1: α=1.0 (100% sensing) · Week 2: α=0.8 · Week 3: α=0.6 · Beyond sensing horizon: α=0 (100% forecast).
      </div>

      {/* KPI summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: "SKUs Active", value: sumLoading ? "…" : (summary?.total_skus ?? 0).toLocaleString(), icon: <Activity size={16} /> },
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
            <div className="p-6">
              <EmptyState
                icon={Layers}
                title="No blended demand data"
                description="Blended demand combines real-time sensing signals with statistical forecasts using a linearly decaying alpha weight: near-term weeks trust sensing more, future weeks trust the statistical model more. Outlier spikes are capped to prevent over-ordering."
                steps={[
                  { label: "Compute demand signals first", command: "make demand-signals-compute" },
                  { label: "Compute blended demand plan", command: "make blended-demand-compute" },
                ]}
              />
            </div>
          ) : (
            <div className="overflow-x-auto px-4 pb-4">
              <table className="w-full text-sm">
                <thead className="bg-muted/50 text-xs uppercase text-muted-foreground">
                  <tr>
                    <th className="px-3 py-2 text-left font-medium">Item</th>
                    <th className="px-3 py-2 text-left font-medium">Loc</th>
                    <th className="px-3 py-2 text-left font-medium">Week Start</th>
                    <th className="px-3 py-2 text-left font-medium cursor-help" title="Weighting factor: 1.0 = trust sensing 100%, 0.0 = trust forecast 100%. Decays linearly over the sensing horizon.">Alpha</th>
                    <th className="px-3 py-2 text-left font-medium cursor-help" title="Short-horizon demand signal: extrapolated from current week's actual sales velocity">Sensing Qty</th>
                    <th className="px-3 py-2 text-left font-medium cursor-help" title="Champion statistical model forecast for this DFU-month">Stat Fcst</th>
                    <th className="px-3 py-2 text-left font-medium cursor-help" title="Alpha-weighted blend of sensing signal and statistical forecast">Blended Qty</th>
                    <th className="px-3 py-2 text-left font-medium cursor-help" title="Sensing ÷ Forecast ratio. Values >1.5× flag a demand surge; blended is capped at 1.5× forecast to prevent over-ordering.">Spike Ratio</th>
                    <th className="px-3 py-2 text-left font-medium">Capped?</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r, i) => (
                    <tr key={i} className="border-t hover:bg-muted/30 transition-colors">
                      <td className="px-3 py-2 font-mono text-xs">{r.item_id}</td>
                      <td className="px-3 py-2 text-xs">{r.loc}</td>
                      <td className="px-3 py-2 text-xs">{r.week_start}</td>
                      <td className="px-3 py-2 text-xs">
                        <span
                          className={`inline-block px-1.5 py-0.5 rounded text-xs font-medium ${
                            r.alpha_weight > 0.7 ? "bg-blue-100 text-blue-700" :
                            r.alpha_weight > 0.3 ? "bg-cyan-100 text-cyan-700" :
                            "bg-muted text-muted-foreground"
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
              <p className="text-xs text-muted-foreground mt-2">
                <strong>Spike Detection:</strong> When sensing signal exceeds 1.5× the statistical forecast, demand is flagged as a surge and the blended quantity is capped to prevent over-ordering.
              </p>
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
