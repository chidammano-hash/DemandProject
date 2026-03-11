import { useState, useRef, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import { Target } from "lucide-react";
import {
  queryKeys,
  fetchProductionForecastSummary,
  fetchProductionForecastVersions,
  fetchProductionForecast,
  STALE,
  type ProductionForecastAbcRow,
  type ProductionForecastPoint,
} from "@/api/queries";

import { EmptyState } from "@/components/EmptyState";
import { formatFixed } from "@/lib/formatters";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";

function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  return iso.slice(0, 10);
}

// ---------------------------------------------------------------------------
// KPI card
// ---------------------------------------------------------------------------
function KpiCard({
  label,
  value,
  sub,
}: {
  label: string;
  value: string;
  sub?: string;
}) {
  return (
    <div className="rounded-lg border bg-card p-4 flex flex-col gap-1">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="text-2xl font-bold text-foreground">{value}</span>
      {sub && <span className="text-xs text-muted-foreground">{sub}</span>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// DFU drill-down chart
// ---------------------------------------------------------------------------
function DfuForecastChart({
  item,
  loc,
  planVersion,
}: {
  item: string;
  loc: string;
  planVersion?: string;
}) {
  const { data, isLoading, error } = useQuery({
    queryKey: queryKeys.productionForecast({ item, loc, planVersion: planVersion ?? "" }),
    queryFn: () =>
      fetchProductionForecast({ item_no: item, loc, plan_version: planVersion }),
    staleTime: STALE.FIVE_MIN,
    enabled: item.trim().length > 0 && loc.trim().length > 0,
  });

  if (!item.trim() || !loc.trim()) {
    return (
      <p className="text-sm text-muted-foreground py-4 text-center">
        Enter item and location above to view the forecast series.
      </p>
    );
  }
  if (isLoading) return <p className="text-sm text-muted-foreground">Loading…</p>;
  if (error)
    return (
      <p className="text-sm text-destructive">
        {error instanceof Error && error.message.includes("404")
          ? "No forecast found. Run 'make forecast-generate' to generate forecasts."
          : "Failed to load forecast."}
      </p>
    );
  if (!data) return null;

  const chartData = data.forecasts.map((p: ProductionForecastPoint) => ({
    month: p.forecast_month?.slice(0, 7) ?? "",
    qty: p.forecast_qty,
    lower: p.forecast_qty_lower,
    upper: p.forecast_qty_upper,
    lag_source: p.lag_source,
  }));

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-4 text-xs text-muted-foreground flex-wrap">
        <span>
          Model: <strong className="text-foreground">{data.model_id}</strong>
        </span>
        <span>
          Version: <strong className="text-foreground">{data.plan_version}</strong>
        </span>
        <span>
          Generated: <strong className="text-foreground">{fmtDate(data.generated_at)}</strong>
        </span>
        {data?.generated_at && new Date(data.generated_at) < new Date(Date.now() - 7 * 24 * 60 * 60 * 1000) && (
          <p className="text-xs text-amber-600 bg-amber-50 dark:bg-amber-900/20 px-2 py-1 rounded mt-1">
            ⚠ Forecast is more than 7 days old. Run <code className="font-mono text-[11px]">make forecast-generate</code> to refresh.
          </p>
        )}
        {data.is_recursive && (
          <span className="rounded-full bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300 px-2 py-0.5 text-[10px] font-semibold">
            RECURSIVE
          </span>
        )}
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={chartData} margin={{ top: 4, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="month" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip
            formatter={(v: number, name: string) => {
              const labelMap: Record<string, string> = {
                "P10 Floor": "P10 Floor",
                "P50 Median": "P50 Median",
                "P90 Ceiling": "P90 Ceiling",
              };
              return [formatFixed(v, 1), labelMap[name] ?? name];
            }}
          />
          <Legend iconSize={12} wrapperStyle={{ fontSize: 11 }} />
          {data.forecasts.some((p) => p.forecast_qty_lower != null) && (
            <Bar dataKey="lower" name="P10 Floor" fill="var(--muted)" opacity={0.4} />
          )}
          <Bar dataKey="qty" name="P50 Median" fill="var(--primary)" radius={[2, 2, 0, 0]} />
          {data.forecasts.some((p) => p.forecast_qty_upper != null) && (
            <Line
              type="monotone"
              dataKey="upper"
              name="P90 Ceiling"
              stroke="var(--muted-foreground)"
              strokeDasharray="4 2"
              dot={false}
            />
          )}
          <ReferenceLine x={chartData[0]?.month ?? ""} stroke="var(--primary)" strokeWidth={1.5} label="" />
        </ComposedChart>
      </ResponsiveContainer>
      <p className="text-xs text-muted-foreground mt-1">
        <strong>P10</strong> (dashed lower) = conservative floor · <strong>P50</strong> (solid) = median forecast · <strong>P90</strong> (dashed upper) = cautious ceiling. Use P10 for safety stock, P50 for consensus demand planning.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------
export function DemandForecastPanel() {
  const { filters: globalFilters } = useGlobalFilterContext();
  const [horizonMonths, setHorizonMonths] = useState(18);
  const [dfuItem, setDfuItem] = useState("");
  const [dfuLoc, setDfuLoc] = useState("");

  const syncedGlobalRef = useRef<string>("");
  useEffect(() => {
    const key = `${globalFilters.item.join(",")}_${globalFilters.location.join(",")}`;
    if (key === syncedGlobalRef.current) return;
    syncedGlobalRef.current = key;
    if (globalFilters.item.length === 1) setDfuItem(globalFilters.item[0]);
    if (globalFilters.location.length === 1) setDfuLoc(globalFilters.location[0]);
  }, [globalFilters.item, globalFilters.location]);

  const { data: versions, isLoading: versionsLoading } = useQuery({
    queryKey: queryKeys.productionForecastVersions(),
    queryFn: fetchProductionForecastVersions,
    staleTime: STALE.TEN_MIN,
  });

  const [selectedVersion, setSelectedVersion] = useState<string | undefined>();
  const latestVersion = versions?.versions[0]?.plan_version;
  const effectiveVersion = selectedVersion ?? latestVersion;

  const { data: summary, isLoading: summaryLoading } = useQuery({
    queryKey: queryKeys.productionForecastSummary({
      plan_version: effectiveVersion ?? "",
      horizon_months: horizonMonths,
    }),
    queryFn: () =>
      fetchProductionForecastSummary({
        plan_version: effectiveVersion,
        horizon_months: horizonMonths,
      }),
    staleTime: STALE.FIVE_MIN,
    enabled: !!effectiveVersion,
  });

  const abcChartData =
    summary?.by_abc_class.map((r: ProductionForecastAbcRow) => ({
      abc: r.abc_class,
      qty: r.forecast_qty,
      dfus: r.dfu_count,
    })) ?? [];

  return (
    <div className="space-y-6">
      {/* Header row */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2 text-sm">
          <label className="text-muted-foreground whitespace-nowrap">Horizon</label>
          <select
            value={horizonMonths}
            onChange={(e) => setHorizonMonths(Number(e.target.value))}
            className="rounded border bg-background px-2 py-1 text-sm"
          >
            {[1, 3, 6, 9, 12, 18].map((m) => (
              <option key={m} value={m}>
                {m}M
              </option>
            ))}
          </select>
        </div>
        {(versions?.versions?.length ?? 0) > 0 && versions && (
          <div className="flex items-center gap-2 text-sm">
            <span className="text-xs text-muted-foreground">Forecast Version:</span>
            <select
              value={effectiveVersion ?? ""}
              onChange={(e) => setSelectedVersion(e.target.value || undefined)}
              className="rounded border bg-background px-2 py-1 text-sm"
            >
              {versions?.versions?.map((v, idx) => (
                <option key={v.plan_version} value={v.plan_version}>
                  {idx === 0 ? `▶ Latest — ${v.plan_version}` : v.plan_version} ({v.dfu_count.toLocaleString()} DFUs)
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Empty state — shown when versions list is loaded but empty */}
      {!versionsLoading && (versions?.versions?.length ?? 0) === 0 && (
        <EmptyState
          icon={Target}
          title="No production forecasts generated"
          description="Production forecasts are generated by running champion ML models (LGBM/CatBoost/XGBoost) forward across the planning horizon. Each DFU gets a per-month forecast with P10/P90 confidence bands derived from backtest residuals."
          steps={[
            { label: "Run backtests to train models", command: "make backtest-all" },
            { label: "Select champion model per DFU", command: "make champion-select" },
            { label: "Apply schema (first time only)", command: "make forecast-prod-schema" },
            { label: "Generate production forecasts", command: "make forecast-generate" },
          ]}
        />
      )}

      {/* KPI cards */}
      {summaryLoading ? (
        <p className="text-sm text-muted-foreground">Loading summary…</p>
      ) : summary ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <KpiCard
            label="Forecast Version"
            value={summary.plan_version ?? "—"}
            sub={`Generated ${fmtDate(summary.generated_at)}`}
          />
          <KpiCard
            label="SKU-Locations Planned"
            value={formatFixed(summary.total_dfu_count)}
            sub="item-location pairs"
          />
          <KpiCard
            label={`Total Forecast (${horizonMonths}M)`}
            value={formatFixed(summary.total_forecast_qty)}
            sub="cumulative units"
          />
          <KpiCard
            label="Horizon"
            value={`${horizonMonths} months`}
            sub="forward-looking window"
          />
        </div>
      ) : null}

      {/* ABC class breakdown chart */}
      {abcChartData.length > 0 && (
        <div className="rounded-lg border bg-card p-4 space-y-3">
          <h4 className="text-sm font-semibold">Forecast by ABC Class</h4>
          <p className="text-xs text-muted-foreground" title="A = top 80% of demand value · B = next 15% · C = bottom 5%">Forecast by ABC Class <span className="cursor-help underline decoration-dotted">(?)</span></p>
          <ResponsiveContainer width="100%" height={180}>
            <ComposedChart data={abcChartData} margin={{ top: 4, right: 12, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="abc" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v: number) => formatFixed(v, 0)} />
              <Legend iconSize={12} wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="qty" name="Forecast Qty" fill="var(--primary)" radius={[2, 2, 0, 0]} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* DFU drill-down */}
      <div className="rounded-lg border bg-card p-4 space-y-3">
        <h4 className="text-sm font-semibold">DFU Forecast Series</h4>
        <div className="flex flex-wrap gap-2">
          <input
            placeholder="e.g. 100320"
            value={dfuItem}
            onChange={(e) => setDfuItem(e.target.value)}
            className="rounded border bg-background px-3 py-1.5 text-sm w-36"
          />
          <input
            placeholder="e.g. 1401-BULK"
            value={dfuLoc}
            onChange={(e) => setDfuLoc(e.target.value)}
            className="rounded border bg-background px-3 py-1.5 text-sm w-36"
          />
        </div>
        <p className="text-xs text-muted-foreground mt-1">Enter exact item and location codes from master data.</p>
        <DfuForecastChart item={dfuItem} loc={dfuLoc} planVersion={effectiveVersion} />
      </div>
    </div>
  );
}
