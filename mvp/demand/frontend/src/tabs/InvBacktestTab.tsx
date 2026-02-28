import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Activity, ChevronLeft, ChevronRight } from "lucide-react";

import {
  queryKeys,
  STALE,
  fetchForecastModels,
  fetchInvBacktestSummary,
  fetchInvBacktestTrend,
  fetchInvBacktestRootCause,
  fetchInvBacktestDetail,
} from "@/api/queries";
import type {
  InvBacktestModelMetrics,
  InvBacktestDetailRow,
} from "@/types";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { KpiCard } from "@/components/KpiCard";
import { LoadingElement } from "@/components/LoadingElement";
import { cn } from "@/lib/utils";
import { useDebounce } from "@/hooks/useDebounce";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";
import { dfuModelColor } from "@/constants/colors";
import { useChartColors } from "@/hooks/useChartColors";
import { formatNumber, formatCompactNumber } from "@/lib/formatters";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const CHART_MARGIN = { top: 8, right: 16, left: 8, bottom: 8 };
const PAGE_SIZE = 50;
const TREND_METRICS = [
  { key: "stockout_rate", label: "Stockout Rate %" },
  { key: "excess_rate", label: "Excess Rate %" },
  { key: "avg_dos", label: "Avg DOS (days)" },
  { key: "wape", label: "WAPE %" },
] as const;

type DetailSortCol =
  | "item_no"
  | "loc"
  | "month_start"
  | "model_id"
  | "forecast"
  | "actual_demand"
  | "eom_qty_on_hand"
  | "dos"
  | "forecast_error";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export default function InvBacktestTab() {
  const { chartColors, trendColors } = useChartColors();

  // ---- State ---------------------------------------------------------------
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set());
  const [itemFilter, setItemFilter] = useState("");
  const [locationFilter, setLocationFilter] = useState("");
  const [clusterFilter, setClusterFilter] = useState("");
  const [rootCauseModel, setRootCauseModel] = useState("");
  const [trendMetric, setTrendMetric] = useState<string>("stockout_rate");
  const [eventType, setEventType] = useState<string>("all");
  const [detailOffset, setDetailOffset] = useState(0);
  const [detailSort, setDetailSort] = useState<DetailSortCol>("month_start");
  const [detailDir, setDetailDir] = useState<"asc" | "desc">("desc");

  // ---- Global filter sync --------------------------------------------------
  const { filters: globalFilters } = useGlobalFilterContext();
  const syncedRef = useRef("");
  useEffect(() => {
    const key = `${globalFilters.item.join(",")}_${globalFilters.location.join(",")}`;
    if (key === syncedRef.current) return;
    syncedRef.current = key;
    if (globalFilters.item.length === 1) setItemFilter(globalFilters.item[0]);
    if (globalFilters.location.length === 1) setLocationFilter(globalFilters.location[0]);
  }, [globalFilters.item, globalFilters.location]);

  const debouncedItem = useDebounce(itemFilter, 400);
  const debouncedLocation = useDebounce(locationFilter, 400);

  // ---- Fetch available models -----------------------------------------------
  const { data: availableModels } = useQuery({
    queryKey: queryKeys.forecastModels(),
    queryFn: fetchForecastModels,
    staleTime: STALE.TEN_MIN,
  });

  // Initialise selected models on first load
  useEffect(() => {
    if (availableModels && availableModels.length > 0 && selectedModels.size === 0) {
      setSelectedModels(new Set(availableModels.slice(0, 5)));
    }
  }, [availableModels, selectedModels.size]);

  // Set default root-cause model
  useEffect(() => {
    if (!rootCauseModel && selectedModels.size > 0) {
      setRootCauseModel([...selectedModels][0]);
    }
  }, [rootCauseModel, selectedModels]);

  // ---- Derived params -------------------------------------------------------
  const modelsStr = useMemo(() => [...selectedModels].join(","), [selectedModels]);

  const filterParams = useMemo(
    () => ({
      models: modelsStr,
      item: debouncedItem,
      location: debouncedLocation,
      cluster_assignment: clusterFilter,
    }),
    [modelsStr, debouncedItem, debouncedLocation, clusterFilter],
  );

  // ---- Queries --------------------------------------------------------------
  const { data: summaryData, isLoading: loadingSummary } = useQuery({
    queryKey: queryKeys.invBacktestSummary(filterParams),
    queryFn: () => fetchInvBacktestSummary(filterParams),
    staleTime: STALE.TWO_MIN,
    enabled: selectedModels.size > 0,
  });

  const { data: trendData, isLoading: loadingTrend } = useQuery({
    queryKey: queryKeys.invBacktestTrend(filterParams),
    queryFn: () => fetchInvBacktestTrend(filterParams),
    staleTime: STALE.TWO_MIN,
    enabled: selectedModels.size > 0,
  });

  const { data: rootCauseData, isLoading: loadingRootCause } = useQuery({
    queryKey: queryKeys.invBacktestRootCause({ ...filterParams, model_id: rootCauseModel }),
    queryFn: () => fetchInvBacktestRootCause({ ...filterParams, model_id: rootCauseModel }),
    staleTime: STALE.TWO_MIN,
    enabled: !!rootCauseModel,
  });

  const detailParams = useMemo(
    () => ({
      ...filterParams,
      event_type: eventType,
      limit: PAGE_SIZE,
      offset: detailOffset,
      sort_by: detailSort,
      sort_dir: detailDir,
    }),
    [filterParams, eventType, detailOffset, detailSort, detailDir],
  );

  const { data: detailData, isLoading: loadingDetail } = useQuery({
    queryKey: queryKeys.invBacktestDetail(detailParams),
    queryFn: () => fetchInvBacktestDetail(detailParams),
    staleTime: STALE.TWO_MIN,
    enabled: selectedModels.size > 0,
  });

  // ---- Computed data --------------------------------------------------------

  // Best model by service level
  const bestModel = useMemo<{
    id: string;
    metrics: InvBacktestModelMetrics;
  } | null>(() => {
    if (!summaryData?.by_model) return null;
    let best: { id: string; metrics: InvBacktestModelMetrics } | null = null;
    for (const [id, m] of Object.entries(summaryData.by_model)) {
      if (!best || m.service_level > best.metrics.service_level) {
        best = { id, metrics: m };
      }
    }
    return best;
  }, [summaryData]);

  // Model comparison bar chart data
  const comparisonData = useMemo(() => {
    if (!summaryData?.by_model) return [];
    return Object.entries(summaryData.by_model).map(([id, m]) => ({
      model: id,
      stockout_rate: m.stockout_rate,
      excess_rate: m.excess_rate,
      wape: m.wape ?? 0,
    }));
  }, [summaryData]);

  // Root cause stacked bar data
  const rootCauseChartData = useMemo(() => {
    if (!rootCauseData) return [];
    return [
      {
        event: "Stockout",
        under_forecast: rootCauseData.stockout_under_forecast,
        over_forecast: rootCauseData.stockout_over_forecast,
        exact: rootCauseData.stockout_exact,
      },
      {
        event: "Excess",
        under_forecast: rootCauseData.excess_under_forecast,
        over_forecast: rootCauseData.excess_over_forecast,
        exact: rootCauseData.excess_exact,
      },
    ];
  }, [rootCauseData]);

  // Trend chart data (flatten by_model per month for the selected metric)
  const trendChartData = useMemo(() => {
    if (!trendData?.trend) return [];
    const models = summaryData?.models ?? [];
    return trendData.trend.map((pt) => {
      const row: Record<string, string | number | null> = { month: pt.month };
      for (const mid of models) {
        const val = pt.by_model[mid];
        row[mid] = val ? (val[trendMetric as keyof typeof val] ?? null) : null;
      }
      return row;
    });
  }, [trendData, summaryData, trendMetric]);

  // ---- Handlers -------------------------------------------------------------
  const toggleModel = useCallback((model: string) => {
    setSelectedModels((prev) => {
      const next = new Set(prev);
      if (next.has(model)) next.delete(model);
      else next.add(model);
      return next;
    });
    setDetailOffset(0);
  }, []);

  const handleDetailSort = useCallback(
    (col: DetailSortCol) => {
      if (detailSort === col) {
        setDetailDir((prev) => (prev === "asc" ? "desc" : "asc"));
      } else {
        setDetailSort(col);
        setDetailDir("desc");
      }
      setDetailOffset(0);
    },
    [detailSort],
  );

  const sortIndicator = useCallback(
    (col: DetailSortCol) => {
      if (detailSort !== col) return null;
      return detailDir === "asc" ? " \u25B2" : " \u25BC";
    },
    [detailSort, detailDir],
  );

  const totalDetailPages = Math.max(1, Math.ceil((detailData?.total ?? 0) / PAGE_SIZE));
  const currentDetailPage = Math.floor(detailOffset / PAGE_SIZE) + 1;

  // ---- Render ---------------------------------------------------------------
  return (
    <section className="mt-4 space-y-4">
      {/* ---- KPI Cards --------------------------------------------------- */}
      {loadingSummary ? (
        <LoadingElement tabKey="invBacktest" message="Loading backtest summary..." />
      ) : bestModel ? (
        <div className="flex flex-wrap gap-3">
          <KpiCard
            label="Best Service Level"
            value={`${formatNumber(bestModel.metrics.service_level)}%`}
            sublabel={bestModel.id}
            severity={
              bestModel.metrics.service_level >= 95
                ? "best"
                : bestModel.metrics.service_level < 90
                  ? "warning"
                  : "neutral"
            }
          />
          <KpiCard
            label="Lowest Stockout Rate"
            value={`${formatNumber(
              Math.min(...Object.values(summaryData!.by_model).map((m) => m.stockout_rate)),
            )}%`}
            severity="best"
          />
          <KpiCard
            label="Lowest Excess Rate"
            value={`${formatNumber(
              Math.min(...Object.values(summaryData!.by_model).map((m) => m.excess_rate)),
            )}%`}
            severity="neutral"
          />
          <KpiCard
            label="Models Compared"
            value={String(summaryData!.models.length)}
          />
          <KpiCard
            label="DFU-Months"
            value={formatCompactNumber(bestModel.metrics.dfu_months)}
          />
        </div>
      ) : null}

      {/* ---- Main Card with filters + charts ----------------------------- */}
      <Card className="animate-fade-in">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            <CardTitle className="text-base">Inventory Planning Backtest</CardTitle>
          </div>
          <CardDescription>
            Connect forecast accuracy to inventory outcomes. Compare which
            algorithms lead to fewer stockouts and less excess inventory.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          {/* ---- Filter Controls ----------------------------------------- */}
          <div className="flex flex-wrap items-end gap-3">
            <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Item
              <input
                className="h-9 w-40 rounded-md border border-input bg-background px-3 text-sm"
                placeholder="Filter item..."
                value={itemFilter}
                onChange={(e) => { setItemFilter(e.target.value); setDetailOffset(0); }}
              />
            </label>
            <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Location
              <input
                className="h-9 w-40 rounded-md border border-input bg-background px-3 text-sm"
                placeholder="Filter location..."
                value={locationFilter}
                onChange={(e) => { setLocationFilter(e.target.value); setDetailOffset(0); }}
              />
            </label>
            <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Cluster
              <input
                className="h-9 w-40 rounded-md border border-input bg-background px-3 text-sm"
                placeholder="Filter cluster..."
                value={clusterFilter}
                onChange={(e) => setClusterFilter(e.target.value)}
              />
            </label>
          </div>

          {/* ---- Model Selector ------------------------------------------ */}
          {availableModels && availableModels.length > 0 && (
            <div className="space-y-1">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">
                Models
              </p>
              <div className="flex flex-wrap gap-2">
                {availableModels.map((m) => (
                  <button
                    key={m}
                    onClick={() => toggleModel(m)}
                    className={cn(
                      "rounded-full border px-3 py-1 text-xs font-medium transition-colors",
                      selectedModels.has(m)
                        ? "border-primary bg-primary/10 text-primary"
                        : "border-input text-muted-foreground hover:bg-muted",
                    )}
                  >
                    {m}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* ---- Model Comparison Chart ---------------------------------- */}
          {comparisonData.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">
                Model Comparison — Stockout vs Excess Rate
              </p>
              <ResponsiveContainer width="100%" height={260}>
                <ComposedChart data={comparisonData} margin={CHART_MARGIN}>
                  <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                  <XAxis
                    dataKey="model"
                    tick={{ fontSize: 10, fill: chartColors.axis }}
                    angle={-20}
                    textAnchor="end"
                    height={50}
                  />
                  <YAxis
                    yAxisId="left"
                    tick={{ fontSize: 11, fill: chartColors.axis }}
                    tickFormatter={(v: number) => `${v}%`}
                  />
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    tick={{ fontSize: 11, fill: chartColors.axis }}
                    tickFormatter={(v: number) => `${v}%`}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: chartColors.tooltip_bg,
                      borderColor: chartColors.tooltip_border,
                    }}
                    formatter={(value: number, name: string) => [
                      `${formatNumber(value)}%`,
                      name === "stockout_rate"
                        ? "Stockout Rate"
                        : name === "excess_rate"
                          ? "Excess Rate"
                          : "WAPE",
                    ]}
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar yAxisId="left" dataKey="stockout_rate" name="Stockout Rate" fill="#ef4444" barSize={20}>
                    {comparisonData.map((_, idx) => (
                      <Cell key={idx} fill="#ef4444" />
                    ))}
                  </Bar>
                  <Bar yAxisId="left" dataKey="excess_rate" name="Excess Rate" fill="#f59e0b" barSize={20}>
                    {comparisonData.map((_, idx) => (
                      <Cell key={idx} fill="#f59e0b" />
                    ))}
                  </Bar>
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="wape"
                    name="WAPE"
                    stroke={trendColors[0]}
                    strokeWidth={2}
                    dot={{ r: 4 }}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* ---- Root Cause Breakdown ------------------------------------- */}
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">
                Root Cause — Why Events Happened
              </p>
              {summaryData?.models && (
                <select
                  className="h-7 rounded border border-input bg-background px-2 text-xs"
                  value={rootCauseModel}
                  onChange={(e) => setRootCauseModel(e.target.value)}
                >
                  {summaryData.models.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              )}
            </div>
            {loadingRootCause ? (
              <LoadingElement tabKey="invBacktest" message="Loading root cause..." />
            ) : rootCauseChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={rootCauseChartData} layout="vertical" margin={CHART_MARGIN}>
                  <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                  <XAxis type="number" tick={{ fontSize: 11, fill: chartColors.axis }} />
                  <YAxis
                    dataKey="event"
                    type="category"
                    tick={{ fontSize: 11, fill: chartColors.axis }}
                    width={60}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: chartColors.tooltip_bg,
                      borderColor: chartColors.tooltip_border,
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="under_forecast" name="Under-Forecast" stackId="a" fill="#ef4444" />
                  <Bar dataKey="over_forecast" name="Over-Forecast" stackId="a" fill="#f59e0b" />
                  <Bar dataKey="exact" name="Exact" stackId="a" fill="#94a3b8" />
                </BarChart>
              </ResponsiveContainer>
            ) : null}
          </div>

          {/* ---- Monthly Trend ------------------------------------------- */}
          {loadingTrend ? (
            <LoadingElement tabKey="invBacktest" message="Loading trend..." />
          ) : trendChartData.length > 0 ? (
            <div className="space-y-2">
              <div className="flex items-center gap-3">
                <p className="text-xs uppercase tracking-wide text-muted-foreground">
                  Monthly Trend
                </p>
                <select
                  className="h-7 rounded border border-input bg-background px-2 text-xs"
                  value={trendMetric}
                  onChange={(e) => setTrendMetric(e.target.value)}
                >
                  {TREND_METRICS.map((tm) => (
                    <option key={tm.key} value={tm.key}>{tm.label}</option>
                  ))}
                </select>
              </div>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={trendChartData} margin={CHART_MARGIN}>
                  <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                  <XAxis dataKey="month" tick={{ fontSize: 11, fill: chartColors.axis }} />
                  <YAxis
                    tick={{ fontSize: 11, fill: chartColors.axis }}
                    tickFormatter={(v: number) =>
                      trendMetric === "avg_dos" ? `${v}d` : `${v}%`
                    }
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: chartColors.tooltip_bg,
                      borderColor: chartColors.tooltip_border,
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {(summaryData?.models ?? []).map((mid, idx) => (
                    <Line
                      key={mid}
                      type="monotone"
                      dataKey={mid}
                      name={mid}
                      stroke={dfuModelColor(mid, idx)}
                      strokeWidth={2}
                      dot={{ r: 2 }}
                      connectNulls
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : null}

          {/* ---- Detail Table --------------------------------------------- */}
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">
                DFU-Level Events ({formatCompactNumber(detailData?.total ?? 0)} rows)
              </p>
              <select
                className="h-7 rounded border border-input bg-background px-2 text-xs"
                value={eventType}
                onChange={(e) => { setEventType(e.target.value); setDetailOffset(0); }}
              >
                <option value="all">All Events</option>
                <option value="stockout">Stockouts Only</option>
                <option value="excess">Excess Only</option>
              </select>
            </div>

            {loadingDetail ? (
              <LoadingElement tabKey="invBacktest" message="Loading events..." />
            ) : (detailData?.rows ?? []).length > 0 ? (
              <>
                <div className="max-h-[400px] overflow-auto rounded-md border border-input">
                  <Table>
                    <TableHeader>
                      <TableRow className="border-muted bg-muted/30">
                        {([
                          { col: "item_no" as DetailSortCol, label: "Item" },
                          { col: "loc" as DetailSortCol, label: "Location" },
                          { col: "month_start" as DetailSortCol, label: "Month" },
                          { col: "model_id" as DetailSortCol, label: "Model" },
                          { col: "forecast" as DetailSortCol, label: "Forecast" },
                          { col: "actual_demand" as DetailSortCol, label: "Actual" },
                          { col: "eom_qty_on_hand" as DetailSortCol, label: "EOM On-Hand" },
                          { col: "dos" as DetailSortCol, label: "DOS" },
                          { col: "forecast_error" as DetailSortCol, label: "Error" },
                        ]).map(({ col, label }) => (
                          <TableHead
                            key={col}
                            className={cn(
                              "text-xs cursor-pointer select-none hover:text-foreground",
                              col !== "item_no" && col !== "loc" && col !== "model_id"
                                ? "text-right"
                                : "",
                            )}
                            onClick={() => handleDetailSort(col)}
                          >
                            {label}{sortIndicator(col)}
                          </TableHead>
                        ))}
                        <TableHead className="text-xs">Event</TableHead>
                        <TableHead className="text-xs">Bias</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {(detailData?.rows ?? []).map((row: InvBacktestDetailRow, idx: number) => (
                        <TableRow
                          key={`${row.item_no}-${row.loc}-${row.month}-${row.model_id}-${idx}`}
                          className={cn(
                            row.event_type === "stockout"
                              ? "bg-red-500/5"
                              : row.event_type === "excess"
                                ? "bg-amber-500/5"
                                : "hover:bg-muted/30",
                          )}
                        >
                          <TableCell className="text-sm font-medium">{row.item_no}</TableCell>
                          <TableCell className="text-sm">{row.loc}</TableCell>
                          <TableCell className="text-sm tabular-nums">{row.month}</TableCell>
                          <TableCell className="text-sm">{row.model_id}</TableCell>
                          <TableCell className="text-sm text-right tabular-nums">{formatNumber(row.forecast)}</TableCell>
                          <TableCell className="text-sm text-right tabular-nums">{formatNumber(row.actual_demand)}</TableCell>
                          <TableCell className="text-sm text-right tabular-nums">{formatNumber(row.eom_qty_on_hand)}</TableCell>
                          <TableCell className="text-sm text-right tabular-nums">{row.dos != null ? formatNumber(row.dos) : "-"}</TableCell>
                          <TableCell className={cn("text-sm text-right tabular-nums", row.forecast_error < 0 ? "text-red-600 dark:text-red-400" : row.forecast_error > 0 ? "text-amber-600 dark:text-amber-400" : "")}>{formatNumber(row.forecast_error)}</TableCell>
                          <TableCell>
                            <span className={cn("inline-block rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase",
                              row.event_type === "stockout" ? "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300" :
                              row.event_type === "excess" ? "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300" :
                              "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400",
                            )}>
                              {row.event_type}
                            </span>
                          </TableCell>
                          <TableCell className="text-xs text-muted-foreground">{row.bias_direction}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>

                {/* Pagination */}
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span>Page {currentDetailPage} of {totalDetailPages}</span>
                  <div className="flex items-center gap-2">
                    <button
                      className={cn(
                        "inline-flex items-center gap-1 rounded-md border border-input bg-background px-2 py-1 text-xs font-medium",
                        detailOffset === 0 ? "opacity-50 cursor-not-allowed" : "hover:bg-muted cursor-pointer",
                      )}
                      disabled={detailOffset === 0}
                      onClick={() => setDetailOffset((o) => Math.max(0, o - PAGE_SIZE))}
                    >
                      <ChevronLeft className="h-3 w-3" /> Prev
                    </button>
                    <button
                      className={cn(
                        "inline-flex items-center gap-1 rounded-md border border-input bg-background px-2 py-1 text-xs font-medium",
                        detailOffset + PAGE_SIZE >= (detailData?.total ?? 0) ? "opacity-50 cursor-not-allowed" : "hover:bg-muted cursor-pointer",
                      )}
                      disabled={detailOffset + PAGE_SIZE >= (detailData?.total ?? 0)}
                      onClick={() => setDetailOffset((o) => o + PAGE_SIZE)}
                    >
                      Next <ChevronRight className="h-3 w-3" />
                    </button>
                  </div>
                </div>
              </>
            ) : (
              <p className="text-sm text-muted-foreground">
                No events found. Ensure inventory and forecast data overlap and the materialized view is refreshed.
              </p>
            )}
          </div>
        </CardContent>
      </Card>
    </section>
  );
}
