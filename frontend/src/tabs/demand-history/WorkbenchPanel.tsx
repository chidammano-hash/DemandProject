import { useCallback, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
  Legend,
} from "recharts";
import {
  ChevronRight,
  BarChart3,
  Search,
  Download,
  PanelLeftClose,
  PanelLeftOpen,
  TrendingUp,
  TrendingDown,
} from "lucide-react";
import { useChartColors } from "@/hooks/useChartColors";
import { useQuery } from "@tanstack/react-query";
import { useWorkbench } from "@/api/queries/demand-history";
import type { WorkbenchGrain, WorkbenchSeries } from "@/api/queries/demand-history";
import { fetchProductionForecast } from "@/api/queries/production-forecast";
import { useDemandHistorySelection } from "../DemandHistoryTab";
import { Skeleton } from "@/components/Skeleton";
import { formatInt, formatCompactNumber } from "@/lib/formatters";

// ---------------------------------------------------------------------------
// Date helpers
// ---------------------------------------------------------------------------

// Parse "YYYY-MM" or "YYYY-MM-DD" as local-time, NEVER UTC.
// `new Date("2025-04-01")` is parsed as UTC midnight per ISO-8601, then
// displayed in local time — west of UTC that becomes "Mar 31", shifting
// every chart label by one month. Force-build a local Date instead.
function parseMonth(v: string): Date | null {
  if (!v) return null;
  const m = v.match(/^(\d{4})-(\d{2})/);
  if (!m) return null;
  const year = Number(m[1]);
  const month = Number(m[2]) - 1; // JS months are 0-indexed
  return new Date(year, month, 1);
}

const SHORT_FMT = new Intl.DateTimeFormat("en-US", { month: "short", year: "2-digit" });
const LONG_FMT = new Intl.DateTimeFormat("en-US", { month: "long", year: "numeric" });

function formatMonthShort(v: string): string {
  const d = parseMonth(v);
  return d ? SHORT_FMT.format(d) : v;
}

function formatMonthLong(v: string): string {
  const d = parseMonth(v);
  return d ? LONG_FMT.format(d) : v;
}

// Canonical "YYYY-MM" key so history ("YYYY-MM") and forecast ("YYYY-MM-DD")
// merge on the same axis.
function monthKey(v: string): string {
  return (v ?? "").slice(0, 7);
}

// Append the raw key (item_id, or item_id||loc, ...) in brackets so planners
// can see the SKU number alongside the description in the chart title.
function formatSeriesTitle(s: WorkbenchSeries): string {
  const label = s.label || s.key;
  if (!s.key || label === s.key) return label;
  const idDisplay = s.key.replace(/\|\|/g, " - ");
  return `${label} (${idDisplay})`;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GRAIN_LABELS: Record<WorkbenchGrain, string> = {
  item: "Item",
  item_loc: "Item + Loc",
  item_loc_customer: "Item + Loc + Cust",
};

type PeriodOption = 12 | 24 | 36 | 0; // 0 = All
const PERIOD_OPTIONS: { value: PeriodOption; label: string }[] = [
  { value: 12, label: "12m" },
  { value: 24, label: "24m" },
  { value: 36, label: "36m" },
  { value: 0, label: "All" },
];
const DEFAULT_PERIOD: PeriodOption = 24;

const MAX_OVERLAY = 3;

// ---------------------------------------------------------------------------
// CSV export
// ---------------------------------------------------------------------------

function downloadCsv(filename: string, rows: string[][]): void {
  const escape = (cell: string) =>
    /[",\n]/.test(cell) ? `"${cell.replace(/"/g, '""')}"` : cell;
  const csv = rows.map((r) => r.map(escape).join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function exportSeriesCsv(seriesList: WorkbenchSeries[]): void {
  if (seriesList.length === 0) return;
  // Wide format: month, series1_label, series2_label, ...
  const monthSet = new Set<string>();
  const valueByMonth = seriesList.map((s) => {
    const map = new Map<string, number>();
    for (const m of s.months) {
      const k = monthKey(m.month);
      monthSet.add(k);
      map.set(k, m.demand_qty);
    }
    return map;
  });
  const months = Array.from(monthSet).sort();
  const header = ["month", ...seriesList.map((s) => s.label || s.key)];
  const rows = [header];
  for (const month of months) {
    rows.push([month, ...valueByMonth.map((m) => String(m.get(month) ?? ""))]);
  }
  const ts = new Date().toISOString().slice(0, 10);
  downloadCsv(`demand-history-${ts}.csv`, rows);
}

// ---------------------------------------------------------------------------
// Sparkline + MoM% — a slim cell that ships per row in the left rail.
// Reads the same months[] payload the chart uses, no extra fetch.
// ---------------------------------------------------------------------------

// Plain SVG sparkline — Recharts ResponsiveContainer-per-row was making the
// main chart fail to lay out when the rail had ~50 series each spinning up
// its own observer. This is a flat, dependency-free render.
function SparkAndDelta({
  months,
  color,
}: {
  months: WorkbenchSeries["months"];
  color: string;
}) {
  const { mom, lastVal, path } = useMemo(() => {
    if (!months || months.length === 0) {
      return { mom: null as number | null, lastVal: null as number | null, path: "" };
    }
    const last = months[months.length - 1].demand_qty ?? 0;
    let momV: number | null = null;
    if (months.length >= 2) {
      const prev = months[months.length - 2].demand_qty ?? 0;
      momV = prev === 0 ? null : ((last - prev) / prev) * 100;
    }
    // Build the SVG polyline path
    const w = 64;
    const h = 16;
    const values = months.map((m) => m.demand_qty ?? 0);
    const max = Math.max(...values, 1);
    const min = Math.min(...values, 0);
    const range = max - min || 1;
    const stepX = months.length > 1 ? w / (months.length - 1) : 0;
    const pts = values.map((v, i) => {
      const x = i * stepX;
      const y = h - ((v - min) / range) * h;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
    return { mom: momV, lastVal: last, path: pts.join(" ") };
  }, [months]);

  if (!months || months.length === 0) return null;

  return (
    <div className="flex items-center gap-2 mt-1 pl-5">
      <svg width={64} height={16} className="flex-shrink-0 overflow-visible">
        <polyline
          points={path}
          fill="none"
          stroke={color}
          strokeWidth={1}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      </svg>
      {lastVal != null && (
        <span className="text-[10px] text-gray-400 tabular-nums">
          {formatCompactNumber(lastVal)}
        </span>
      )}
      {mom != null && (
        <span
          className={`text-[10px] tabular-nums inline-flex items-center gap-0.5 ${
            mom >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"
          }`}
        >
          {mom >= 0 ? <TrendingUp className="h-2.5 w-2.5" /> : <TrendingDown className="h-2.5 w-2.5" />}
          {Math.abs(mom).toFixed(1)}%
        </span>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tree row
// ---------------------------------------------------------------------------

function TreeNode({
  series,
  isSelected,
  onToggle,
  canDrillDown,
  sparkColor,
  rowDense,
}: {
  series: WorkbenchSeries;
  isSelected: boolean;
  onToggle: (key: string, additive: boolean) => void;
  canDrillDown: boolean;
  sparkColor: string;
  rowDense: boolean;
}) {
  return (
    <button
      onClick={(e) => onToggle(series.key, e.metaKey || e.ctrlKey || e.shiftKey)}
      className={`group w-full text-left px-3 py-2 text-sm rounded-lg transition-all ${
        isSelected
          ? "bg-blue-50 text-blue-700 ring-1 ring-blue-200 dark:bg-blue-900/30 dark:text-blue-300 dark:ring-blue-800"
          : "hover:bg-gray-50 dark:hover:bg-gray-800/60 text-gray-700 dark:text-gray-300"
      }`}
      title={isSelected ? "Click to deselect (cmd/ctrl-click for overlay)" : "Click to select (cmd/ctrl-click to add overlay, max 3)"}
    >
      <div className="flex items-center gap-2">
        {canDrillDown && (
          <ChevronRight className={`h-3.5 w-3.5 flex-shrink-0 transition-transform ${
            isSelected ? "text-blue-500 rotate-90" : "text-gray-400 group-hover:text-gray-500"
          }`} />
        )}
        <span className="truncate font-medium flex-1">{series.label || series.key}</span>
        <span className={`text-xs tabular-nums flex-shrink-0 ${
          isSelected ? "text-blue-500 font-medium" : "text-gray-400"
        }`}>
          {formatInt(series.total_demand)}
        </span>
      </div>
      {!rowDense && <SparkAndDelta months={series.months} color={sparkColor} />}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

export function DemandWorkbenchPanel() {
  const { itemId, loc, setSelection } = useDemandHistorySelection();
  const { trendColors, chartColors } = useChartColors();

  const [grain, setGrain] = useState<WorkbenchGrain>("item");
  const [search, setSearch] = useState("");
  const [selectedKeys, setSelectedKeys] = useState<string[]>([]);
  const [period, setPeriod] = useState<PeriodOption>(DEFAULT_PERIOD);
  const [showForecast, setShowForecast] = useState(false);
  const [railCollapsed, setRailCollapsed] = useState(false);

  // Derive query params from grain + selection.
  // item_loc requires item_id; item_loc_customer requires both item_id and loc.
  const queryItemId = grain !== "item" ? (itemId || undefined) : undefined;
  const queryLoc = grain === "item_loc_customer" ? (loc || undefined) : undefined;

  const queryEnabled =
    grain === "item" ||
    (grain === "item_loc" && !!queryItemId) ||
    (grain === "item_loc_customer" && !!queryItemId && !!queryLoc);

  const monthsArg = period === 0 ? undefined : period;
  const { data, isLoading } = useWorkbench(
    grain,
    queryItemId,
    queryLoc,
    undefined,
    monthsArg,
    undefined,
    undefined,
    queryEnabled,
  );

  // Filter series by search
  const filteredSeries = useMemo(() => {
    if (!data?.series) return [];
    if (!search) return data.series;
    const q = search.toLowerCase();
    return data.series.filter(
      (s) => s.key.toLowerCase().includes(q) || s.label.toLowerCase().includes(q),
    );
  }, [data?.series, search]);

  // Resolve selected keys to series objects (preserving select order so the
  // first picked series gets trendColors[0], second gets [1], etc.)
  const selectedSeriesList = useMemo<WorkbenchSeries[]>(() => {
    if (!data?.series || selectedKeys.length === 0) return [];
    const byKey = new Map(data.series.map((s) => [s.key, s]));
    return selectedKeys.map((k) => byKey.get(k)).filter((s): s is WorkbenchSeries => !!s);
  }, [data?.series, selectedKeys]);

  // Forecast overlay availability:
  //  - Item grain      : aggregated across all locations (loc omitted)
  //  - Item+Loc grain  : single DFU
  //  - Item+Loc+Cust   : not available (no customer-grain forecast exists)
  const forecastEnabled =
    showForecast &&
    selectedSeriesList.length === 1 &&
    grain !== "item_loc_customer" &&
    !!itemId;

  const forecastLoc = grain === "item" ? undefined : loc || undefined;

  const forecastQuery = useQuery({
    queryKey: ["wb-production-forecast", itemId, forecastLoc ?? "ALL"],
    queryFn: () => fetchProductionForecast({ item_id: itemId, loc: forecastLoc }),
    enabled: forecastEnabled,
  });

  const handleToggle = useCallback((key: string, additive: boolean) => {
    setSelectedKeys((prev) => {
      const has = prev.includes(key);
      if (additive) {
        if (has) return prev.filter((k) => k !== key);
        return prev.length >= MAX_OVERLAY ? prev : [...prev, key];
      }
      // Plain click: replace selection (or deselect if same single)
      if (prev.length === 1 && has) return [];
      return [key];
    });

    // Propagate item+loc context for the Decomposition / Comparison tabs.
    const parts = key.split("||");
    if (parts.length >= 2) setSelection(parts[0], parts[1]);
    else if (parts.length === 1) setSelection(parts[0], "");
  }, [setSelection]);

  function handleDrillDown(childKey: string) {
    const nextGrain: Record<WorkbenchGrain, WorkbenchGrain> = {
      item: "item_loc",
      item_loc: "item_loc_customer",
      item_loc_customer: "item_loc_customer",
    };
    const parts = childKey.split("||");
    if (parts.length >= 1) setSelection(parts[0], parts[1] ?? "");
    setGrain(nextGrain[grain]);
    setSelectedKeys([]);
  }

  const canDrillDown = grain !== "item_loc_customer" && !!data?.hierarchy_children;
  const railWidth = railCollapsed ? "w-12" : "w-80";

  return (
    <div className="flex gap-4">
      {/* Left rail — capped to chart height so 50 rows scroll inside the rail
          rather than stretching the page. */}
      <div className={`${railWidth} flex-shrink-0 border-r dark:border-gray-700 flex flex-col transition-all duration-150 h-[480px]`}>
        {/* Collapse toggle */}
        <div className="flex items-center justify-between mb-2">
          {!railCollapsed && (
            <span className="text-[10px] uppercase tracking-wider text-gray-400 px-1">
              Series
            </span>
          )}
          <button
            onClick={() => setRailCollapsed((v) => !v)}
            className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-400 ml-auto"
            title={railCollapsed ? "Expand rail" : "Collapse rail"}
          >
            {railCollapsed ? <PanelLeftOpen className="h-4 w-4" /> : <PanelLeftClose className="h-4 w-4" />}
          </button>
        </div>

        {!railCollapsed && (
          <>
            {/* Grain selector */}
            <div className="flex gap-1 mb-3 pr-3">
              {(["item", "item_loc", "item_loc_customer"] as WorkbenchGrain[]).map((g) => (
                <button
                  key={g}
                  onClick={() => { setGrain(g); setSelectedKeys([]); }}
                  className={`px-2.5 py-1 text-xs rounded-md font-medium transition-colors ${
                    grain === g
                      ? "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300"
                      : "text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800"
                  }`}
                >
                  {GRAIN_LABELS[g]}
                </button>
              ))}
            </div>

            {/* Period selector */}
            <div className="flex gap-1 mb-3 pr-3">
              <span className="text-[10px] uppercase tracking-wider text-gray-400 self-center">Period</span>
              {PERIOD_OPTIONS.map((p) => (
                <button
                  key={p.value}
                  onClick={() => setPeriod(p.value)}
                  className={`px-2 py-0.5 text-xs rounded font-medium transition-colors ${
                    period === p.value
                      ? "bg-gray-200 text-gray-800 dark:bg-gray-700 dark:text-gray-100"
                      : "text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800"
                  }`}
                >
                  {p.label}
                </button>
              ))}
            </div>

            {/* Search */}
            <div className="relative mb-3 pr-3">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-gray-400" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search items..."
                className="w-full pl-8 pr-3 py-1.5 text-sm border dark:border-gray-700 rounded-md bg-white dark:bg-gray-900"
              />
            </div>

            {/* Series count + multi-select hint */}
            {!isLoading && data?.series && (
              <div className="flex items-center justify-between text-[10px] mb-2 px-1 pr-3">
                <span className="text-gray-400 uppercase tracking-wider">
                  {filteredSeries.length} of {data.series.length} series
                </span>
                {selectedKeys.length > 0 && (
                  <span className="text-blue-500">
                    {selectedKeys.length}/{MAX_OVERLAY} selected
                  </span>
                )}
              </div>
            )}

            {/* Series list */}
            <div className="flex-1 overflow-y-auto pr-3 space-y-0.5">
              {isLoading && (
                <div className="space-y-1.5 p-1">
                  {Array.from({ length: 8 }).map((_, i) => (
                    <Skeleton key={i} className="h-9 rounded-lg" />
                  ))}
                </div>
              )}
              {filteredSeries.map((s) => {
                const idx = selectedKeys.indexOf(s.key);
                return (
                  <TreeNode
                    key={s.key}
                    series={s}
                    isSelected={idx >= 0}
                    onToggle={handleToggle}
                    canDrillDown={canDrillDown}
                    sparkColor={idx >= 0 ? trendColors[idx % trendColors.length] : chartColors.axis}
                    rowDense={false}
                  />
                );
              })}
              {!isLoading && filteredSeries.length === 0 && (
                <div className="flex flex-col items-center py-8 text-gray-400">
                  <Search className="h-8 w-8 mb-2 opacity-40" />
                  <p className="text-sm">No series found</p>
                  {search && (
                    <button
                      onClick={() => setSearch("")}
                      className="mt-1 text-xs text-blue-500 hover:underline"
                    >
                      Clear search
                    </button>
                  )}
                </div>
              )}
            </div>

            {/* Drill-down (only when a single series is selected) */}
            {canDrillDown && selectedKeys.length === 1 && (
              <div className="mt-3 pt-3 border-t dark:border-gray-700 pr-3">
                <button
                  onClick={() => handleDrillDown(selectedKeys[0])}
                  className="w-full flex items-center justify-between px-3 py-2.5 text-xs rounded-lg bg-gradient-to-r from-blue-50 to-blue-100/50 hover:from-blue-100 hover:to-blue-100 dark:from-blue-900/30 dark:to-blue-900/20 dark:hover:from-blue-900/50 dark:hover:to-blue-900/30 text-blue-700 dark:text-blue-300 font-medium transition-all"
                >
                  <span>Drill down to {GRAIN_LABELS[grain === "item" ? "item_loc" : "item_loc_customer"]}</span>
                  <ChevronRight className="h-4 w-4" />
                </button>
              </div>
            )}
          </>
        )}
      </div>

      {/* Right: chart */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Chart toolbar */}
        {selectedSeriesList.length > 0 && (
          <div className="flex items-center justify-between mb-2 gap-3">
            <div className="text-xs text-gray-500 truncate">
              {selectedSeriesList.length === 1
                ? formatSeriesTitle(selectedSeriesList[0])
                : `Comparing ${selectedSeriesList.length} series`}
            </div>
            <div className="flex items-center gap-1">
              <label
                className={`flex items-center gap-1.5 text-xs px-2 py-1 rounded ${
                  selectedSeriesList.length === 1 && grain !== "item_loc_customer"
                    ? "text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                    : "text-gray-400 cursor-not-allowed"
                }`}
                title={
                  selectedSeriesList.length === 1 && grain !== "item_loc_customer"
                    ? grain === "item"
                      ? "Overlay production forecast (sum across all locations)"
                      : "Overlay production forecast for this DFU"
                    : "Forecast needs a single item or item+loc selection"
                }
              >
                <input
                  type="checkbox"
                  className="h-3 w-3"
                  checked={showForecast}
                  onChange={(e) => setShowForecast(e.target.checked)}
                  disabled={selectedSeriesList.length !== 1 || grain === "item_loc_customer"}
                />
                Forecast
              </label>
              <button
                onClick={() => exportSeriesCsv(selectedSeriesList)}
                className="flex items-center gap-1 text-xs px-2 py-1 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
                title="Download visible series as CSV"
              >
                <Download className="h-3.5 w-3.5" />
                CSV
              </button>
            </div>
          </div>
        )}

        {isLoading && selectedKeys.length > 0 ? (
          <Skeleton className="flex-1 rounded-lg" />
        ) : selectedSeriesList.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center text-gray-400">
            <BarChart3 className="h-12 w-12 mb-3 opacity-30" />
            <p className="text-sm font-medium">No series selected</p>
            <p className="text-xs mt-1">Click a series to view, cmd/ctrl-click to overlay (max {MAX_OVERLAY})</p>
          </div>
        ) : (
          <ChartView
            seriesList={selectedSeriesList}
            forecastPoints={forecastEnabled ? forecastQuery.data?.forecasts ?? [] : []}
            forecastLoading={forecastEnabled && forecastQuery.isLoading}
            chartColors={chartColors}
            trendColors={trendColors}
          />
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Chart view — handles 1 series (Area + optional forecast Line) or 2-3
// series (Lines, no fill). Forecast overlay only applies in the 1-series case.
// ---------------------------------------------------------------------------

interface ForecastPoint {
  forecast_month: string;
  forecast_qty: number | null;
}

interface ChartViewProps {
  seriesList: WorkbenchSeries[];
  forecastPoints: ForecastPoint[];
  forecastLoading: boolean;
  chartColors: { grid: string; axis: string; tooltip_bg: string; tooltip_border: string };
  trendColors: string[];
}

function ChartView({ seriesList, forecastPoints, forecastLoading, chartColors, trendColors }: ChartViewProps) {
  // Build a unified row[] keyed by month with one column per selected series
  // and an optional `forecast_qty` column.
  const { rows, headerStats } = useMemo(() => {
    const monthSet = new Set<string>();
    const seriesMaps = seriesList.map((s) => {
      const m = new Map<string, number>();
      for (const point of s.months) {
        const k = monthKey(point.month);
        monthSet.add(k);
        m.set(k, point.demand_qty);
      }
      return m;
    });
    const forecastMap = new Map<string, number>();
    for (const f of forecastPoints) {
      if (f.forecast_qty == null) continue;
      const k = monthKey(f.forecast_month);
      monthSet.add(k);
      forecastMap.set(k, f.forecast_qty);
    }
    const months = Array.from(monthSet).sort();
    const built = months.map((m) => {
      const row: Record<string, string | number | null> = { month: m };
      seriesList.forEach((s, i) => {
        row[`s${i}`] = seriesMaps[i].get(m) ?? null;
      });
      row.forecast_qty = forecastMap.has(m) ? forecastMap.get(m)! : null;
      return row;
    });

    // Header stats only for single-series mode
    let header = null as null | {
      mean: number; peak: number; peakMonth: string;
      first: string; last: string; count: number;
    };
    if (seriesList.length === 1) {
      const ms = seriesList[0].months;
      if (ms.length > 0) {
        let total = 0;
        let peak = -Infinity;
        let peakMonth = ms[0].month;
        for (const m of ms) {
          const q = m.demand_qty ?? 0;
          total += q;
          if (q > peak) {
            peak = q;
            peakMonth = m.month;
          }
        }
        header = {
          mean: total / ms.length,
          peak,
          peakMonth,
          first: ms[0].month,
          last: ms[ms.length - 1].month,
          count: ms.length,
        };
      }
    }
    return { rows: built, headerStats: header };
  }, [seriesList, forecastPoints]);

  const isSingle = seriesList.length === 1;
  const single = seriesList[0];
  const stroke0 = trendColors[0];
  const gradientId = isSingle ? `wbGrad-${single.key.replace(/[^a-zA-Z0-9]/g, "_")}` : "";

  const seriesNameByDataKey = useMemo(() => {
    const map: Record<string, string> = { forecast_qty: "Forecast" };
    seriesList.forEach((s, i) => { map[`s${i}`] = s.label || s.key; });
    return map;
  }, [seriesList]);

  return (
    <>
      {/* Stats header (single-series only) */}
      {isSingle && headerStats && (
        <div className="mb-2">
          <div className="flex items-baseline gap-3">
            <span className="text-xs text-gray-400 tabular-nums">
              Total: {formatInt(single.total_demand)}
            </span>
            {forecastLoading && (
              <span className="text-[11px] text-gray-400">Loading forecast...</span>
            )}
          </div>
          <div className="mt-0.5 flex flex-wrap gap-x-4 gap-y-0.5 text-[11px] text-gray-500 dark:text-gray-400">
            <span>
              Period: <span className="text-gray-700 dark:text-gray-200 tabular-nums">{formatMonthShort(headerStats.first)} - {formatMonthShort(headerStats.last)}</span> ({headerStats.count} mo)
            </span>
            <span>
              Mean: <span className="text-gray-700 dark:text-gray-200 tabular-nums">{formatCompactNumber(headerStats.mean)}</span>/mo
            </span>
            <span>
              Peak: <span className="text-gray-700 dark:text-gray-200 tabular-nums">{formatCompactNumber(headerStats.peak)}</span>{" "}
              <span className="text-gray-400">({formatMonthShort(headerStats.peakMonth)})</span>
            </span>
          </div>
        </div>
      )}

      <div className="h-[420px] w-full" key={seriesList.map((s) => s.key).join("|") + (forecastPoints.length > 0 ? ":fc" : "")}>
        <ResponsiveContainer width="100%" height="100%">
          {isSingle ? (
            // Unified single-series path: ComposedChart that always works
            // for history-only AND forecast-overlay. Avoids the dual-path
            // brittleness that previously hid the forecast Line.
            <ComposedChart data={rows} margin={{ top: 10, right: 16, left: 0, bottom: 4 }}>
              <CartesianGrid stroke={chartColors.grid} strokeDasharray="3 3" />
              <XAxis
                dataKey="month"
                tick={{ fontSize: 11, fill: chartColors.axis }}
                tickFormatter={formatMonthShort}
                interval="preserveStartEnd"
                minTickGap={32}
                angle={-30}
                textAnchor="end"
                height={50}
                padding={{ left: 8, right: 8 }}
              />
              <YAxis
                tick={{ fontSize: 11, fill: chartColors.axis }}
                tickFormatter={formatCompactNumber}
                domain={[0, "auto"]}
                tickCount={6}
                width={56}
                allowDecimals={false}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: chartColors.tooltip_bg,
                  border: `1px solid ${chartColors.tooltip_border}`,
                  fontSize: 12,
                }}
                labelFormatter={formatMonthLong}
                formatter={(v: number, key: string) => [formatInt(v), seriesNameByDataKey[key] ?? key]}
              />
              <defs>
                <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={stroke0} stopOpacity={0.18} />
                  <stop offset="95%" stopColor={stroke0} stopOpacity={0} />
                </linearGradient>
              </defs>
              {headerStats && headerStats.mean > 0 && (
                <ReferenceLine
                  y={headerStats.mean}
                  stroke={chartColors.axis}
                  strokeDasharray="4 4"
                  strokeOpacity={0.5}
                  label={{ value: "avg", position: "right", fontSize: 10, fill: chartColors.axis }}
                />
              )}
              <Area
                type="linear"
                dataKey="s0"
                stroke={stroke0}
                fill={`url(#${gradientId})`}
                strokeWidth={2}
                dot={{ r: 2, fill: stroke0 }}
                activeDot={{ r: 4 }}
                name={single.label || single.key}
                connectNulls
              />
              {forecastPoints.length > 0 && (
                <Line
                  type="linear"
                  dataKey="forecast_qty"
                  stroke="#059669"
                  strokeWidth={2}
                  strokeDasharray="6 4"
                  dot={{ r: 3, fill: "#059669" }}
                  name="Forecast"
                  connectNulls
                  isAnimationActive={false}
                />
              )}
              {forecastPoints.length > 0 && <Legend wrapperStyle={{ fontSize: 11 }} />}
            </ComposedChart>
          ) : (
            <ComposedChart data={rows} margin={{ top: 10, right: 16, left: 0, bottom: 4 }}>
              <CartesianGrid stroke={chartColors.grid} strokeDasharray="3 3" />
              <XAxis
                dataKey="month"
                tick={{ fontSize: 11, fill: chartColors.axis }}
                tickFormatter={formatMonthShort}
                interval="preserveStartEnd"
                minTickGap={32}
                angle={-30}
                textAnchor="end"
                height={50}
                padding={{ left: 8, right: 8 }}
              />
              <YAxis
                tick={{ fontSize: 11, fill: chartColors.axis }}
                tickFormatter={formatCompactNumber}
                domain={[0, "auto"]}
                tickCount={6}
                width={56}
                allowDecimals={false}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: chartColors.tooltip_bg,
                  border: `1px solid ${chartColors.tooltip_border}`,
                  fontSize: 12,
                }}
                labelFormatter={formatMonthLong}
                formatter={(v: number, key: string) => [formatInt(v), seriesNameByDataKey[key] ?? key]}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {seriesList.map((s, i) => (
                <Line
                  key={s.key}
                  type="linear"
                  dataKey={`s${i}`}
                  stroke={trendColors[i % trendColors.length]}
                  strokeWidth={2}
                  dot={false}
                  name={s.label || s.key}
                  connectNulls
                />
              ))}
            </ComposedChart>
          )}
        </ResponsiveContainer>
      </div>
    </>
  );
}
