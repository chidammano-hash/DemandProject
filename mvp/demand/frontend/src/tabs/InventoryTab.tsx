import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Package, ChevronLeft, ChevronRight, ChevronDown, ChevronUp } from "lucide-react";

import {
  queryKeys,
  STALE,
  fetchInventoryPosition,
  fetchInventoryKpis,
  fetchInventoryTrend,
  fetchInventoryItemDetail,
  fetchVariabilitySummary,
  fetchVariabilityDetail,
} from "@/api/queries";
import type { VariabilityDetailRow } from "@/api/queries";
import type {
  InventoryPosition,
  InventoryKpis,
  InventoryTrendPoint,
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
import { useChartColors } from "@/hooks/useChartColors";
import { formatNumber, formatCompactNumber } from "@/lib/formatters";

// ---------------------------------------------------------------------------
// Hoisted constants
// ---------------------------------------------------------------------------
const CHART_MARGIN = { top: 8, right: 16, left: 8, bottom: 8 };
const CHART_DOT = { r: 3 };
const PAGE_SIZE = 50;

type SortCol =
  | "item_no"
  | "loc"
  | "snapshot_date"
  | "qty_on_hand"
  | "qty_on_hand_on_order"
  | "qty_on_order"
  | "lead_time_days"
  | "mtd_sales";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function InventoryTab() {
  const { chartColors, trendColors } = useChartColors();

  // ---- Local state --------------------------------------------------------
  const [itemFilter, setItemFilter] = useState("");
  const [locationFilter, setLocationFilter] = useState("");
  const [months, setMonths] = useState(12);
  const [offset, setOffset] = useState(0);
  const [sortBy, setSortBy] = useState<SortCol>("snapshot_date");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [selectedRow, setSelectedRow] = useState<{
    item: string;
    location: string;
  } | null>(null);

  // ---- Global filter sync --------------------------------------------------
  const { filters: globalFilters } = useGlobalFilterContext();

  // Sync global item/location filter into local inputs
  const syncedGlobalRef = useRef<string>("");
  useEffect(() => {
    const key = `${globalFilters.item.join(",")}_${globalFilters.location.join(",")}`;
    if (key === syncedGlobalRef.current) return;
    syncedGlobalRef.current = key;
    if (globalFilters.item.length === 1) setItemFilter(globalFilters.item[0]);
    if (globalFilters.location.length === 1) setLocationFilter(globalFilters.location[0]);
  }, [globalFilters.item, globalFilters.location]);

  const debouncedItem = useDebounce(itemFilter, 400);
  const debouncedLocation = useDebounce(locationFilter, 400);

  // ---- Derived params -----------------------------------------------------
  const kpiParams = useMemo(
    () => ({
      item: debouncedItem,
      location: debouncedLocation,
      months,
    }),
    [debouncedItem, debouncedLocation, months],
  );

  const trendParams = useMemo(
    () => ({
      item: debouncedItem,
      location: debouncedLocation,
      months,
    }),
    [debouncedItem, debouncedLocation, months],
  );

  const positionParams = useMemo(
    () => ({
      item: debouncedItem,
      location: debouncedLocation,
      limit: PAGE_SIZE,
      offset,
      sort_by: sortBy,
      sort_dir: sortDir,
    }),
    [debouncedItem, debouncedLocation, offset, sortBy, sortDir],
  );

  // ---- Data fetching: KPIs ------------------------------------------------
  const { data: kpiData, isLoading: loadingKpis } = useQuery({
    queryKey: queryKeys.inventoryKpis(kpiParams),
    queryFn: () => fetchInventoryKpis(kpiParams),
    staleTime: STALE.FIVE_MIN,
  });

  // ---- Data fetching: Trend -----------------------------------------------
  const { data: trendPayload, isLoading: loadingTrend } = useQuery({
    queryKey: queryKeys.inventoryTrend(trendParams),
    queryFn: () => fetchInventoryTrend(trendParams),
    staleTime: STALE.FIVE_MIN,
  });

  const trendData: InventoryTrendPoint[] = trendPayload?.trend ?? [];

  // ---- Data fetching: Position table --------------------------------------
  const { data: positionPayload, isLoading: loadingPosition } = useQuery({
    queryKey: queryKeys.inventoryPosition(positionParams),
    queryFn: () => fetchInventoryPosition(positionParams),
    staleTime: STALE.TWO_MIN,
  });

  const positions: InventoryPosition[] = positionPayload?.positions ?? [];
  const totalPositions = positionPayload?.total ?? 0;

  // ---- Variability panel state --------------------------------------------
  const [varPanelOpen, setVarPanelOpen] = useState(false);
  const [varClassFilter, setVarClassFilter] = useState("");
  const [varOffset, setVarOffset] = useState(0);
  const VAR_PAGE = 20;

  const { data: varSummary, isLoading: loadingVarSummary } = useQuery({
    queryKey: queryKeys.variabilitySummary({ abc_vol: "" }),
    queryFn: () => fetchVariabilitySummary({}),
    staleTime: STALE.FIVE_MIN,
    enabled: varPanelOpen,
  });

  const varDetailParams = useMemo(
    () => ({
      variability_class: varClassFilter || undefined,
      limit: VAR_PAGE,
      offset: varOffset,
      sort_by: "demand_cv",
      sort_dir: "desc",
    }),
    [varClassFilter, varOffset],
  );

  const { data: varDetail, isLoading: loadingVarDetail } = useQuery({
    queryKey: queryKeys.variabilityDetail(varDetailParams),
    queryFn: () => fetchVariabilityDetail(varDetailParams),
    staleTime: STALE.FIVE_MIN,
    enabled: varPanelOpen,
  });

  const varRows: VariabilityDetailRow[] = varDetail?.rows ?? [];

  // ---- Data fetching: Item detail -----------------------------------------
  const { data: detailPayload, isLoading: loadingDetail } = useQuery({
    queryKey: queryKeys.inventoryItemDetail({
      item: selectedRow?.item ?? "",
      location: selectedRow?.location ?? "",
    }),
    queryFn: () =>
      fetchInventoryItemDetail({
        item: selectedRow!.item,
        location: selectedRow!.location,
        months,
      }),
    staleTime: STALE.TWO_MIN,
    enabled: selectedRow !== null,
  });

  // ---- Pagination helpers -------------------------------------------------
  const totalPages = Math.max(1, Math.ceil(totalPositions / PAGE_SIZE));
  const currentPage = Math.floor(offset / PAGE_SIZE) + 1;

  const handlePrevPage = useCallback(() => {
    setOffset((prev) => Math.max(0, prev - PAGE_SIZE));
  }, []);

  const handleNextPage = useCallback(() => {
    setOffset((prev) =>
      prev + PAGE_SIZE < totalPositions ? prev + PAGE_SIZE : prev,
    );
  }, [totalPositions]);

  // ---- Sort handler -------------------------------------------------------
  const handleSort = useCallback(
    (col: SortCol) => {
      if (sortBy === col) {
        setSortDir((prev) => (prev === "asc" ? "desc" : "asc"));
      } else {
        setSortBy(col);
        setSortDir("desc");
      }
      setOffset(0);
    },
    [sortBy],
  );

  // ---- Filter change handlers (reset offset + deselect) -------------------
  const handleItemChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setItemFilter(e.target.value);
      setOffset(0);
      setSelectedRow(null);
    },
    [],
  );

  const handleLocationChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setLocationFilter(e.target.value);
      setOffset(0);
      setSelectedRow(null);
    },
    [],
  );

  const handleMonthsChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      setMonths(Number(e.target.value));
      setOffset(0);
    },
    [],
  );

  // ---- Row click handler --------------------------------------------------
  const handleRowClick = useCallback(
    (row: InventoryPosition) => {
      if (
        selectedRow?.item === row.item_no &&
        selectedRow?.location === row.loc
      ) {
        setSelectedRow(null);
      } else {
        setSelectedRow({ item: row.item_no, location: row.loc });
      }
    },
    [selectedRow],
  );

  // ---- Sort indicator helper ----------------------------------------------
  const sortIndicator = useCallback(
    (col: SortCol) => {
      if (sortBy !== col) return null;
      return sortDir === "asc" ? " \u25B2" : " \u25BC";
    },
    [sortBy, sortDir],
  );

  // ---- Trend chart data (memoised) ----------------------------------------
  const chartData = useMemo(() => {
    return trendData.map((pt) => ({
      month: pt.month,
      total_on_hand: pt.total_on_hand,
      total_on_order: pt.total_on_order,
      monthly_sales: pt.monthly_sales,
      avg_lead_time: pt.avg_lead_time,
      dos: pt.dos,
    }));
  }, [trendData]);

  // ---- Render -------------------------------------------------------------
  return (
    <section className="mt-4 space-y-4">
      {/* ---- KPI Cards --------------------------------------------------- */}
      {loadingKpis ? (
        <LoadingElement tabKey="inventory" message="Loading inventory KPIs..." />
      ) : kpiData ? (
        <div className="flex flex-wrap gap-3">
          <KpiCard
            label="Total On-Hand"
            value={formatCompactNumber(kpiData.total_on_hand)}
          />
          <KpiCard
            label="Total On-Order"
            value={formatCompactNumber(kpiData.total_on_order)}
          />
          <KpiCard
            label="Avg Lead Time"
            value={
              kpiData.avg_lead_time_days != null
                ? `${formatNumber(kpiData.avg_lead_time_days)} days`
                : "-"
            }
          />
          <KpiCard
            label="Days of Supply"
            value={kpiData.dos != null ? formatNumber(kpiData.dos) : "-"}
            sublabel="days"
            severity={
              kpiData.dos != null
                ? kpiData.dos >= 14 && kpiData.dos <= 60
                  ? "best"
                  : kpiData.dos < 7 || kpiData.dos > 90
                    ? "warning"
                    : "neutral"
                : undefined
            }
          />
          <KpiCard
            label="Weeks of Cover"
            value={kpiData.woc != null ? formatNumber(kpiData.woc) : "-"}
            sublabel="weeks"
            severity={
              kpiData.woc != null
                ? kpiData.woc >= 2 && kpiData.woc <= 8
                  ? "best"
                  : kpiData.woc < 1 || kpiData.woc > 12
                    ? "warning"
                    : "neutral"
                : undefined
            }
          />
          <KpiCard
            label="Inventory Turns"
            value={
              kpiData.inventory_turns != null
                ? formatNumber(kpiData.inventory_turns)
                : "-"
            }
            sublabel="/yr"
            severity={
              kpiData.inventory_turns != null
                ? kpiData.inventory_turns > 8
                  ? "best"
                  : kpiData.inventory_turns < 4
                    ? "warning"
                    : "neutral"
                : undefined
            }
          />
          <KpiCard
            label="LT Coverage"
            value={
              kpiData.lt_coverage != null
                ? `${formatNumber(kpiData.lt_coverage)}x`
                : "-"
            }
            severity={
              kpiData.lt_coverage != null
                ? kpiData.lt_coverage > 1.5
                  ? "best"
                  : kpiData.lt_coverage < 1.0
                    ? "warning"
                    : "neutral"
                : undefined
            }
          />
        </div>
      ) : null}

      {/* ---- Filter Controls --------------------------------------------- */}
      <Card className="animate-fade-in">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Package className="h-5 w-5" />
            <CardTitle className="text-base">Inventory Position</CardTitle>
          </div>
          <CardDescription>
            Browse inventory snapshots by item and location. Click a row to view
            snapshot history.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="flex flex-wrap items-end gap-3">
            <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Item
              <input
                className="h-9 w-44 rounded-md border border-input bg-background px-3 text-sm"
                placeholder="Filter by item..."
                value={itemFilter}
                onChange={handleItemChange}
              />
            </label>
            <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Location
              <input
                className="h-9 w-44 rounded-md border border-input bg-background px-3 text-sm"
                placeholder="Filter by location..."
                value={locationFilter}
                onChange={handleLocationChange}
              />
            </label>
            <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Months
              <select
                className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                value={months}
                onChange={handleMonthsChange}
              >
                <option value={3}>3 months</option>
                <option value={6}>6 months</option>
                <option value={12}>12 months</option>
                <option value={14}>14 months</option>
              </select>
            </label>
          </div>

          {/* ---- Inventory Trend Chart ----------------------------------- */}
          {loadingTrend ? (
            <LoadingElement
              tabKey="inventory"
              message="Loading trend data..."
            />
          ) : chartData.length > 0 ? (
            <div className="space-y-2">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">
                Monthly Inventory Trend
              </p>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={chartData} margin={CHART_MARGIN}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke={chartColors.grid}
                  />
                  <XAxis
                    dataKey="month"
                    tick={{ fontSize: 11, fill: chartColors.axis }}
                  />
                  <YAxis
                    yAxisId="left"
                    tick={{ fontSize: 11, fill: chartColors.axis }}
                    tickFormatter={(v: number) => formatCompactNumber(v)}
                  />
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    tick={{ fontSize: 11, fill: chartColors.axis }}
                    tickFormatter={(v: number) =>
                      `${Number(v).toFixed(0)}d`
                    }
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: chartColors.tooltip_bg,
                      borderColor: chartColors.tooltip_border,
                    }}
                    formatter={(value: number, name: string) => {
                      const labels: Record<string, string> = {
                        total_on_hand: "On Hand",
                        total_on_order: "On Order",
                        monthly_sales: "Monthly Sales",
                        avg_lead_time: "Avg Lead Time",
                        dos: "Days of Supply",
                      };
                      const formatted =
                        name === "avg_lead_time" || name === "dos"
                          ? `${Number(value).toFixed(1)} days`
                          : formatCompactNumber(value);
                      return [formatted, labels[name] ?? name];
                    }}
                  />
                  <Legend
                    wrapperStyle={{ fontSize: 11 }}
                    formatter={(value: string) => {
                      const labels: Record<string, string> = {
                        total_on_hand: "On Hand",
                        total_on_order: "On Order",
                        monthly_sales: "Monthly Sales",
                        avg_lead_time: "Avg Lead Time",
                        dos: "Days of Supply",
                      };
                      return labels[value] ?? value;
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="total_on_hand"
                    yAxisId="left"
                    stroke={trendColors[0]}
                    strokeWidth={2}
                    dot={CHART_DOT}
                    connectNulls
                  />
                  <Line
                    type="monotone"
                    dataKey="total_on_order"
                    yAxisId="left"
                    stroke={trendColors[1]}
                    strokeWidth={2}
                    dot={CHART_DOT}
                    connectNulls
                  />
                  <Line
                    type="monotone"
                    dataKey="monthly_sales"
                    yAxisId="left"
                    stroke={trendColors[3]}
                    strokeWidth={2}
                    dot={CHART_DOT}
                    connectNulls
                  />
                  <Line
                    type="monotone"
                    dataKey="dos"
                    yAxisId="right"
                    stroke={trendColors[4]}
                    strokeWidth={2.5}
                    dot={CHART_DOT}
                    connectNulls
                  />
                  <Line
                    type="monotone"
                    dataKey="avg_lead_time"
                    yAxisId="right"
                    stroke={trendColors[2]}
                    strokeWidth={2}
                    strokeDasharray="5 3"
                    dot={CHART_DOT}
                    connectNulls
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : null}

          {/* ---- Position Table ------------------------------------------- */}
          {loadingPosition ? (
            <LoadingElement
              tabKey="inventory"
              message="Loading position data..."
            />
          ) : positions.length > 0 ? (
            <div className="space-y-2">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">
                Position ({totalPositions.toLocaleString()} row
                {totalPositions !== 1 ? "s" : ""})
              </p>
              <div className="max-h-[420px] overflow-auto rounded-md border border-input">
                <Table>
                  <TableHeader>
                    <TableRow className="border-muted bg-muted/30">
                      {(
                        [
                          { col: "item_no" as SortCol, label: "Item" },
                          { col: "loc" as SortCol, label: "Location" },
                          {
                            col: "snapshot_date" as SortCol,
                            label: "Snapshot Date",
                          },
                          {
                            col: "qty_on_hand" as SortCol,
                            label: "On Hand",
                          },
                          {
                            col: "qty_on_hand_on_order" as SortCol,
                            label: "On Hand+Order",
                          },
                          {
                            col: "qty_on_order" as SortCol,
                            label: "On Order",
                          },
                          {
                            col: "lead_time_days" as SortCol,
                            label: "Lead Time",
                          },
                          {
                            col: "mtd_sales" as SortCol,
                            label: "MTD Sales",
                          },
                        ] as { col: SortCol; label: string }[]
                      ).map(({ col, label }) => (
                        <TableHead
                          key={col}
                          className={cn(
                            "text-xs cursor-pointer select-none hover:text-foreground",
                            col !== "item_no" && col !== "loc"
                              ? "text-right"
                              : "",
                          )}
                          onClick={() => handleSort(col)}
                        >
                          {label}
                          {sortIndicator(col)}
                        </TableHead>
                      ))}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {positions.map((row, idx) => {
                      const isSelected =
                        selectedRow?.item === row.item_no &&
                        selectedRow?.location === row.loc;
                      return (
                        <TableRow
                          key={`${row.item_no}-${row.loc}-${row.snapshot_date}-${idx}`}
                          className={cn(
                            "cursor-pointer",
                            isSelected
                              ? "bg-primary/10"
                              : "hover:bg-muted/30",
                          )}
                          onClick={() => handleRowClick(row)}
                        >
                          <TableCell className="text-sm font-medium">
                            {row.item_no}
                          </TableCell>
                          <TableCell className="text-sm">
                            {row.loc}
                          </TableCell>
                          <TableCell className="text-sm text-right tabular-nums">
                            {row.snapshot_date}
                          </TableCell>
                          <TableCell className="text-sm text-right tabular-nums">
                            {formatNumber(row.qty_on_hand)}
                          </TableCell>
                          <TableCell className="text-sm text-right tabular-nums">
                            {formatNumber(row.qty_on_hand_on_order)}
                          </TableCell>
                          <TableCell className="text-sm text-right tabular-nums">
                            {formatNumber(row.qty_on_order)}
                          </TableCell>
                          <TableCell className="text-sm text-right tabular-nums">
                            {row.lead_time_days != null
                              ? formatNumber(row.lead_time_days)
                              : "-"}
                          </TableCell>
                          <TableCell className="text-sm text-right tabular-nums">
                            {formatNumber(row.mtd_sales)}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </div>

              {/* Pagination controls */}
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>
                  Page {currentPage} of {totalPages}
                </span>
                <div className="flex items-center gap-2">
                  <button
                    className={cn(
                      "inline-flex items-center gap-1 rounded-md border border-input bg-background px-2 py-1 text-xs font-medium",
                      offset === 0
                        ? "opacity-50 cursor-not-allowed"
                        : "hover:bg-muted cursor-pointer",
                    )}
                    disabled={offset === 0}
                    onClick={handlePrevPage}
                  >
                    <ChevronLeft className="h-3 w-3" /> Prev
                  </button>
                  <button
                    className={cn(
                      "inline-flex items-center gap-1 rounded-md border border-input bg-background px-2 py-1 text-xs font-medium",
                      offset + PAGE_SIZE >= totalPositions
                        ? "opacity-50 cursor-not-allowed"
                        : "hover:bg-muted cursor-pointer",
                    )}
                    disabled={offset + PAGE_SIZE >= totalPositions}
                    onClick={handleNextPage}
                  >
                    Next <ChevronRight className="h-3 w-3" />
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">
              No inventory data found. Check that inventory data has been loaded
              into the database.
            </p>
          )}

          {/* ---- Item Detail Panel --------------------------------------- */}
          {selectedRow ? (
            <Card className="animate-fade-in border-primary/20">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">
                  Item Detail: {selectedRow.item} @ {selectedRow.location}
                </CardTitle>
                <CardDescription>
                  Snapshot history for this item-location pair.
                </CardDescription>
              </CardHeader>
              <CardContent>
                {loadingDetail ? (
                  <LoadingElement
                    tabKey="inventory"
                    message="Loading item detail..."
                  />
                ) : detailPayload?.snapshots &&
                  detailPayload.snapshots.length > 0 ? (
                  <div className="max-h-[300px] overflow-auto rounded-md border border-input">
                    <Table>
                      <TableHeader>
                        <TableRow className="border-muted bg-muted/30">
                          <TableHead className="text-xs">
                            Snapshot Date
                          </TableHead>
                          <TableHead className="text-xs text-right">
                            On Hand
                          </TableHead>
                          <TableHead className="text-xs text-right">
                            On Hand+Order
                          </TableHead>
                          <TableHead className="text-xs text-right">
                            On Order
                          </TableHead>
                          <TableHead className="text-xs text-right">
                            Lead Time
                          </TableHead>
                          <TableHead className="text-xs text-right">
                            MTD Sales
                          </TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {detailPayload.snapshots.map((snap, idx) => (
                          <TableRow
                            key={`${snap.snapshot_date}-${idx}`}
                            className="hover:bg-muted/30"
                          >
                            <TableCell className="text-sm tabular-nums">
                              {snap.snapshot_date}
                            </TableCell>
                            <TableCell className="text-sm text-right tabular-nums">
                              {formatNumber(snap.qty_on_hand)}
                            </TableCell>
                            <TableCell className="text-sm text-right tabular-nums">
                              {formatNumber(snap.qty_on_hand_on_order)}
                            </TableCell>
                            <TableCell className="text-sm text-right tabular-nums">
                              {formatNumber(snap.qty_on_order)}
                            </TableCell>
                            <TableCell className="text-sm text-right tabular-nums">
                              {snap.lead_time_days != null
                                ? formatNumber(snap.lead_time_days)
                                : "-"}
                            </TableCell>
                            <TableCell className="text-sm text-right tabular-nums">
                              {formatNumber(snap.mtd_sales)}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    No snapshot history available for this item-location pair.
                  </p>
                )}
              </CardContent>
            </Card>
          ) : null}
        </CardContent>
      </Card>

      {/* ---- Demand Variability Panel (IPfeature1) ----------------------- */}
      <Card>
        <CardHeader
          className="cursor-pointer select-none"
          onClick={() => setVarPanelOpen((o) => !o)}
        >
          <div className="flex items-center justify-between">
            <CardTitle className="text-base">Demand Variability Profile</CardTitle>
            {varPanelOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </div>
          <CardDescription>
            CV-based variability class per DFU — low / medium / high / lumpy
          </CardDescription>
        </CardHeader>

        {varPanelOpen && (
          <CardContent className="space-y-4">
            {/* Summary class breakdown */}
            {loadingVarSummary ? (
              <LoadingElement tabKey="variability" message="Loading variability summary..." />
            ) : varSummary ? (
              <div className="space-y-3">
                <div className="flex flex-wrap gap-3">
                  {(["low", "medium", "high", "lumpy"] as const).map((cls) => {
                    const colors: Record<string, string> = {
                      low: "text-green-600 dark:text-green-400",
                      medium: "text-yellow-600 dark:text-yellow-400",
                      high: "text-orange-600 dark:text-orange-400",
                      lumpy: "text-red-600 dark:text-red-400",
                    };
                    return (
                      <button
                        key={cls}
                        onClick={() => setVarClassFilter(varClassFilter === cls ? "" : cls)}
                        className={cn(
                          "flex flex-col items-center rounded-lg border px-4 py-2 text-sm transition-colors",
                          varClassFilter === cls
                            ? "border-primary bg-primary/10"
                            : "border-border hover:border-primary/50",
                        )}
                      >
                        <span className={cn("text-xl font-bold tabular-nums", colors[cls])}>
                          {varSummary.by_class[cls]}
                        </span>
                        <span className="capitalize text-muted-foreground">{cls}</span>
                      </button>
                    );
                  })}
                  <div className="flex flex-col items-center rounded-lg border border-border px-4 py-2 text-sm">
                    <span className="text-xl font-bold tabular-nums">
                      {varSummary.avg_cv != null ? Number(varSummary.avg_cv).toFixed(2) : "—"}
                    </span>
                    <span className="text-muted-foreground">Avg CV</span>
                  </div>
                  <div className="flex flex-col items-center rounded-lg border border-border px-4 py-2 text-sm">
                    <span className="text-xl font-bold tabular-nums">
                      {varSummary.avg_intermittency_ratio != null
                        ? `${(Number(varSummary.avg_intermittency_ratio) * 100).toFixed(1)}%`
                        : "—"}
                    </span>
                    <span className="text-muted-foreground">Avg Intermittency</span>
                  </div>
                </div>

                {/* CV percentile bar */}
                <div className="text-xs text-muted-foreground">
                  CV percentiles — p25:{" "}
                  {varSummary.cv_percentiles.p25 != null ? Number(varSummary.cv_percentiles.p25).toFixed(2) : "—"} · p50:{" "}
                  {varSummary.cv_percentiles.p50 != null ? Number(varSummary.cv_percentiles.p50).toFixed(2) : "—"} · p75:{" "}
                  {varSummary.cv_percentiles.p75 != null ? Number(varSummary.cv_percentiles.p75).toFixed(2) : "—"} · p95:{" "}
                  {varSummary.cv_percentiles.p95 != null ? Number(varSummary.cv_percentiles.p95).toFixed(2) : "—"}
                </div>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                No variability data. Run <code>make variability-compute</code> first.
              </p>
            )}

            {/* Detail table — top volatile DFUs */}
            {loadingVarDetail ? (
              <LoadingElement tabKey="variability-detail" message="Loading detail..." />
            ) : varRows.length > 0 ? (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium">
                    {varClassFilter
                      ? `${varClassFilter.charAt(0).toUpperCase() + varClassFilter.slice(1)} variability DFUs`
                      : "Top volatile DFUs"}{" "}
                    <span className="text-muted-foreground">({varDetail?.total ?? 0} total)</span>
                  </p>
                  {varClassFilter && (
                    <button
                      onClick={() => setVarClassFilter("")}
                      className="text-xs text-primary underline"
                    >
                      Clear filter
                    </button>
                  )}
                </div>
                <div className="overflow-x-auto rounded-md border">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Item</TableHead>
                        <TableHead>Loc</TableHead>
                        <TableHead>ABC</TableHead>
                        <TableHead className="text-right">CV</TableHead>
                        <TableHead className="text-right">Std</TableHead>
                        <TableHead className="text-right">Mean</TableHead>
                        <TableHead className="text-right">Intermittency</TableHead>
                        <TableHead>Class</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {varRows.map((r) => {
                        const tierColor: Record<string, string> = {
                          low: "bg-green-50 dark:bg-green-950/20",
                          medium: "bg-yellow-50 dark:bg-yellow-950/20",
                          high: "bg-orange-50 dark:bg-orange-950/20",
                          lumpy: "bg-red-50 dark:bg-red-950/20",
                        };
                        return (
                          <TableRow
                            key={`${r.item_no}-${r.loc}`}
                            className={tierColor[r.variability_class ?? ""] ?? ""}
                          >
                            <TableCell className="font-mono text-xs">{r.item_no}</TableCell>
                            <TableCell className="text-xs">{r.loc}</TableCell>
                            <TableCell className="text-xs">{r.abc_vol ?? "—"}</TableCell>
                            <TableCell className="text-right tabular-nums text-xs">
                              {r.demand_cv != null ? Number(r.demand_cv).toFixed(3) : "—"}
                            </TableCell>
                            <TableCell className="text-right tabular-nums text-xs">
                              {r.demand_std != null ? formatNumber(r.demand_std) : "—"}
                            </TableCell>
                            <TableCell className="text-right tabular-nums text-xs">
                              {r.demand_mean != null ? formatNumber(r.demand_mean) : "—"}
                            </TableCell>
                            <TableCell className="text-right tabular-nums text-xs">
                              {r.intermittency_ratio != null
                                ? `${(Number(r.intermittency_ratio) * 100).toFixed(1)}%`
                                : "—"}
                            </TableCell>
                            <TableCell>
                              <span
                                className={cn(
                                  "rounded px-1.5 py-0.5 text-xs font-medium capitalize",
                                  {
                                    "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300":
                                      r.variability_class === "low",
                                    "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300":
                                      r.variability_class === "medium",
                                    "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300":
                                      r.variability_class === "high",
                                    "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300":
                                      r.variability_class === "lumpy",
                                  },
                                )}
                              >
                                {r.variability_class ?? "—"}
                              </span>
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </div>

                {/* Pagination */}
                {(varDetail?.total ?? 0) > VAR_PAGE && (
                  <div className="flex items-center justify-between text-sm">
                    <button
                      disabled={varOffset === 0}
                      onClick={() => setVarOffset(Math.max(0, varOffset - VAR_PAGE))}
                      className="flex items-center gap-1 disabled:opacity-40"
                    >
                      <ChevronLeft className="h-4 w-4" /> Prev
                    </button>
                    <span className="text-muted-foreground">
                      {varOffset + 1}–{Math.min(varOffset + VAR_PAGE, varDetail?.total ?? 0)} of{" "}
                      {varDetail?.total}
                    </span>
                    <button
                      disabled={varOffset + VAR_PAGE >= (varDetail?.total ?? 0)}
                      onClick={() => setVarOffset(varOffset + VAR_PAGE)}
                      className="flex items-center gap-1 disabled:opacity-40"
                    >
                      Next <ChevronRight className="h-4 w-4" />
                    </button>
                  </div>
                )}
              </div>
            ) : varPanelOpen && !loadingVarDetail ? (
              <p className="text-sm text-muted-foreground">No variability data available.</p>
            ) : null}
          </CardContent>
        )}
      </Card>
    </section>
  );
}
