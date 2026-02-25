import { useCallback, useMemo, useState } from "react";
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
import { Package, ChevronLeft, ChevronRight } from "lucide-react";

import {
  queryKeys,
  STALE,
  fetchInventoryPosition,
  fetchInventoryKpis,
  fetchInventoryTrend,
  fetchInventoryItemDetail,
} from "@/api/queries";
import type {
  Theme,
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
import { CHART_COLORS, TREND_COLORS_BY_THEME } from "@/constants/colors";
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
type InventoryTabProps = {
  theme: Theme;
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function InventoryTab({ theme }: InventoryTabProps) {
  const trendColors = TREND_COLORS_BY_THEME[theme];

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
      avg_on_hand: pt.avg_on_hand,
      avg_on_order: pt.avg_on_order,
      avg_lead_time: pt.avg_lead_time,
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
            label="Distinct Items"
            value={formatCompactNumber(kpiData.distinct_items)}
          />
          <KpiCard
            label="Distinct Locations"
            value={formatCompactNumber(kpiData.distinct_locations)}
          />
          <KpiCard
            label="Snapshot Count"
            value={formatCompactNumber(kpiData.snapshot_count)}
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
                    stroke={CHART_COLORS[theme].grid}
                  />
                  <XAxis
                    dataKey="month"
                    tick={{ fontSize: 11, fill: CHART_COLORS[theme].axis }}
                  />
                  <YAxis
                    yAxisId="left"
                    tick={{ fontSize: 11, fill: CHART_COLORS[theme].axis }}
                    tickFormatter={(v: number) => formatCompactNumber(v)}
                  />
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    tick={{ fontSize: 11, fill: CHART_COLORS[theme].axis }}
                    tickFormatter={(v: number) =>
                      `${Number(v).toFixed(0)}d`
                    }
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: CHART_COLORS[theme].tooltip_bg,
                      borderColor: CHART_COLORS[theme].tooltip_border,
                    }}
                    formatter={(value: number, name: string) => [
                      name === "avg_lead_time"
                        ? `${Number(value).toFixed(1)} days`
                        : formatCompactNumber(value),
                      name === "avg_on_hand"
                        ? "Avg On Hand"
                        : name === "avg_on_order"
                          ? "Avg On Order"
                          : "Avg Lead Time",
                    ]}
                  />
                  <Legend
                    wrapperStyle={{ fontSize: 11 }}
                    formatter={(value: string) =>
                      value === "avg_on_hand"
                        ? "Avg On Hand"
                        : value === "avg_on_order"
                          ? "Avg On Order"
                          : "Avg Lead Time"
                    }
                  />
                  <Line
                    type="monotone"
                    dataKey="avg_on_hand"
                    yAxisId="left"
                    stroke={trendColors[0]}
                    strokeWidth={2}
                    dot={CHART_DOT}
                    connectNulls
                  />
                  <Line
                    type="monotone"
                    dataKey="avg_on_order"
                    yAxisId="left"
                    stroke={trendColors[1]}
                    strokeWidth={2}
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
    </section>
  );
}
