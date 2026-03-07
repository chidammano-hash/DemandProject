import { useCallback, useState } from "react";
import { ChevronLeft, ChevronRight, Package, Columns } from "lucide-react";

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
import { LoadingElement } from "@/components/LoadingElement";
import { cn } from "@/lib/utils";
import { formatNumber } from "@/lib/formatters";
import type { InventoryPosition } from "@/types";
import { TrendChartPanel } from "./TrendChartPanel";
import { ItemDetailPanel } from "./ItemDetailPanel";
import type { InventoryTrendPoint } from "@/types";

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

interface DetailSnapshot {
  snapshot_date: string;
  qty_on_hand: number;
  qty_on_hand_on_order: number;
  qty_on_order: number;
  lead_time_days: number | null;
  mtd_sales: number;
}

interface PositionTablePanelProps {
  // Filter state
  itemFilter: string;
  locationFilter: string;
  months: number;
  onItemChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onLocationChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onMonthsChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;

  // Table data
  positions: InventoryPosition[];
  totalPositions: number;
  isLoadingPosition: boolean;

  // Pagination state
  offset: number;
  onPrevPage: () => void;
  onNextPage: () => void;

  // Sort state
  sortBy: SortCol;
  sortDir: "asc" | "desc";
  onSort: (col: SortCol) => void;

  // Row selection
  selectedRow: { item: string; location: string } | null;
  onRowClick: (row: InventoryPosition) => void;

  // Trend chart
  trendData: InventoryTrendPoint[];
  isLoadingTrend: boolean;

  // Item detail
  detailSnapshots: DetailSnapshot[] | undefined;
  isLoadingDetail: boolean;
}

// Default visible columns (5 primary + 3 optional)
const DEFAULT_VISIBLE = new Set<SortCol>(["item_no", "loc", "qty_on_hand", "lead_time_days", "mtd_sales"]);

export function PositionTablePanel({
  itemFilter,
  locationFilter,
  months,
  onItemChange,
  onLocationChange,
  onMonthsChange,
  positions,
  totalPositions,
  isLoadingPosition,
  offset,
  onPrevPage,
  onNextPage,
  sortBy,
  sortDir,
  onSort,
  selectedRow,
  onRowClick,
  trendData,
  isLoadingTrend,
  detailSnapshots,
  isLoadingDetail,
}: PositionTablePanelProps) {
  const totalPages = Math.max(1, Math.ceil(totalPositions / PAGE_SIZE));
  const currentPage = Math.floor(offset / PAGE_SIZE) + 1;
  const [visibleCols, setVisibleCols] = useState<Set<SortCol>>(DEFAULT_VISIBLE);
  const [colPickerOpen, setColPickerOpen] = useState(false);

  const toggleCol = useCallback((col: SortCol) => {
    setVisibleCols((prev) => {
      const next = new Set(prev);
      if (next.has(col)) { next.delete(col); } else { next.add(col); }
      return next;
    });
  }, []);

  const sortIndicator = useCallback(
    (col: SortCol) => {
      if (sortBy !== col) return null;
      return sortDir === "asc" ? " \u25B2" : " \u25BC";
    },
    [sortBy, sortDir],
  );

  const COLUMNS: { col: SortCol; label: string }[] = [
    { col: "item_no", label: "Item" },
    { col: "loc", label: "Location" },
    { col: "snapshot_date", label: "Snapshot Date" },
    { col: "qty_on_hand", label: "On Hand" },
    { col: "qty_on_hand_on_order", label: "On Hand+Order" },
    { col: "qty_on_order", label: "On Order" },
    { col: "lead_time_days", label: "Lead Time" },
    { col: "mtd_sales", label: "MTD Sales" },
  ];

  return (
    <Card className="animate-fade-in">
      <CardHeader>
        <div className="flex items-center gap-2">
          <Package className="h-5 w-5" />
          <CardTitle className="text-base">Inventory Position</CardTitle>
          <div className="relative ml-auto">
            <button
              onClick={() => setColPickerOpen((v) => !v)}
              className="flex items-center gap-1 rounded border border-input bg-background px-2 py-1 text-xs hover:bg-muted"
              title="Choose columns"
            >
              <Columns className="h-3.5 w-3.5" /> Columns
            </button>
            {colPickerOpen && (
              <div className="absolute right-0 top-full z-20 mt-1 w-48 rounded-md border bg-card shadow-lg p-2 space-y-1">
                {COLUMNS.map(({ col, label }) => (
                  <label key={col} className="flex items-center gap-2 text-xs cursor-pointer hover:bg-muted px-1 py-0.5 rounded">
                    <input
                      type="checkbox"
                      checked={visibleCols.has(col)}
                      onChange={() => toggleCol(col)}
                      className="h-3 w-3"
                    />
                    {label}
                  </label>
                ))}
                <button
                  onClick={() => setColPickerOpen(false)}
                  className="mt-1 w-full text-center text-[10px] text-muted-foreground hover:text-foreground"
                >
                  Close
                </button>
              </div>
            )}
          </div>
        </div>
        <CardDescription>
          Browse inventory snapshots by item and location. Click a row to view
          snapshot history.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* Filter controls */}
        <div className="flex flex-wrap items-end gap-3">
          <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Item
            <input
              className="h-9 w-44 rounded-md border border-input bg-background px-3 text-sm"
              placeholder="Filter by item..."
              value={itemFilter}
              onChange={onItemChange}
            />
          </label>
          <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Location
            <input
              className="h-9 w-44 rounded-md border border-input bg-background px-3 text-sm"
              placeholder="Filter by location..."
              value={locationFilter}
              onChange={onLocationChange}
            />
          </label>
          <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Months
            <select
              className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
              value={months}
              onChange={onMonthsChange}
            >
              <option value={3}>3 months</option>
              <option value={6}>6 months</option>
              <option value={12}>12 months</option>
              <option value={14}>14 months</option>
            </select>
          </label>
        </div>

        {/* Trend chart */}
        <TrendChartPanel trendData={trendData} isLoading={isLoadingTrend} />

        {/* Position table */}
        {isLoadingPosition ? (
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
                    {COLUMNS.filter(({ col }) => visibleCols.has(col)).map(({ col, label }) => (
                      <TableHead
                        key={col}
                        className={cn(
                          "text-xs cursor-pointer select-none hover:text-foreground",
                          col !== "item_no" && col !== "loc"
                            ? "text-right"
                            : "",
                        )}
                        onClick={() => onSort(col)}
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
                    // PL-016: compute approximate DOS and apply threshold color
                    const avgDailySales = row.mtd_sales > 0 ? row.mtd_sales / 30 : null;
                    const dos = avgDailySales != null ? row.qty_on_hand / avgDailySales : null;
                    const dosColor = dos == null ? "" :
                      dos < 7   ? "bg-red-50 dark:bg-red-900/20" :
                      dos < 14  ? "bg-amber-50 dark:bg-amber-900/20" :
                      dos > 180 ? "bg-blue-50 dark:bg-blue-900/20" :
                      "";
                    return (
                      <TableRow
                        key={`${row.item_no}-${row.loc}-${row.snapshot_date}-${idx}`}
                        className={cn(
                          "cursor-pointer",
                          isSelected ? "bg-primary/10" : cn(dosColor, "hover:bg-muted/30"),
                        )}
                        onClick={() => onRowClick(row)}
                      >
                        {visibleCols.has("item_no") && (
                          <TableCell className="text-sm font-medium">{row.item_no}</TableCell>
                        )}
                        {visibleCols.has("loc") && (
                          <TableCell className="text-sm">{row.loc}</TableCell>
                        )}
                        {visibleCols.has("snapshot_date") && (
                          <TableCell className="text-sm text-right tabular-nums">{row.snapshot_date}</TableCell>
                        )}
                        {visibleCols.has("qty_on_hand") && (
                          <TableCell className="text-sm text-right tabular-nums">{formatNumber(row.qty_on_hand)}</TableCell>
                        )}
                        {visibleCols.has("qty_on_hand_on_order") && (
                          <TableCell className="text-sm text-right tabular-nums">{formatNumber(row.qty_on_hand_on_order)}</TableCell>
                        )}
                        {visibleCols.has("qty_on_order") && (
                          <TableCell className="text-sm text-right tabular-nums">{formatNumber(row.qty_on_order)}</TableCell>
                        )}
                        {visibleCols.has("lead_time_days") && (
                          <TableCell className="text-sm text-right tabular-nums">
                            {row.lead_time_days != null ? formatNumber(row.lead_time_days) : "-"}
                          </TableCell>
                        )}
                        {visibleCols.has("mtd_sales") && (
                          <TableCell className="text-sm text-right tabular-nums">{formatNumber(row.mtd_sales)}</TableCell>
                        )}
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>

            {/* Pagination */}
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
                  onClick={onPrevPage}
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
                  onClick={onNextPage}
                >
                  Next <ChevronRight className="h-3 w-3" />
                </button>
              </div>
            </div>
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">
            No inventory data found
            {itemFilter || locationFilter
              ? " matching current filters. Try broadening the item or location filter."
              : ". Check that inventory data has been loaded into the database."}
          </p>
        )}

        {/* Item detail panel */}
        {selectedRow && (
          <ItemDetailPanel
            selectedRow={selectedRow}
            snapshots={detailSnapshots}
            isLoading={isLoadingDetail}
          />
        )}
      </CardContent>
    </Card>
  );
}
