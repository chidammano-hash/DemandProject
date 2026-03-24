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
import { ItemDetailPanel } from "./ItemDetailPanel";

const PAGE_SIZE = 50;

type SortCol =
  | "item_id"
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
  // Table data
  positions: InventoryPosition[];
  totalPositions: number;
  isLoadingPosition: boolean;

  // Months window
  months: number;
  onMonthsChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;

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

  // Item detail
  detailSnapshots: DetailSnapshot[] | undefined;
  isLoadingDetail: boolean;
}

// Default visible columns (5 primary + 3 optional)
const DEFAULT_VISIBLE = new Set<SortCol>(["item_id", "loc", "qty_on_hand", "lead_time_days", "mtd_sales"]);

export function PositionTablePanel({
  positions,
  totalPositions,
  isLoadingPosition,
  months,
  onMonthsChange,
  offset,
  onPrevPage,
  onNextPage,
  sortBy,
  sortDir,
  onSort,
  selectedRow,
  onRowClick,
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
      return (
        <span className="ml-1 text-primary">
          {sortDir === "asc" ? "↑" : "↓"}
        </span>
      );
    },
    [sortBy, sortDir],
  );

  const COLUMNS: { col: SortCol; label: string; tooltip?: string }[] = [
    { col: "item_id", label: "Item" },
    { col: "loc", label: "Location" },
    { col: "snapshot_date", label: "Snapshot Date" },
    { col: "qty_on_hand", label: "On Hand", tooltip: "Quantity on hand at end of month" },
    { col: "qty_on_hand_on_order", label: "On Hand+Order", tooltip: "On-hand + on-order combined position" },
    { col: "qty_on_order", label: "On Order", tooltip: "Open purchase orders not yet received" },
    { col: "lead_time_days", label: "Lead Time", tooltip: "Average supplier lead time in days for this item-location" },
    { col: "mtd_sales", label: "MTD Sales", tooltip: "Month-to-date sales quantity" },
  ];

  return (
    <Card className="animate-fade-in">
      <CardHeader>
        <div className="flex items-center gap-2">
          <Package className="h-5 w-5" />
          <CardTitle className="text-base">Inventory Position</CardTitle>
          <select
            className="ml-auto h-7 rounded border border-input bg-background px-2 text-xs"
            value={months}
            onChange={onMonthsChange}
          >
            <option value={3}>3 mo</option>
            <option value={6}>6 mo</option>
            <option value={12}>12 mo</option>
            <option value={14}>14 mo</option>
          </select>
          <div className="relative">
            <button
              onClick={() => setColPickerOpen((v) => !v)}
              className="flex items-center gap-1 rounded border border-input bg-background px-2 py-1 text-xs hover:bg-muted"
              title="Show/hide columns"
            >
              <Columns className="h-3.5 w-3.5" /> Columns
              {COLUMNS.length - visibleCols.size > 0 && (
                <span className="ml-1 rounded bg-muted px-1 text-[10px] text-muted-foreground">
                  +{COLUMNS.length - visibleCols.size} hidden
                </span>
              )}
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
          Monitor stock levels, days of supply, and on-order positions across all item-locations.
          Rows are color-coded by urgency: red = critical stockout risk (&lt;7d), amber = at risk (7–14d), blue = excess (&gt;180d).
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Position table */}
        {isLoadingPosition ? (
          <LoadingElement
            tabKey="inventory"
            message="Loading position data..."
          />
        ) : (
          <div className="space-y-2">
            {positions.length > 0 && (
              <p className="text-xs uppercase tracking-wide text-muted-foreground">
                Position ({totalPositions.toLocaleString()} row
                {totalPositions !== 1 ? "s" : ""})
              </p>
            )}
            {positions.length > 0 && (
              <div className="flex gap-3 text-xs items-center mb-2">
                <span className="font-medium text-muted-foreground">Row color:</span>
                <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded bg-red-100 border border-red-300" /> Critical (&lt;7d DOS)</span>
                <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded bg-amber-100 border border-amber-300" /> At Risk (7–14d)</span>
                <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded bg-blue-100 border border-blue-300" /> Excess (&gt;180d)</span>
              </div>
            )}
            <div className="max-h-[420px] overflow-auto rounded-md border border-input">
              <Table>
                <TableHeader>
                  <TableRow className="border-muted bg-muted/30">
                    {COLUMNS.filter(({ col }) => visibleCols.has(col)).map(({ col, label, tooltip }) => (
                      <TableHead
                        key={col}
                        className={cn(
                          "text-xs cursor-pointer select-none hover:bg-muted/50 hover:text-foreground",
                          col !== "item_id" && col !== "loc" ? "text-right" : "",
                          tooltip ? "cursor-help" : "",
                        )}
                        title={tooltip}
                        onClick={() => onSort(col)}
                      >
                        {label}
                        {sortIndicator(col)}
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {positions.length === 0 ? (
                    <TableRow>
                      <TableCell
                        colSpan={visibleCols.size}
                        className="py-8 text-center text-sm text-muted-foreground"
                      >
                        No inventory positions found. Try adjusting the item or location filter.
                      </TableCell>
                    </TableRow>
                  ) : (
                    positions.map((row, idx) => {
                      const isSelected =
                        selectedRow?.item === row.item_id &&
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
                          key={`${row.item_id}-${row.loc}-${row.snapshot_date}-${idx}`}
                          className={cn(
                            "cursor-pointer",
                            isSelected
                              ? "bg-primary/10 ring-1 ring-primary/50"
                              : cn(dosColor, "hover:bg-muted/30 transition-colors"),
                          )}
                          onClick={() => onRowClick(row)}
                        >
                          {visibleCols.has("item_id") && (
                            <TableCell className="text-sm font-medium">{row.item_id}</TableCell>
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
                    })
                  )}
                </TableBody>
              </Table>
            </div>

            {/* Pagination — only shown when there are rows */}
            {positions.length > 0 && (
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>
                  Page {currentPage} of {totalPages}
                </span>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted-foreground">
                    {offset + 1}–{Math.min(offset + PAGE_SIZE, totalPositions)} of {totalPositions.toLocaleString()}
                  </span>
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
            )}
          </div>
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
