/**
 * DetailTablePanel — DFU-level event detail table with sorting and pagination.
 */
import { ChevronLeft, ChevronRight } from "lucide-react";
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
import { formatNumber, formatCompactNumber } from "@/lib/formatters";
import type { InvBacktestDetailRow } from "@/types";
import { PAGE_SIZE, type DetailSortCol } from "./invBacktestShared";

export function DetailTablePanel({
  detailData,
  loadingDetail,
  eventType,
  detailOffset,
  detailSort,
  detailDir,
  onEventTypeChange,
  onDetailSort,
  onOffsetChange,
}: {
  detailData: { rows: InvBacktestDetailRow[]; total: number } | undefined;
  loadingDetail: boolean;
  eventType: string;
  detailOffset: number;
  detailSort: DetailSortCol;
  detailDir: "asc" | "desc";
  onEventTypeChange: (type: string) => void;
  onDetailSort: (col: DetailSortCol) => void;
  onOffsetChange: (offset: number) => void;
}) {
  const totalDetailPages = Math.max(1, Math.ceil((detailData?.total ?? 0) / PAGE_SIZE));
  const currentDetailPage = Math.floor(detailOffset / PAGE_SIZE) + 1;

  const sortIndicator = (col: DetailSortCol) => {
    if (detailSort !== col) return null;
    return detailDir === "asc" ? " \u25B2" : " \u25BC";
  };

  const COLUMNS: { col: DetailSortCol; label: string }[] = [
    { col: "item_id", label: "Item" },
    { col: "loc", label: "Location" },
    { col: "month_start", label: "Month" },
    { col: "model_id", label: "Model" },
    { col: "forecast", label: "Forecast" },
    { col: "actual_demand", label: "Actual" },
    { col: "eom_qty_on_hand", label: "EOM On-Hand" },
    { col: "dos", label: "DOS" },
    { col: "forecast_error", label: "Error" },
  ];

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-3">
        <p className="text-xs uppercase tracking-wide text-muted-foreground">
          DFU-Level Events ({formatCompactNumber(detailData?.total ?? 0)} rows)
        </p>
        <select
          className="h-7 rounded border border-input bg-background px-2 text-xs"
          value={eventType}
          onChange={(e) => onEventTypeChange(e.target.value)}
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
                  {COLUMNS.map(({ col, label }) => (
                    <TableHead
                      key={col}
                      className={cn(
                        "text-xs cursor-pointer select-none hover:text-foreground",
                        col !== "item_id" && col !== "loc" && col !== "model_id"
                          ? "text-right"
                          : "",
                      )}
                      onClick={() => onDetailSort(col)}
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
                    key={`${row.item_id}-${row.loc}-${row.month}-${row.model_id}-${idx}`}
                    className={cn(
                      row.event_type === "stockout"
                        ? "bg-red-500/5"
                        : row.event_type === "excess"
                          ? "bg-amber-500/5"
                          : "hover:bg-muted/30",
                    )}
                  >
                    <TableCell className="text-sm font-medium">{row.item_id}</TableCell>
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
                        "bg-muted text-muted-foreground",
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
                onClick={() => onOffsetChange(Math.max(0, detailOffset - PAGE_SIZE))}
              >
                <ChevronLeft className="h-3 w-3" /> Prev
              </button>
              <button
                className={cn(
                  "inline-flex items-center gap-1 rounded-md border border-input bg-background px-2 py-1 text-xs font-medium",
                  detailOffset + PAGE_SIZE >= (detailData?.total ?? 0) ? "opacity-50 cursor-not-allowed" : "hover:bg-muted cursor-pointer",
                )}
                disabled={detailOffset + PAGE_SIZE >= (detailData?.total ?? 0)}
                onClick={() => onOffsetChange(detailOffset + PAGE_SIZE)}
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
  );
}
