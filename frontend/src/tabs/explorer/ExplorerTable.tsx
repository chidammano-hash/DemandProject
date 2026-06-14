/**
 * Sortable, filterable data table with sticky first column for the Explorer tab.
 */
import {
  ArrowDownWideNarrow,
  ArrowUpWideNarrow,
  ChevronsUpDown,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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
import { titleCase, formatCell, isEmptyCell } from "@/lib/formatters";

import type { DomainPageRow, SortDir } from "./types";

export interface ExplorerTableProps {
  domain: string;
  visibleCols: string[];
  rows: DomainPageRow[];
  offset: number;
  loading: boolean;
  sortBy: string;
  sortDir: SortDir;
  columnFilters: Record<string, string>;
  columnSuggestions: Record<string, string[]>;
  onToggleSort: (column: string) => void;
  onColumnFilterChange: (column: string, value: string) => void;
}

export function ExplorerTable({
  domain,
  visibleCols,
  rows,
  offset,
  loading,
  sortBy,
  sortDir,
  columnFilters,
  columnSuggestions,
  onToggleSort,
  onColumnFilterChange,
}: ExplorerTableProps) {
  return (
    <div className="relative">
      {loading && (
        <LoadingElement
          message={`Querying ${titleCase(domain)}...`}
          overlay
          size="md"
        />
      )}
      <div className="max-h-[680px] overflow-x-scroll overflow-y-auto rounded-md border pb-2 [scrollbar-gutter:stable]">
        <Table
          style={{
            minWidth: `${Math.max(visibleCols.length * 260, 1800)}px`,
          }}
        >
          <TableHeader className="sticky top-0 z-20 bg-muted/80 backdrop-blur">
            <TableRow>
              {visibleCols.map((col, colIdx) => (
                <TableHead
                  key={col}
                  className={cn(
                    "min-w-[180px] bg-muted/70 align-top",
                    colIdx === 0 && "sticky left-0 z-10 bg-muted",
                  )}
                >
                  <Button
                    variant={sortBy === col ? "secondary" : "ghost"}
                    size="sm"
                    className="mb-1 h-7 w-full justify-between px-2"
                    onClick={() => onToggleSort(col)}
                  >
                    <span>{titleCase(col)}</span>
                    {sortBy === col ? (
                      sortDir === "asc" ? (
                        <ArrowUpWideNarrow className="h-3.5 w-3.5" />
                      ) : (
                        <ArrowDownWideNarrow className="h-3.5 w-3.5" />
                      )
                    ) : (
                      <ChevronsUpDown className="h-3.5 w-3.5" />
                    )}
                  </Button>
                  <Input
                    className="h-7 text-xs"
                    placeholder="Filter (=exact)"
                    list={`col-suggest-${domain}-${col}`}
                    value={columnFilters[col] || ""}
                    onChange={(e) => onColumnFilterChange(col, e.target.value)}
                  />
                  {(columnSuggestions[col]?.length ?? 0) > 0 && (
                    <datalist id={`col-suggest-${domain}-${col}`}>
                      {columnSuggestions[col].map((v) => (
                        <option key={v} value={v} />
                      ))}
                    </datalist>
                  )}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.length === 0 && !loading ? (
              <TableRow>
                <TableCell
                  colSpan={Math.max(visibleCols.length, 1)}
                  className="h-24 text-center text-muted-foreground"
                >
                  No records found. Try adjusting your filters or selecting a different domain.
                </TableCell>
              </TableRow>
            ) : (
              rows.map((row, idx) => (
                <TableRow key={`row-${offset + idx}`}>
                  {visibleCols.map((col, colIdx) => (
                    <TableCell
                      key={`${offset + idx}-${col}`}
                      className={cn(
                        "whitespace-nowrap max-w-[300px] truncate",
                        colIdx === 0 && "sticky left-0 z-10 bg-card",
                      )}
                      title={isEmptyCell(row[col]) ? "" : String(row[col])}
                    >
                      {formatCell(row[col])}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
