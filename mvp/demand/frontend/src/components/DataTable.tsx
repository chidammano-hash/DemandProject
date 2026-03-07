import { useCallback, useMemo, useRef, useState } from "react";
import {
  useReactTable,
  getCoreRowModel,
  flexRender,
  type ColumnDef,
  type ColumnResizeMode,
} from "@tanstack/react-table";
import { useVirtualizer } from "@tanstack/react-virtual";
import { ArrowDownWideNarrow, ArrowUpWideNarrow, ChevronsUpDown, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { titleCase, formatCell } from "@/lib/formatters";
import { downloadCsv } from "@/lib/export";

interface DataTableProps {
  data: Record<string, unknown>[];
  columns: string[];
  visibleColumns: Record<string, boolean>;
  onToggleColumn: (col: string, checked: boolean) => void;
  showFieldPanel: boolean;
  onToggleFieldPanel: () => void;
  sortBy: string;
  sortDir: "asc" | "desc";
  onSortChange: (col: string) => void;
  columnFilters: Record<string, string>;
  onColumnFilterChange: (col: string, value: string) => void;
  columnSuggestions: Record<string, string[]>;
  total: number;
  totalApproximate: boolean;
  offset: number;
  limit: number;
  onOffsetChange: (offset: number) => void;
  onLimitChange: (limit: number) => void;
  isLoading: boolean;
  domain: string;
  loadingElement?: React.ReactNode;
}

export function DataTable({
  data,
  columns: allColumns,
  visibleColumns,
  onToggleColumn,
  showFieldPanel,
  onToggleFieldPanel,
  sortBy,
  sortDir,
  onSortChange,
  columnFilters,
  onColumnFilterChange,
  columnSuggestions,
  total,
  totalApproximate,
  offset,
  limit,
  onOffsetChange,
  onLimitChange,
  isLoading,
  domain,
  loadingElement,
}: DataTableProps) {
  const tableContainerRef = useRef<HTMLDivElement>(null);
  const [columnResizeMode] = useState<ColumnResizeMode>("onChange");
  const [rowSelection, setRowSelection] = useState<Record<string, boolean>>({});

  const visibleCols = useMemo(
    () => allColumns.filter((col) => visibleColumns[col] !== false),
    [allColumns, visibleColumns],
  );

  const columnDefs = useMemo<ColumnDef<Record<string, unknown>>[]>(() => {
    const defs: ColumnDef<Record<string, unknown>>[] = [
      {
        id: "_select",
        size: 40,
        enableResizing: false,
        header: ({ table }) => (
          <Checkbox
            checked={table.getIsAllRowsSelected()}
            onCheckedChange={(v) => table.toggleAllRowsSelected(!!v)}
            aria-label="Select all"
          />
        ),
        cell: ({ row }) => (
          <Checkbox
            checked={row.getIsSelected()}
            onCheckedChange={(v) => row.toggleSelected(!!v)}
            aria-label={`Select row ${row.index + 1}`}
          />
        ),
      },
    ];
    for (const col of visibleCols) {
      defs.push({
        accessorKey: col,
        id: col,
        size: 200,
        minSize: 100,
        header: () => (
          <div className="space-y-1">
            <Button
              variant={sortBy === col ? "secondary" : "ghost"}
              size="sm"
              className="mb-1 h-7 w-full justify-between px-2"
              onClick={() => onSortChange(col)}
            >
              <span>{titleCase(col)}</span>
              {sortBy === col ? (
                sortDir === "asc" ? <ArrowUpWideNarrow className="h-3.5 w-3.5" /> : <ArrowDownWideNarrow className="h-3.5 w-3.5" />
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
          </div>
        ),
        cell: (info) => formatCell(info.getValue()),
      });
    }
    return defs;
  }, [visibleCols, sortBy, sortDir, columnFilters, columnSuggestions, domain, onSortChange, onColumnFilterChange]);

  const table = useReactTable({
    data,
    columns: columnDefs,
    state: { rowSelection },
    onRowSelectionChange: setRowSelection,
    getCoreRowModel: getCoreRowModel(),
    manualSorting: true,
    manualFiltering: true,
    manualPagination: true,
    columnResizeMode,
    enableRowSelection: true,
  });

  const { rows: tableRows } = table.getRowModel();

  const virtualizer = useVirtualizer({
    count: tableRows.length,
    getScrollElement: () => tableContainerRef.current,
    estimateSize: () => 40,
    overscan: 10,
  });

  const virtualItems = virtualizer.getVirtualItems();
  const totalSize = virtualizer.getTotalSize();

  const paddingTop = virtualItems.length > 0 ? virtualItems[0].start : 0;
  const paddingBottom = virtualItems.length > 0 ? totalSize - virtualItems[virtualItems.length - 1].end : 0;

  const start = total === 0 ? 0 : offset + 1;
  const end = Math.min(offset + limit, total);
  const selectedCount = Object.keys(rowSelection).length;

  const handleExport = useCallback(() => {
    const exportData = selectedCount > 0
      ? table.getSelectedRowModel().rows.map((r) => r.original)
      : data;
    const ts = new Date().toISOString().slice(0, 10);
    downloadCsv(exportData, `${domain}_export_${ts}.csv`, visibleCols);
  }, [selectedCount, table, data, domain, visibleCols]);

  return (
    <div className="space-y-3">
      {/* Column visibility panel */}
      {showFieldPanel && (
        <div className="rounded-md border p-2">
          <div className="flex gap-2 mb-2">
            <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={() => {
              allColumns.forEach((c) => onToggleColumn(c, true));
            }}>Select All</Button>
            <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={() => {
              allColumns.forEach((c) => onToggleColumn(c, false));
            }}>Deselect All</Button>
          </div>
          <div className="grid max-h-40 grid-cols-2 gap-2 overflow-y-auto overflow-x-hidden lg:grid-cols-3">
            {allColumns.map((col) => (
              <label key={col} className="flex items-center gap-2 text-sm">
                <Checkbox
                  checked={visibleColumns[col] !== false}
                  onCheckedChange={(checked) => onToggleColumn(col, checked === true)}
                />
                <span>{titleCase(col)}</span>
              </label>
            ))}
          </div>
        </div>
      )}

      {/* Table with virtualization */}
      <div className="relative">
        {isLoading && loadingElement}
        <div
          ref={tableContainerRef}
          className="max-h-[680px] overflow-auto rounded-md border pb-2 [scrollbar-gutter:stable]"
        >
          <Table style={{ minWidth: `${Math.max((visibleCols.length + 1) * 200, 1800)}px` }}>
            <TableHeader className="sticky top-0 z-20 bg-muted/80 backdrop-blur">
              {table.getHeaderGroups().map((headerGroup) => (
                <TableRow key={headerGroup.id}>
                  {headerGroup.headers.map((header, colIdx) => (
                    <TableHead
                      key={header.id}
                      scope="col"
                      className={cn(
                        "align-top",
                        colIdx === 0 && "w-[40px] sticky left-0 z-10 bg-muted",
                        colIdx === 1 && "sticky left-[40px] z-10 bg-muted",
                      )}
                      style={{ width: header.getSize() }}
                    >
                      {header.isPlaceholder ? null : flexRender(header.column.columnDef.header, header.getContext())}
                      {header.column.getCanResize() && (
                        <div
                          onMouseDown={header.getResizeHandler()}
                          onTouchStart={header.getResizeHandler()}
                          className={cn(
                            "absolute right-0 top-0 h-full w-1 cursor-col-resize select-none touch-none bg-transparent hover:bg-primary/30",
                            header.column.getIsResizing() && "bg-primary/50",
                          )}
                        />
                      )}
                    </TableHead>
                  ))}
                </TableRow>
              ))}
            </TableHeader>
            <TableBody>
              {tableRows.length === 0 && !isLoading ? (
                <TableRow>
                  <TableCell colSpan={columnDefs.length} className="h-24 text-center text-muted-foreground">
                    No records
                  </TableCell>
                </TableRow>
              ) : (
                <>
                  {paddingTop > 0 && (
                    <tr><td style={{ height: `${paddingTop}px` }} /></tr>
                  )}
                  {virtualItems.map((virtualRow) => {
                    const row = tableRows[virtualRow.index];
                    return (
                      <TableRow
                        key={row.id}
                        data-index={virtualRow.index}
                        className={cn(row.getIsSelected() && "bg-primary/5")}
                      >
                        {row.getVisibleCells().map((cell, colIdx) => (
                          <TableCell
                            key={cell.id}
                            className={cn(
                              "whitespace-nowrap max-w-[300px] truncate",
                              colIdx === 0 && "sticky left-0 z-10 bg-card w-[40px]",
                              colIdx === 1 && "sticky left-[40px] z-10 bg-card",
                            )}
                            title={cell.getValue() != null ? String(cell.getValue()) : ""}
                          >
                            {flexRender(cell.column.columnDef.cell, cell.getContext())}
                          </TableCell>
                        ))}
                      </TableRow>
                    );
                  })}
                  {paddingBottom > 0 && (
                    <tr><td style={{ height: `${paddingBottom}px` }} /></tr>
                  )}
                </>
              )}
            </TableBody>
          </Table>
        </div>
      </div>

      {/* Pagination + export */}
      <div className="flex items-center justify-between gap-2 text-sm">
        <div className="flex items-center gap-3">
          <span className="text-muted-foreground">
            Showing {start}-{end} of {totalApproximate ? `${(total - 1).toLocaleString()}+` : total.toLocaleString()}
            {total > 0 && (
              <span className="ml-2 tabular-nums">
                (Page {Math.floor(offset / limit) + 1} of {totalApproximate ? `${Math.ceil((total - 1) / limit)}+` : Math.ceil(total / limit)})
              </span>
            )}
          </span>
          {selectedCount > 0 && (
            <span className="text-xs text-primary font-medium">{selectedCount} selected</span>
          )}
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={handleExport} title="Export to CSV">
            <Download className="mr-1 h-3.5 w-3.5" /> CSV
          </Button>
          <Button variant="outline" size="sm" disabled={offset === 0} onClick={() => onOffsetChange(Math.max(0, offset - limit))}>
            Previous
          </Button>
          <Button
            variant="outline"
            size="sm"
            disabled={offset + limit >= total}
            onClick={() => onOffsetChange(offset + limit)}
          >
            Next
          </Button>
        </div>
      </div>
    </div>
  );
}
