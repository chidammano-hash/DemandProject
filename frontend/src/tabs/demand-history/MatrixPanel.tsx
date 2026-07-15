import { useState, useMemo, useCallback } from "react";
import { Grid3x3 } from "lucide-react";
import { useMatrix } from "@/api/queries/demand-history";
import { DemandReferencePanel } from "@/components/DemandReferencePanel";
import { TableSkeleton } from "@/components/Skeleton";
import { formatInt } from "@/lib/formatters";
import { interactiveRowProps } from "@/lib/interactiveRow";
import type { MatrixDim, MatrixMetric } from "@/api/queries/demand-history";

/** 6-stop heatmap from gray (zero) through cool blue to deep indigo. */
function cellStyle(value: number, max: number): { bg: string; text: string } {
  if (max === 0 || value === 0) return { bg: "", text: "text-muted-foreground/50" };
  const pct = Math.min(value / max, 1);
  if (pct > 0.85) return { bg: "bg-indigo-700", text: "text-white font-semibold" };
  if (pct > 0.65) return { bg: "bg-blue-600", text: "text-white" };
  if (pct > 0.45) return { bg: "bg-blue-400", text: "text-white" };
  if (pct > 0.25) return { bg: "bg-blue-200 dark:bg-blue-800", text: "text-foreground" };
  if (pct > 0.1)  return { bg: "bg-blue-100 dark:bg-blue-900/40", text: "text-foreground/80" };
  return { bg: "bg-blue-50 dark:bg-blue-950/30", text: "text-muted-foreground" };
}

const METRIC_LABELS: Record<MatrixMetric, string> = {
  demand_qty: "Demand Qty",
  sales_qty: "Sales Qty",
  fill_rate: "Fill Rate",
};

export function MatrixPanel() {
  const [rowDim, setRowDim] = useState<MatrixDim>("item");
  const [colDim, setColDim] = useState<MatrixDim>("location");
  const [metric, setMetric] = useState<MatrixMetric>("demand_qty");
  const [months, setMonths] = useState(6);

  // Reference panel state
  const [refOpen, setRefOpen] = useState(false);
  const [refItem, setRefItem] = useState("");
  const [refLoc, setRefLoc] = useState("");

  const { data, isLoading, isError } = useMatrix(rowDim, colDim, metric, months);

  const maxVal = useMemo(() => {
    if (!data?.cells) return 0;
    let m = 0;
    for (const row of data.cells) {
      for (const v of row) {
        if (v > m) m = v;
      }
    }
    return m;
  }, [data?.cells]);

  const handleCellClick = useCallback(
    (rowIdx: number, colIdx: number) => {
      if (!data) return;
      const rowKey = data.rows[rowIdx];
      const colKey = data.cols[colIdx];

      // Determine item and loc from dimensions
      let item = "";
      let loc = "";
      if (rowDim === "item") item = rowKey;
      if (colDim === "item") item = colKey;
      if (rowDim === "location") loc = rowKey;
      if (colDim === "location") loc = colKey;

      if (item && loc) {
        setRefItem(item);
        setRefLoc(loc);
        setRefOpen(true);
      }
    },
    [data, rowDim, colDim],
  );

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex items-center gap-4 flex-wrap">
        <label className="flex items-center gap-1.5 text-sm">
          <span className="text-muted-foreground">Rows:</span>
          <select
            value={rowDim}
            onChange={(e) => setRowDim(e.target.value as MatrixDim)}
            className="px-2 py-1 text-sm border rounded bg-white"
          >
            <option value="item">Item</option>
            <option value="location">Location</option>
            <option value="customer">Customer</option>
          </select>
        </label>
        <label className="flex items-center gap-1.5 text-sm">
          <span className="text-muted-foreground">Cols:</span>
          <select
            value={colDim}
            onChange={(e) => setColDim(e.target.value as MatrixDim)}
            className="px-2 py-1 text-sm border rounded bg-white"
          >
            <option value="item">Item</option>
            <option value="location">Location</option>
            <option value="customer">Customer</option>
          </select>
        </label>
        <label className="flex items-center gap-1.5 text-sm">
          <span className="text-muted-foreground">Metric:</span>
          <select
            value={metric}
            onChange={(e) => setMetric(e.target.value as MatrixMetric)}
            className="px-2 py-1 text-sm border rounded bg-white"
          >
            <option value="demand_qty">Demand Qty</option>
            <option value="sales_qty">Sales Qty</option>
            <option value="fill_rate">Fill Rate</option>
          </select>
        </label>
        <label className="flex items-center gap-1.5 text-sm">
          <span className="text-muted-foreground">Months:</span>
          <select
            value={months}
            onChange={(e) => setMonths(Number(e.target.value))}
            className="px-2 py-1 text-sm border rounded bg-white"
          >
            <option value={3}>3</option>
            <option value={6}>6</option>
            <option value={12}>12</option>
            <option value={24}>24</option>
          </select>
        </label>
      </div>

      {isLoading && <TableSkeleton rows={10} cols={8} />}
      {isError && <div className="text-center text-red-500 text-sm py-10">Failed to load data</div>}

      {!isLoading && !isError && !data && (
        <div className="flex flex-col items-center py-16 text-muted-foreground">
          <Grid3x3 className="h-12 w-12 mb-3 opacity-30" />
          <p className="text-sm font-medium">No matrix data</p>
          <p className="text-xs mt-1">Adjust dimensions or time range to load the pivot grid</p>
        </div>
      )}

      {data && (
        <div className="overflow-auto max-h-[600px] border rounded-lg">
          {/* Legend */}
          <div className="sticky top-0 z-20 bg-white border-b px-3 py-1.5 flex items-center gap-3 text-[10px] text-muted-foreground">
            <span className="font-medium">{METRIC_LABELS[metric]}:</span>
            <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-sm bg-blue-50 dark:bg-blue-950/30 border" /> Low</span>
            <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-sm bg-blue-200" /> Medium</span>
            <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-sm bg-blue-600" /> High</span>
            <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-sm bg-indigo-700" /> Max</span>
            <span className="ml-auto text-muted-foreground">{data.rows.length} rows x {data.cols.length} cols</span>
          </div>
          <table className="text-xs w-full">
            <thead className="sticky top-8 bg-white z-10">
              <tr>
                <th className="px-3 py-2 text-left font-medium text-muted-foreground border-b min-w-[120px]">
                  {rowDim} / {colDim}
                </th>
                {data.cols.map((col) => (
                  <th
                    key={col}
                    className="px-2 py-2 text-center font-medium text-muted-foreground border-b min-w-[70px]"
                    title={data.col_labels[col] || col}
                  >
                    <span className="truncate block max-w-[80px]">
                      {data.col_labels[col] || col}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.rows.map((row, ri) => {
                const rowTotal = data.cells[ri]?.reduce((a, b) => a + b, 0) ?? 0;
                return (
                  <tr key={row} className="hover:bg-muted dark:hover:bg-muted">
                    <td
                      className="px-3 py-1.5 font-medium text-foreground/80 border-b truncate max-w-[140px]"
                      title={`${data.row_labels[row] || row} — Total: ${formatInt(rowTotal)}`}
                    >
                      {data.row_labels[row] || row}
                    </td>
                    {data.cells[ri]?.map((val, ci) => {
                      const style = cellStyle(val, maxVal);
                      return (
                        <td
                          key={ci}
                          {...interactiveRowProps(() => handleCellClick(ri, ci))}
                          className={`px-2 py-1.5 text-center border-b cursor-pointer tabular-nums transition-colors ${style.bg} ${style.text}`}
                          title={`${data.row_labels[row] || row} x ${data.col_labels[data.cols[ci]] || data.cols[ci]}: ${formatInt(val)}`}
                        >
                          {val > 0 ? formatInt(val) : "—"}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Reference panel for drill-through */}
      <DemandReferencePanel
        itemId={refItem}
        loc={refLoc}
        open={refOpen}
        onClose={() => setRefOpen(false)}
      />
    </div>
  );
}