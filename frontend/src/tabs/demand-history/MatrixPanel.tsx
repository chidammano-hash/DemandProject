import { useState, useMemo, useCallback } from "react";
import { useMatrix } from "@/api/queries/demand-history";
import { DemandReferencePanel } from "@/components/DemandReferencePanel";
import type { MatrixDim, MatrixMetric } from "@/api/queries/demand-history";

function cellColor(value: number, max: number): string {
  if (max === 0) return "bg-gray-50 dark:bg-gray-800";
  const intensity = Math.min(value / max, 1);
  if (intensity > 0.75) return "bg-blue-600 text-white";
  if (intensity > 0.5) return "bg-blue-400 text-white";
  if (intensity > 0.25) return "bg-blue-200 dark:bg-blue-800 text-gray-900 dark:text-gray-100";
  if (intensity > 0) return "bg-blue-50 dark:bg-blue-900/30";
  return "bg-gray-50 dark:bg-gray-800 text-gray-400";
}

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
          <span className="text-gray-500">Rows:</span>
          <select
            value={rowDim}
            onChange={(e) => setRowDim(e.target.value as MatrixDim)}
            className="px-2 py-1 text-sm border dark:border-gray-700 rounded bg-white dark:bg-gray-900"
          >
            <option value="item">Item</option>
            <option value="location">Location</option>
            <option value="customer">Customer</option>
          </select>
        </label>
        <label className="flex items-center gap-1.5 text-sm">
          <span className="text-gray-500">Cols:</span>
          <select
            value={colDim}
            onChange={(e) => setColDim(e.target.value as MatrixDim)}
            className="px-2 py-1 text-sm border dark:border-gray-700 rounded bg-white dark:bg-gray-900"
          >
            <option value="item">Item</option>
            <option value="location">Location</option>
            <option value="customer">Customer</option>
          </select>
        </label>
        <label className="flex items-center gap-1.5 text-sm">
          <span className="text-gray-500">Metric:</span>
          <select
            value={metric}
            onChange={(e) => setMetric(e.target.value as MatrixMetric)}
            className="px-2 py-1 text-sm border dark:border-gray-700 rounded bg-white dark:bg-gray-900"
          >
            <option value="demand_qty">Demand Qty</option>
            <option value="sales_qty">Sales Qty</option>
            <option value="fill_rate">Fill Rate</option>
          </select>
        </label>
        <label className="flex items-center gap-1.5 text-sm">
          <span className="text-gray-500">Months:</span>
          <select
            value={months}
            onChange={(e) => setMonths(Number(e.target.value))}
            className="px-2 py-1 text-sm border dark:border-gray-700 rounded bg-white dark:bg-gray-900"
          >
            <option value={3}>3</option>
            <option value={6}>6</option>
            <option value={12}>12</option>
            <option value={24}>24</option>
          </select>
        </label>
      </div>

      {isLoading && <div className="text-center text-gray-500 text-sm py-10">Loading...</div>}
      {isError && <div className="text-center text-red-500 text-sm py-10">Failed to load data</div>}

      {data && (
        <div className="overflow-auto max-h-[600px] border dark:border-gray-700 rounded-lg">
          <table className="text-xs w-full">
            <thead className="sticky top-0 bg-white dark:bg-gray-900 z-10">
              <tr>
                <th className="px-2 py-2 text-left font-medium text-gray-500 border-b dark:border-gray-700 min-w-[100px]">
                  {rowDim} / {colDim}
                </th>
                {data.cols.map((col) => (
                  <th
                    key={col}
                    className="px-2 py-2 text-center font-medium text-gray-500 border-b dark:border-gray-700 min-w-[70px]"
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
              {data.rows.map((row, ri) => (
                <tr key={row} className="hover:bg-gray-50 dark:hover:bg-gray-800/50">
                  <td
                    className="px-2 py-1.5 font-medium text-gray-700 dark:text-gray-300 border-b dark:border-gray-700/50 truncate max-w-[120px]"
                    title={data.row_labels[row] || row}
                  >
                    {data.row_labels[row] || row}
                  </td>
                  {data.cells[ri]?.map((val, ci) => (
                    <td
                      key={ci}
                      className={`px-2 py-1.5 text-center border-b dark:border-gray-700/50 cursor-pointer ${cellColor(val, maxVal)}`}
                      onClick={() => handleCellClick(ri, ci)}
                      title={`${val.toLocaleString()}`}
                    >
                      {val > 0 ? val.toLocaleString(undefined, { maximumFractionDigits: 0 }) : "—"}
                    </td>
                  ))}
                </tr>
              ))}
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