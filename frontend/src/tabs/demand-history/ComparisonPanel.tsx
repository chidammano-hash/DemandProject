import { useState } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from "recharts";
import { GitCompare } from "lucide-react";
import { useChartColors } from "@/hooks/useChartColors";
import { useComparison } from "@/api/queries/demand-history";
import { useDemandHistorySelection } from "../DemandHistoryTab";
import { Skeleton } from "@/components/Skeleton";

const SERIES_CONFIG = [
  { key: "actual_qty", label: "Actual", color: "#1e293b", dash: undefined },
  { key: "bottom_up_qty", label: "Bottom-Up", color: "#2563eb", dash: "6 3" },
  { key: "top_down_qty", label: "Top-Down", color: "#d97706", dash: "6 3" },
  { key: "reconciled_qty", label: "Reconciled", color: "#059669", dash: undefined },
] as const;

export function ComparisonPanel() {
  const { itemId, loc, setSelection } = useDemandHistorySelection();
  const { chartColors } = useChartColors();
  const [inputItem, setInputItem] = useState(itemId);
  const [inputLoc, setInputLoc] = useState(loc);
  const [visibleSeries, setVisibleSeries] = useState<Set<string>>(
    new Set(SERIES_CONFIG.map((s) => s.key)),
  );

  const activeItem = itemId || inputItem;
  const activeLoc = loc || inputLoc;

  const { data, isLoading, isError } = useComparison(activeItem, activeLoc);

  function handleApply() {
    setSelection(inputItem, inputLoc);
  }

  function toggleSeries(key: string) {
    setVisibleSeries((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  const hasSelection = activeItem && activeLoc;
  const hasHierarchical = data?.comparison?.some(
    (m) => m.bottom_up_qty != null || m.top_down_qty != null,
  );

  return (
    <div className="space-y-4">
      {/* Item+Loc selector */}
      <div className="flex items-center gap-3">
        <input
          type="text"
          value={inputItem}
          onChange={(e) => setInputItem(e.target.value)}
          placeholder="Item ID"
          className="px-3 py-1.5 text-sm border dark:border-gray-700 rounded-md bg-white dark:bg-gray-900 w-40"
        />
        <input
          type="text"
          value={inputLoc}
          onChange={(e) => setInputLoc(e.target.value)}
          placeholder="Location"
          className="px-3 py-1.5 text-sm border dark:border-gray-700 rounded-md bg-white dark:bg-gray-900 w-40"
        />
        <button
          onClick={handleApply}
          className="px-3 py-1.5 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700"
        >
          Apply
        </button>
      </div>

      {!hasSelection && (
        <div className="flex flex-col items-center py-16 text-gray-400">
          <GitCompare className="h-12 w-12 mb-3 opacity-30" />
          <p className="text-sm font-medium">No item selected</p>
          <p className="text-xs mt-1">Enter an Item ID and Location, or select from the Workbench panel</p>
        </div>
      )}

      {isLoading && (
        <div className="space-y-3">
          <Skeleton className="h-6 w-64" />
          <Skeleton className="h-[420px] rounded-lg" />
        </div>
      )}
      {isError && <div className="text-center text-red-500 text-sm py-10">Failed to load data</div>}

      {hasSelection && data && (
        <>
          {/* Series toggles */}
          <div className="flex gap-4">
            {SERIES_CONFIG.map((s) => (
              <label key={s.key} className="flex items-center gap-1.5 text-xs cursor-pointer">
                <input
                  type="checkbox"
                  checked={visibleSeries.has(s.key)}
                  onChange={() => toggleSeries(s.key)}
                  className="rounded"
                />
                <span
                  className="w-3 h-0.5 inline-block"
                  style={{
                    backgroundColor: s.color,
                    borderTop: s.dash ? `2px dashed ${s.color}` : `2px solid ${s.color}`,
                  }}
                />
                <span className="text-gray-700 dark:text-gray-300">{s.label}</span>
              </label>
            ))}
          </div>

          {!hasHierarchical && (
            <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-md px-4 py-2 text-sm text-amber-700 dark:text-amber-400">
              No hierarchical forecast data available yet. Run the Bolt hierarchical backtest to see bottom-up vs top-down comparison.
            </div>
          )}

          {/* Chart */}
          <ResponsiveContainer width="100%" height={420}>
            <ComposedChart data={data.comparison}>
              <CartesianGrid stroke={chartColors.grid} strokeDasharray="3 3" />
              <XAxis
                dataKey="month"
                tick={{ fontSize: 11, fill: chartColors.axis }}
                tickFormatter={(v: string) => v.slice(0, 7)}
              />
              <YAxis tick={{ fontSize: 11, fill: chartColors.axis }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: chartColors.tooltip_bg,
                  border: `1px solid ${chartColors.tooltip_border}`,
                  fontSize: 12,
                }}
              />
              <Legend />

              {/* Gap shading between BU and TD */}
              {visibleSeries.has("bottom_up_qty") && visibleSeries.has("top_down_qty") && (
                <Area
                  type="monotone"
                  dataKey="bottom_up_qty"
                  stroke="none"
                  fill="#2563eb"
                  fillOpacity={0.08}
                  name=""
                  legendType="none"
                />
              )}

              {SERIES_CONFIG.map((s) =>
                visibleSeries.has(s.key) ? (
                  <Line
                    key={s.key}
                    type="monotone"
                    dataKey={s.key}
                    stroke={s.color}
                    strokeWidth={s.key === "actual_qty" || s.key === "reconciled_qty" ? 2.5 : 1.5}
                    strokeDasharray={s.dash}
                    dot={false}
                    name={s.label}
                    connectNulls
                  />
                ) : null,
              )}
            </ComposedChart>
          </ResponsiveContainer>
        </>
      )}
    </div>
  );
}
