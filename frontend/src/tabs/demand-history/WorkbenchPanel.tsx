import { useState, useMemo } from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";
import { useChartColors } from "@/hooks/useChartColors";
import { useWorkbench } from "@/api/queries/demand-history";
import type { WorkbenchGrain, WorkbenchSeries } from "@/api/queries/demand-history";
import { useDemandHistorySelection } from "../DemandHistoryTab";

// ---------------------------------------------------------------------------
// Hierarchy tree node
// ---------------------------------------------------------------------------

function TreeNode({
  series,
  isSelected,
  onSelect,
}: {
  series: WorkbenchSeries;
  isSelected: boolean;
  onSelect: (key: string) => void;
}) {
  return (
    <button
      onClick={() => onSelect(series.key)}
      className={`w-full text-left px-3 py-2 text-sm rounded-md transition-colors ${
        isSelected
          ? "bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300"
          : "hover:bg-gray-50 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300"
      }`}
    >
      <div className="flex items-center justify-between">
        <span className="truncate font-medium">{series.label || series.key}</span>
        <span className="text-xs text-gray-500 ml-2 flex-shrink-0">
          {series.total_demand.toLocaleString()}
        </span>
      </div>
    </button>
  );
}

// ---------------------------------------------------------------------------
// Main workbench panel
// ---------------------------------------------------------------------------

export function DemandWorkbenchPanel() {
  const { itemId, loc, setSelection } = useDemandHistorySelection();
  const { trendColors, chartColors } = useChartColors();

  const [grain, setGrain] = useState<WorkbenchGrain>("item");
  const [search, setSearch] = useState("");
  const [selectedKey, setSelectedKey] = useState("");

  // Derive query params from grain + selection
  const queryItemId = grain === "item" ? undefined : itemId || undefined;
  const queryLoc = grain === "item_loc_customer" ? loc || undefined : undefined;

  const { data, isLoading } = useWorkbench(grain, queryItemId, queryLoc);

  // Filter series by search
  const filteredSeries = useMemo(() => {
    if (!data?.series) return [];
    if (!search) return data.series;
    const q = search.toLowerCase();
    return data.series.filter(
      (s) =>
        s.key.toLowerCase().includes(q) || s.label.toLowerCase().includes(q),
    );
  }, [data?.series, search]);

  // Selected series data for chart
  const selectedSeries = useMemo(
    () => data?.series.find((s) => s.key === selectedKey),
    [data?.series, selectedKey],
  );

  function handleNodeSelect(key: string) {
    setSelectedKey(key);
    // Parse key to propagate item+loc context
    const parts = key.split("__");
    if (parts.length >= 2) {
      setSelection(parts[0], parts[1]);
    } else if (parts.length === 1) {
      setSelection(parts[0], "");
    }
  }

  function handleDrillDown(childKey: string) {
    const nextGrain: Record<WorkbenchGrain, WorkbenchGrain> = {
      item: "item_loc",
      item_loc: "item_loc_customer",
      item_loc_customer: "item_loc_customer",
    };
    const parts = childKey.split("__");
    if (parts.length >= 1) setSelection(parts[0], parts[1] ?? "");
    setGrain(nextGrain[grain]);
    setSelectedKey("");
  }

  return (
    <div className="flex gap-4 h-full min-h-[500px]">
      {/* Left: hierarchy tree */}
      <div className="w-72 flex-shrink-0 border-r dark:border-gray-700 pr-4 flex flex-col">
        {/* Grain selector */}
        <div className="flex gap-1 mb-3">
          {(["item", "item_loc", "item_loc_customer"] as WorkbenchGrain[]).map((g) => (
            <button
              key={g}
              onClick={() => { setGrain(g); setSelectedKey(""); }}
              className={`px-2 py-1 text-xs rounded ${
                grain === g
                  ? "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300"
                  : "text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800"
              }`}
            >
              {g.replace(/_/g, " + ")}
            </button>
          ))}
        </div>

        {/* Search */}
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search..."
          className="mb-3 px-3 py-1.5 text-sm border dark:border-gray-700 rounded-md bg-white dark:bg-gray-900 w-full"
        />

        {/* Series list */}
        <div className="flex-1 overflow-y-auto space-y-0.5">
          {isLoading && (
            <p className="text-sm text-gray-500 p-3">Loading...</p>
          )}
          {filteredSeries.map((s) => (
            <TreeNode
              key={s.key}
              series={s}
              isSelected={selectedKey === s.key}
              onSelect={handleNodeSelect}
            />
          ))}
          {!isLoading && filteredSeries.length === 0 && (
            <p className="text-sm text-gray-400 p-3">No series found</p>
          )}
        </div>

        {/* Children for drill-down */}
        {data?.hierarchy_children && data.hierarchy_children.length > 0 && selectedKey && grain !== "item_loc_customer" && (
          <div className="mt-3 pt-3 border-t dark:border-gray-700">
            <p className="text-xs text-gray-500 mb-2 font-medium">Drill down</p>
            <div className="space-y-0.5 max-h-40 overflow-y-auto">
              {data.hierarchy_children.map((c) => (
                <button
                  key={c.key}
                  onClick={() => handleDrillDown(c.key)}
                  className="w-full text-left px-3 py-1.5 text-xs rounded hover:bg-gray-50 dark:hover:bg-gray-800 text-gray-600 dark:text-gray-400 flex justify-between"
                >
                  <span className="truncate">{c.label || c.key}</span>
                  <span>{c.total_demand.toLocaleString()}</span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Right: chart */}
      <div className="flex-1 flex flex-col">
        {selectedSeries ? (
          <>
            <h3 className="text-sm font-semibold mb-1 text-gray-800 dark:text-gray-200">
              {selectedSeries.label || selectedSeries.key}
            </h3>
            <p className="text-xs text-gray-500 mb-3">
              Total: {selectedSeries.total_demand.toLocaleString()}
            </p>
            <div className="flex-1">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={selectedSeries.months}>
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
                  <defs>
                    <linearGradient id="wbGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={trendColors[0]} stopOpacity={0.3} />
                      <stop offset="95%" stopColor={trendColors[0]} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <Area
                    type="monotone"
                    dataKey="demand_qty"
                    stroke={trendColors[0]}
                    fill="url(#wbGrad)"
                    strokeWidth={2}
                    name="Demand"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-gray-400 text-sm">
            Select a series from the tree to view demand history
          </div>
        )}
      </div>
    </div>
  );
}