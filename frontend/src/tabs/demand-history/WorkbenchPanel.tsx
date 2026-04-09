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
import { ChevronRight, BarChart3, Search } from "lucide-react";
import { useChartColors } from "@/hooks/useChartColors";
import { useWorkbench } from "@/api/queries/demand-history";
import type { WorkbenchGrain, WorkbenchSeries } from "@/api/queries/demand-history";
import { useDemandHistorySelection } from "../DemandHistoryTab";
import { Skeleton } from "@/components/Skeleton";
import { formatInt } from "@/lib/formatters";

// ---------------------------------------------------------------------------
// Grain labels
// ---------------------------------------------------------------------------

const GRAIN_LABELS: Record<WorkbenchGrain, string> = {
  item: "Item",
  item_loc: "Item + Loc",
  item_loc_customer: "Item + Loc + Cust",
};

// ---------------------------------------------------------------------------
// Hierarchy tree node
// ---------------------------------------------------------------------------

function TreeNode({
  series,
  isSelected,
  onSelect,
  canDrillDown,
}: {
  series: WorkbenchSeries;
  isSelected: boolean;
  onSelect: (key: string) => void;
  canDrillDown: boolean;
}) {
  return (
    <button
      onClick={() => onSelect(series.key)}
      className={`group w-full text-left px-3 py-2 text-sm rounded-lg transition-all ${
        isSelected
          ? "bg-blue-50 text-blue-700 ring-1 ring-blue-200 dark:bg-blue-900/30 dark:text-blue-300 dark:ring-blue-800"
          : "hover:bg-gray-50 dark:hover:bg-gray-800/60 text-gray-700 dark:text-gray-300"
      }`}
    >
      <div className="flex items-center gap-2">
        {canDrillDown && (
          <ChevronRight className={`h-3.5 w-3.5 flex-shrink-0 transition-transform ${
            isSelected ? "text-blue-500 rotate-90" : "text-gray-400 group-hover:text-gray-500"
          }`} />
        )}
        <span className="truncate font-medium flex-1">{series.label || series.key}</span>
        <span className={`text-xs tabular-nums flex-shrink-0 ${
          isSelected ? "text-blue-500 font-medium" : "text-gray-400"
        }`}>
          {formatInt(series.total_demand)}
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

  const canDrillDown = grain !== "item_loc_customer" && !!data?.hierarchy_children;

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
              className={`px-2.5 py-1 text-xs rounded-md font-medium transition-colors ${
                grain === g
                  ? "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300"
                  : "text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800"
              }`}
            >
              {GRAIN_LABELS[g]}
            </button>
          ))}
        </div>

        {/* Search */}
        <div className="relative mb-3">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-gray-400" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search items..."
            className="w-full pl-8 pr-3 py-1.5 text-sm border dark:border-gray-700 rounded-md bg-white dark:bg-gray-900"
          />
        </div>

        {/* Series count */}
        {!isLoading && data?.series && (
          <p className="text-[10px] text-gray-400 uppercase tracking-wider mb-2 px-1">
            {filteredSeries.length} of {data.series.length} series
          </p>
        )}

        {/* Series list */}
        <div className="flex-1 overflow-y-auto space-y-0.5">
          {isLoading && (
            <div className="space-y-1.5 p-1">
              {Array.from({ length: 8 }).map((_, i) => (
                <Skeleton key={i} className="h-9 rounded-lg" />
              ))}
            </div>
          )}
          {filteredSeries.map((s) => (
            <TreeNode
              key={s.key}
              series={s}
              isSelected={selectedKey === s.key}
              onSelect={handleNodeSelect}
              canDrillDown={canDrillDown}
            />
          ))}
          {!isLoading && filteredSeries.length === 0 && (
            <div className="flex flex-col items-center py-8 text-gray-400">
              <Search className="h-8 w-8 mb-2 opacity-40" />
              <p className="text-sm">No series found</p>
              {search && (
                <button
                  onClick={() => setSearch("")}
                  className="mt-1 text-xs text-blue-500 hover:underline"
                >
                  Clear search
                </button>
              )}
            </div>
          )}
        </div>

        {/* Drill-down button */}
        {canDrillDown && selectedKey && (
          <div className="mt-3 pt-3 border-t dark:border-gray-700">
            <button
              onClick={() => handleDrillDown(selectedKey)}
              className="w-full flex items-center justify-between px-3 py-2.5 text-xs rounded-lg bg-gradient-to-r from-blue-50 to-blue-100/50 hover:from-blue-100 hover:to-blue-100 dark:from-blue-900/30 dark:to-blue-900/20 dark:hover:from-blue-900/50 dark:hover:to-blue-900/30 text-blue-700 dark:text-blue-300 font-medium transition-all"
            >
              <span>Drill down to {GRAIN_LABELS[grain === "item" ? "item_loc" : "item_loc_customer"]}</span>
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        )}
      </div>

      {/* Right: chart */}
      <div className="flex-1 flex flex-col">
        {selectedSeries ? (
          <>
            <div className="flex items-baseline gap-3 mb-3">
              <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-200">
                {selectedSeries.label || selectedSeries.key}
              </h3>
              <span className="text-xs text-gray-400 tabular-nums">
                Total: {formatInt(selectedSeries.total_demand)}
              </span>
            </div>
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
                    formatter={(v: number) => [formatInt(v), "Demand"]}
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
          <div className="flex-1 flex flex-col items-center justify-center text-gray-400">
            <BarChart3 className="h-12 w-12 mb-3 opacity-30" />
            <p className="text-sm font-medium">No series selected</p>
            <p className="text-xs mt-1">Select a series from the tree to view demand history</p>
          </div>
        )}
      </div>
    </div>
  );
}