import { useState, useMemo } from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  BarChart,
  Bar,
  Cell,
  Line,
  ComposedChart,
} from "recharts";
import { useChartColors } from "@/hooks/useChartColors";
import { useDecomposition } from "@/api/queries/demand-history";
import { useDemandHistorySelection } from "../DemandHistoryTab";

type ViewMode = "absolute" | "percent";

export function DecompositionPanel() {
  const { itemId, loc, setSelection } = useDemandHistorySelection();
  const { trendColors, chartColors } = useChartColors();
  const [viewMode, setViewMode] = useState<ViewMode>("percent");
  const [inputItem, setInputItem] = useState(itemId);
  const [inputLoc, setInputLoc] = useState(loc);

  const activeItem = itemId || inputItem;
  const activeLoc = loc || inputLoc;

  const { data, isLoading, isError } = useDecomposition(activeItem, activeLoc);

  // Pivot monthly data: month → { month, cust1: qty, cust2: qty, ... }
  const { pivotData, customerKeys } = useMemo(() => {
    if (!data?.monthly?.length) return { pivotData: [], customerKeys: [] };

    const custSet = new Set<string>();
    const byMonth = new Map<string, Record<string, string | number>>();

    for (const row of data.monthly) {
      custSet.add(row.customer_no);
      const rec = byMonth.get(row.month) ?? { month: row.month } as Record<string, string | number>;
      rec[row.customer_no] = viewMode === "percent" ? row.pct_share * 100 : row.demand_qty;
      byMonth.set(row.month, rec);
    }

    const keys = Array.from(custSet);
    return { pivotData: Array.from(byMonth.values()), customerKeys: keys };
  }, [data?.monthly, viewMode]);

  function handleApply() {
    setSelection(inputItem, inputLoc);
  }

  const hasSelection = activeItem && activeLoc;

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

        <div className="ml-auto flex gap-1">
          <button
            onClick={() => setViewMode("absolute")}
            className={`px-2 py-1 text-xs rounded ${viewMode === "absolute" ? "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300" : "text-gray-500 hover:bg-gray-100"}`}
          >
            Absolute
          </button>
          <button
            onClick={() => setViewMode("percent")}
            className={`px-2 py-1 text-xs rounded ${viewMode === "percent" ? "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300" : "text-gray-500 hover:bg-gray-100"}`}
          >
            % Share
          </button>
        </div>
      </div>

      {!hasSelection && (
        <div className="text-center text-gray-400 text-sm py-20">
          Enter an Item ID and Location, or select from the Workbench panel
        </div>
      )}

      {isLoading && <div className="text-center text-gray-500 text-sm py-10">Loading...</div>}
      {isError && <div className="text-center text-red-500 text-sm py-10">Failed to load data</div>}

      {hasSelection && data && (
        <div className="flex gap-6">
          {/* Stacked area chart */}
          <div className="flex-1">
            <h4 className="text-xs font-medium text-gray-500 mb-2">
              Customer Demand {viewMode === "percent" ? "(% Share)" : "(Qty)"}
            </h4>
            <ResponsiveContainer width="100%" height={360}>
              <AreaChart
                data={pivotData}
                stackOffset={viewMode === "percent" ? "expand" : "none"}
              >
                <CartesianGrid stroke={chartColors.grid} strokeDasharray="3 3" />
                <XAxis
                  dataKey="month"
                  tick={{ fontSize: 10, fill: chartColors.axis }}
                  tickFormatter={(v: string) => v.slice(5, 7)}
                />
                <YAxis
                  tick={{ fontSize: 10, fill: chartColors.axis }}
                  tickFormatter={viewMode === "percent" ? (v: number) => `${(v * 100).toFixed(0)}%` : undefined}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: chartColors.tooltip_bg,
                    border: `1px solid ${chartColors.tooltip_border}`,
                    fontSize: 11,
                  }}
                />
                {customerKeys.map((key, i) => (
                  <Area
                    key={key}
                    type="monotone"
                    dataKey={key}
                    stackId="1"
                    stroke={trendColors[i % trendColors.length]}
                    fill={trendColors[i % trendColors.length]}
                    fillOpacity={0.6}
                    name={data.pareto.find((p) => p.customer_no === key)?.customer_name ?? key}
                  />
                ))}
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Pareto sidebar */}
          <div className="w-72 flex-shrink-0">
            <h4 className="text-xs font-medium text-gray-500 mb-2">Pareto Analysis</h4>
            <ResponsiveContainer width="100%" height={360}>
              <ComposedChart data={data.pareto} layout="vertical">
                <XAxis type="number" tick={{ fontSize: 10, fill: chartColors.axis }} />
                <YAxis
                  type="category"
                  dataKey="customer_name"
                  width={90}
                  tick={{ fontSize: 9, fill: chartColors.axis }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: chartColors.tooltip_bg,
                    border: `1px solid ${chartColors.tooltip_border}`,
                    fontSize: 11,
                  }}
                />
                <Bar dataKey="pct_share" name="Share %" radius={[0, 4, 4, 0]}>
                  {data.pareto.map((_, i) => (
                    <Cell key={i} fill={trendColors[i % trendColors.length]} />
                  ))}
                </Bar>
                <Line
                  type="monotone"
                  dataKey="cumulative_pct"
                  stroke="#94a3b8"
                  strokeWidth={1.5}
                  dot={false}
                  name="Cumulative %"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
