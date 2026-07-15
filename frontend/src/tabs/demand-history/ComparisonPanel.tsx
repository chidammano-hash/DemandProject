import { useMemo, useState } from "react";
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
import { useCustomerBlendOverlay } from "@/hooks/useCustomerBlendOverlay";
import { useComparison } from "@/api/queries/demand-history";
import { useDemandHistorySelection } from "../DemandHistoryTab";
import { Skeleton } from "@/components/Skeleton";
import { CustomerBlendLegend, CustomerBlendLines } from "@/components/CustomerBlendOverlay";
import { hasCustomerBlendData, mergeCustomerBlendOverlay } from "@/lib/customer-blend-overlay";

const SERIES_CONFIG = [
  { key: "actual_qty", label: "Actual", dash: undefined },
  { key: "bottom_up_qty", label: "Hierarchy Bottom-Up", dash: "6 3" },
  { key: "top_down_qty", label: "Hierarchy Top-Down", dash: "6 3" },
  { key: "reconciled_qty", label: "Hierarchy Reconciled", dash: undefined },
] as const;

export function ComparisonPanel() {
  const { itemId, loc, setSelection } = useDemandHistorySelection();
  const { chartColors, okabeIto } = useChartColors();
  const [inputItem, setInputItem] = useState(itemId);
  const [inputLoc, setInputLoc] = useState(loc);
  const [visibleSeries, setVisibleSeries] = useState<Set<string>>(
    new Set(SERIES_CONFIG.map((s) => s.key))
  );

  const activeItem = itemId || inputItem;
  const activeLoc = loc || inputLoc;

  const { data, isLoading, isError } = useComparison(activeItem, activeLoc);
  const hasSelection = Boolean(activeItem && activeLoc);
  const customerBlend = useCustomerBlendOverlay(activeItem, activeLoc, hasSelection);
  const comparisonData = useMemo(
    () =>
      mergeCustomerBlendOverlay(
        (data?.comparison ?? []).map((point) => ({ ...point })),
        customerBlend.points
      ),
    [customerBlend.points, data?.comparison]
  );
  const seriesConfig = useMemo(
    () => [
      { ...SERIES_CONFIG[0], color: chartColors.axis },
      { ...SERIES_CONFIG[1], color: okabeIto[4] },
      { ...SERIES_CONFIG[2], color: okabeIto[0] },
      { ...SERIES_CONFIG[3], color: okabeIto[2] },
    ],
    [chartColors.axis, okabeIto]
  );

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

  const hasHierarchical = data?.comparison?.some(
    (m) => m.bottom_up_qty != null || m.top_down_qty != null
  );

  return (
    <div className="space-y-4">
      {/* Item+Loc selector */}
      <div className="flex flex-wrap items-center gap-3">
        <input
          type="text"
          aria-label="Item ID"
          value={inputItem}
          onChange={(e) => setInputItem(e.target.value)}
          placeholder="Item ID"
          className="px-3 py-1.5 text-sm border dark:border-gray-700 rounded-md bg-white dark:bg-gray-900 w-40"
        />
        <input
          type="text"
          aria-label="Location"
          value={inputLoc}
          onChange={(e) => setInputLoc(e.target.value)}
          placeholder="Location"
          className="px-3 py-1.5 text-sm border dark:border-gray-700 rounded-md bg-white dark:bg-gray-900 w-40"
        />
        <button
          type="button"
          onClick={handleApply}
          className="px-3 py-1.5 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700"
        >
          Apply
        </button>
      </div>

      {!hasSelection && (
        <div className="flex flex-col items-center py-16 text-muted-foreground">
          <GitCompare className="h-12 w-12 mb-3 opacity-30" />
          <p className="text-sm font-medium">No item selected</p>
          <p className="text-xs mt-1">
            Enter an Item ID and Location, or select from the Workbench panel
          </p>
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
          <div className="flex flex-wrap gap-4">
            {seriesConfig.map((s) => (
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
              No reconciled hierarchy forecast is available for this DFU and period.
            </div>
          )}

          <CustomerBlendLegend
            months={customerBlend.months}
            status={customerBlend.status}
            emptyMessage={
              customerBlend.runId
                ? "No customer blend is available for this item and location."
                : "No staged customer blend draft exists yet. Generate a blend draft first."
            }
            runId={customerBlend.runId}
            planningMonth={customerBlend.planningMonth}
          />

          {/* Chart */}
          <ResponsiveContainer width="100%" height={420}>
            <ComposedChart data={comparisonData}>
              <CartesianGrid stroke={chartColors.grid} strokeDasharray="3 3" />
              <XAxis
                dataKey="month"
                tick={{ fontSize: 11, fill: chartColors.axis }}
                tickFormatter={(v: string) => v.slice(0, 7)}
              />
              <YAxis
                tick={{ fontSize: 11, fill: chartColors.axis }}
                label={{
                  value: "Units",
                  angle: -90,
                  position: "insideLeft",
                  fontSize: 10,
                  fill: chartColors.axis,
                }}
              />
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
                  fill={seriesConfig[1].color}
                  fillOpacity={0.08}
                  name=""
                  legendType="none"
                />
              )}

              {seriesConfig.map((s) =>
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
                ) : null
              )}
              {hasCustomerBlendData(comparisonData) && <CustomerBlendLines />}
            </ComposedChart>
          </ResponsiveContainer>
        </>
      )}
    </div>
  );
}
