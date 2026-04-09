import { useRef, useEffect } from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  BarChart,
  Bar,
  Cell,
} from "recharts";
import { X } from "lucide-react";
import { useChartColors } from "@/hooks/useChartColors";
import { useReference } from "@/api/queries/demand-history";
import { Skeleton } from "@/components/Skeleton";
import { formatInt, formatPct, formatFixed } from "@/lib/formatters";

interface Props {
  itemId: string;
  loc: string;
  open: boolean;
  onClose: () => void;
}

const kpiFmt = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 1,
  notation: "compact",
});

function formatKpi(n: number | null | undefined): string {
  if (n == null) return "--";
  return kpiFmt.format(n);
}

export function DemandReferencePanel({ itemId, loc, open, onClose }: Props) {
  const panelRef = useRef<HTMLDivElement>(null);
  const { trendColors, chartColors } = useChartColors();
  const { data, isLoading, isError } = useReference(itemId, loc);

  // Close on Escape
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/20 z-40"
        onClick={onClose}
      />

      {/* Panel */}
      <div
        ref={panelRef}
        className="fixed top-0 right-0 h-full w-96 bg-white dark:bg-gray-900 shadow-xl z-50 overflow-y-auto transition-transform duration-200"
        style={{ transform: open ? "translateX(0)" : "translateX(100%)" }}
      >
        {/* Header */}
        <div className="sticky top-0 bg-white dark:bg-gray-900 border-b dark:border-gray-700 px-4 py-3 flex items-center justify-between">
          <div className="min-w-0 flex-1">
            <h3 className="font-semibold text-sm truncate">
              {data?.item_description ?? itemId}
            </h3>
            <p className="text-xs text-gray-500 truncate">
              {data?.location_name ?? loc}
            </p>
          </div>
          <button
            onClick={onClose}
            className="flex-shrink-0 ml-2 p-1 rounded-md text-gray-400 hover:text-gray-600 hover:bg-gray-100 dark:hover:text-gray-300 dark:hover:bg-gray-800 transition-colors"
            aria-label="Close panel"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {isLoading && (
          <div className="p-4 space-y-4">
            <div className="grid grid-cols-2 gap-3">
              {Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-16 rounded-lg" />
              ))}
            </div>
            <Skeleton className="h-[120px] rounded-lg" />
            <Skeleton className="h-[100px] rounded-lg" />
          </div>
        )}
        {isError && (
          <div className="p-6 text-center text-sm text-red-500">Failed to load data</div>
        )}

        {data && (
          <div className="p-4 space-y-5">
            {/* KPI Cards */}
            <div className="grid grid-cols-2 gap-3">
              <KpiCard
                label="MoM Trend"
                value={`${data.trend_mom_pct >= 0 ? "+" : ""}${formatFixed(data.trend_mom_pct)}%`}
                color={data.trend_mom_pct >= 0 ? "text-green-600" : "text-red-600"}
              />
              <KpiCard
                label="Accuracy"
                value={data.forecast_accuracy != null ? formatPct(data.forecast_accuracy) : "--"}
              />
              <KpiCard
                label="Inventory"
                value={formatKpi(data.current_inventory)}
              />
              <KpiCard
                label="Lead Time"
                value={data.avg_lead_time != null ? `${formatFixed(data.avg_lead_time, 0)}d` : "--"}
              />
            </div>

            {/* Demand History Sparkline */}
            <div>
              <h4 className="text-xs font-medium text-gray-500 mb-2">24-Month Demand</h4>
              <ResponsiveContainer width="100%" height={120}>
                <AreaChart data={data.history}>
                  <defs>
                    <linearGradient id="refGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={trendColors[0]} stopOpacity={0.3} />
                      <stop offset="95%" stopColor={trendColors[0]} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis
                    dataKey="month"
                    tick={{ fontSize: 9, fill: chartColors.axis }}
                    tickFormatter={(v: string) => v.slice(5, 7)}
                    interval="preserveStartEnd"
                  />
                  <YAxis hide />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: chartColors.tooltip_bg,
                      border: `1px solid ${chartColors.tooltip_border}`,
                      fontSize: 11,
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="demand_qty"
                    stroke={trendColors[0]}
                    fill="url(#refGrad)"
                    strokeWidth={1.5}
                    name="Demand"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Top Customers */}
            <div>
              <h4 className="text-xs font-medium text-gray-500 mb-2">Top Customers</h4>
              <ResponsiveContainer width="100%" height={Math.max(80, data.top_customers.length * 28)}>
                <BarChart
                  data={data.top_customers}
                  layout="vertical"
                  margin={{ left: 0, right: 8, top: 0, bottom: 0 }}
                >
                  <XAxis type="number" hide />
                  <YAxis
                    type="category"
                    dataKey="customer_name"
                    width={100}
                    tick={{ fontSize: 10, fill: chartColors.axis }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: chartColors.tooltip_bg,
                      border: `1px solid ${chartColors.tooltip_border}`,
                      fontSize: 11,
                    }}
                    formatter={(v: number) =>
                      [formatInt(v), "Demand"]
                    }
                  />
                  <Bar dataKey="demand_qty" radius={[0, 4, 4, 0]}>
                    {data.top_customers.map((_, i) => (
                      <Cell key={i} fill={trendColors[i % trendColors.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Open full analysis link */}
            <div className="pt-2 border-t dark:border-gray-700">
              <p className="text-xs text-gray-400 text-center">
                {itemId} @ {loc}
              </p>
            </div>
          </div>
        )}
      </div>
    </>
  );
}

function KpiCard({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="rounded-lg border border-gray-100 dark:border-gray-800 bg-gray-50 dark:bg-gray-800/50 px-3 py-2.5">
      <p className="text-[10px] text-gray-500 uppercase tracking-wider font-medium">{label}</p>
      <p className={`text-lg font-bold tabular-nums ${color ?? "text-gray-900 dark:text-gray-100"}`}>
        {value}
      </p>
    </div>
  );
}