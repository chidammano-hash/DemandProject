import { useMemo } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ModularReactECharts as ReactECharts } from "@/components/echarts-modular";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsOosImpact,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters } from "@/api/queries/customer-analytics";
import { ExportButtons } from "./ExportButtons";
import { EmptyState } from "./EmptyState";

type Grain = "customer" | "state";

interface Props {
  filters: CustomerAnalyticsFilters;
  grain: Grain;
  onGrainChange: (g: Grain) => void;
}

const CHANNEL_COLORS: Record<string, string> = {
  "On Premise": "#6366f1",
  "Off Premise": "#f59e0b",
  Unknown: "#94a3b8",
  All: "#3b82f6",
};

function getColor(channel: string): string {
  return CHANNEL_COLORS[channel] || `hsl(${Math.abs(hashCode(channel)) % 360}, 60%, 55%)`;
}

function hashCode(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (Math.imul(31, h) + s.charCodeAt(i)) | 0;
  return h;
}

export function OosImpactBubble({ filters, grain, onGrainChange }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.oosImpact(grain, filters),
    queryFn: () => fetchCustomerAnalyticsOosImpact(grain, filters),
    staleTime: 60 * 60_000, // monthly data; pin to 1h to suppress thundering-herd refetches
    placeholderData: keepPreviousData, // keep prior chart visible during filter-change refetch
  });

  const totalOos = useMemo(() => {
    const bubbles = data?.bubbles ?? [];
    return bubbles.reduce((sum, b) => sum + b.oos_qty, 0);
  }, [data]);

  const option = useMemo(() => {
    const bubbles = data?.bubbles ?? [];
    if (!bubbles.length) return {};

    const maxOos = Math.max(...bubbles.map((b) => b.oos_qty), 1);
    const maxDemand = Math.max(...bubbles.map((b) => b.demand_qty), 1);
    const midDemand = maxDemand / 2;

    const seriesMap: Record<string, Array<[number, number, number, string]>> = {};
    for (const b of bubbles) {
      const ch = b.channel || "Unknown";
      if (!seriesMap[ch]) seriesMap[ch] = [];
      seriesMap[ch].push([b.demand_qty, b.fill_rate, b.oos_qty, b.label]);
    }

    const scatterSeries = Object.entries(seriesMap).map(([ch, pts]) => ({
      name: ch,
      type: "scatter" as const,
      data: pts,
      symbolSize: (val: number[]) => 8 + 30 * Math.sqrt(val[2] / maxOos),
      itemStyle: { color: getColor(ch), opacity: 0.7 },
      emphasis: { itemStyle: { opacity: 1 } },
    }));

    return {
      tooltip: {
        formatter: (p: { value: [number, number, number, string]; seriesName: string }) => {
          const [demand, fr, oos, label] = p.value;
          return `<b>${label}</b><br/>Channel: ${p.seriesName}<br/>Demand: ${demand.toLocaleString()}<br/>Fill Rate: ${fr}%<br/>OOS: ${oos.toLocaleString()}`;
        },
      },
      legend: { top: 5, type: "scroll" as const },
      grid: { left: 60, right: 20, top: 40, bottom: 50 },
      xAxis: {
        name: "Demand (cases)",
        nameLocation: "center" as const,
        nameGap: 30,
        type: "value" as const,
        axisLabel: { formatter: (v: number) => v >= 1000 ? `${(v / 1000).toFixed(0)}K` : String(v) },
      },
      yAxis: { name: "Fill Rate %", nameLocation: "center" as const, nameGap: 40, type: "value" as const, min: 0, max: 105 },
      series: [
        // Quadrant shading: bottom-right = critical zone (high demand, low fill rate)
        {
          type: "scatter" as const,
          data: [],
          silent: true,
          markArea: {
            silent: true,
            data: [
              [
                { xAxis: midDemand, yAxis: 0, itemStyle: { color: "rgba(239,68,68,0.08)" } },
                { xAxis: maxDemand * 1.1, yAxis: 90 },
              ],
            ],
            label: {
              show: true,
              position: "insideBottomRight" as const,
              formatter: "Critical",
              fontSize: 11,
              color: "#dc2626",
              fontStyle: "italic" as const,
            },
          },
          markLine: {
            silent: true,
            lineStyle: { type: "dashed" as const, color: "#94a3b8" },
            data: [{ yAxis: 90 }],
          },
        },
        ...scatterSeries,
      ],
    };
  }, [data]);

  return (
    <Card aria-label="OOS impact bubble chart">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">OOS Impact Analysis</CardTitle>
          <ExportButtons panelId="oos-impact" getData={() => data?.bubbles ?? []} />
        </div>
        <div className="flex gap-1 mt-1">
          {(["customer", "state"] as const).map((g) => (
            <button
              key={g}
              onClick={() => onGrainChange(g)}
              className={`px-2 py-0.5 text-xs rounded ${grain === g ? "bg-indigo-600 text-white" : "bg-gray-100 text-gray-600"}`}
            >
              By {g}
            </button>
          ))}
        </div>
        <p className="text-xs text-muted-foreground mt-1">Bubble size = OOS volume. Below the line = action needed.</p>
        {totalOos > 0 && (
          <div className="mt-1 px-2 py-1 bg-red-50 rounded text-xs font-medium text-red-700">
            Total OOS Volume: {totalOos.toLocaleString()} cases
          </div>
        )}
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="h-[400px] flex items-center justify-center text-sm text-muted-foreground">Loading...</div>
        ) : !data?.bubbles || data.bubbles.length === 0 ? (
          <EmptyState height={400} />
        ) : (
          <div role="img" aria-roledescription="OOS impact bubble scatter chart">
            <ReactECharts option={option} style={{ height: 400 }} lazyUpdate notMerge={false} />
          </div>
        )}
      </CardContent>
    </Card>
  );
}
