import { useMemo } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ModularReactECharts as ReactECharts } from "@/components/echarts-modular";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsTreemap,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters } from "@/api/queries/customer-analytics";
import { ExportButtons } from "./ExportButtons";

interface Props {
  filters: CustomerAnalyticsFilters;
}

export function CustomerTreemap({ filters }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.treemap(filters),
    queryFn: () => fetchCustomerAnalyticsTreemap(filters),
    staleTime: 60 * 60_000, // monthly data; pin to 1h to suppress thundering-herd refetches
    placeholderData: keepPreviousData, // keep prior chart visible during filter-change refetch
  });

  // Memoize so the option object identity is stable across parent re-renders
  // (filter keystrokes, theme toggles, etc.). Without this ECharts re-runs the
  // full treemap layout on every render even when `data` is unchanged.
  const option = useMemo(() => ({
    tooltip: {
      formatter: (p: { name: string; value: number; data?: { fill_rate?: number } }) => {
        const fr = p.data?.fill_rate;
        return `<b>${p.name}</b><br/>Demand: ${(p.value ?? 0).toLocaleString()} cases${fr != null ? `<br/>Fill Rate: ${fr}%` : ""}`;
      },
    },
    visualMap: {
      show: true,
      min: 0,
      max: 100,
      text: ["100%", "0%"],
      dimension: "fill_rate",
      inRange: {
        color: ["#ef4444", "#eab308", "#22c55e"],
      },
      calculable: true,
      orient: "horizontal" as const,
      left: "center",
      bottom: 0,
    },
    series: [
      {
        type: "treemap",
        data: data?.tree ?? [],
        leafDepth: 2,
        roam: false,
        animation: false, // skip layout animation on large trees
        label: { show: true, formatter: "{b}", fontSize: 11 },
        breadcrumb: { show: true },
        levels: [
          { itemStyle: { borderWidth: 2, borderColor: "#fff", gapWidth: 2 } },
          { itemStyle: { borderWidth: 1, borderColor: "#ddd", gapWidth: 1 } },
          { itemStyle: { borderWidth: 0 } },
        ],
      },
    ],
  }), [data]);

  return (
    <Card aria-label="Customer concentration treemap">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Customer Concentration</CardTitle>
          <ExportButtons panelId="treemap" getData={() => data?.tree ?? []} />
        </div>
        <p className="text-xs text-muted-foreground">State &gt; Channel &gt; Customer by demand. Color = fill rate.</p>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="h-[360px] flex items-center justify-center text-sm text-muted-foreground">Loading...</div>
        ) : !data?.tree || data.tree.length === 0 ? (
          <div className="h-[360px] flex flex-col items-center justify-center text-sm text-muted-foreground gap-1">
            <span className="font-medium">No data for the selected filters</span>
            <span className="text-xs">Try a different item or widen the date range</span>
          </div>
        ) : (
          <div role="img" aria-roledescription="Customer concentration treemap chart">
            <ReactECharts option={option} style={{ height: 360 }} lazyUpdate notMerge={false} />
          </div>
        )}
      </CardContent>
    </Card>
  );
}
