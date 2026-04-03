import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import ReactECharts from "echarts-for-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsHeatmap,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters } from "@/api/queries/customer-analytics";

type HeatmapMetric = "demand_qty" | "customer_count" | "fill_rate";

interface Props {
  filters: CustomerAnalyticsFilters;
  metric: HeatmapMetric;
  topN: number;
}

export function CustomerHeatmap({ filters, metric, topN }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.heatmap(metric, topN, filters),
    queryFn: () => fetchCustomerAnalyticsHeatmap(metric, topN, filters),
    staleTime: 5 * 60_000,
  });

  const option = useMemo(() => {
    if (!data) return {};
    const items = data.items.map((i) => i.item_desc);
    const states = data.states;

    const cellData = data.cells.map((c) => {
      const x = states.indexOf(c.state);
      const y = items.indexOf(data.items.find((i) => i.item_id === c.item_id)?.item_desc ?? "");
      const val = metric === "fill_rate" ? c.fill_rate : metric === "customer_count" ? c.customer_count : c.demand_qty;
      return [x, y, val];
    });

    const maxVal = Math.max(...cellData.map((c) => c[2] as number), 1);

    return {
      tooltip: {
        position: "top",
        formatter: (p: { value: [number, number, number] }) => {
          const [x, y, v] = p.value;
          return `<b>${items[y]}</b> — ${states[x]}<br/>${metric === "fill_rate" ? `${v}%` : v.toLocaleString()}`;
        },
      },
      grid: { left: 180, right: 20, top: 10, bottom: 60 },
      xAxis: { type: "category" as const, data: states, axisLabel: { fontSize: 10, rotate: 45 } },
      yAxis: { type: "category" as const, data: items, axisLabel: { fontSize: 10, width: 160, overflow: "truncate" as const } },
      visualMap: {
        min: 0,
        max: maxVal,
        calculable: true,
        orient: "horizontal" as const,
        left: "center",
        bottom: 0,
        inRange: {
          color: metric === "fill_rate"
            ? ["#ef4444", "#eab308", "#22c55e"]
            : ["#eff6ff", "#3b82f6", "#1e3a5f"],
        },
      },
      series: [
        {
          type: "heatmap",
          data: cellData,
          label: { show: false },
          emphasis: { itemStyle: { shadowBlur: 10, shadowColor: "rgba(0,0,0,0.5)" } },
        },
      ],
    };
  }, [data, metric]);

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Item x State Heatmap</CardTitle>
        <p className="text-xs text-muted-foreground">Top {topN} items by demand across states</p>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="h-[400px] flex items-center justify-center text-sm text-muted-foreground">Loading...</div>
        ) : (
          <ReactECharts option={option} style={{ height: 400 }} />
        )}
      </CardContent>
    </Card>
  );
}
