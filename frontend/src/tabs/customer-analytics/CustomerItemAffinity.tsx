import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import ReactECharts from "echarts-for-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsAffinity,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters } from "@/api/queries/customer-analytics";
import { ExportButtons } from "./ExportButtons";

interface Props {
  filters: CustomerAnalyticsFilters;
}

export function CustomerItemAffinity({ filters }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.affinity(filters),
    queryFn: () => fetchCustomerAnalyticsAffinity(filters),
    staleTime: 5 * 60_000,
  });

  const option = useMemo(() => {
    if (!data) return {};
    const customers = data.customers;
    const items = data.items;

    const cellData = data.cells.map((c) => {
      const x = items.indexOf(c.item);
      const y = customers.indexOf(c.customer);
      return [x, y, c.demand_qty];
    });

    const maxVal = cellData.length > 0
      ? Math.max(...cellData.map((c) => c[2] as number), 1)
      : 1;

    return {
      tooltip: {
        position: "top",
        formatter: (p: { value: [number, number, number] }) => {
          const [x, y, v] = p.value;
          return `<b>${customers[y]}</b> x ${items[x]}<br/>Demand: ${v.toLocaleString()}`;
        },
      },
      grid: { left: 120, right: 20, top: 10, bottom: 60 },
      xAxis: {
        type: "category" as const,
        data: items,
        axisLabel: { fontSize: 9, rotate: 45 },
      },
      yAxis: {
        type: "category" as const,
        data: customers,
        axisLabel: { fontSize: 9, width: 110, overflow: "truncate" as const },
      },
      visualMap: {
        min: 0,
        max: maxVal,
        calculable: true,
        orient: "horizontal" as const,
        left: "center",
        bottom: 0,
        inRange: { color: ["#eff6ff", "#3b82f6", "#1e3a5f"] },
      },
      series: [{
        type: "heatmap",
        data: cellData,
        label: { show: false },
        emphasis: { itemStyle: { shadowBlur: 10, shadowColor: "rgba(0,0,0,0.5)" } },
      }],
    };
  }, [data]);

  return (
    <Card aria-label="Customer item affinity heatmap">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Customer-Item Affinity</CardTitle>
          <ExportButtons panelId="affinity" getData={() => data?.cells ?? []} />
        </div>
        <p className="text-xs text-muted-foreground">Customers vs items by demand volume</p>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="h-[400px] flex items-center justify-center text-sm text-muted-foreground">Loading...</div>
        ) : (
          <div role="img" aria-roledescription="Customer-item affinity heatmap chart">
            <ReactECharts option={option} style={{ height: 400 }} />
          </div>
        )}
      </CardContent>
    </Card>
  );
}
