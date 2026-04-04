import { useQuery } from "@tanstack/react-query";
import ReactECharts from "echarts-for-react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsOrderPatterns,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters } from "@/api/queries/customer-analytics";
import { useMemo } from "react";
import { ExportButtons } from "./ExportButtons";

interface Props {
  filters: CustomerAnalyticsFilters;
}

export function OrderPatterns({ filters }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.orderPatterns(filters),
    queryFn: () => fetchCustomerAnalyticsOrderPatterns(filters),
    staleTime: 5 * 60_000,
  });

  const scatterOption = useMemo(() => {
    if (!data) return {};
    const points = data.regularity;
    const maxOrders = points.length > 0 ? Math.max(...points.map((p) => p.total_orders), 1) : 1;

    return {
      tooltip: {
        formatter: (p: { value: [number, number, number, string] }) => {
          const [interval, cv, orders, name] = p.value;
          return `<b>${name}</b><br/>Avg interval: ${interval.toFixed(1)} days<br/>CV: ${cv.toFixed(2)}<br/>Orders: ${orders}`;
        },
      },
      grid: { left: 50, right: 20, top: 10, bottom: 40 },
      xAxis: {
        name: "Avg Interval (days)",
        nameLocation: "center" as const,
        nameGap: 25,
        type: "value" as const,
      },
      yAxis: {
        name: "CV (regularity)",
        nameLocation: "center" as const,
        nameGap: 35,
        type: "value" as const,
      },
      series: [{
        type: "scatter",
        data: points.map((p) => [p.avg_interval, p.cv, p.total_orders, p.customer]),
        symbolSize: (val: number[]) => 6 + 16 * Math.sqrt(val[2] / maxOrders),
        itemStyle: { color: "#6366f1", opacity: 0.65 },
        emphasis: { itemStyle: { opacity: 1 } },
      }],
    };
  }, [data]);

  const freqData = data?.frequency ?? [];

  return (
    <Card aria-label="Order patterns analysis">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Order Patterns</CardTitle>
          <ExportButtons panelId="order-patterns" getData={() => data?.frequency ?? []} />
        </div>
        <p className="text-xs text-muted-foreground">Order frequency histogram and regularity scatter</p>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="h-[400px] flex items-center justify-center text-sm text-muted-foreground">Loading...</div>
        ) : (
          <div className="space-y-4">
            <div>
              <p className="text-xs font-medium text-muted-foreground mb-1">Order Frequency</p>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={freqData} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
                  <XAxis dataKey="bin" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div>
              <p className="text-xs font-medium text-muted-foreground mb-1">Order Regularity</p>
              <div role="img" aria-roledescription="Order regularity scatter chart">
                <ReactECharts option={scatterOption} style={{ height: 200 }} />
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
