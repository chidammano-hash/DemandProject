import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import ReactECharts from "echarts-for-react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsLifecycle,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters } from "@/api/queries/customer-analytics";
import { ExportButtons } from "./ExportButtons";

interface Props {
  filters: CustomerAnalyticsFilters;
}

export function CustomerLifecycle({ filters }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.lifecycle(filters),
    queryFn: () => fetchCustomerAnalyticsLifecycle(filters),
    staleTime: 5 * 60_000,
  });

  const heatmapOption = useMemo(() => {
    if (!data) return {};
    const cohorts = data.cohort_months;
    const maxMonths = data.max_months_since;
    const monthCols = Array.from({ length: maxMonths + 1 }, (_, i) => `M${i}`);

    const cellData = data.cohort_heatmap.map((c) => [
      c.months_since,
      cohorts.indexOf(c.cohort_month),
      c.retention_pct,
    ]);

    return {
      tooltip: {
        position: "top",
        formatter: (p: { value: [number, number, number] }) => {
          const [ms, ci, pct] = p.value;
          return `<b>${cohorts[ci]}</b> + ${ms} months<br/>Retention: ${pct.toFixed(1)}%`;
        },
      },
      grid: { left: 100, right: 20, top: 10, bottom: 60 },
      xAxis: { type: "category" as const, data: monthCols, axisLabel: { fontSize: 10 } },
      yAxis: { type: "category" as const, data: cohorts, axisLabel: { fontSize: 10 } },
      visualMap: {
        min: 0,
        max: 100,
        calculable: true,
        orient: "horizontal" as const,
        left: "center",
        bottom: 0,
        inRange: { color: ["#fee2e2", "#fef08a", "#bbf7d0", "#22c55e"] },
      },
      series: [{
        type: "heatmap",
        data: cellData,
        label: { show: false },
        emphasis: { itemStyle: { shadowBlur: 10, shadowColor: "rgba(0,0,0,0.5)" } },
      }],
    };
  }, [data]);

  const waterfallData = useMemo(() => {
    if (!data) return [];
    return data.waterfall;
  }, [data]);

  const waterfallColors: Record<string, string> = {
    new: "#22c55e",
    churned: "#ef4444",
    net: "#94a3b8",
  };

  return (
    <Card aria-label="Customer lifecycle analysis">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Customer Lifecycle</CardTitle>
          <ExportButtons panelId="lifecycle" getData={() => data?.waterfall ?? []} />
        </div>
        <p className="text-xs text-muted-foreground">Cohort retention heatmap and new/churned waterfall</p>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="h-[400px] flex items-center justify-center text-sm text-muted-foreground">Loading...</div>
        ) : (
          <div className="space-y-4">
            <div role="img" aria-roledescription="Cohort retention heatmap">
              <ReactECharts option={heatmapOption} style={{ height: 260 }} />
            </div>
            <div>
              <p className="text-xs font-medium text-muted-foreground mb-1">New vs Churned Customers</p>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={waterfallData} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
                  <XAxis dataKey="label" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip formatter={(v: number) => v.toLocaleString()} />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {waterfallData.map((d, i) => (
                      <Cell key={i} fill={waterfallColors[d.type] ?? "#94a3b8"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
