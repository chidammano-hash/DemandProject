import { useMemo } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ModularReactECharts as ReactECharts } from "@/components/echarts-modular";
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
import { EmptyState } from "./EmptyState";

interface Props {
  filters: CustomerAnalyticsFilters;
}

// Backend response shape:
//   { cohorts: [{cohort_month, months_since: number[], retention_pct: number[]}],
//     waterfall: [{month, new_customers, churned_customers, net_change}] }
// Flatten into heatmap cells + waterfall bars in the shape the chart needs.
interface LifecycleResponse {
  cohorts: Array<{ cohort_month: string; months_since: number[]; retention_pct: number[] }>;
  waterfall: Array<{ month: string; new_customers: number; churned_customers: number; net_change: number }>;
}

export function CustomerLifecycle({ filters }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.lifecycle(filters),
    queryFn: () => fetchCustomerAnalyticsLifecycle(filters),
    staleTime: 60 * 60_000, // monthly data; pin to 1h to suppress thundering-herd refetches
    placeholderData: keepPreviousData, // keep prior chart visible during filter-change refetch
  });

  const { cohortMonths, maxMonths, heatmapCells, waterfallBars } = useMemo(() => {
    const resp = data as LifecycleResponse | null | undefined;
    const cohorts = resp?.cohorts ?? [];
    const waterfall = resp?.waterfall ?? [];
    const months = cohorts.map((c) => c.cohort_month);
    let mMax = 0;
    const cells: Array<[number, number, number]> = [];
    cohorts.forEach((c, ci) => {
      c.months_since.forEach((ms, i) => {
        if (ms > mMax) mMax = ms;
        cells.push([ms, ci, c.retention_pct[i] ?? 0]);
      });
    });
    const bars = waterfall.flatMap((w) => [
      { label: w.month, value: w.new_customers, type: "new" as const },
      { label: w.month, value: -w.churned_customers, type: "churned" as const, displayValue: w.churned_customers },
    ]);
    return { cohortMonths: months, maxMonths: mMax, heatmapCells: cells, waterfallBars: bars };
  }, [data]);

  const heatmapOption = useMemo(() => {
    if (cohortMonths.length === 0) return {};
    const monthCols = Array.from({ length: maxMonths + 1 }, (_, i) => `M${i}`);
    return {
      tooltip: {
        position: "top",
        formatter: (p: { value: [number, number, number] }) => {
          const [ms, ci, pct] = p.value;
          return `<b>${cohortMonths[ci]}</b> + ${ms} months<br/>Retention: ${pct.toFixed(1)}%`;
        },
      },
      grid: { left: 100, right: 20, top: 10, bottom: 60 },
      xAxis: { type: "category" as const, data: monthCols, axisLabel: { fontSize: 10 } },
      yAxis: { type: "category" as const, data: cohortMonths, axisLabel: { fontSize: 10 } },
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
        data: heatmapCells,
        label: { show: false },
        emphasis: { itemStyle: { shadowBlur: 10, shadowColor: "rgba(0,0,0,0.5)" } },
      }],
    };
  }, [cohortMonths, maxMonths, heatmapCells]);

  const waterfallData = waterfallBars;

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
          <ExportButtons panelId="lifecycle" getData={() => waterfallBars as unknown as Record<string, unknown>[]} />
        </div>
        <p className="text-xs text-muted-foreground">Cohort retention heatmap and new/churned waterfall</p>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="h-[400px] flex items-center justify-center text-sm text-muted-foreground">Loading...</div>
        ) : cohortMonths.length === 0 && waterfallBars.length === 0 ? (
          <EmptyState height={400} />
        ) : (
          <div className="space-y-4">
            <div role="img" aria-roledescription="Cohort retention heatmap">
              <ReactECharts option={heatmapOption} style={{ height: 260 }} lazyUpdate notMerge={false} />
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
