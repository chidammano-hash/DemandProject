import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ModularReactECharts as ReactECharts } from "@/components/echarts-modular";
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
import { EmptyState } from "./EmptyState";

interface Props {
  filters: CustomerAnalyticsFilters;
}

export function OrderPatterns({ filters }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.orderPatterns(filters),
    queryFn: () => fetchCustomerAnalyticsOrderPatterns(filters),
    staleTime: 60 * 60_000, // monthly data; pin to 1h to suppress thundering-herd refetches
    placeholderData: keepPreviousData, // keep prior chart visible during filter-change refetch
  });

  // Backend returns frequency_histogram: [{bucket, count, pct}] and
  // regularity_scatter: [{customer_no, customer_name, avg_interval_months,
  // interval_cv, total_demand}]. Normalize to the shape the charts use,
  // tolerating the legacy {frequency, regularity} naming as a fallback.
  type RawFreq = { bucket?: string; bin?: string; count: number };
  type RawReg = {
    customer_no?: string;
    customer_name?: string;
    customer?: string;
    avg_interval_months?: number;
    avg_interval?: number;
    interval_cv?: number;
    cv?: number;
    total_demand?: number;
    total_orders?: number;
  };
  const raw = data as
    | { frequency_histogram?: RawFreq[]; regularity_scatter?: RawReg[]; frequency?: RawFreq[]; regularity?: RawReg[] }
    | null
    | undefined;
  const freqRaw = raw?.frequency_histogram ?? raw?.frequency ?? [];
  const regRaw = raw?.regularity_scatter ?? raw?.regularity ?? [];

  const freqData = useMemo(
    () => freqRaw.map((f) => ({ bin: f.bin ?? f.bucket ?? "", count: f.count })),
    [freqRaw],
  );

  const scatterOption = useMemo(() => {
    if (!data) return {};
    const points = regRaw.map((p) => ({
      customer: p.customer ?? p.customer_name ?? p.customer_no ?? "",
      avg_interval: p.avg_interval ?? p.avg_interval_months ?? 0,
      cv: p.cv ?? p.interval_cv ?? 0,
      total_orders: p.total_orders ?? p.total_demand ?? 0,
    }));
    const maxOrders = points.length > 0 ? Math.max(...points.map((p) => p.total_orders), 1) : 1;

    return {
      tooltip: {
        formatter: (p: { value: [number, number, number, string] }) => {
          const [interval, cv, orders, name] = p.value;
          return `<b>${name}</b><br/>Avg interval: ${interval.toFixed(1)} months<br/>CV: ${cv.toFixed(2)}<br/>Demand: ${orders.toLocaleString()}`;
        },
      },
      grid: { left: 50, right: 20, top: 10, bottom: 40 },
      xAxis: {
        name: "Avg Interval (months)",
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
  }, [data, regRaw]);

  return (
    <Card aria-label="Order patterns analysis">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Order Patterns</CardTitle>
          <ExportButtons panelId="order-patterns" getData={() => freqRaw as unknown as Record<string, unknown>[]} />
        </div>
        <p className="text-xs text-muted-foreground">Order frequency histogram and regularity scatter</p>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="h-[400px] flex items-center justify-center text-sm text-muted-foreground">Loading...</div>
        ) : freqRaw.length === 0 && regRaw.length === 0 ? (
          <EmptyState height={400} />
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
                <ReactECharts option={scatterOption} style={{ height: 200 }} lazyUpdate notMerge={false} />
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
