import { useQuery } from "@tanstack/react-query";
import ReactECharts from "echarts-for-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsTreemap,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters } from "@/api/queries/customer-analytics";

interface Props {
  filters: CustomerAnalyticsFilters;
}

export function CustomerTreemap({ filters }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.treemap(filters),
    queryFn: () => fetchCustomerAnalyticsTreemap(filters),
    staleTime: 5 * 60_000,
  });

  const option = {
    tooltip: {
      formatter: (p: { name: string; value: number; data?: { fill_rate?: number } }) => {
        const fr = p.data?.fill_rate;
        return `<b>${p.name}</b><br/>Demand: ${(p.value ?? 0).toLocaleString()} cases${fr != null ? `<br/>Fill Rate: ${fr}%` : ""}`;
      },
    },
    series: [
      {
        type: "treemap",
        data: data?.tree ?? [],
        leafDepth: 2,
        roam: false,
        label: { show: true, formatter: "{b}", fontSize: 11 },
        breadcrumb: { show: true },
        levels: [
          { itemStyle: { borderWidth: 2, borderColor: "#fff", gapWidth: 2 } },
          { itemStyle: { borderWidth: 1, borderColor: "#ddd", gapWidth: 1 }, colorSaturation: [0.3, 0.7] },
          { itemStyle: { borderWidth: 0 }, colorSaturation: [0.3, 0.6] },
        ],
      },
    ],
  };

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Customer Concentration</CardTitle>
        <p className="text-xs text-muted-foreground">State &gt; Channel &gt; Customer by demand volume</p>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="h-[360px] flex items-center justify-center text-sm text-muted-foreground">Loading...</div>
        ) : (
          <ReactECharts option={option} style={{ height: 360 }} />
        )}
      </CardContent>
    </Card>
  );
}
