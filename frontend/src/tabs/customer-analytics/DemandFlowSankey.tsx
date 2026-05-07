import { useMemo } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ModularReactECharts as ReactECharts } from "@/components/echarts-modular";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsDemandFlow,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters } from "@/api/queries/customer-analytics";
import { ExportButtons } from "./ExportButtons";
import { PanelStateGate } from "@/components/PanelStateGate";

interface Props {
  filters: CustomerAnalyticsFilters;
}

export function DemandFlowSankey({ filters }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.demandFlow(filters),
    queryFn: () => fetchCustomerAnalyticsDemandFlow(filters),
    staleTime: 60 * 60_000, // monthly data; pin to 1h to suppress thundering-herd refetches
    placeholderData: keepPreviousData, // keep prior chart visible during filter-change refetch
  });

  const option = useMemo(() => {
    if (!data) return {};
    return {
      animation: false,  // sankey can have hundreds of links — no point animating
      tooltip: {
        trigger: "item" as const,
        triggerOn: "mousemove" as const,
      },
      series: [{
        type: "sankey",
        data: data.nodes,
        links: data.links,
        orient: "horizontal" as const,
        emphasis: { focus: "adjacency" as const },
        lineStyle: { color: "gradient" as const, curveness: 0.5 },
        label: { fontSize: 11 },
        nodeWidth: 20,
        nodeGap: 12,
      }],
    };
  }, [data]);

  return (
    <Card aria-label="Demand flow sankey diagram">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Demand Flow</CardTitle>
          <ExportButtons panelId="demand-flow" getData={() => data?.links ?? []} />
        </div>
        <p className="text-xs text-muted-foreground">Warehouse to State to Channel demand flow</p>
      </CardHeader>
      <CardContent>
        <PanelStateGate
          isLoading={isLoading}
          isEmpty={!data?.links || data.links.length === 0}
          height={400}
        >
          <div role="img" aria-roledescription="Demand flow sankey diagram">
            <ReactECharts option={option} style={{ height: 400 }} lazyUpdate notMerge={false} />
          </div>
        </PanelStateGate>
      </CardContent>
    </Card>
  );
}
