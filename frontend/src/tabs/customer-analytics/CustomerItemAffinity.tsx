import { useMemo } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ModularReactECharts as ReactECharts } from "@/components/echarts-modular";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsAffinity,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters } from "@/api/queries/customer-analytics";
import { ExportButtons } from "./ExportButtons";
import { EmptyState } from "./EmptyState";

interface Props {
  filters: CustomerAnalyticsFilters;
}

export function CustomerItemAffinity({ filters }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.affinity(filters),
    queryFn: () => fetchCustomerAnalyticsAffinity(filters),
    staleTime: 60 * 60_000, // monthly data; pin to 1h to suppress thundering-herd refetches
    placeholderData: keepPreviousData, // keep prior chart visible during filter-change refetch
  });

  const option = useMemo(() => {
    if (!data) return {};
    // Backend returns objects ({customer_no, customer_name} and
    // {item_id, item_desc}); the cells reference customer_no/item_id.
    // Normalize to parallel label arrays + key-indexed maps so the heatmap
    // axes show human-readable names while cells look up by id.
    type CustObj = { customer_no: string; customer_name: string };
    type ItemObj = { item_id: string; item_desc: string };
    type Cell = { customer_no?: string; customer?: string; item_id?: string; item?: string; demand_qty: number };

    const rawCust = (data.customers ?? []) as unknown as Array<string | CustObj>;
    const rawItems = (data.items ?? []) as unknown as Array<string | ItemObj>;
    const rawCells = (data.cells ?? []) as unknown as Cell[];

    // Downsample if cell count blows past the rendering budget. We keep the
    // top customers and top items by total demand and drop everything else;
    // the high-signal cells the chart actually communicates stay intact.
    const MAX_CELLS = 2000;
    const MAX_AXIS = 50;
    let custKeys: string[] = rawCust.map((c) => (typeof c === "string" ? c : c.customer_no));
    let itemKeys: string[] = rawItems.map((i) => (typeof i === "string" ? i : i.item_id));
    let activeCells: Cell[] = rawCells;
    if (custKeys.length * itemKeys.length > MAX_CELLS) {
      const custDemand = new Map<string, number>();
      const itemDemand = new Map<string, number>();
      for (const c of rawCells) {
        const ck = c.customer_no ?? c.customer ?? "";
        const ik = c.item_id ?? c.item ?? "";
        custDemand.set(ck, (custDemand.get(ck) ?? 0) + c.demand_qty);
        itemDemand.set(ik, (itemDemand.get(ik) ?? 0) + c.demand_qty);
      }
      custKeys = [...custDemand.entries()]
        .sort((a, b) => b[1] - a[1]).slice(0, MAX_AXIS).map((e) => e[0]);
      itemKeys = [...itemDemand.entries()]
        .sort((a, b) => b[1] - a[1]).slice(0, MAX_AXIS).map((e) => e[0]);
      const custSet = new Set(custKeys);
      const itemSet = new Set(itemKeys);
      activeCells = rawCells.filter((c) => {
        const ck = c.customer_no ?? c.customer ?? "";
        const ik = c.item_id ?? c.item ?? "";
        return custSet.has(ck) && itemSet.has(ik);
      });
    }

    // Re-derive labels in sync with possibly-downsampled keys.
    const custLabel = new Map<string, string>(
      rawCust.map((c) => typeof c === "string" ? [c, c] : [c.customer_no, c.customer_name || c.customer_no]),
    );
    const itemLabel = new Map<string, string>(
      rawItems.map((i) => typeof i === "string" ? [i, i] : [i.item_id, i.item_desc || i.item_id]),
    );
    const customers = custKeys.map((k) => custLabel.get(k) ?? k);
    const items = itemKeys.map((k) => itemLabel.get(k) ?? k);

    const custKeyIdx = new Map<string, number>(custKeys.map((k, i) => [k, i]));
    const itemKeyIdx = new Map<string, number>(itemKeys.map((k, i) => [k, i]));

    const cellData = activeCells.map((c) => {
      const custKey = c.customer_no ?? c.customer ?? "";
      const itemKey = c.item_id ?? c.item ?? "";
      const x = itemKeyIdx.get(itemKey) ?? -1;
      const y = custKeyIdx.get(custKey) ?? -1;
      return [x, y, c.demand_qty];
    });

    const maxVal = cellData.length > 0
      ? Math.max(...cellData.map((c) => c[2] as number), 1)
      : 1;

    return {
      animation: false,  // up to ~10K cells; animation chokes the main thread
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
        ) : !data?.cells || data.cells.length === 0 ? (
          <EmptyState height={400} />
        ) : (
          <div role="img" aria-roledescription="Customer-item affinity heatmap chart">
            <ReactECharts option={option} style={{ height: 400 }} lazyUpdate notMerge={false} />
          </div>
        )}
      </CardContent>
    </Card>
  );
}
