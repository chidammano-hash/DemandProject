import { useMemo } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ResponsiveContainer, Treemap, Tooltip } from "recharts";
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

interface TreeNode {
  name: string;
  size?: number;
  value?: number;
  fill_rate?: number;
  fill?: string;
  children?: TreeNode[];
}

/** Map fill_rate (0-100) to a red->yellow->green color (matches the prior ECharts visualMap). */
function fillRateColor(fr: number): string {
  // Red #ef4444 -> Yellow #eab308 -> Green #22c55e
  const clamped = Math.max(0, Math.min(100, fr));
  if (clamped <= 50) {
    // Red -> Yellow on [0, 50]
    const t = clamped / 50;
    const r = Math.round(0xef + (0xea - 0xef) * t);
    const g = Math.round(0x44 + (0xb3 - 0x44) * t);
    const b = Math.round(0x44 + (0x08 - 0x44) * t);
    return `rgb(${r}, ${g}, ${b})`;
  }
  // Yellow -> Green on [50, 100]
  const t = (clamped - 50) / 50;
  const r = Math.round(0xea + (0x22 - 0xea) * t);
  const g = Math.round(0xb3 + (0xc5 - 0xb3) * t);
  const b = Math.round(0x08 + (0x5e - 0x08) * t);
  return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Recharts Treemap expects `{name, size}` (or whatever the dataKey is) at
 * leaf nodes. Recursively coerce the API tree (which uses `value`) to the
 * recharts shape and attach a per-leaf fill computed from fill_rate.
 */
function toRechartsTree(nodes: TreeNode[]): TreeNode[] {
  return nodes.map((n) => {
    const kids = n.children ? toRechartsTree(n.children) : undefined;
    const size = n.value ?? n.size ?? 0;
    const fr = n.fill_rate;
    const fill = fr != null ? fillRateColor(fr) : undefined;
    return {
      name: n.name,
      size,
      value: size,
      fill_rate: fr,
      fill,
      children: kids,
    };
  });
}

interface RechartsTooltipProps {
  active?: boolean;
  payload?: Array<{ payload: TreeNode & { root?: TreeNode } }>;
}

function TreemapTooltip({ active, payload }: RechartsTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;
  const node = payload[0].payload;
  const value = node.value ?? node.size ?? 0;
  return (
    <div className="rounded-md border bg-background p-2 text-xs shadow-sm">
      <div className="font-semibold">{node.name}</div>
      <div>Demand: {value.toLocaleString()} cases</div>
      {node.fill_rate != null && <div>Fill Rate: {node.fill_rate}%</div>}
    </div>
  );
}

export function CustomerTreemap({ filters }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.treemap(filters),
    queryFn: () => fetchCustomerAnalyticsTreemap(filters),
    staleTime: 60 * 60_000,
    placeholderData: keepPreviousData,
  });

  // Recharts wants a single root node — wrap the API forest under a synthetic
  // root so the chart shows the full set of channels at the top level.
  const treeData = useMemo<TreeNode[]>(() => {
    const tree = (data?.tree ?? []) as TreeNode[];
    if (tree.length === 0) return [];
    return toRechartsTree(tree);
  }, [data]);

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
            <ResponsiveContainer width="100%" height={360}>
              <Treemap
                data={treeData}
                dataKey="size"
                nameKey="name"
                stroke="#fff"
                isAnimationActive={false}
              >
                <Tooltip content={<TreemapTooltip />} />
              </Treemap>
            </ResponsiveContainer>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
