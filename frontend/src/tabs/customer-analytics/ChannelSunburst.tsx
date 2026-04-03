import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import ReactECharts from "echarts-for-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsChannelMix,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters } from "@/api/queries/customer-analytics";

interface Props {
  filters: CustomerAnalyticsFilters;
}

// Distinct palette for top-level channels
const CHANNEL_PALETTE = [
  "#3b82f6", // blue
  "#8b5cf6", // violet
  "#06b6d4", // cyan
  "#f59e0b", // amber
  "#10b981", // emerald
  "#ef4444", // red
  "#ec4899", // pink
  "#64748b", // slate
];

function fmtNum(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toFixed(0);
}

export function ChannelSunburst({ filters }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.channelMix(filters),
    queryFn: () => fetchCustomerAnalyticsChannelMix(filters),
    staleTime: 5 * 60_000,
  });

  const option = useMemo(() => {
    const tree = data?.tree ?? [];
    const grandTotal = data?.grand_total ?? 0;

    // Assign colors to top-level channels
    const coloredTree = tree.map((ch, i) => ({
      ...ch,
      itemStyle: { color: CHANNEL_PALETTE[i % CHANNEL_PALETTE.length] },
      children: (ch.children ?? []).map((st: Record<string, unknown>) => ({
        ...st,
        // Store type children don't need explicit color — ECharts will derive
      })),
    }));

    return {
      tooltip: {
        trigger: "item" as const,
        formatter: (p: { name: string; value: number; data?: { customer_count?: number }; treePathInfo?: Array<{ name: string }> }) => {
          const cc = p.data?.customer_count;
          const pct = grandTotal > 0 ? ((p.value / grandTotal) * 100).toFixed(1) : "0";
          const path = (p.treePathInfo ?? []).map((x) => x.name).filter(Boolean).join(" > ");
          return `<div style="max-width:260px">
            <div style="font-size:11px;color:#666;margin-bottom:2px">${path}</div>
            <b>${p.name}</b><br/>
            Demand: ${fmtNum(p.value)} cases (${pct}%)
            ${cc != null ? `<br/>Customers: ${cc.toLocaleString()}` : ""}
          </div>`;
        },
      },
      series: [
        {
          type: "sunburst",
          data: coloredTree,
          radius: ["18%", "92%"],
          sort: "desc" as const,
          nodeClick: "rootToNode" as const,
          label: {
            rotate: "tangential" as const,
            fontSize: 11,
            fontWeight: "normal" as const,
            overflow: "truncate" as const,
            ellipsis: "...",
            minAngle: 8, // hide labels on arcs < 8 degrees
          },
          itemStyle: {
            borderWidth: 2,
            borderColor: "#fff",
            borderRadius: 4,
          },
          emphasis: {
            focus: "ancestor" as const,
            itemStyle: { shadowBlur: 10, shadowColor: "rgba(0,0,0,0.15)" },
          },
          levels: [
            {}, // root (invisible)
            {
              // Level 1: Channels
              r0: "18%",
              r: "42%",
              label: {
                fontSize: 13,
                fontWeight: "bold" as const,
                rotate: "tangential" as const,
                minAngle: 15,
              },
              itemStyle: { borderWidth: 3, borderColor: "#fff" },
            },
            {
              // Level 2: Store Types
              r0: "42%",
              r: "68%",
              label: {
                fontSize: 11,
                rotate: "tangential" as const,
                minAngle: 10,
              },
              itemStyle: { borderWidth: 2, borderColor: "#fff" },
            },
            {
              // Level 3: Sub-Channels
              r0: "68%",
              r: "92%",
              label: {
                fontSize: 9,
                rotate: "tangential" as const,
                minAngle: 12, // stricter threshold — hide most small outer labels
                color: "#555",
              },
              itemStyle: { borderWidth: 1, borderColor: "rgba(255,255,255,0.6)" },
            },
          ],
        },
      ],
    };
  }, [data]);

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Channel Mix</CardTitle>
        <p className="text-xs text-muted-foreground">
          Channel &gt; Store Type &gt; Sub-Channel by demand. Click a segment to zoom in.
        </p>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="h-[420px] flex items-center justify-center text-sm text-muted-foreground">Loading...</div>
        ) : (
          <ReactECharts option={option} style={{ height: 420 }} />
        )}
      </CardContent>
    </Card>
  );
}
