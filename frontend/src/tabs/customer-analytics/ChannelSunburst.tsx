import { useMemo, useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ModularReactECharts as ReactECharts } from "@/components/echarts-modular";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsChannelMix,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters } from "@/api/queries/customer-analytics";
import { ExportButtons } from "./ExportButtons";
import { EmptyState } from "./EmptyState";

type SunburstMetric = "demand" | "customers";

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
  const [sunburstMetric, setSunburstMetric] = useState<SunburstMetric>("demand");

  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.channelMix(filters),
    queryFn: () => fetchCustomerAnalyticsChannelMix(filters),
    staleTime: 60 * 60_000, // monthly data; pin to 1h to suppress thundering-herd refetches
    placeholderData: keepPreviousData, // keep prior chart visible during filter-change refetch
  });

  const topChannelName = useMemo(() => {
    const tree = data?.tree ?? [];
    if (tree.length === 0) return "";
    const sorted = [...tree].sort((a, b) => b.value - a.value);
    return sorted[0].name;
  }, [data]);

  const option = useMemo(() => {
    const tree = data?.tree ?? [];
    const grandTotal = data?.grand_total ?? 0;
    const totalCustomers = data?.total_customers ?? 0;

    // For customer count metric, use customer_count values if available
    const mapNodeForMetric = (node: Record<string, unknown>): Record<string, unknown> => {
      if (sunburstMetric === "customers" && node.customer_count != null) {
        return {
          ...node,
          value: node.customer_count,
          children: ((node.children ?? []) as Record<string, unknown>[]).map(mapNodeForMetric),
        };
      }
      return {
        ...node,
        children: ((node.children ?? []) as Record<string, unknown>[]).map(mapNodeForMetric),
      };
    };

    // Assign colors to top-level channels
    const coloredTree = tree.map((ch, i) => ({
      ...mapNodeForMetric(ch as unknown as Record<string, unknown>),
      itemStyle: { color: CHANNEL_PALETTE[i % CHANNEL_PALETTE.length] },
    }));

    const centerLabel = sunburstMetric === "demand"
      ? `${fmtNum(grandTotal)} cases`
      : `${fmtNum(totalCustomers)} customers`;

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
      graphic: [
        {
          type: "text",
          left: "center",
          top: "center",
          style: {
            text: `${centerLabel}\n${topChannelName ? topChannelName : ""}`,
            textAlign: "center",
            fontSize: 13,
            fontWeight: "bold",
            fill: "#333",
            lineHeight: 18,
          },
        },
      ],
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
  }, [data, sunburstMetric, topChannelName]);

  return (
    <Card aria-label="Channel mix sunburst chart">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Channel Mix</CardTitle>
          <ExportButtons panelId="channel-mix" getData={() => data?.tree ?? []} />
        </div>
        <p className="text-xs text-muted-foreground">
          Channel &gt; Store Type &gt; Sub-Channel. Click a segment to zoom in.
        </p>
        <div className="flex gap-1 mt-1">
          {(["demand", "customers"] as SunburstMetric[]).map((m) => (
            <button
              key={m}
              onClick={() => setSunburstMetric(m)}
              className={`px-2 py-0.5 text-xs rounded ${sunburstMetric === m ? "bg-indigo-600 text-white" : "bg-gray-100 text-gray-600"}`}
            >
              {m === "demand" ? "Demand Volume" : "Customer Count"}
            </button>
          ))}
        </div>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="h-[420px] flex items-center justify-center text-sm text-muted-foreground">Loading...</div>
        ) : !data?.tree || data.tree.length === 0 ? (
          <EmptyState height={420} />
        ) : (
          <div role="img" aria-roledescription="Channel mix sunburst chart">
            <ReactECharts option={option} style={{ height: 420 }} lazyUpdate notMerge={false} />
          </div>
        )}
      </CardContent>
    </Card>
  );
}
