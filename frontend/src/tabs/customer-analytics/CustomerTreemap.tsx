import { useMemo } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ModularReactECharts as ReactECharts } from "@/components/echarts-modular";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsTreemap,
} from "@/api/queries/customer-analytics";
import type {
  CustomerAnalyticsFilters,
  TreemapNode,
} from "@/api/queries/customer-analytics";
import { ExportButtons } from "./ExportButtons";

interface Props {
  filters: CustomerAnalyticsFilters;
}

// Fill-rate color ramp: red -> amber -> green. Applied per node via
// itemStyle.color rather than ECharts `visualMap.dimension`, which expects a
// numeric index into a node's `value` array — a treemap node's `value` is a
// scalar, so the old `dimension: "fill_rate"` resolved every node out-of-range
// and painted the whole treemap transparent/blank (U7.1).
const RAMP_LOW = "#ef4444"; // red — band floor
const RAMP_MID = "#eab308"; // amber — band midpoint
const RAMP_HIGH = "#22c55e"; // green — band ceiling
const NEUTRAL = "#94a3b8"; // slate — node with no fill_rate

// Business band the ramp spans. Real fill rates cluster in ~95-100%, so a 0-100
// ramp painted every node the same green and the legend (0%..100%) implied a
// dynamic range the data never uses — a 95.1% node (worth attention) looked
// identical to a 99.9% node (U3.11). Anchoring the ramp to 90-100% restores
// perceivable variation across the band the data actually occupies; values
// outside the band clamp to the endpoints.
export const FILL_RATE_BAND: readonly [number, number] = [90, 100];

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function lerp(a: number, b: number, t: number): number {
  return Math.round(a + (b - a) * t);
}

/** Map a fill rate to a ramp color across {@link FILL_RATE_BAND}. Values below
 *  the band floor clamp to red; above the ceiling clamp to green. Exported for
 *  unit testing the band mapping (U3.11). */
export function fillRateColor(fr: number | undefined): string {
  if (fr == null || Number.isNaN(fr)) return NEUTRAL;
  const [lo, hi] = FILL_RATE_BAND;
  // Normalize within the band rather than 0-100 so the 95-100 spread is visible.
  const pct = Math.max(0, Math.min(1, (fr - lo) / (hi - lo)));
  // Two-segment ramp: low->mid for the bottom half, mid->high for the top half.
  const [from, to, t] =
    pct <= 0.5
      ? ([hexToRgb(RAMP_LOW), hexToRgb(RAMP_MID), pct / 0.5] as const)
      : ([hexToRgb(RAMP_MID), hexToRgb(RAMP_HIGH), (pct - 0.5) / 0.5] as const);
  const [r, g, b] = from.map((c, i) => lerp(c, to[i], t));
  return `rgb(${r}, ${g}, ${b})`;
}

type ColoredNode = Omit<TreemapNode, "children"> & {
  itemStyle: { color: string };
  children?: ColoredNode[];
};

/** Recursively attach an explicit `itemStyle.color` (mapped from `fill_rate`) to
 *  every node so the treemap draws without a fragile `visualMap` binding. */
function colorizeTree(nodes: TreemapNode[]): ColoredNode[] {
  return nodes.map((node) => {
    const { children, ...rest } = node;
    const colored: ColoredNode = {
      ...rest,
      itemStyle: { color: fillRateColor(node.fill_rate) },
    };
    if (children) colored.children = colorizeTree(children);
    return colored;
  });
}

export function CustomerTreemap({ filters }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.treemap(filters),
    queryFn: () => fetchCustomerAnalyticsTreemap(filters),
    staleTime: 60 * 60_000, // monthly data; pin to 1h to suppress thundering-herd refetches
    placeholderData: keepPreviousData, // keep prior chart visible during filter-change refetch
  });

  // Memoize so the option object identity is stable across parent re-renders
  // (filter keystrokes, theme toggles, etc.). Without this ECharts re-runs the
  // full treemap layout on every render even when `data` is unchanged.
  const option = useMemo(() => ({
    tooltip: {
      formatter: (p: { name: string; value: number; data?: { fill_rate?: number } }) => {
        const fr = p.data?.fill_rate;
        return `<b>${p.name}</b><br/>Demand: ${(p.value ?? 0).toLocaleString()} cases${fr != null ? `<br/>Fill Rate: ${fr}%` : ""}`;
      },
    },
    series: [
      {
        type: "treemap",
        // Each node carries an explicit itemStyle.color mapped from fill_rate —
        // no visualMap.dimension (U7.1). A static legend below shows the ramp.
        data: colorizeTree(data?.tree ?? []),
        leafDepth: 2,
        roam: false,
        animation: false, // skip layout animation on large trees
        label: { show: true, formatter: "{b}", fontSize: 11 },
        breadcrumb: { show: true },
        levels: [
          { itemStyle: { borderWidth: 2, borderColor: "#fff", gapWidth: 2 } },
          { itemStyle: { borderWidth: 1, borderColor: "#ddd", gapWidth: 1 } },
          { itemStyle: { borderWidth: 0 } },
        ],
      },
    ],
  }), [data]);

  return (
    <Card aria-label="Customer concentration treemap">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Customer Concentration</CardTitle>
          <ExportButtons panelId="treemap" getData={() => data?.tree ?? []} />
        </div>
        <p className="text-xs text-muted-foreground">State &gt; Channel &gt; Customer by demand. Color = fill rate ({FILL_RATE_BAND[0]}&ndash;{FILL_RATE_BAND[1]}% band).</p>
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
          <div role="img" aria-roledescription="Customer concentration treemap chart" className="w-full">
            <ReactECharts option={option} style={{ height: 360, width: "100%" }} lazyUpdate notMerge={false} />
            {/* Static fill-rate legend — replaces the removed ECharts visualMap
                (U7.1). Mirrors the red->amber->green node ramp, labeled with the
                real business band endpoints (90-100%) so the legend reflects the
                data's actual range rather than a misleading 0-100 (U3.11). */}
            <div className="mt-2 flex items-center justify-center gap-2 text-xs text-muted-foreground">
              <span>{FILL_RATE_BAND[0]}%</span>
              <span
                aria-hidden
                className="h-2 w-40 rounded"
                style={{ background: `linear-gradient(to right, ${RAMP_LOW}, ${RAMP_MID}, ${RAMP_HIGH})` }}
              />
              <span>{FILL_RATE_BAND[1]}% fill rate</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
