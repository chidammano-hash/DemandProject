import { useMemo, useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { SunburstChart } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsChannelMix,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters } from "@/api/queries/customer-analytics";
import { ExportButtons } from "./ExportButtons";
import { PanelStateGate } from "@/components/PanelStateGate";
import { formatCompactKMB as fmtNum } from "@/lib/formatters";
import { useChartColors } from "@/hooks/useChartColors";

type SunburstMetric = "demand" | "customers";

interface Props {
  filters: CustomerAnalyticsFilters;
}

interface ApiTreeNode {
  name: string;
  value: number;
  customer_count?: number;
  children?: ApiTreeNode[];
}

interface SunburstNode {
  name: string;
  value?: number;
  fill?: string;
  children?: SunburstNode[];
}

/**
 * Recharts SunburstChart wants `{name, value, children}` rooted at a single
 * node. We synthesize a "root" wrapper with `value: 0` so children draw
 * around it; the visualMap-style fill_rate -> color is replaced with a
 * deterministic per-channel palette to keep parity with the prior design.
 */
function toSunburstTree(
  nodes: ApiTreeNode[],
  metric: SunburstMetric,
  palette: string[],
): SunburstNode[] {
  return nodes.map((n, i) => {
    const fill = palette[i % palette.length];
    const value = metric === "customers" && n.customer_count != null
      ? n.customer_count
      : n.value;
    const children = n.children
      ? toSunburstTree(n.children, metric, palette).map((c) => ({ ...c, fill }))
      : undefined;
    return { name: n.name, value, fill, children };
  });
}

export function ChannelSunburst({ filters }: Props) {
  const [sunburstMetric, setSunburstMetric] = useState<SunburstMetric>("demand");
  const { okabeIto } = useChartColors();

  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.channelMix(filters),
    queryFn: () => fetchCustomerAnalyticsChannelMix(filters),
    staleTime: 60 * 60_000,
    placeholderData: keepPreviousData,
  });

  const topChannelName = useMemo(() => {
    const tree = data?.tree ?? [];
    if (tree.length === 0) return "";
    const sorted = [...tree].sort((a, b) => b.value - a.value);
    return sorted[0].name;
  }, [data]);

  // Pull totals once. The legacy API exposes `grand_total` / `total_customers`
  // even though the typed `ChannelMixPayload` doesn't yet declare them — keep
  // the runtime access via a typed shim so we get the totals without leaning
  // on `any`.
  const totals = data as
    | { grand_total?: number; total_customers?: number }
    | null
    | undefined;
  const grandTotal = totals?.grand_total ?? 0;
  const totalCustomers = totals?.total_customers ?? 0;

  const rootData = useMemo<SunburstNode>(() => {
    const tree = (data?.tree ?? []) as ApiTreeNode[];
    const rootValue = sunburstMetric === "demand" ? grandTotal : totalCustomers;
    return {
      name: "All",
      value: rootValue,
      children: toSunburstTree(tree, sunburstMetric, okabeIto),
    };
  }, [data, sunburstMetric, okabeIto, grandTotal, totalCustomers]);

  const centerLabel = sunburstMetric === "demand"
    ? `${fmtNum(grandTotal)} cases`
    : `${fmtNum(totalCustomers)} customers`;

  return (
    <Card aria-label="Channel mix sunburst chart">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Channel Mix</CardTitle>
          <ExportButtons panelId="channel-mix" getData={() => data?.tree ?? []} />
        </div>
        <p className="text-xs text-muted-foreground">
          Channel &gt; Store Type &gt; Sub-Channel.
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
        <PanelStateGate
          isLoading={isLoading}
          isEmpty={!data?.tree || data.tree.length === 0}
          height={420}
        >
          <div role="img" aria-roledescription="Channel mix sunburst chart" className="relative">
            <SunburstChart
              data={rootData}
              width={420}
              height={420}
              dataKey="value"
              innerRadius={70}
              outerRadius={200}
            />
            {/* Center label overlay (recharts SunburstChart has no built-in center text). */}
            <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center text-center">
              <div className="text-sm font-bold text-foreground">{centerLabel}</div>
              {topChannelName && (
                <div className="text-xs text-muted-foreground mt-1">{topChannelName}</div>
              )}
            </div>
          </div>
        </PanelStateGate>
      </CardContent>
    </Card>
  );
}
