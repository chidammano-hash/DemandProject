import { useMemo } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ResponsiveContainer, Sankey, Tooltip } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsDemandFlow,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters } from "@/api/queries/customer-analytics";
import { ExportButtons } from "./ExportButtons";
import { PanelStateGate } from "@/components/PanelStateGate";
import { useChartColors } from "@/hooks/useChartColors";

interface Props {
  filters: CustomerAnalyticsFilters;
}

interface RechartsSankeyData {
  nodes: { name: string }[];
  links: { source: number; target: number; value: number }[];
}

interface SankeyPayloadShape {
  name?: string;
  source?: { name: string };
  target?: { name: string };
  value?: number;
  payload?: SankeyPayloadShape;
}

interface SankeyTooltipProps {
  active?: boolean;
  payload?: Array<{ payload: SankeyPayloadShape }>;
}

function SankeyTooltip({ active, payload }: SankeyTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;
  const top = payload[0].payload;
  // Recharts double-wraps each tooltip entry: payload[0].payload.payload is
  // the raw node/link, payload[0].payload itself can also be the raw node.
  const wrapped: SankeyPayloadShape = top.payload ?? top;
  if (wrapped.source && wrapped.target) {
    return (
      <div className="rounded-md border bg-background p-2 text-xs shadow-sm">
        <div className="font-semibold">
          {wrapped.source.name} -&gt; {wrapped.target.name}
        </div>
        <div>Value: {(wrapped.value ?? 0).toLocaleString()}</div>
      </div>
    );
  }
  if (wrapped.name) {
    return (
      <div className="rounded-md border bg-background p-2 text-xs shadow-sm">
        <div className="font-semibold">{wrapped.name}</div>
      </div>
    );
  }
  return null;
}

export function DemandFlowSankey({ filters }: Props) {
  const { okabeIto } = useChartColors();
  const nodeFill = okabeIto[0];
  const linkFill = okabeIto[2];

  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.demandFlow(filters),
    queryFn: () => fetchCustomerAnalyticsDemandFlow(filters),
    staleTime: 60 * 60_000,
    placeholderData: keepPreviousData,
  });

  // Recharts Sankey needs numeric source/target indices into the nodes array.
  // The API returns string-keyed links — convert in one pass via a name->index
  // map. Drop links whose endpoints aren't in the node list.
  const sankeyData = useMemo<RechartsSankeyData>(() => {
    if (!data) return { nodes: [], links: [] };
    const nodes = data.nodes.map((n) => ({ name: n.name }));
    const idx = new Map(nodes.map((n, i) => [n.name, i]));
    const links = data.links
      .map((l) => {
        const s = idx.get(l.source);
        const t = idx.get(l.target);
        if (s == null || t == null) return null;
        return { source: s, target: t, value: l.value };
      })
      .filter((l): l is NonNullable<typeof l> => l !== null);
    return { nodes, links };
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
            <ResponsiveContainer width="100%" height={400}>
              <Sankey
                data={sankeyData}
                node={{ fill: nodeFill, stroke: nodeFill }}
                link={{ stroke: linkFill, strokeOpacity: 0.4 }}
                nodePadding={20}
                nodeWidth={14}
                margin={{ top: 10, bottom: 10, left: 10, right: 10 }}
              >
                <Tooltip content={<SankeyTooltip />} />
              </Sankey>
            </ResponsiveContainer>
          </div>
        </PanelStateGate>
      </CardContent>
    </Card>
  );
}
