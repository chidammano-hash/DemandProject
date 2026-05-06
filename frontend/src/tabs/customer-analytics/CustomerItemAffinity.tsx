import { useMemo } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { HeatmapGrid } from "@/components/HeatmapGrid";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsAffinity,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters } from "@/api/queries/customer-analytics";
import { ExportButtons } from "./ExportButtons";
import { PanelStateGate } from "@/components/PanelStateGate";

interface Props {
  filters: CustomerAnalyticsFilters;
}

const AFFINITY_STOPS = ["#eff6ff", "#3b82f6", "#1e3a5f"];

function lerp(a: string, b: string, t: number): string {
  const ar = parseInt(a.slice(1, 3), 16);
  const ag = parseInt(a.slice(3, 5), 16);
  const ab = parseInt(a.slice(5, 7), 16);
  const br = parseInt(b.slice(1, 3), 16);
  const bg = parseInt(b.slice(3, 5), 16);
  const bb = parseInt(b.slice(5, 7), 16);
  const r = Math.round(ar + (br - ar) * t);
  const g = Math.round(ag + (bg - ag) * t);
  const bl = Math.round(ab + (bb - ab) * t);
  return `rgb(${r}, ${g}, ${bl})`;
}

function buildScale(maxVal: number) {
  const max = Math.max(maxVal, 1);
  return (v: number) => {
    const t = Math.max(0, Math.min(1, v / max));
    const segments = AFFINITY_STOPS.length - 1;
    const segIdx = Math.min(Math.floor(t * segments), segments - 1);
    const segT = t * segments - segIdx;
    return lerp(AFFINITY_STOPS[segIdx], AFFINITY_STOPS[segIdx + 1], segT);
  };
}

export function CustomerItemAffinity({ filters }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.affinity(filters),
    queryFn: () => fetchCustomerAnalyticsAffinity(filters),
    staleTime: 60 * 60_000,
    placeholderData: keepPreviousData,
  });

  const { rows, columnLabels, scale } = useMemo(() => {
    if (!data) return { rows: [], columnLabels: [] as string[], scale: buildScale(1) };

    type CustObj = { customer_no: string; customer_name: string };
    type ItemObj = { item_id: string; item_desc: string };
    type RawCell = {
      customer_no?: string;
      customer?: string;
      item_id?: string;
      item?: string;
      demand_qty: number;
    };

    const rawCust = (data.customers ?? []) as unknown as Array<string | CustObj>;
    const rawItems = (data.items ?? []) as unknown as Array<string | ItemObj>;
    const rawCells = (data.cells ?? []) as unknown as RawCell[];

    // Downsample to keep render under control (max 50 rows x 50 cols).
    const MAX_CELLS = 2000;
    const MAX_AXIS = 50;
    let custKeys = rawCust.map((c) => (typeof c === "string" ? c : c.customer_no));
    let itemKeys = rawItems.map((i) => (typeof i === "string" ? i : i.item_id));
    let activeCells: RawCell[] = rawCells;
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
        .sort((a, b) => b[1] - a[1])
        .slice(0, MAX_AXIS)
        .map((e) => e[0]);
      itemKeys = [...itemDemand.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, MAX_AXIS)
        .map((e) => e[0]);
      const custSet = new Set(custKeys);
      const itemSet = new Set(itemKeys);
      activeCells = rawCells.filter((c) => {
        const ck = c.customer_no ?? c.customer ?? "";
        const ik = c.item_id ?? c.item ?? "";
        return custSet.has(ck) && itemSet.has(ik);
      });
    }

    const custLabel = new Map<string, string>(
      rawCust.map((c) => typeof c === "string" ? [c, c] : [c.customer_no, c.customer_name || c.customer_no]),
    );
    const itemLabel = new Map<string, string>(
      rawItems.map((i) => typeof i === "string" ? [i, i] : [i.item_id, i.item_desc || i.item_id]),
    );

    // Build a (custKey -> itemKey -> qty) map.
    const grid = new Map<string, Map<string, number>>();
    let maxVal = 0;
    for (const c of activeCells) {
      const ck = c.customer_no ?? c.customer ?? "";
      const ik = c.item_id ?? c.item ?? "";
      const inner = grid.get(ck) ?? new Map<string, number>();
      inner.set(ik, c.demand_qty);
      grid.set(ck, inner);
      if (c.demand_qty > maxVal) maxVal = c.demand_qty;
    }

    const builtRows = custKeys.map((ck) => ({
      label: custLabel.get(ck) ?? ck,
      values: itemKeys.map((ik) => grid.get(ck)?.get(ik) ?? 0),
    }));
    const cols = itemKeys.map((ik) => itemLabel.get(ik) ?? ik);
    return { rows: builtRows, columnLabels: cols, scale: buildScale(maxVal) };
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
        <PanelStateGate
          isLoading={isLoading}
          isEmpty={!data?.cells || data.cells.length === 0}
          height={400}
        >
          <div role="img" aria-roledescription="Customer-item affinity heatmap chart">
            <HeatmapGrid
              rows={rows}
              columnLabels={columnLabels}
              colorScale={scale}
              valueFormat={(v) => v.toLocaleString()}
              compact
              compactCellHeight={14}
              rowLabelWidth={140}
              cellMinWidth={14}
              disablePruning
            />
          </div>
        </PanelStateGate>
      </CardContent>
    </Card>
  );
}
