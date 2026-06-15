import { useState, useMemo } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { AreaChart, Area, ResponsiveContainer } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsSegmentTrends,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters, SegmentRow } from "@/api/queries/customer-analytics";
import { ExportButtons } from "./ExportButtons";
import { formatCompactKMB as fmtNum, isEmptyCell } from "@/lib/formatters";
import { useChartColors } from "@/hooks/useChartColors";
import { togglePillClass } from "./togglePill";

/**
 * Derive the sparkline series stroke + fill from a theme series color so the
 * panel honors Soft/Dark themes (CLAUDE.md: no inline hex in tabs) — U5.12.
 * The fill is the same hue at low opacity (hex 8-digit alpha "26" ≈ 15%).
 */
export function sparklineColors(seriesColor: string): { stroke: string; fill: string } {
  return { stroke: seriesColor, fill: `${seriesColor}26` };
}

// Customers with no store-type (or other dimension) code surface as a nullish
// segment. For store_type this single bucket is ~86% of demand, so we label it
// rather than drop it (hiding the majority of demand would mislead) — U5.10.
const UNCLASSIFIED_LABEL = "Unclassified";

/** Display label for a segment, mapping nullish/sentinel names to "Unclassified". */
export function segmentDisplayLabel(segment: string): string {
  return isEmptyCell(segment) ? UNCLASSIFIED_LABEL : segment;
}

type SegmentBy = "rpt_channel_desc" | "store_type_desc" | "chain_type_desc" | "state";

const SEGMENT_LABELS: Record<SegmentBy, string> = {
  rpt_channel_desc: "Channel",
  store_type_desc: "Store Type",
  chain_type_desc: "Chain Type",
  state: "State",
};

interface Props {
  filters: CustomerAnalyticsFilters;
  segmentBy: SegmentBy;
  onSegmentByChange: (s: SegmentBy) => void;
}

type SortField = "segment" | "total_customers" | "total_demand" | "fill_rate" | "mom_change";

export function SegmentSparklines({ filters, segmentBy, onSegmentByChange }: Props) {
  const [sortField, setSortField] = useState<SortField>("total_demand");
  const [sortAsc, setSortAsc] = useState(false);
  const { trendColors } = useChartColors();
  const { stroke: areaStroke, fill: areaFill } = sparklineColors(trendColors[0]);

  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.segmentTrends(segmentBy, filters),
    queryFn: () => fetchCustomerAnalyticsSegmentTrends(segmentBy, filters),
    staleTime: 60 * 60_000, // monthly data; pin to 1h to suppress thundering-herd refetches
    placeholderData: keepPreviousData, // keep prior chart visible during filter-change refetch
  });

  const rawSegments = useMemo(() => data?.segments ?? [], [data?.segments]);

  const hasUnclassified = useMemo(
    () => rawSegments.some((s) => isEmptyCell(s.segment)),
    [rawSegments],
  );

  const segments = useMemo(() => {
    const sorted = [...rawSegments].sort((a, b) => {
      const av = a[sortField as keyof SegmentRow];
      const bv = b[sortField as keyof SegmentRow];
      if (typeof av === "number" && typeof bv === "number") {
        return sortAsc ? av - bv : bv - av;
      }
      return sortAsc
        ? String(av).localeCompare(String(bv))
        : String(bv).localeCompare(String(av));
    });
    return sorted;
  }, [rawSegments, sortField, sortAsc]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  const sortIndicator = (field: SortField) => {
    if (sortField !== field) return "";
    return sortAsc ? " \u25B2" : " \u25BC";
  };

  return (
    <Card aria-label="Segment trends sparklines">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Segment Trends</CardTitle>
          <ExportButtons panelId="segment-trends" getData={() => rawSegments as unknown as Record<string, unknown>[]} />
        </div>
        <div className="flex gap-1 mt-1">
          {(Object.keys(SEGMENT_LABELS) as SegmentBy[]).map((s) => (
            <button
              key={s}
              onClick={() => onSegmentByChange(s)}
              aria-pressed={segmentBy === s}
              className={togglePillClass(segmentBy === s)}
            >
              {SEGMENT_LABELS[s]}
            </button>
          ))}
        </div>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="h-[300px] flex items-center justify-center text-sm text-muted-foreground">Loading...</div>
        ) : segments.length === 0 ? (
          <div className="h-[300px] flex items-center justify-center text-sm text-muted-foreground">No data</div>
        ) : (
          <div className="space-y-1 max-h-[400px] overflow-y-auto">
            <div className="grid grid-cols-[1fr_80px_80px_100px_60px_60px] gap-2 text-xs font-medium text-muted-foreground px-1 pb-1 border-b">
              <button className="text-left hover:text-foreground" onClick={() => handleSort("segment")}>Segment{sortIndicator("segment")}</button>
              <button className="text-right hover:text-foreground" onClick={() => handleSort("total_customers")}>Customers{sortIndicator("total_customers")}</button>
              <button className="text-right hover:text-foreground" onClick={() => handleSort("total_demand")}>Demand{sortIndicator("total_demand")}</button>
              <span className="text-center">Trend</span>
              <button className="text-right hover:text-foreground" onClick={() => handleSort("fill_rate")}>Fill %{sortIndicator("fill_rate")}</button>
              <button className="text-right hover:text-foreground" onClick={() => handleSort("mom_change")}>MoM{sortIndicator("mom_change")}</button>
            </div>
            {segments.map((seg) => (
              <div
                key={seg.segment}
                className={`grid grid-cols-[1fr_80px_80px_100px_60px_60px] gap-2 items-center text-xs px-1 py-1 rounded ${seg.mom_change < 0 ? "bg-destructive/10" : "hover:bg-accent"}`}
              >
                <span className="truncate font-medium">{segmentDisplayLabel(seg.segment)}</span>
                <span className="text-right">{seg.total_customers.toLocaleString()}</span>
                <span className="text-right">{fmtNum(seg.total_demand)}</span>
                <div className="h-6">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={seg.trend}>
                      <Area
                        type="monotone"
                        dataKey="demand_qty"
                        stroke={areaStroke}
                        fill={areaFill}
                        strokeWidth={1.5}
                        dot={false}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                <span className={`text-right ${seg.fill_rate < 90 ? "text-red-600" : "text-green-600"}`}>
                  {seg.fill_rate}%
                </span>
                <span className={`text-right ${seg.mom_change >= 0 ? "text-green-600" : "text-red-600"}`}>
                  {seg.mom_change >= 0 ? "+" : ""}{seg.mom_change}%
                </span>
              </div>
            ))}
            {hasUnclassified && (
              <p className="text-[11px] text-muted-foreground pt-2">
                Unclassified = customers with no {SEGMENT_LABELS[segmentBy].toLowerCase()} code.
              </p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
