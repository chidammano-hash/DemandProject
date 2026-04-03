import { useQuery } from "@tanstack/react-query";
import { AreaChart, Area, ResponsiveContainer } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsSegmentTrends,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters } from "@/api/queries/customer-analytics";

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

function fmtNum(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toFixed(0);
}

export function SegmentSparklines({ filters, segmentBy, onSegmentByChange }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.segmentTrends(segmentBy, filters),
    queryFn: () => fetchCustomerAnalyticsSegmentTrends(segmentBy, filters),
    staleTime: 5 * 60_000,
  });

  const segments = data?.segments ?? [];

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Segment Trends</CardTitle>
        <div className="flex gap-1 mt-1">
          {(Object.keys(SEGMENT_LABELS) as SegmentBy[]).map((s) => (
            <button
              key={s}
              onClick={() => onSegmentByChange(s)}
              className={`px-2 py-0.5 text-xs rounded ${segmentBy === s ? "bg-indigo-600 text-white" : "bg-gray-100 text-gray-600"}`}
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
              <span>Segment</span>
              <span className="text-right">Customers</span>
              <span className="text-right">Demand</span>
              <span className="text-center">Trend</span>
              <span className="text-right">Fill %</span>
              <span className="text-right">MoM</span>
            </div>
            {segments.map((seg) => (
              <div
                key={seg.segment}
                className="grid grid-cols-[1fr_80px_80px_100px_60px_60px] gap-2 items-center text-xs px-1 py-1 hover:bg-gray-50 rounded"
              >
                <span className="truncate font-medium">{seg.segment}</span>
                <span className="text-right">{seg.total_customers.toLocaleString()}</span>
                <span className="text-right">{fmtNum(seg.total_demand)}</span>
                <div className="h-6">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={seg.trend}>
                      <Area
                        type="monotone"
                        dataKey="demand_qty"
                        stroke="#6366f1"
                        fill="#e0e7ff"
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
          </div>
        )}
      </CardContent>
    </Card>
  );
}
