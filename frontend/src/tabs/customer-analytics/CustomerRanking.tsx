import { useQuery } from "@tanstack/react-query";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsRanking,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters } from "@/api/queries/customer-analytics";

type SortMode = "demand_desc" | "fill_rate_asc";

interface Props {
  filters: CustomerAnalyticsFilters;
  sort: SortMode;
  topN: number;
  onSortChange: (s: SortMode) => void;
}

function fillRateColor(fr: number): string {
  if (fr >= 95) return "#22c55e";
  if (fr >= 90) return "#84cc16";
  if (fr >= 85) return "#eab308";
  if (fr >= 80) return "#f97316";
  return "#ef4444";
}

export function CustomerRanking({ filters, sort, topN, onSortChange }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.ranking(sort, topN, filters),
    queryFn: () => fetchCustomerAnalyticsRanking(sort, topN, filters),
    staleTime: 5 * 60_000,
  });

  const customers = data?.customers ?? [];
  const chartData = customers.map((c) => ({
    name: c.customer_name.length > 25 ? c.customer_name.slice(0, 22) + "..." : c.customer_name,
    value: sort === "demand_desc" ? c.demand_qty : c.fill_rate,
    fill_rate: c.fill_rate,
    demand_qty: c.demand_qty,
    state: c.state,
    channel: c.channel,
  }));

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Customer Ranking</CardTitle>
        <div className="flex gap-1 mt-1">
          <button
            onClick={() => onSortChange("demand_desc")}
            className={`px-2 py-0.5 text-xs rounded ${sort === "demand_desc" ? "bg-indigo-600 text-white" : "bg-gray-100 text-gray-600"}`}
          >
            Top by Demand
          </button>
          <button
            onClick={() => onSortChange("fill_rate_asc")}
            className={`px-2 py-0.5 text-xs rounded ${sort === "fill_rate_asc" ? "bg-red-600 text-white" : "bg-gray-100 text-gray-600"}`}
          >
            Worst Fill Rate
          </button>
        </div>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="h-[400px] flex items-center justify-center text-sm text-muted-foreground">Loading...</div>
        ) : chartData.length === 0 ? (
          <div className="h-[400px] flex items-center justify-center text-sm text-muted-foreground">No data</div>
        ) : (
          <ResponsiveContainer width="100%" height={Math.max(300, chartData.length * 28)}>
            <BarChart data={chartData} layout="vertical" margin={{ left: 140, right: 20, top: 5, bottom: 5 }}>
              <XAxis type="number" tickFormatter={(v: number) => v >= 1000 ? `${(v / 1000).toFixed(0)}K` : String(v)} />
              <YAxis type="category" dataKey="name" width={135} tick={{ fontSize: 11 }} />
              <Tooltip
                formatter={(v: number, _name: string, props: { payload: { fill_rate: number; demand_qty: number; state: string; channel: string } }) => {
                  const p = props.payload;
                  return [
                    `${sort === "demand_desc" ? v.toLocaleString() + " cases" : v + "%"}`,
                    `Fill: ${p.fill_rate}% | Demand: ${p.demand_qty.toLocaleString()} | ${p.state} | ${p.channel}`,
                  ];
                }}
              />
              <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                {chartData.map((d, i) => (
                  <Cell key={i} fill={fillRateColor(d.fill_rate)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}
