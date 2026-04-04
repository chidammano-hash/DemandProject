import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsRanking,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters } from "@/api/queries/customer-analytics";
import { ExportButtons } from "./ExportButtons";

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

function fmtNum(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toFixed(0);
}

export function CustomerRanking({ filters, sort, topN, onSortChange }: Props) {
  const [search, setSearch] = useState("");

  const { data, isLoading } = useQuery({
    queryKey: customerAnalyticsKeys.ranking(sort, topN, filters),
    queryFn: () => fetchCustomerAnalyticsRanking(sort, topN, filters),
    staleTime: 5 * 60_000,
  });

  const customers = data?.customers ?? [];
  const maxDemand = useMemo(
    () => Math.max(...customers.map((c) => c.demand_qty), 1),
    [customers],
  );

  const filtered = useMemo(() => {
    if (!search) return customers;
    const q = search.toLowerCase();
    return customers.filter(
      (c) => c.customer_name.toLowerCase().includes(q) || c.customer_no.toLowerCase().includes(q),
    );
  }, [customers, search]);

  return (
    <Card aria-label="Customer ranking table">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Customer Ranking</CardTitle>
          <ExportButtons panelId="ranking" getData={() => customers as unknown as Record<string, unknown>[]} />
        </div>
        <div className="flex gap-2 mt-1 items-center flex-wrap">
          <div className="flex gap-1">
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
          <input
            type="text"
            placeholder="Search customer..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-48 px-2 py-1 text-xs border rounded"
          />
        </div>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="h-[400px] flex items-center justify-center text-sm text-muted-foreground">Loading...</div>
        ) : filtered.length === 0 ? (
          <div className="h-[400px] flex items-center justify-center text-sm text-muted-foreground">No data</div>
        ) : (
          <div className="max-h-[500px] overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-white">
                <tr className="border-b text-left text-muted-foreground">
                  <th className="py-1.5 px-2 font-medium">Customer</th>
                  <th className="py-1.5 px-2 font-medium">State</th>
                  <th className="py-1.5 px-2 font-medium">Channel</th>
                  <th className="py-1.5 px-2 font-medium text-right">Demand</th>
                  <th className="py-1.5 px-2 font-medium w-40"></th>
                  <th className="py-1.5 px-2 font-medium text-right">Fill Rate</th>
                  <th className="py-1.5 px-2 font-medium text-right">OOS</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((c) => {
                  const barWidth = maxDemand > 0 ? Math.round((c.demand_qty / maxDemand) * 100) : 0;
                  return (
                    <tr key={c.customer_no} className="border-b hover:bg-gray-50">
                      <td className="py-1.5 px-2 font-medium truncate max-w-[200px]" title={c.customer_name}>
                        {c.customer_name}
                      </td>
                      <td className="py-1.5 px-2">{c.state}</td>
                      <td className="py-1.5 px-2">{c.channel}</td>
                      <td className="py-1.5 px-2 text-right">{fmtNum(c.demand_qty)}</td>
                      <td className="py-1.5 px-2">
                        <div className="w-full bg-gray-100 rounded-sm h-3">
                          <div
                            className="h-3 rounded-sm"
                            style={{
                              width: `${barWidth}%`,
                              backgroundColor: fillRateColor(c.fill_rate),
                            }}
                          />
                        </div>
                      </td>
                      <td className={`py-1.5 px-2 text-right font-medium ${c.fill_rate < 90 ? "text-red-600" : "text-green-600"}`}>
                        {c.fill_rate}%
                      </td>
                      <td className="py-1.5 px-2 text-right text-red-600">{fmtNum(c.oos_qty)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
