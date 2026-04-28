import { useRef, useState, useMemo } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useVirtualizer } from "@tanstack/react-virtual";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsRanking,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters, RankedCustomer } from "@/api/queries/customer-analytics";
import { ExportButtons } from "./ExportButtons";

const ROW_HEIGHT = 28;
const VIEWPORT_HEIGHT = 480;

// Virtualized rows: render only what's in the viewport. With ~50 rows the
// gain is small, but ranking grows to topN=200+ quickly when planners change
// the sort, and DOM-row count was the largest paint-time cost on this card.
function VirtualizedRankingTable({ rows, maxDemand }: { rows: RankedCustomer[]; maxDemand: number }) {
  const parentRef = useRef<HTMLDivElement>(null);
  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 8,
  });
  return (
    <div
      ref={parentRef}
      className="overflow-y-auto"
      style={{ maxHeight: VIEWPORT_HEIGHT }}
    >
      <div className="grid grid-cols-[1fr_50px_120px_80px_160px_60px_60px] gap-2 text-xs font-medium text-muted-foreground sticky top-0 bg-white z-10 px-2 py-1 border-b">
        <span>Customer</span>
        <span>State</span>
        <span>Channel</span>
        <span className="text-right">Demand</span>
        <span></span>
        <span className="text-right">Fill</span>
        <span className="text-right">OOS</span>
      </div>
      <div style={{ height: virtualizer.getTotalSize(), position: "relative" }}>
        {virtualizer.getVirtualItems().map((vRow) => {
          const c = rows[vRow.index];
          const barWidth = maxDemand > 0 ? Math.round((c.demand_qty / maxDemand) * 100) : 0;
          return (
            <div
              key={c.customer_no}
              className="grid grid-cols-[1fr_50px_120px_80px_160px_60px_60px] gap-2 items-center text-xs px-2 border-b hover:bg-gray-50 absolute left-0 right-0"
              style={{ height: ROW_HEIGHT, transform: `translateY(${vRow.start}px)` }}
            >
              <span className="truncate font-medium" title={c.customer_name}>{c.customer_name}</span>
              <span>{c.state}</span>
              <span className="truncate">{c.channel}</span>
              <span className="text-right tabular-nums">{fmtNum(c.demand_qty)}</span>
              <div className="bg-gray-100 rounded-sm h-3">
                <div
                  className="h-3 rounded-sm"
                  style={{ width: `${barWidth}%`, backgroundColor: fillRateColor(c.fill_rate) }}
                />
              </div>
              <span className={`text-right tabular-nums font-medium ${c.fill_rate < 90 ? "text-red-600" : "text-green-600"}`}>
                {c.fill_rate}%
              </span>
              <span className="text-right tabular-nums text-red-600">{fmtNum(c.oos_qty)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

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
    staleTime: 60 * 60_000, // monthly data; pin to 1h to suppress thundering-herd refetches
    placeholderData: keepPreviousData, // keep prior chart visible during filter-change refetch
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
          <VirtualizedRankingTable rows={filtered} maxDemand={maxDemand} />
        )}
      </CardContent>
    </Card>
  );
}
