import { useState, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent } from "@/components/ui/card";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsItems,
  fetchCustomerAnalyticsFilterOptions,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters } from "@/api/queries/customer-analytics";
import { DashboardFilterProvider, useDashboardFilter } from "./customer-analytics/DashboardFilterContext";
import { KpiSummaryCards } from "./customer-analytics/KpiSummaryCards";
import { CustomerDemandMap } from "./customer-analytics/CustomerDemandMap";
import { CustomerTreemap } from "./customer-analytics/CustomerTreemap";
import { CustomerHeatmap } from "./customer-analytics/CustomerHeatmap";
import { ChannelSunburst } from "./customer-analytics/ChannelSunburst";
import { SegmentSparklines } from "./customer-analytics/SegmentSparklines";
import { CustomerRanking } from "./customer-analytics/CustomerRanking";
import { OosImpactBubble } from "./customer-analytics/OosImpactBubble";
import { CustomerLifecycle } from "./customer-analytics/CustomerLifecycle";
import { DemandAtRisk } from "./customer-analytics/DemandAtRisk";
import { CustomerItemAffinity } from "./customer-analytics/CustomerItemAffinity";
import { OrderPatterns } from "./customer-analytics/OrderPatterns";
import { DemandFlowSankey } from "./customer-analytics/DemandFlowSankey";

type MapMetric = "customer_count" | "demand_qty" | "sales_qty" | "oos_qty" | "fill_rate";
type GroupBy = "state" | "city" | "zip";
type SegmentBy = "rpt_channel_desc" | "store_type_desc" | "chain_type_desc" | "state";
type SortMode = "demand_desc" | "fill_rate_asc";
type Grain = "customer" | "state";

function CustomerAnalyticsContent() {
  const { state: dashFilter, dispatch } = useDashboardFilter();

  // Shared filters
  const [itemId, setItemId] = useState<string>("");
  const [itemSearch, setItemSearch] = useState("");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [channel, setChannel] = useState("");
  const [storeType, setStoreType] = useState("");
  const [stateFilter, setStateFilter] = useState("");

  // Panel-specific state
  const [mapMetric, setMapMetric] = useState<MapMetric>("demand_qty");
  const [groupBy, setGroupBy] = useState<GroupBy>("state");
  const [segmentBy, setSegmentBy] = useState<SegmentBy>("rpt_channel_desc");
  const [rankSort, setRankSort] = useState<SortMode>("demand_desc");
  const [oosGrain, setOosGrain] = useState<Grain>("customer");

  // Merge local and context filters
  const effectiveChannel = dashFilter.selectedChannel || channel;
  const effectiveState = dashFilter.selectedState || stateFilter;

  const filters: CustomerAnalyticsFilters = {
    item_id: itemId || undefined,
    date_from: dateFrom || undefined,
    date_to: dateTo || undefined,
    channel: effectiveChannel || undefined,
    store_type: storeType || undefined,
    state: effectiveState || undefined,
  };

  // Item search typeahead
  const { data: itemsData } = useQuery({
    queryKey: customerAnalyticsKeys.items(itemSearch),
    queryFn: () => fetchCustomerAnalyticsItems(itemSearch),
    staleTime: 5 * 60_000,
    enabled: itemSearch.length >= 1 || itemSearch === "",
  });

  // Filter options for dropdowns
  const { data: filterOptions } = useQuery({
    queryKey: customerAnalyticsKeys.filterOptions(),
    queryFn: () => fetchCustomerAnalyticsFilterOptions(),
    staleTime: 10 * 60_000,
  });

  const handleItemSelect = useCallback((val: string) => {
    setItemId(val);
    setItemSearch(val ? (itemsData?.items.find((i) => i.item_id === val)?.item_desc ?? val) : "");
  }, [itemsData]);

  const handleClear = () => {
    setItemId(""); setItemSearch(""); setDateFrom(""); setDateTo("");
    setChannel(""); setStoreType(""); setStateFilter("");
    dispatch({ type: "CLEAR_ALL" });
  };

  return (
    <div className="space-y-4 p-4">
      {/* 1. KPI Summary Cards */}
      <KpiSummaryCards filters={filters} />

      {/* 2. Filter Bar */}
      <Card>
        <CardContent className="py-3">
          <div className="flex flex-wrap gap-3 items-end">
            {/* Item picker */}
            <div className="flex flex-col gap-1">
              <label className="text-xs font-medium text-muted-foreground">Item</label>
              <div className="relative">
                <input
                  type="text"
                  placeholder="Search item..."
                  value={itemSearch}
                  onChange={(e) => {
                    setItemSearch(e.target.value);
                    if (!e.target.value) setItemId("");
                  }}
                  className="w-52 px-2 py-1 text-sm border rounded"
                  list="ca-items"
                />
                <datalist id="ca-items">
                  {(itemsData?.items ?? []).map((it) => (
                    <option key={it.item_id} value={it.item_id}>
                      {it.item_desc}
                    </option>
                  ))}
                </datalist>
                {itemId && (
                  <button
                    onClick={() => { setItemId(""); setItemSearch(""); }}
                    className="absolute right-1 top-1 text-xs text-gray-400 hover:text-gray-600"
                  >
                    x
                  </button>
                )}
              </div>
              {itemSearch && !itemId && (itemsData?.items ?? []).length > 0 && (
                <div className="absolute z-50 mt-14 bg-white border rounded shadow-lg max-h-40 overflow-y-auto w-52">
                  {itemsData!.items.slice(0, 10).map((it) => (
                    <button
                      key={it.item_id}
                      onClick={() => handleItemSelect(it.item_id)}
                      className="block w-full text-left px-2 py-1 text-xs hover:bg-gray-100"
                    >
                      <span className="font-medium">{it.item_id}</span>
                      <span className="text-muted-foreground ml-1">{it.item_desc}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Date range */}
            <div className="flex flex-col gap-1">
              <label className="text-xs font-medium text-muted-foreground">From</label>
              <input type="date" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} className="px-2 py-1 text-sm border rounded" />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs font-medium text-muted-foreground">To</label>
              <input type="date" value={dateTo} onChange={(e) => setDateTo(e.target.value)} className="px-2 py-1 text-sm border rounded" />
            </div>

            {/* State dropdown */}
            <div className="flex flex-col gap-1">
              <label className="text-xs font-medium text-muted-foreground">State</label>
              <select
                value={effectiveState}
                onChange={(e) => {
                  setStateFilter(e.target.value);
                  dispatch({ type: "SET_STATE", payload: e.target.value });
                }}
                className="w-36 px-2 py-1 text-sm border rounded"
              >
                <option value="">All states</option>
                {(filterOptions?.states ?? []).map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>

            {/* Channel dropdown */}
            <div className="flex flex-col gap-1">
              <label className="text-xs font-medium text-muted-foreground">Channel</label>
              <select
                value={effectiveChannel}
                onChange={(e) => {
                  setChannel(e.target.value);
                  dispatch({ type: "SET_CHANNEL", payload: e.target.value });
                }}
                className="w-36 px-2 py-1 text-sm border rounded"
              >
                <option value="">All channels</option>
                {(filterOptions?.channels ?? []).map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>

            {/* Store Type dropdown */}
            <div className="flex flex-col gap-1">
              <label className="text-xs font-medium text-muted-foreground">Store Type</label>
              <select
                value={storeType}
                onChange={(e) => setStoreType(e.target.value)}
                className="w-36 px-2 py-1 text-sm border rounded"
              >
                <option value="">All types</option>
                {(filterOptions?.store_types ?? []).map((st) => (
                  <option key={st} value={st}>{st}</option>
                ))}
              </select>
            </div>

            {/* Active cross-filter badges */}
            {(dashFilter.selectedState || dashFilter.selectedChannel || dashFilter.selectedCustomer || dashFilter.selectedSegment) && (
              <div className="flex gap-1 items-center">
                {dashFilter.selectedState && (
                  <span className="px-2 py-0.5 text-xs bg-teal-100 text-teal-700 rounded-full">
                    State: {dashFilter.selectedState}
                  </span>
                )}
                {dashFilter.selectedChannel && (
                  <span className="px-2 py-0.5 text-xs bg-indigo-100 text-indigo-700 rounded-full">
                    Channel: {dashFilter.selectedChannel}
                  </span>
                )}
                {dashFilter.selectedCustomer && (
                  <span className="px-2 py-0.5 text-xs bg-amber-100 text-amber-700 rounded-full">
                    Customer: {dashFilter.selectedCustomer}
                  </span>
                )}
              </div>
            )}

            {/* Clear */}
            <button
              onClick={handleClear}
              className="px-3 py-1 text-xs bg-gray-100 text-gray-600 rounded hover:bg-gray-200"
            >
              Clear
            </button>
          </div>
        </CardContent>
      </Card>

      {/* 3. Demand Map | Treemap */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <CustomerDemandMap
          filters={filters}
          metric={mapMetric}
          groupBy={groupBy}
          onMetricChange={setMapMetric}
          onGroupByChange={setGroupBy}
        />
        <CustomerTreemap filters={filters} />
      </div>

      {/* 4. Heatmap | Sunburst */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <CustomerHeatmap filters={filters} metric="demand_qty" topN={25} />
        <ChannelSunburst filters={filters} />
      </div>

      {/* 5. Sparklines | OOS Bubble */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <SegmentSparklines
          filters={filters}
          segmentBy={segmentBy}
          onSegmentByChange={setSegmentBy}
        />
        <OosImpactBubble
          filters={filters}
          grain={oosGrain}
          onGrainChange={setOosGrain}
        />
      </div>

      {/* 6. Full-width ranking */}
      <CustomerRanking
        filters={filters}
        sort={rankSort}
        topN={20}
        onSortChange={setRankSort}
      />

      {/* 7. Lifecycle | Demand at Risk */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <CustomerLifecycle filters={filters} />
        <DemandAtRisk filters={filters} />
      </div>

      {/* 8. Affinity | Order Patterns */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <CustomerItemAffinity filters={filters} />
        <OrderPatterns filters={filters} />
      </div>

      {/* 9. Full-width Demand Flow Sankey */}
      <DemandFlowSankey filters={filters} />
    </div>
  );
}

export function CustomerAnalyticsTab() {
  return (
    <DashboardFilterProvider>
      <CustomerAnalyticsContent />
    </DashboardFilterProvider>
  );
}

export default CustomerAnalyticsTab;
