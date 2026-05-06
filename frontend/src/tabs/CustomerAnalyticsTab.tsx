import { lazy, Suspense, useCallback, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent } from "@/components/ui/card";
import { LazyPanel } from "@/components/LazyPanel";
import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsItems,
  fetchCustomerAnalyticsFilterOptions,
} from "@/api/queries/customer-analytics";
import type { CustomerAnalyticsFilters } from "@/api/queries/customer-analytics";
import { useDebounce } from "@/hooks/useDebounce";
import { DashboardFilterProvider, useDashboardFilter } from "./customer-analytics/DashboardFilterContext";
// Eager: above-the-fold panels visible on first paint.
import { KpiSummaryCards } from "./customer-analytics/KpiSummaryCards";
import { CustomerDemandMap } from "./customer-analytics/CustomerDemandMap";
import { CustomerTreemap } from "./customer-analytics/CustomerTreemap";

// Lazy: below-the-fold or chart-heavy panels. Each gets its own JS chunk so
// initial bundle stays small and ECharts init is deferred until scrolled.
const CustomerHeatmap = lazy(() =>
  import("./customer-analytics/CustomerHeatmap").then((m) => ({ default: m.CustomerHeatmap })),
);
const ChannelSunburst = lazy(() =>
  import("./customer-analytics/ChannelSunburst").then((m) => ({ default: m.ChannelSunburst })),
);
const SegmentSparklines = lazy(() =>
  import("./customer-analytics/SegmentSparklines").then((m) => ({ default: m.SegmentSparklines })),
);
const CustomerRanking = lazy(() =>
  import("./customer-analytics/CustomerRanking").then((m) => ({ default: m.CustomerRanking })),
);
const OosImpactBubble = lazy(() =>
  import("./customer-analytics/OosImpactBubble").then((m) => ({ default: m.OosImpactBubble })),
);
const CustomerLifecycle = lazy(() =>
  import("./customer-analytics/CustomerLifecycle").then((m) => ({ default: m.CustomerLifecycle })),
);
const DemandAtRisk = lazy(() =>
  import("./customer-analytics/DemandAtRisk").then((m) => ({ default: m.DemandAtRisk })),
);
const CustomerItemAffinity = lazy(() =>
  import("./customer-analytics/CustomerItemAffinity").then((m) => ({ default: m.CustomerItemAffinity })),
);
const OrderPatterns = lazy(() =>
  import("./customer-analytics/OrderPatterns").then((m) => ({ default: m.OrderPatterns })),
);
const DemandFlowSankey = lazy(() =>
  import("./customer-analytics/DemandFlowSankey").then((m) => ({ default: m.DemandFlowSankey })),
);

function PanelFallback({ height = 300 }: { height?: number }) {
  return (
    <div
      className="flex items-center justify-center text-sm text-muted-foreground rounded-md border border-dashed"
      style={{ height }}
    >
      Loading...
    </div>
  );
}

// Last 12 months window, aligned to the first of the month — matches the
// backend's _default_date_range() so the picker shows what the API would
// have used implicitly anyway.
function defaultDateRange(): { from: string; to: string } {
  const now = new Date();
  const to = new Date(now.getFullYear(), now.getMonth(), 1);
  const from = new Date(to.getFullYear(), to.getMonth() - 12, 1);
  const fmt = (d: Date) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-01`;
  return { from: fmt(from), to: fmt(to) };
}

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
  const initialRange = useMemo(defaultDateRange, []);
  const [dateFrom, setDateFrom] = useState(initialRange.from);
  const [dateTo, setDateTo] = useState(initialRange.to);
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

  // Stable identity across unrelated renders — otherwise every keystroke
  // recreates `filters` and cache-busts the 13 panels that spread it into
  // their React Query keys.
  const filters: CustomerAnalyticsFilters = useMemo(
    () => ({
      item_id: itemId || undefined,
      date_from: dateFrom || undefined,
      date_to: dateTo || undefined,
      channel: effectiveChannel || undefined,
      store_type: storeType || undefined,
      state: effectiveState || undefined,
    }),
    [itemId, dateFrom, dateTo, effectiveChannel, storeType, effectiveState],
  );

  // Debounced typeahead — each keystroke hits an ILIKE scan on dim_item.
  const debouncedItemSearch = useDebounce(itemSearch, 300);
  const { data: itemsData } = useQuery({
    queryKey: customerAnalyticsKeys.items(debouncedItemSearch),
    queryFn: () => fetchCustomerAnalyticsItems(debouncedItemSearch),
    staleTime: 5 * 60_000,
    enabled: debouncedItemSearch.length >= 1 || debouncedItemSearch === "",
  });

  // Filter options for dropdowns
  const { data: filterOptions } = useQuery({
    queryKey: customerAnalyticsKeys.filterOptions(),
    queryFn: () => fetchCustomerAnalyticsFilterOptions(),
    staleTime: 60 * 60_000, // enums are essentially static; 1h is plenty
  });

  const handleItemSelect = useCallback((val: string) => {
    setItemId(val);
    setItemSearch(val ? (itemsData?.items.find((i) => i.item_id === val)?.item_desc ?? val) : "");
  }, [itemsData]);

  const handleClear = () => {
    const r = defaultDateRange();
    setItemId(""); setItemSearch(""); setDateFrom(r.from); setDateTo(r.to);
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

      {/* 4. Heatmap | Sunburst — viewport-gated: useQuery only fires when scrolled into view */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <LazyPanel fallback={<PanelFallback height={400} />} minHeight={400}>
          <Suspense fallback={<PanelFallback height={400} />}>
            <CustomerHeatmap filters={filters} metric="demand_qty" topN={25} />
          </Suspense>
        </LazyPanel>
        <LazyPanel fallback={<PanelFallback height={420} />} minHeight={420}>
          <Suspense fallback={<PanelFallback height={420} />}>
            <ChannelSunburst filters={filters} />
          </Suspense>
        </LazyPanel>
      </div>

      {/* 5. Sparklines | OOS Bubble */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <LazyPanel fallback={<PanelFallback height={300} />} minHeight={300}>
          <Suspense fallback={<PanelFallback height={300} />}>
            <SegmentSparklines
              filters={filters}
              segmentBy={segmentBy}
              onSegmentByChange={setSegmentBy}
            />
          </Suspense>
        </LazyPanel>
        <LazyPanel fallback={<PanelFallback height={400} />} minHeight={400}>
          <Suspense fallback={<PanelFallback height={400} />}>
            <OosImpactBubble
              filters={filters}
              grain={oosGrain}
              onGrainChange={setOosGrain}
            />
          </Suspense>
        </LazyPanel>
      </div>

      {/* 6. Full-width ranking */}
      <LazyPanel fallback={<PanelFallback height={400} />} minHeight={400}>
        <Suspense fallback={<PanelFallback height={400} />}>
          <CustomerRanking
            filters={filters}
            sort={rankSort}
            topN={20}
            onSortChange={setRankSort}
          />
        </Suspense>
      </LazyPanel>

      {/* 7. Lifecycle | Demand at Risk */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <LazyPanel fallback={<PanelFallback />} minHeight={300}>
          <Suspense fallback={<PanelFallback />}>
            <CustomerLifecycle filters={filters} />
          </Suspense>
        </LazyPanel>
        <LazyPanel fallback={<PanelFallback />} minHeight={300}>
          <Suspense fallback={<PanelFallback />}>
            <DemandAtRisk filters={filters} />
          </Suspense>
        </LazyPanel>
      </div>

      {/* 8. Affinity | Order Patterns */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <LazyPanel fallback={<PanelFallback height={400} />} minHeight={400}>
          <Suspense fallback={<PanelFallback height={400} />}>
            <CustomerItemAffinity filters={filters} />
          </Suspense>
        </LazyPanel>
        <LazyPanel fallback={<PanelFallback />} minHeight={300}>
          <Suspense fallback={<PanelFallback />}>
            <OrderPatterns filters={filters} />
          </Suspense>
        </LazyPanel>
      </div>

      {/* 9. Full-width Demand Flow Sankey */}
      <LazyPanel fallback={<PanelFallback height={500} />} minHeight={500}>
        <Suspense fallback={<PanelFallback height={500} />}>
          <DemandFlowSankey filters={filters} />
        </Suspense>
      </LazyPanel>
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
