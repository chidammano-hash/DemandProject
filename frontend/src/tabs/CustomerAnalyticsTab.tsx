import { useCallback, useMemo, useState, type KeyboardEvent } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Filter,
  MapPinned,
  Network,
  Search,
  ShieldAlert,
  SlidersHorizontal,
  Users,
  X,
  type LucideIcon,
} from "lucide-react";

import {
  customerAnalyticsKeys,
  fetchCustomerAnalyticsFilterOptions,
  fetchCustomerAnalyticsItems,
  type CustomerAnalyticsFilters,
  type CustomerAnalyticsView,
} from "@/api/queries/customer-analytics";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { SearchableSelect } from "@/components/SearchableSelect";
import { useDebounce } from "@/hooks/useDebounce";
import { cn } from "@/lib/utils";
import { CustomerAnalyticsAssistant } from "./customer-analytics/CustomerAnalyticsAssistant";
import { CustomerAnalyticsWorkspace } from "./customer-analytics/CustomerAnalyticsWorkspace";
import {
  DashboardFilterProvider,
  useDashboardFilter,
} from "./customer-analytics/DashboardFilterContext";
import { KpiSummaryCards } from "./customer-analytics/KpiSummaryCards";
import { RecalculateButton } from "./customer-analytics/RecalculateButton";

interface ViewDefinition {
  id: CustomerAnalyticsView;
  label: string;
  description: string;
  icon: LucideIcon;
}

const VIEWS: ViewDefinition[] = [
  { id: "overview", label: "Overview", description: "Demand footprint", icon: MapPinned },
  { id: "customers", label: "Customers", description: "Rank and retain", icon: Users },
  { id: "segments", label: "Segments", description: "Mix and trends", icon: Network },
  { id: "service", label: "Service risk", description: "OOS and fill rate", icon: ShieldAlert },
  { id: "behavior", label: "Buying behavior", description: "Cadence and affinity", icon: SlidersHorizontal },
];

function defaultDateRange(): { from: string; to: string } {
  const now = new Date();
  const to = new Date(now.getFullYear(), now.getMonth(), 1);
  const from = new Date(to.getFullYear(), to.getMonth() - 12, 1);
  const format = (date: Date) =>
    `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-01`;
  return { from: format(from), to: format(to) };
}

function CustomerAnalyticsContent() {
  const { state: dashboardFilter, dispatch } = useDashboardFilter();
  const initialRange = useMemo(defaultDateRange, []);
  const [activeView, setActiveView] = useState<CustomerAnalyticsView>("overview");
  const [filtersOpen, setFiltersOpen] = useState(true);
  const [itemId, setItemId] = useState("");
  const [itemSearch, setItemSearch] = useState("");
  const [dateFrom, setDateFrom] = useState(initialRange.from);
  const [dateTo, setDateTo] = useState(initialRange.to);
  const [channel, setChannel] = useState("");
  const [storeType, setStoreType] = useState("");
  const [stateFilter, setStateFilter] = useState("");

  const effectiveChannel = dashboardFilter.selectedChannel || channel;
  const effectiveState = dashboardFilter.selectedState || stateFilter;
  const filters = useMemo<CustomerAnalyticsFilters>(
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

  const debouncedItemSearch = useDebounce(itemSearch, 300);
  const { data: itemsData } = useQuery({
    queryKey: customerAnalyticsKeys.items(debouncedItemSearch),
    queryFn: () => fetchCustomerAnalyticsItems(debouncedItemSearch),
    staleTime: 5 * 60_000,
    enabled: debouncedItemSearch.length >= 1 || debouncedItemSearch === "",
  });
  const { data: filterOptions } = useQuery({
    queryKey: customerAnalyticsKeys.filterOptions(),
    queryFn: () => fetchCustomerAnalyticsFilterOptions(),
    staleTime: 60 * 60_000,
  });

  const handleItemSelect = useCallback(
    (value: string) => {
      setItemId(value);
      setItemSearch(
        value ? (itemsData?.items.find((item) => item.item_id === value)?.item_desc ?? value) : "",
      );
    },
    [itemsData],
  );

  const clearFilters = () => {
    const range = defaultDateRange();
    setItemId("");
    setItemSearch("");
    setDateFrom(range.from);
    setDateTo(range.to);
    setChannel("");
    setStoreType("");
    setStateFilter("");
    dispatch({ type: "CLEAR_ALL" });
  };

  const activeFilterCount = [itemId, effectiveState, effectiveChannel, storeType].filter(Boolean).length;
  const assistantScopeKey = [
    activeView,
    itemId,
    dateFrom,
    dateTo,
    effectiveState,
    effectiveChannel,
    storeType,
  ].join("|");

  const handleViewKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    let nextIndex: number | null = null;
    if (event.key === "ArrowRight") nextIndex = (index + 1) % VIEWS.length;
    if (event.key === "ArrowLeft") nextIndex = (index - 1 + VIEWS.length) % VIEWS.length;
    if (event.key === "Home") nextIndex = 0;
    if (event.key === "End") nextIndex = VIEWS.length - 1;
    if (nextIndex == null) return;

    event.preventDefault();
    const nextView = VIEWS[nextIndex];
    setActiveView(nextView.id);
    document.getElementById(`customer-analytics-tab-${nextView.id}`)?.focus();
  };

  return (
    <div className="space-y-4 p-4 sm:p-5">
      <header className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="max-w-2xl">
          <div className="mb-1 flex items-center gap-2 text-xs font-medium text-primary">
            <span className="h-1.5 w-1.5 rounded-full bg-primary" />
            Customer intelligence workspace
          </div>
          <h1 className="text-xl font-semibold tracking-tight text-foreground">Customer Analytics</h1>
          <p className="mt-1 text-sm leading-6 text-muted-foreground">
            Move from demand footprint to customer risk, segment performance, and buying behavior
            without losing your filter context.
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Button
            type="button"
            variant="outline"
            className="h-10"
            onClick={() => setFiltersOpen((open) => !open)}
            aria-expanded={filtersOpen}
            aria-controls="customer-analytics-filters"
          >
            <Filter className="h-4 w-4" aria-hidden="true" />
            Filters
            {activeFilterCount > 0 && (
              <span className="rounded-full bg-primary px-1.5 py-0.5 text-[10px] text-primary-foreground">
                {activeFilterCount}
              </span>
            )}
          </Button>
          <RecalculateButton />
        </div>
      </header>

      <nav
        className="overflow-x-auto overscroll-x-none rounded-xl border bg-card p-1 shadow-card"
        aria-label="Customer Analytics views"
      >
        <div className="grid min-w-[660px] grid-cols-5 gap-1" role="tablist">
          {VIEWS.map((view, index) => {
            const Icon = view.icon;
            const selected = activeView === view.id;
            return (
              <button
                key={view.id}
                id={`customer-analytics-tab-${view.id}`}
                type="button"
                role="tab"
                aria-label={view.label}
                aria-selected={selected}
                aria-controls={`customer-analytics-panel-${view.id}`}
                onClick={() => setActiveView(view.id)}
                onKeyDown={(event) => handleViewKeyDown(event, index)}
                tabIndex={selected ? 0 : -1}
                className={cn(
                  "flex min-h-14 items-center gap-2 rounded-lg px-3 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  selected
                    ? "bg-primary text-primary-foreground shadow-sm"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground",
                )}
              >
                <Icon className="h-4 w-4 shrink-0" aria-hidden="true" />
                <span className="min-w-0">
                  <span className="block text-xs font-semibold">{view.label}</span>
                  <span className={cn("block truncate text-[10px]", selected ? "text-primary-foreground/75" : "text-muted-foreground")}>
                    {view.description}
                  </span>
                </span>
              </button>
            );
          })}
        </div>
      </nav>

      <Card id="customer-analytics-filters" hidden={!filtersOpen}>
        <CardContent className="p-4">
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-semibold text-foreground">
              <Search className="h-4 w-4 text-primary" aria-hidden="true" />
              Focus the analysis
            </div>
            <Button type="button" variant="ghost" size="sm" onClick={clearFilters}>
              Clear filters
            </Button>
          </div>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
            <div className="relative space-y-1 sm:col-span-2 xl:col-span-1">
              <label htmlFor="customer-analytics-item" className="text-xs font-medium text-muted-foreground">
                Item
              </label>
              <Input
                id="customer-analytics-item"
                type="text"
                placeholder="Search item..."
                value={itemSearch}
                onChange={(event) => {
                  setItemSearch(event.target.value);
                  if (!event.target.value) setItemId("");
                }}
                className="pr-9"
              />
              {itemId && (
                <button
                  type="button"
                  onClick={() => handleItemSelect("")}
                  className="absolute right-2 top-[27px] flex h-7 w-7 items-center justify-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  aria-label="Clear selected item"
                >
                  <X className="h-3.5 w-3.5" aria-hidden="true" />
                </button>
              )}
              {itemSearch && !itemId && (itemsData?.items.length ?? 0) > 0 && (
                <div className="absolute z-50 mt-1 max-h-52 w-full overflow-y-auto rounded-lg border bg-popover p-1 text-popover-foreground shadow-lg">
                  {itemsData?.items.slice(0, 10).map((item) => (
                    <button
                      key={item.item_id}
                      type="button"
                      onClick={() => handleItemSelect(item.item_id)}
                      className="block min-h-10 w-full rounded-md px-2 py-1.5 text-left text-xs hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    >
                      <span className="block font-mono font-medium">{item.item_id}</span>
                      <span className="block truncate text-muted-foreground">{item.item_desc}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            <div className="space-y-1">
              <label htmlFor="customer-analytics-from" className="text-xs font-medium text-muted-foreground">From</label>
              <Input id="customer-analytics-from" type="date" value={dateFrom} onChange={(event) => setDateFrom(event.target.value)} />
            </div>
            <div className="space-y-1">
              <label htmlFor="customer-analytics-to" className="text-xs font-medium text-muted-foreground">To</label>
              <Input id="customer-analytics-to" type="date" value={dateTo} onChange={(event) => setDateTo(event.target.value)} />
            </div>
            <div className="space-y-1">
              <label htmlFor="customer-analytics-state" className="text-xs font-medium text-muted-foreground">State</label>
              <select
                id="customer-analytics-state"
                value={effectiveState}
                onChange={(event) => {
                  setStateFilter(event.target.value);
                  dispatch({ type: "SET_STATE", payload: event.target.value });
                }}
                className="flex h-9 w-full rounded-md border border-input bg-background px-3 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              >
                <option value="">All states</option>
                {(filterOptions?.states ?? []).map((state) => <option key={state} value={state}>{state}</option>)}
              </select>
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground">Channel</label>
              <SearchableSelect
                value={effectiveChannel}
                options={filterOptions?.channels ?? []}
                placeholder="All channels"
                ariaLabel="Channel"
                className="w-full"
                onChange={(value) => {
                  setChannel(value);
                  dispatch({ type: "SET_CHANNEL", payload: value });
                }}
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground">Store Type</label>
              <SearchableSelect
                value={storeType}
                options={filterOptions?.store_types ?? []}
                placeholder="All types"
                ariaLabel="Store Type"
                className="w-full"
                onChange={setStoreType}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      <KpiSummaryCards filters={filters} />
      <CustomerAnalyticsAssistant
        key={assistantScopeKey}
        filters={filters}
        activeView={activeView}
      />
      <CustomerAnalyticsWorkspace activeView={activeView} filters={filters} />
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
