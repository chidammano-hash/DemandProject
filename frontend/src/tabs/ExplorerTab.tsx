import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowDownWideNarrow,
  ArrowUpWideNarrow,
  ChevronsUpDown,
  RefreshCcw,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { LoadingElement } from "@/components/LoadingElement";

import {
  queryKeys,
  STALE,
  fetchDomainMeta,
  fetchDomainPage,
  fetchForecastModels,
  fetchSkuClusters,
  fetchSamplePair,
  fetchDomainSuggest,
} from "@/api/queries";
import type {
  DomainMeta,
  ClusterInfo,
} from "@/types";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";
import { useDebounce } from "@/hooks/useDebounce";
import { titleCase, formatCell, formatNumber, formatCompactNumber } from "@/lib/formatters";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DIMENSION_DOMAINS = [
  "item",
  "location",
  "customer",
  "time",
  "sku",
  "sales",
  "forecast",
];

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

type ExplorerTabProps = {
  domain: string;
  onDomainChange: (domain: string) => void;
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ExplorerTab({ domain, onDomainChange }: ExplorerTabProps) {
  // -----------------------------------------------------------------------
  // Local state
  // -----------------------------------------------------------------------
  const [offset, setOffset] = useState(0);
  const [limit, setLimit] = useState(100);
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState("");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");
  const [columnFilters, setColumnFilters] = useState<Record<string, string>>({});
  const [visibleColumns, setVisibleColumns] = useState<Record<string, boolean>>({});
  const [showFieldPanel, setShowFieldPanel] = useState(false);

  const [itemFilter, setItemFilter] = useState("");
  const [locationFilter, setLocationFilter] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedCluster, setSelectedCluster] = useState("");
  const [clusterSource, setClusterSource] = useState<"ml" | "source">("ml");
  const [autoSampledDomain, setAutoSampledDomain] = useState("");
  const [columnSuggestions, setColumnSuggestions] = useState<Record<string, string[]>>({});

  // -----------------------------------------------------------------------
  // Global filter sync
  // -----------------------------------------------------------------------
  const { filters: globalFilters } = useGlobalFilterContext();

  // Sync global item/location filter into local inputs
  const syncedGlobalRef = useRef<string>("");
  useEffect(() => {
    const key = `${globalFilters.item.join(",")}_${globalFilters.location.join(",")}`;
    if (key === syncedGlobalRef.current) return;
    syncedGlobalRef.current = key;
    if (globalFilters.item.length === 1) setItemFilter(globalFilters.item[0]);
    if (globalFilters.location.length === 1) setLocationFilter(globalFilters.location[0]);
  }, [globalFilters.item, globalFilters.location]);

  // -----------------------------------------------------------------------
  // Derived values
  // -----------------------------------------------------------------------
  const isFactDomain = domain === "sales" || domain === "forecast";
  const filterDebounceMs = isFactDomain ? 500 : 300;

  const debouncedSearch = useDebounce(search, filterDebounceMs);
  const debouncedColumnFilters = useDebounce(columnFilters, filterDebounceMs);
  const debouncedItemFilter = useDebounce(itemFilter, filterDebounceMs);
  const debouncedLocationFilter = useDebounce(locationFilter, filterDebounceMs);

  // -----------------------------------------------------------------------
  // 1. Domain metadata (useQuery)
  // -----------------------------------------------------------------------
  const {
    data: meta,
    isLoading: isLoadingMeta,
    error: metaError,
  } = useQuery<DomainMeta>({
    queryKey: queryKeys.domainMeta(domain),
    queryFn: () => fetchDomainMeta(domain),
    staleTime: STALE.TEN_MIN,
  });

  // Reset local state when domain or meta changes
  useEffect(() => {
    if (!meta) return;
    setOffset(0);
    setSearch("");
    setColumnFilters({});
    setSortBy(meta.default_sort);
    setSortDir("asc");
    setItemFilter("");
    setLocationFilter("");
    setSelectedModel("");
    setAutoSampledDomain("");
    setColumnSuggestions({});
    setVisibleColumns(Object.fromEntries(meta.columns.map((c) => [c, true])));
  }, [meta]);

  // -----------------------------------------------------------------------
  // Derived field names
  // -----------------------------------------------------------------------
  const itemField = useMemo(() => {
    if (!meta) return "";
    if (meta.columns.includes("item_id")) return "item_id";
    if (meta.columns.includes("item_id")) return "item_id";
    return "";
  }, [meta]);

  const locationField = useMemo(() => {
    if (!meta) return "";
    if (meta.columns.includes("loc")) return "loc";
    if (meta.columns.includes("location_id")) return "location_id";
    return "";
  }, [meta]);

  // Whether to show item/location pair filters.
  // In the original App.tsx this was gated by `activeTab !== "explorer"`,
  // meaning pair filters were hidden when exploring the raw grid. The model
  // and cluster dropdown filters are still shown in the explorer tab.
  const showFactFilters =
    (domain === "sales" || domain === "forecast") &&
    Boolean(itemField) &&
    Boolean(locationField);

  const formatPairFilterValue = useCallback((value: string): string => {
    const trimmed = value.trim();
    if (!trimmed) return "";
    return `=${trimmed}`;
  }, []);

  const effectiveFilters = useMemo(() => {
    const out = Object.fromEntries(
      Object.entries(debouncedColumnFilters).filter(([, value]) => value.trim() !== ""),
    );
    if (showFactFilters && debouncedItemFilter.trim() && itemField) {
      out[itemField] = formatPairFilterValue(debouncedItemFilter);
    }
    if (showFactFilters && debouncedLocationFilter.trim() && locationField) {
      out[locationField] = formatPairFilterValue(debouncedLocationFilter);
    }
    if (domain === "forecast" && selectedModel.trim()) {
      out["model_id"] = `=${selectedModel.trim()}`;
    }
    if (domain === "sku" && selectedCluster.trim()) {
      const filterCol = clusterSource === "ml" ? "ml_cluster" : "cluster_assignment";
      out[filterCol] = `=${selectedCluster.trim()}`;
    }
    return out;
  }, [
    debouncedColumnFilters,
    showFactFilters,
    debouncedItemFilter,
    debouncedLocationFilter,
    itemField,
    locationField,
    domain,
    selectedModel,
    selectedCluster,
    clusterSource,
    formatPairFilterValue,
  ]);

  const visibleCols = useMemo(() => {
    if (!meta) return [];
    return meta.columns.filter((col) => visibleColumns[col] !== false);
  }, [meta, visibleColumns]);

  // -----------------------------------------------------------------------
  // 2. Paginated table data (useQuery)
  // -----------------------------------------------------------------------
  const pageParams = useMemo(
    () => ({
      limit,
      offset,
      q: debouncedSearch,
      sort_by: sortBy,
      sort_dir: sortDir,
      filters: Object.keys(effectiveFilters).length > 0 ? effectiveFilters : undefined,
    }),
    [limit, offset, debouncedSearch, sortBy, sortDir, effectiveFilters],
  );

  const {
    data: pageData,
    isLoading: isLoadingPage,
    isFetching: isFetchingPage,
    error: pageError,
  } = useQuery({
    queryKey: queryKeys.domainPage(domain, pageParams),
    queryFn: () => fetchDomainPage(domain, {
      limit: pageParams.limit,
      offset: pageParams.offset,
      q: pageParams.q,
      sort_by: pageParams.sort_by,
      sort_dir: pageParams.sort_dir,
      filters: pageParams.filters,
    }),
    enabled: !!meta,
    staleTime: STALE.THIRTY_SEC,
    placeholderData: (prev) => prev,
  });

  const rows = useMemo<Record<string, unknown>[]>(() => {
    if (!pageData || !meta) return [];
    return (pageData[meta.plural] || []) as Record<string, unknown>[];
  }, [pageData, meta]);

  const total = pageData?.total ?? 0;
  const totalApproximate = Boolean(pageData?.total_approximate);

  // -----------------------------------------------------------------------
  // 3. Forecast models (useQuery)
  // -----------------------------------------------------------------------
  const { data: availableModels = [] } = useQuery<string[]>({
    queryKey: queryKeys.forecastModels(),
    queryFn: fetchForecastModels,
    enabled: domain === "forecast" && !!meta,
    staleTime: STALE.FIVE_MIN,
  });

  // -----------------------------------------------------------------------
  // 4. DFU clusters (useQuery)
  // -----------------------------------------------------------------------
  const { data: clustersPayload } = useQuery({
    queryKey: queryKeys.skuClusters(clusterSource),
    queryFn: () => fetchSkuClusters(clusterSource),
    enabled: domain === "sku",
    staleTime: STALE.FIVE_MIN,
  });
  const clusterSummary: ClusterInfo[] = clustersPayload?.clusters ?? [];

  // Cluster profiles (clusterMeta) are fetched in the Clusters tab, not here.

  // -----------------------------------------------------------------------
  // 5. Auto-sample item+location pair (useQuery)
  // -----------------------------------------------------------------------
  const shouldAutoSample =
    !!meta &&
    showFactFilters &&
    autoSampledDomain !== domain &&
    !itemFilter.trim() &&
    !locationFilter.trim();

  const { data: samplePair } = useQuery({
    queryKey: queryKeys.samplePair(domain),
    queryFn: () => fetchSamplePair(domain),
    enabled: shouldAutoSample,
    staleTime: STALE.TEN_MIN,
  });

  // Apply auto-sampled pair once
  useEffect(() => {
    if (!shouldAutoSample) {
      // Mark as sampled if user already has filters
      if (meta && showFactFilters && autoSampledDomain !== domain) {
        if (itemFilter.trim() || locationFilter.trim()) {
          setAutoSampledDomain(domain);
        }
      }
      return;
    }
    if (samplePair) {
      if (samplePair.item) setItemFilter(String(samplePair.item));
      if (samplePair.location) setLocationFilter(String(samplePair.location));
      setOffset(0);
      setAutoSampledDomain(domain);
    }
  }, [samplePair, shouldAutoSample, domain, meta, showFactFilters, autoSampledDomain, itemFilter, locationFilter]);

  // -----------------------------------------------------------------------
  // 6. Column-level typeahead suggestions (useEffect with manual fetching)
  //    We keep this as useEffect because it involves multiple parallel requests
  //    for different columns with staggered timing.
  // -----------------------------------------------------------------------
  useEffect(() => {
    if (!meta) return;
    const textCols = new Set(
      meta.columns.filter(
        (c) => !meta.numeric_fields.includes(c) && !meta.date_fields.includes(c),
      ),
    );
    const active = Object.entries(debouncedColumnFilters).filter(
      ([col, val]) => val.trim() !== "" && !val.startsWith("=") && textCols.has(col),
    );

    // Clear stale suggestions
    setColumnSuggestions((prev) => {
      const staleCols = Object.keys(prev).filter(
        (col) => !debouncedColumnFilters[col]?.trim(),
      );
      if (staleCols.length === 0) return prev;
      const next = { ...prev };
      staleCols.forEach((col) => delete next[col]);
      return next;
    });

    if (active.length === 0) return;

    let cancelled = false;
    const timers: number[] = [];

    for (const [col, val] of active) {
      const tid = window.setTimeout(async () => {
        try {
          const otherFilters: Record<string, string> = {};
          for (const [k, v] of Object.entries(debouncedColumnFilters)) {
            if (k !== col && v.trim()) otherFilters[k] = v.trim();
          }
          const values = await fetchDomainSuggest(
            domain,
            col,
            val.trim(),
            Object.keys(otherFilters).length > 0 ? otherFilters : undefined,
            12,
          );
          if (!cancelled) {
            setColumnSuggestions((prev) => ({ ...prev, [col]: values }));
          }
        } catch {
          if (!cancelled) {
            setColumnSuggestions((prev) => ({ ...prev, [col]: [] }));
          }
        }
      }, 180);
      timers.push(tid);
    }

    return () => {
      cancelled = true;
      timers.forEach((t) => window.clearTimeout(t));
    };
  }, [debouncedColumnFilters, domain, meta]);

  // -----------------------------------------------------------------------
  // Computed display values
  // -----------------------------------------------------------------------
  const start = total === 0 ? 0 : offset + 1;
  const end = Math.min(offset + limit, total);

  const loadingTable = isLoadingMeta || isLoadingPage || isFetchingPage;

  const error = useMemo(() => {
    if (metaError) return metaError instanceof Error ? metaError.message : "Failed to load domain metadata";
    if (pageError) return pageError instanceof Error ? pageError.message : "Failed to load records";
    return "";
  }, [metaError, pageError]);

  // -----------------------------------------------------------------------
  // Handlers
  // -----------------------------------------------------------------------
  const toggleSort = useCallback(
    (column: string) => {
      if (sortBy === column) {
        setSortDir((prev) => (prev === "asc" ? "desc" : "asc"));
        return;
      }
      setSortBy(column);
      setSortDir("asc");
    },
    [sortBy],
  );

  const toggleColumn = useCallback((column: string, checked: boolean) => {
    setVisibleColumns((prev) => ({ ...prev, [column]: checked }));
  }, []);

  const handleDomainChange = useCallback(
    (newDomain: string) => {
      onDomainChange(newDomain);
    },
    [onDomainChange],
  );

  const handleSearchChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setOffset(0);
    setSearch(e.target.value);
  }, []);

  const handleLimitChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    setOffset(0);
    setLimit(Number(e.target.value));
  }, []);

  const handleColumnFilterChange = useCallback(
    (col: string, value: string) => {
      setOffset(0);
      setColumnFilters((prev) => ({ ...prev, [col]: value }));
    },
    [],
  );

  const handlePreviousPage = useCallback(() => {
    setOffset((prev) => Math.max(0, prev - limit));
  }, [limit]);

  const handleNextPage = useCallback(() => {
    setOffset((prev) => prev + limit);
  }, [limit]);

  const handleSelectAllColumns = useCallback(() => {
    if (!meta) return;
    const all: Record<string, boolean> = {};
    meta.columns.forEach((c) => {
      all[c] = true;
    });
    setVisibleColumns(all);
  }, [meta]);

  const handleDeselectAllColumns = useCallback(() => {
    if (!meta) return;
    const none: Record<string, boolean> = {};
    meta.columns.forEach((c) => {
      none[c] = false;
    });
    setVisibleColumns(none);
  }, [meta]);

  // -----------------------------------------------------------------------
  // Render
  // -----------------------------------------------------------------------
  return (
    <section className="mt-4 grid gap-4 [&>*]:min-w-0 xl:grid-cols-1">
      <Card className="animate-fade-in">
        <CardHeader className="space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <CardTitle className="text-base">Data Explorer</CardTitle>
              <CardDescription className="max-w-2xl">
                Browse raw data across all domains. Select a <strong>Domain</strong> (item, location, DFU, sales,
                forecast, inventory) to explore its records. Use column filters to narrow results — prefix
                with <code className="text-[10px] bg-muted px-1 rounded">=</code> for exact match, or type
                plain text for fuzzy substring search.
              </CardDescription>
            </div>
            <div className="flex items-center gap-3">
              <select
                className="h-9 w-[180px] rounded-md border border-input bg-background px-3 text-sm"
                value={domain}
                onChange={(e) => handleDomainChange(e.target.value)}
              >
                {DIMENSION_DOMAINS.map((d) => {
                  return (
                    <option key={d} value={d}>
                      {titleCase(d)}
                    </option>
                  );
                })}
              </select>
              <Badge variant="secondary">
                {meta
                  ? `${titleCase(meta.name)} (${totalApproximate ? `${formatNumber(total - 1)}+` : formatNumber(total)})`
                  : "Loading"}
              </Badge>
            </div>
          </div>

          <div className="grid gap-2 md:grid-cols-[2fr_120px_1fr]">
            <Input
              placeholder="Search across configured fields"
              value={search}
              onChange={handleSearchChange}
              disabled={!meta}
            />
            <select
              className="h-9 rounded-md border border-input bg-background px-3 text-sm"
              value={limit}
              onChange={handleLimitChange}
            >
              {[50, 100, 250, 500].map((v) => (
                <option key={v} value={v}>
                  {v}/page
                </option>
              ))}
            </select>
            <Button variant="outline" onClick={() => setShowFieldPanel((v) => !v)}>
              <ChevronsUpDown className="mr-2 h-4 w-4" /> Fields
            </Button>
          </div>

          {showFieldPanel && meta ? (
            <div className="rounded-md border p-2">
              <div className="flex gap-2 mb-2">
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 text-xs"
                  onClick={handleSelectAllColumns}
                >
                  Select All
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 text-xs"
                  onClick={handleDeselectAllColumns}
                >
                  Deselect All
                </Button>
              </div>
              <div className="grid max-h-40 grid-cols-2 gap-2 overflow-y-auto overflow-x-hidden lg:grid-cols-3">
                {meta.columns.map((col) => (
                  <label key={col} className="flex items-center gap-2 text-sm">
                    <Checkbox
                      checked={visibleColumns[col] !== false}
                      onCheckedChange={(checked) => toggleColumn(col, checked === true)}
                    />
                    <span>{titleCase(col)}</span>
                  </label>
                ))}
              </div>
            </div>
          ) : null}

          {/* Fact-domain filters: item/location for sales/forecast, model for forecast, cluster for sku */}
          {showFactFilters || domain === "forecast" || domain === "sku" ? (
            <div className="flex flex-wrap items-end gap-3">
              {showFactFilters ? (
                <>
                  <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    {titleCase(itemField)}
                    <Input
                      className="h-9 w-40"
                      placeholder={`Filter ${itemField}...`}
                      value={itemFilter}
                      onChange={(e) => {
                        setOffset(0);
                        setItemFilter(e.target.value);
                      }}
                    />
                  </label>
                  <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    {titleCase(locationField)}
                    <Input
                      className="h-9 w-40"
                      placeholder={`Filter ${locationField}...`}
                      value={locationFilter}
                      onChange={(e) => {
                        setOffset(0);
                        setLocationFilter(e.target.value);
                      }}
                    />
                  </label>
                </>
              ) : null}
              {domain === "forecast" && availableModels.length > 0 ? (
                <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Model
                  <select
                    className="h-9 w-44 rounded-md border border-input bg-background px-3 text-sm"
                    value={selectedModel}
                    onChange={(e) => {
                      setOffset(0);
                      setSelectedModel(e.target.value);
                    }}
                  >
                    <option value="">All Models</option>
                    {availableModels.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </label>
              ) : null}
              {domain === "sku" && clusterSummary.length > 0 ? (
                <>
                  <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Cluster Source
                    <select
                      className="h-9 w-36 rounded-md border border-input bg-background px-3 text-sm"
                      value={clusterSource}
                      onChange={(e) => {
                        setClusterSource(e.target.value as "ml" | "source");
                        setSelectedCluster("");
                        setOffset(0);
                      }}
                    >
                      <option value="ml">ML Pipeline</option>
                      <option value="source">Source (sku.txt)</option>
                    </select>
                  </label>
                  <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Cluster
                    <select
                      className="h-9 w-52 rounded-md border border-input bg-background px-3 text-sm"
                      value={selectedCluster}
                      onChange={(e) => {
                        setOffset(0);
                        setSelectedCluster(e.target.value);
                      }}
                    >
                      <option value="">All Clusters</option>
                      {clusterSummary.map((c) => (
                        <option key={c.label} value={c.label}>
                          {c.label} ({formatCompactNumber(c.count)})
                        </option>
                      ))}
                    </select>
                  </label>
                </>
              ) : null}
            </div>
          ) : null}
        </CardHeader>

        <CardContent>
          {error ? (
            <Card className="mb-4 border-destructive/30 bg-destructive/10">
              <CardContent className="pt-4 flex items-center justify-between gap-2">
                <span className="text-sm text-destructive">{error}</span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleDomainChange(domain)}
                >
                  <RefreshCcw className="mr-1 h-3.5 w-3.5" /> Retry
                </Button>
              </CardContent>
            </Card>
          ) : null}

          <div className="relative">
            {/* Loading overlay */}
            {loadingTable && (
              <LoadingElement
                message={`Querying ${titleCase(domain)}...`}
                overlay
                size="md"
              />
            )}
            <div className="max-h-[680px] overflow-x-scroll overflow-y-auto rounded-md border pb-2 [scrollbar-gutter:stable]">
              <Table
                style={{
                  minWidth: `${Math.max(visibleCols.length * 260, 1800)}px`,
                }}
              >
                <TableHeader className="sticky top-0 z-20 bg-muted/80 backdrop-blur">
                  <TableRow>
                    {visibleCols.map((col, colIdx) => (
                      <TableHead
                        key={col}
                        className={cn(
                          "min-w-[180px] bg-muted/70 align-top",
                          colIdx === 0 && "sticky left-0 z-10 bg-muted",
                        )}
                      >
                        <Button
                          variant={sortBy === col ? "secondary" : "ghost"}
                          size="sm"
                          className="mb-1 h-7 w-full justify-between px-2"
                          onClick={() => toggleSort(col)}
                        >
                          <span>{titleCase(col)}</span>
                          {sortBy === col ? (
                            sortDir === "asc" ? (
                              <ArrowUpWideNarrow className="h-3.5 w-3.5" />
                            ) : (
                              <ArrowDownWideNarrow className="h-3.5 w-3.5" />
                            )
                          ) : (
                            <ChevronsUpDown className="h-3.5 w-3.5" />
                          )}
                        </Button>
                        <Input
                          className="h-7 text-xs"
                          placeholder="Filter (=exact)"
                          list={`col-suggest-${domain}-${col}`}
                          value={columnFilters[col] || ""}
                          onChange={(e) => handleColumnFilterChange(col, e.target.value)}
                        />
                        {(columnSuggestions[col]?.length ?? 0) > 0 && (
                          <datalist id={`col-suggest-${domain}-${col}`}>
                            {columnSuggestions[col].map((v) => (
                              <option key={v} value={v} />
                            ))}
                          </datalist>
                        )}
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {rows.length === 0 && !loadingTable ? (
                    <TableRow>
                      <TableCell
                        colSpan={Math.max(visibleCols.length, 1)}
                        className="h-24 text-center text-muted-foreground"
                      >
                        No records found. Try adjusting your filters or selecting a different domain.
                      </TableCell>
                    </TableRow>
                  ) : (
                    rows.map((row, idx) => (
                      <TableRow key={`row-${offset + idx}`}>
                        {visibleCols.map((col, colIdx) => (
                          <TableCell
                            key={`${offset + idx}-${col}`}
                            className={cn(
                              "whitespace-nowrap max-w-[300px] truncate",
                              colIdx === 0 && "sticky left-0 z-10 bg-card",
                            )}
                            title={row[col] != null ? String(row[col]) : ""}
                          >
                            {formatCell(row[col])}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </div>
          </div>

          <div className="mt-3 flex items-center justify-between gap-2 text-sm">
            <span className="text-muted-foreground">
              Showing {start}-{end} of{" "}
              {totalApproximate
                ? `${formatNumber(total - 1)}+`
                : formatNumber(total)}
              {total > 0 && (
                <span className="ml-2 tabular-nums">
                  (Page {Math.floor(offset / limit) + 1} of{" "}
                  {totalApproximate
                    ? `${Math.ceil((total - 1) / limit)}+`
                    : Math.ceil(total / limit)}
                  )
                </span>
              )}
            </span>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={offset === 0}
                onClick={handlePreviousPage}
              >
                Previous
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={offset + limit >= total}
                onClick={handleNextPage}
              >
                Next
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </section>
  );
}
