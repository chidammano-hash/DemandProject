import { useEffect, useMemo } from "react";

import { Card, CardContent } from "@/components/ui/card";

import { DomainFiltersPanel } from "./explorer/DomainFiltersPanel";
import { ExplorerErrorBanner } from "./explorer/ExplorerErrorBanner";
import { ExplorerHeader } from "./explorer/ExplorerHeader";
import { ExplorerPagination } from "./explorer/ExplorerPagination";
import { ExplorerTable } from "./explorer/ExplorerTable";
import { FieldVisibilityPanel } from "./explorer/FieldVisibilityPanel";
import { buildEffectiveFilters, deriveMetaFields } from "./explorer/_helpers";
import { useExplorerState } from "./explorer/useExplorerState";
import {
  useAutoSamplePair,
  useDomainMeta,
  useDomainPage,
  useForecastModels,
  useSkuClusters,
} from "./explorer/useExplorerQueries";
import { useColumnSuggestions } from "./explorer/useColumnSuggestions";
import type { DomainPageRow } from "./explorer/types";

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
  const state = useExplorerState(domain);

  // 1. Domain metadata
  const { meta, isLoadingMeta, metaError } = useDomainMeta(domain);

  // Reset local state when meta changes
  useEffect(() => {
    if (!meta) return;
    state.resetForMeta(meta);
    // We intentionally exclude `state` from deps; resetForMeta is stable.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meta]);

  // Derived meta-dependent fields
  const { itemField, locationField, showFactFilters } = useMemo(
    () => deriveMetaFields(meta, domain),
    [meta, domain],
  );

  const effectiveFilters = useMemo(
    () =>
      buildEffectiveFilters({
        debouncedColumnFilters: state.debouncedColumnFilters,
        showFactFilters,
        debouncedItemFilter: state.debouncedItemFilter,
        debouncedLocationFilter: state.debouncedLocationFilter,
        itemField,
        locationField,
        domain,
        selectedModel: state.selectedModel,
        selectedCluster: state.selectedCluster,
        clusterSource: state.clusterSource,
      }),
    [
      state.debouncedColumnFilters,
      state.debouncedItemFilter,
      state.debouncedLocationFilter,
      state.selectedModel,
      state.selectedCluster,
      state.clusterSource,
      showFactFilters,
      itemField,
      locationField,
      domain,
    ],
  );

  const visibleCols = useMemo(() => {
    if (!meta) return [];
    return meta.columns.filter((col) => state.visibleColumns[col] !== false);
  }, [meta, state.visibleColumns]);

  // 2. Paginated data
  const pageParams = useMemo(
    () => ({
      limit: state.limit,
      offset: state.offset,
      q: state.debouncedSearch,
      sort_by: state.sortBy,
      sort_dir: state.sortDir,
      filters: Object.keys(effectiveFilters).length > 0 ? effectiveFilters : undefined,
    }),
    [
      state.limit,
      state.offset,
      state.debouncedSearch,
      state.sortBy,
      state.sortDir,
      effectiveFilters,
    ],
  );

  const { pageData, isLoadingPage, isFetchingPage, pageError } = useDomainPage(
    domain,
    pageParams,
    !!meta,
  );

  const rows = useMemo<DomainPageRow[]>(() => {
    if (!pageData || !meta) return [];
    return (pageData[meta.plural] || []) as DomainPageRow[];
  }, [pageData, meta]);

  const total = pageData?.total ?? 0;
  const totalApproximate = Boolean(pageData?.total_approximate);

  // 3. Forecast models
  const availableModels = useForecastModels(domain === "forecast" && !!meta);

  // 4. SKU clusters
  const clusterSummary = useSkuClusters(state.clusterSource, domain === "sku");

  // 5. Auto-sample item+location pair on first visit to a fact domain
  useAutoSamplePair({
    domain,
    meta,
    showFactFilters,
    itemFilter: state.itemFilter,
    locationFilter: state.locationFilter,
    autoSampledDomain: state.autoSampledDomain,
    setItemFilter: state.setItemFilter,
    setLocationFilter: state.setLocationFilter,
    setOffset: state.setOffset,
    setAutoSampledDomain: state.setAutoSampledDomain,
  });

  // 6. Column-level typeahead suggestions
  useColumnSuggestions(
    domain,
    meta,
    state.debouncedColumnFilters,
    state.setColumnSuggestions,
  );

  // Computed display values
  const loadingTable = isLoadingMeta || isLoadingPage || isFetchingPage;

  const error = useMemo(() => {
    if (metaError)
      return metaError instanceof Error
        ? metaError.message
        : "Failed to load domain metadata";
    if (pageError)
      return pageError instanceof Error
        ? pageError.message
        : "Failed to load records";
    return "";
  }, [metaError, pageError]);

  return (
    <section className="mt-4 grid gap-4 [&>*]:min-w-0 xl:grid-cols-1">
      <Card className="animate-fade-in">
        <ExplorerHeader
          domain={domain}
          meta={meta}
          total={total}
          totalApproximate={totalApproximate}
          search={state.search}
          limit={state.limit}
          onDomainChange={onDomainChange}
          onSearchChange={state.handleSearchChange}
          onLimitChange={state.handleLimitChange}
          onToggleFieldPanel={state.handleToggleFieldPanel}
        >
          {state.showFieldPanel && meta ? (
            <FieldVisibilityPanel
              meta={meta}
              visibleColumns={state.visibleColumns}
              onToggleColumn={state.toggleColumn}
              onSelectAll={state.handleSelectAllColumns}
              onDeselectAll={state.handleDeselectAllColumns}
            />
          ) : null}

          <DomainFiltersPanel
            domain={domain}
            showFactFilters={showFactFilters}
            itemField={itemField}
            locationField={locationField}
            itemFilter={state.itemFilter}
            locationFilter={state.locationFilter}
            selectedModel={state.selectedModel}
            selectedCluster={state.selectedCluster}
            clusterSource={state.clusterSource}
            availableModels={availableModels}
            clusterSummary={clusterSummary}
            onItemFilterChange={state.handleItemFilterInput}
            onLocationFilterChange={state.handleLocationFilterInput}
            onModelChange={state.handleModelInput}
            onClusterChange={state.handleClusterInput}
            onClusterSourceChange={state.handleClusterSourceInput}
          />
        </ExplorerHeader>

        <CardContent>
          {error ? (
            <ExplorerErrorBanner
              message={error}
              onRetry={() => onDomainChange(domain)}
            />
          ) : null}

          <ExplorerTable
            domain={domain}
            visibleCols={visibleCols}
            rows={rows}
            offset={state.offset}
            loading={loadingTable}
            sortBy={state.sortBy}
            sortDir={state.sortDir}
            columnFilters={state.columnFilters}
            columnSuggestions={state.columnSuggestions}
            onToggleSort={state.toggleSort}
            onColumnFilterChange={state.handleColumnFilterChange}
          />

          <ExplorerPagination
            offset={state.offset}
            limit={state.limit}
            total={total}
            totalApproximate={totalApproximate}
            onPrev={state.handlePreviousPage}
            onNext={state.handleNextPage}
          />
        </CardContent>
      </Card>
    </section>
  );
}
