/**
 * SKU Features Tab — computed feature explorer with rich visualizations.
 *
 * Layout (top to bottom):
 *  1. Header + Compute button + status banner       (ComputeButton)
 *  2. Summary cards row                             (SummaryCards)
 *  3. Categorical distribution charts (3 horizontal) (CategoricalDistributions)
 *  4. Continuous-feature histograms (2x3 grid)      (FeatureHistograms)
 *  5. Filterable, sortable, paginated table         (FeatureTable)
 */
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  skuFeatureKeys,
  fetchSkuFeaturesSummary,
  fetchSkuFeaturesList,
  fetchSkuFeaturesDistributions,
  STALE_SKU_FEATURES,
} from "@/api/queries/sku-features";
import type {
  SkuFeatureRow,
  SkuFeaturesListParams,
} from "@/api/queries/sku-features";
import { ComputeButton } from "./sku-features/ComputeButton";
import { SummaryCards } from "./sku-features/SummaryCards";
import { CategoricalDistributions } from "./sku-features/CategoricalDistributions";
import { FeatureHistograms } from "./sku-features/FeatureHistograms";
import { FeatureTable } from "./sku-features/FeatureTable";
import { PAGE_SIZE } from "./sku-features/constants";

export default function SkuFeaturesTab() {
  // Filters & pagination state (lives at the orchestrator level so that
  // the table's controls can drive the query the orchestrator owns).
  const [search, setSearch] = useState("");
  const [seasonalityFilter, setSeasonalityFilter] = useState("");
  const [variabilityFilter, setVariabilityFilter] = useState("");
  const [trendFilter, setTrendFilter] = useState("");
  const [sortBy, setSortBy] = useState("sku_ck");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");
  const [page, setPage] = useState(0);

  const listParams: SkuFeaturesListParams = useMemo(
    () => ({
      limit: PAGE_SIZE,
      offset: page * PAGE_SIZE,
      sort_by: sortBy,
      sort_dir: sortDir,
      seasonality_profile: seasonalityFilter || undefined,
      variability_class: variabilityFilter || undefined,
      trend_direction: trendFilter || undefined,
      search: search || undefined,
    }),
    [page, sortBy, sortDir, seasonalityFilter, variabilityFilter, trendFilter, search],
  );

  // Queries
  // refetchOnMount: "always" so re-opening the tab always reflects the latest
  // compute job — without it a stale empty ("0 / Never") cache from before a
  // successful run lingers until staleTime elapses, hiding freshly computed data.
  const { data: summary, isLoading: summaryLoading } = useQuery({
    queryKey: skuFeatureKeys.summary,
    queryFn: fetchSkuFeaturesSummary,
    staleTime: STALE_SKU_FEATURES.SUMMARY,
    refetchOnMount: "always",
  });

  const { data: distributions, isLoading: distLoading } = useQuery({
    queryKey: skuFeatureKeys.distributions,
    queryFn: () => fetchSkuFeaturesDistributions(),
    staleTime: STALE_SKU_FEATURES.DISTRIBUTIONS,
    refetchOnMount: "always",
  });

  const { data: listData, isLoading: listLoading } = useQuery({
    queryKey: skuFeatureKeys.list(listParams as Record<string, unknown>),
    queryFn: () => fetchSkuFeaturesList(listParams),
    staleTime: STALE_SKU_FEATURES.LIST,
    refetchOnMount: "always",
  });

  const rows: SkuFeatureRow[] = listData?.rows ?? [];
  const totalRows = listData?.total ?? 0;

  return (
    <div className="space-y-6">
      <ComputeButton />

      <SummaryCards summary={summary} isLoading={summaryLoading} />

      <CategoricalDistributions summary={summary} isLoading={summaryLoading} />

      <FeatureHistograms distributions={distributions} isLoading={distLoading} />

      <FeatureTable
        rows={rows}
        totalRows={totalRows}
        isLoading={listLoading}
        state={{
          search,
          seasonalityFilter,
          variabilityFilter,
          trendFilter,
          sortBy,
          sortDir,
          page,
        }}
        handlers={{
          setSearch,
          setSeasonalityFilter,
          setVariabilityFilter,
          setTrendFilter,
          setSortBy,
          setSortDir,
          setPage,
        }}
      />
    </div>
  );
}
