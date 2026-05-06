/**
 * Pure helpers for the Data Explorer tab.
 */
import type { DomainMeta } from "@/types";

import type { ClusterSource } from "./types";

/**
 * Detect canonical item / location field names for a domain meta payload.
 * Returns blank strings when the domain has no fact-table-style key columns.
 * `showFactFilters` is true only for the sales/forecast domains where both
 * item and location fields are resolvable.
 */
export function deriveMetaFields(meta: DomainMeta | undefined, domain: string) {
  const itemField = (() => {
    if (!meta) return "";
    if (meta.columns.includes("item_id")) return "item_id";
    return "";
  })();

  const locationField = (() => {
    if (!meta) return "";
    if (meta.columns.includes("loc")) return "loc";
    if (meta.columns.includes("location_id")) return "location_id";
    return "";
  })();

  const showFactFilters =
    (domain === "sales" || domain === "forecast") &&
    Boolean(itemField) &&
    Boolean(locationField);

  return { itemField, locationField, showFactFilters };
}

/**
 * Build the column-filter map sent to /domains/{domain}/page based on the
 * combination of debounced text filters, item/location pair filters, model
 * filter (forecast domain), and cluster filter (sku domain).
 */
export interface BuildEffectiveFiltersArgs {
  debouncedColumnFilters: Record<string, string>;
  showFactFilters: boolean;
  debouncedItemFilter: string;
  debouncedLocationFilter: string;
  itemField: string;
  locationField: string;
  domain: string;
  selectedModel: string;
  selectedCluster: string;
  clusterSource: ClusterSource;
}

export function buildEffectiveFilters(
  args: BuildEffectiveFiltersArgs,
): Record<string, string> {
  const {
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
  } = args;
  const out = Object.fromEntries(
    Object.entries(debouncedColumnFilters).filter(
      ([, value]) => value.trim() !== "",
    ),
  );
  const formatPair = (value: string) => `=${value.trim()}`;
  if (showFactFilters && debouncedItemFilter.trim() && itemField) {
    out[itemField] = formatPair(debouncedItemFilter);
  }
  if (showFactFilters && debouncedLocationFilter.trim() && locationField) {
    out[locationField] = formatPair(debouncedLocationFilter);
  }
  if (domain === "forecast" && selectedModel.trim()) {
    out["model_id"] = `=${selectedModel.trim()}`;
  }
  if (domain === "sku" && selectedCluster.trim()) {
    const filterCol = clusterSource === "ml" ? "ml_cluster" : "cluster_assignment";
    out[filterCol] = `=${selectedCluster.trim()}`;
  }
  return out;
}
