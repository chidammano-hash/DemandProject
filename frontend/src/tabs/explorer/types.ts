/**
 * Shared types for the Data Explorer tab.
 */
import type { DomainMeta, ClusterInfo } from "@/types";

export type SortDir = "asc" | "desc";
export type ClusterSource = "ml" | "source";

export const DIMENSION_DOMAINS = [
  "item",
  "location",
  "customer",
  "time",
  "sku",
  "sales",
  "forecast",
  "customer_demand",
  "customer_features",
];

export interface ExplorerState {
  offset: number;
  limit: number;
  search: string;
  sortBy: string;
  sortDir: SortDir;
  columnFilters: Record<string, string>;
  visibleColumns: Record<string, boolean>;
  showFieldPanel: boolean;
  itemFilter: string;
  locationFilter: string;
  selectedModel: string;
  selectedCluster: string;
  clusterSource: ClusterSource;
  autoSampledDomain: string;
  columnSuggestions: Record<string, string[]>;
}

export type DomainPageRow = Record<string, unknown>;

export interface ExplorerMetaInfo {
  meta: DomainMeta | undefined;
  rows: DomainPageRow[];
  total: number;
  totalApproximate: boolean;
  visibleCols: string[];
  itemField: string;
  locationField: string;
  showFactFilters: boolean;
  availableModels: string[];
  clusterSummary: ClusterInfo[];
  loadingTable: boolean;
  error: string;
}
