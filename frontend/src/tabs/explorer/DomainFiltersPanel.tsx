/**
 * Domain-aware filter strip: item/location for fact tables, model for forecast,
 * cluster + cluster-source for sku domain.
 */
import { Input } from "@/components/ui/input";
import type { ClusterInfo } from "@/types";
import {
  titleCase,
  formatCompactNumber,
  formatClusterLabel,
} from "@/lib/formatters";

import type { ClusterSource } from "./types";

export interface DomainFiltersPanelProps {
  domain: string;
  showFactFilters: boolean;
  itemField: string;
  locationField: string;
  itemFilter: string;
  locationFilter: string;
  selectedModel: string;
  selectedCluster: string;
  clusterSource: ClusterSource;
  availableModels: string[];
  clusterSummary: ClusterInfo[];
  onItemFilterChange: (value: string) => void;
  onLocationFilterChange: (value: string) => void;
  onModelChange: (value: string) => void;
  onClusterChange: (value: string) => void;
  onClusterSourceChange: (value: ClusterSource) => void;
}

export function DomainFiltersPanel({
  domain,
  showFactFilters,
  itemField,
  locationField,
  itemFilter,
  locationFilter,
  selectedModel,
  selectedCluster,
  clusterSource,
  availableModels,
  clusterSummary,
  onItemFilterChange,
  onLocationFilterChange,
  onModelChange,
  onClusterChange,
  onClusterSourceChange,
}: DomainFiltersPanelProps) {
  if (!(showFactFilters || domain === "forecast" || domain === "sku")) {
    return null;
  }

  return (
    <div className="flex flex-wrap items-end gap-3">
      {showFactFilters ? (
        <>
          <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {titleCase(itemField)}
            <Input
              className="h-9 w-40"
              placeholder={`Filter ${itemField}...`}
              value={itemFilter}
              onChange={(e) => onItemFilterChange(e.target.value)}
            />
          </label>
          <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {titleCase(locationField)}
            <Input
              className="h-9 w-40"
              placeholder={`Filter ${locationField}...`}
              value={locationFilter}
              onChange={(e) => onLocationFilterChange(e.target.value)}
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
            onChange={(e) => onModelChange(e.target.value)}
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
              onChange={(e) =>
                onClusterSourceChange(e.target.value as ClusterSource)
              }
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
              onChange={(e) => onClusterChange(e.target.value)}
            >
              <option value="">All Clusters</option>
              {clusterSummary.map((c) => (
                <option key={c.label} value={c.label}>
                  {formatClusterLabel(c.label)} ({formatCompactNumber(c.count)})
                </option>
              ))}
            </select>
          </label>
        </>
      ) : null}
    </div>
  );
}
