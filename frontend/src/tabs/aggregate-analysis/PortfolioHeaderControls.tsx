import type { Dispatch, SetStateAction } from "react";
import { CalendarClock, RotateCcw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { formatDate } from "@/lib/formatters";
import { cn } from "@/lib/utils";
import { PageHeader } from "@/components/PageHeader";
import { FilterDropdown, SearchableFilterDropdown, TimeGrainToggle } from "./FilterDropdowns";
import {
  buildCascade,
  EMPTY_FILTERS,
  FILTERS,
  hasActiveFilters,
  PANELS,
  type FilterConfig,
  type LocalFilters,
} from "./aggregateShared";

interface PortfolioHeaderControlsProps {
  filters: LocalFilters;
  setFilters: Dispatch<SetStateAction<LocalFilters>>;
  onFilterChange: (key: FilterConfig["key"], values: string[]) => void;
  planningDate?: string | null;
  skuCount?: number | null;
  visiblePanels: Record<string, boolean>;
  onTogglePanel: (key: string) => void;
}

export function PortfolioHeaderControls({
  filters,
  setFilters,
  onFilterChange,
  planningDate,
  skuCount,
  visiblePanels,
  onTogglePanel,
}: PortfolioHeaderControlsProps) {
  const filtered = hasActiveFilters(filters);
  return (
    <>
      <PageHeader
        title="Portfolio Analysis"
        description="Forecast performance and accuracy analytics across your portfolio. Filter by brand, category, item, location."
        actions={
          <>
            {planningDate && (
              <span className="flex items-center gap-1 rounded bg-muted/50 px-2 py-1 text-2xs text-muted-foreground">
                <CalendarClock className="h-3 w-3" />
                Plan as of {formatDate(planningDate)}
              </span>
            )}
            {filtered && skuCount != null && (
              <span className="rounded bg-primary/10 px-2 py-1 text-2xs font-medium text-primary">
                {skuCount.toLocaleString()} SKUs
              </span>
            )}
          </>
        }
      />

      <div className="flex flex-wrap items-center gap-2" aria-label="Portfolio filters">
        {FILTERS.map((config) =>
          config.searchable ? (
            <SearchableFilterDropdown
              key={config.key}
              config={config}
              selected={filters[config.key]}
              onSelect={(values) => onFilterChange(config.key, values)}
              cascade={buildCascade(filters, config.key)}
            />
          ) : (
            <FilterDropdown
              key={config.key}
              config={config}
              selected={filters[config.key]}
              onSelect={(values) => onFilterChange(config.key, values)}
              cascade={buildCascade(filters, config.key)}
            />
          )
        )}
        <TimeGrainToggle
          value={filters.timeGrain}
          onChange={(timeGrain) => setFilters((current) => ({ ...current, timeGrain }))}
        />
        {filtered && (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-7 gap-1 text-xs"
            onClick={() => setFilters(EMPTY_FILTERS)}
          >
            <RotateCcw className="h-3 w-3" /> Reset
          </Button>
        )}
      </div>

      <div
        className="flex flex-wrap items-center gap-3 rounded-md border border-border bg-muted/30 px-3 py-2"
        aria-label="Portfolio panels"
      >
        {PANELS.map((panel) => (
          <label key={panel.key} className="flex items-center gap-1.5 text-xs">
            <Checkbox
              checked={visiblePanels[panel.key]}
              onCheckedChange={() => onTogglePanel(panel.key)}
              aria-label={`Toggle ${panel.label}`}
              className="h-3.5 w-3.5"
            />
            <span
              className={cn(visiblePanels[panel.key] ? "text-foreground" : "text-muted-foreground")}
            >
              {panel.label}
            </span>
          </label>
        ))}
      </div>
    </>
  );
}
