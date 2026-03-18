import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, X, RotateCcw, Search, CalendarClock } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { cn } from "@/lib/utils";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";
import { fetchDistinctValues, fetchPlanningDate, queryKeys, STALE, filterMetaKeys, fetchDfuCount } from "@/api/queries";
import type { CascadeFilterParams } from "@/api/queries";
import type { GlobalFilters } from "@/types/theme";
import { useDebounce } from "@/hooks/useDebounce";

// ---------------------------------------------------------------------------
// Filter dropdown config
// ---------------------------------------------------------------------------
interface FilterConfig {
  key: "brand" | "category" | "market" | "channel" | "item" | "location" | "cluster";
  label: string;
  domain: string;
  column: string;
  searchable?: boolean;
}

const FILTERS: FilterConfig[] = [
  { key: "brand", label: "Brand", domain: "item", column: "brand_name" },
  { key: "category", label: "Category", domain: "item", column: "class_" },
  { key: "item", label: "Item", domain: "item", column: "item_no", searchable: true },
  { key: "location", label: "Location", domain: "location", column: "location_id", searchable: true },
  { key: "market", label: "Market", domain: "location", column: "state_id" },
  { key: "channel", label: "Channel", domain: "customer", column: "rpt_channel_desc" },
  { key: "cluster", label: "Cluster", domain: "dfu", column: "cluster_assignment" },
];

/** Build cascade params from all filters EXCEPT the one being queried. */
function buildCascade(
  filters: GlobalFilters,
  excludeKey: FilterConfig["key"],
): CascadeFilterParams | undefined {
  const c: CascadeFilterParams = {};
  if (excludeKey !== "brand" && filters.brand.length > 0) c.brand = filters.brand.join(",");
  if (excludeKey !== "category" && filters.category.length > 0) c.category = filters.category.join(",");
  if (excludeKey !== "item" && filters.item.length > 0) c.item = filters.item.join(",");
  if (excludeKey !== "location" && filters.location.length > 0) c.location = filters.location.join(",");
  if (excludeKey !== "market" && filters.market.length > 0) c.market = filters.market.join(",");
  if (excludeKey !== "channel" && filters.channel.length > 0) c.channel = filters.channel.join(",");
  if (excludeKey !== "cluster" && filters.cluster.length > 0) c.cluster = filters.cluster.join(",");
  return Object.keys(c).length > 0 ? c : undefined;
}

// ---------------------------------------------------------------------------
// Multi-select dropdown (low-cardinality — preloads all values)
// ---------------------------------------------------------------------------
function FilterDropdown({ config, selected, onSelect, cascade }: { config: FilterConfig; selected: string[]; onSelect: (vals: string[]) => void; cascade?: CascadeFilterParams }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const { data } = useQuery({
    queryKey: [...queryKeys.distinctValues(config.domain, config.column), cascade ?? null],
    queryFn: () => fetchDistinctValues(config.domain, config.column, undefined, 100, cascade),
    staleTime: STALE.THIRTY_SEC,
  });
  const values = data?.values ?? [];

  // Click outside to close
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const toggle = useCallback((val: string) => {
    if (selected.includes(val)) {
      onSelect(selected.filter((v) => v !== val));
    } else {
      onSelect([...selected, val]);
    }
  }, [selected, onSelect]);

  const label = selected.length === 0
    ? config.label
    : selected.length === 1
      ? selected[0]
      : `${config.label} (${selected.length})`;

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className={cn(
          "flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs transition-colors",
          selected.length > 0
            ? "border-primary/30 bg-primary/8 text-primary font-medium"
            : "border-border bg-card text-muted-foreground hover:bg-muted/50",
        )}
      >
        <span className="max-w-[120px] truncate">{label}</span>
        <ChevronDown className="h-3 w-3 flex-shrink-0 opacity-60" />
      </button>

      {open && (
        <div className="absolute left-0 top-full z-50 mt-1 max-h-60 min-w-[180px] overflow-y-auto rounded-md border border-border bg-card p-1 shadow-lg">
          {selected.length > 0 && (
            <button
              onClick={() => { onSelect([]); setOpen(false); }}
              className="flex w-full items-center gap-1.5 rounded px-2 py-1.5 text-xs text-muted-foreground hover:bg-muted/50"
            >
              <X className="h-3 w-3" /> Clear
            </button>
          )}
          {values.map((val) => (
            <button
              key={val}
              onClick={() => toggle(val)}
              className={cn(
                "flex w-full items-center gap-2 rounded px-2 py-1.5 text-xs transition-colors",
                selected.includes(val)
                  ? "bg-primary/10 text-primary"
                  : "text-foreground hover:bg-muted/50",
              )}
            >
              <span className={cn(
                "flex h-3.5 w-3.5 items-center justify-center rounded-sm border",
                selected.includes(val) ? "border-primary bg-primary text-primary-foreground" : "border-border",
              )}>
                {selected.includes(val) && <span className="text-[10px]">&#10003;</span>}
              </span>
              <span className="truncate">{val}</span>
            </button>
          ))}
          {values.length === 0 && (
            <p className="px-2 py-1.5 text-xs text-muted-foreground">No values</p>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Searchable multi-select dropdown (high-cardinality — search-as-you-type)
// ---------------------------------------------------------------------------
function SearchableFilterDropdown({ config, selected, onSelect, cascade }: { config: FilterConfig; selected: string[]; onSelect: (vals: string[]) => void; cascade?: CascadeFilterParams }) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const ref = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const debouncedSearch = useDebounce(search, 250);

  const { data, isFetching } = useQuery({
    queryKey: [...queryKeys.distinctValues(config.domain, config.column + ":" + debouncedSearch), cascade ?? null],
    queryFn: () => fetchDistinctValues(config.domain, config.column, debouncedSearch || undefined, 20, cascade),
    enabled: open,
    staleTime: STALE.THIRTY_SEC,
  });
  const suggestions = (data?.values ?? []).filter((v) => !selected.includes(v));

  // Click outside to close
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  // Focus input when opening
  useEffect(() => {
    if (open) {
      setTimeout(() => inputRef.current?.focus(), 0);
    } else {
      setSearch("");
    }
  }, [open]);

  const addItem = useCallback((val: string) => {
    if (!selected.includes(val)) {
      onSelect([...selected, val]);
    }
    setSearch("");
  }, [selected, onSelect]);

  const removeItem = useCallback((val: string) => {
    onSelect(selected.filter((v) => v !== val));
  }, [selected, onSelect]);

  const label = selected.length === 0
    ? config.label
    : selected.length === 1
      ? selected[0]
      : `${config.label} (${selected.length})`;

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className={cn(
          "flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs transition-colors",
          selected.length > 0
            ? "border-primary/30 bg-primary/8 text-primary font-medium"
            : "border-border bg-card text-muted-foreground hover:bg-muted/50",
        )}
      >
        <span className="max-w-[120px] truncate">{label}</span>
        <ChevronDown className="h-3 w-3 flex-shrink-0 opacity-60" />
      </button>

      {open && (
        <div className="absolute left-0 top-full z-50 mt-1 min-w-[220px] rounded-md border border-border bg-card shadow-lg">
          {/* Search input */}
          <div className="flex items-center gap-1.5 border-b border-border px-2 py-1.5">
            <Search className="h-3 w-3 text-muted-foreground" />
            <input
              ref={inputRef}
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder={`Search ${config.label.toLowerCase()}...`}
              className="flex-1 bg-transparent text-xs text-foreground placeholder:text-muted-foreground outline-none"
            />
            {isFetching && (
              <span className="h-3 w-3 animate-spin rounded-full border border-primary/30 border-t-primary" />
            )}
          </div>

          {/* Selected chips */}
          {selected.length > 0 && (
            <div className="flex flex-wrap gap-1 border-b border-border px-2 py-1.5">
              {selected.map((val) => (
                <span
                  key={val}
                  className="inline-flex items-center gap-1 rounded bg-primary/10 px-1.5 py-0.5 text-[10px] text-primary"
                >
                  {val}
                  <button onClick={() => removeItem(val)} className="hover:text-primary/70">
                    <X className="h-2.5 w-2.5" />
                  </button>
                </span>
              ))}
              <button
                onClick={() => { onSelect([]); }}
                className="text-[10px] text-muted-foreground hover:text-foreground"
              >
                Clear all
              </button>
            </div>
          )}

          {/* Suggestions list */}
          <div className="max-h-48 overflow-y-auto p-1">
            {suggestions.map((val) => (
              <button
                key={val}
                onClick={() => addItem(val)}
                className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-xs text-foreground transition-colors hover:bg-muted/50"
              >
                <span className="flex h-3.5 w-3.5 items-center justify-center rounded-sm border border-border" />
                <span className="truncate">{val}</span>
              </button>
            ))}
            {suggestions.length === 0 && !isFetching && (
              <p className="px-2 py-1.5 text-xs text-muted-foreground">No matches</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Time grain toggle
// ---------------------------------------------------------------------------
function TimeGrainToggle({ value, onChange }: { value: "month" | "quarter"; onChange: (v: "month" | "quarter") => void }) {
  return (
    <div className="flex rounded-md border border-border">
      <button
        onClick={() => onChange("month")}
        className={cn(
          "px-2.5 py-1.5 text-xs transition-colors",
          value === "month" ? "bg-primary/10 text-primary" : "text-muted-foreground hover:bg-muted/50",
        )}
      >
        Mo
      </button>
      <button
        onClick={() => onChange("quarter")}
        className={cn(
          "border-l border-border px-2.5 py-1.5 text-xs transition-colors",
          value === "quarter" ? "bg-primary/10 text-primary" : "text-muted-foreground hover:bg-muted/50",
        )}
      >
        Qtr
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// GlobalFilterBar
// ---------------------------------------------------------------------------
export function GlobalFilterBar() {
  const { filters, setFilters, resetFilters, hasActiveFilters } = useGlobalFilterContext();

  const { data: planningDateInfo } = useQuery({
    queryKey: queryKeys.planningDate(),
    queryFn: fetchPlanningDate,
    staleTime: STALE.TEN_MIN,
  });

  const { data: dfuCountData } = useQuery({
    queryKey: filterMetaKeys.dfuCount(filters),
    queryFn: () => fetchDfuCount(filters),
    staleTime: STALE.FIVE_MIN,
    enabled: hasActiveFilters,
  });

  // Memoize cascade params per filter key so dropdown queries re-fetch when other filters change
  const cascades = useMemo(() => {
    const result: Record<string, CascadeFilterParams | undefined> = {};
    for (const cfg of FILTERS) {
      result[cfg.key] = buildCascade(filters, cfg.key);
    }
    return result;
  }, [filters]);

  return (
    <div className="flex items-center gap-2 border-b border-border bg-card/80 px-4 py-2 shadow-sm backdrop-blur-sm" role="toolbar" aria-label="Global filters">
      {FILTERS.map((cfg) =>
        cfg.searchable ? (
          <SearchableFilterDropdown
            key={cfg.key}
            config={cfg}
            selected={filters[cfg.key]}
            onSelect={(vals) => setFilters({ [cfg.key]: vals })}
            cascade={cascades[cfg.key]}
          />
        ) : (
          <FilterDropdown
            key={cfg.key}
            config={cfg}
            selected={filters[cfg.key]}
            onSelect={(vals) => setFilters({ [cfg.key]: vals })}
            cascade={cascades[cfg.key]}
          />
        ),
      )}

      <div className="mx-1 h-5 w-px bg-border" />

      <TimeGrainToggle
        value={filters.timeGrain}
        onChange={(v) => setFilters({ timeGrain: v })}
      />

      {hasActiveFilters && dfuCountData != null && (
        <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[11px] font-medium text-primary">
          {dfuCountData.count.toLocaleString()} DFUs
        </span>
      )}

      {hasActiveFilters && (
        <button
          onClick={resetFilters}
          className="ml-1 flex items-center gap-1 rounded-md px-2 py-1.5 text-xs text-muted-foreground hover:text-foreground"
          title="Reset all filters"
        >
          <RotateCcw className="h-3 w-3" />
          Reset
        </button>
      )}

      {planningDateInfo && (
        <div className="ml-auto flex items-center">
          <div
            className={cn(
              "flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-medium",
              planningDateInfo.is_frozen
                ? "border-amber-300 bg-amber-50 text-amber-800 dark:border-amber-700 dark:bg-amber-950/40 dark:text-amber-300"
                : "border-border bg-muted/50 text-muted-foreground",
            )}
            title={
              planningDateInfo.is_frozen
                ? `Frozen planning date — ${planningDateInfo.days_behind} day(s) behind system date (${planningDateInfo.system_date})`
                : "Using live system date"
            }
          >
            <CalendarClock className="h-3 w-3 flex-shrink-0" />
            <span>
              {planningDateInfo.is_frozen ? "Plan: " : ""}
              {new Date(planningDateInfo.planning_date + "T00:00:00").toLocaleDateString("en-GB", {
                day: "2-digit", month: "short", year: "numeric",
              })}
            </span>
            {planningDateInfo.is_frozen && (
              <span className="opacity-70">⚠</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
