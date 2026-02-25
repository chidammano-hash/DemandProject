import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, X, RotateCcw } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { cn } from "@/lib/utils";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";
import { fetchDistinctValues, queryKeys, STALE } from "@/api/queries";

// ---------------------------------------------------------------------------
// Filter dropdown config
// ---------------------------------------------------------------------------
interface FilterConfig {
  key: "brand" | "category" | "market" | "channel";
  label: string;
  domain: string;
  column: string;
}

const FILTERS: FilterConfig[] = [
  { key: "brand", label: "Brand", domain: "item", column: "brand_name" },
  { key: "category", label: "Category", domain: "item", column: "class_" },
  { key: "market", label: "Market", domain: "location", column: "state_id" },
  { key: "channel", label: "Channel", domain: "customer", column: "rpt_channel_desc" },
];

// ---------------------------------------------------------------------------
// Multi-select dropdown
// ---------------------------------------------------------------------------
function FilterDropdown({ config, selected, onSelect }: { config: FilterConfig; selected: string[]; onSelect: (vals: string[]) => void }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const { data } = useQuery({
    queryKey: queryKeys.distinctValues(config.domain, config.column),
    queryFn: () => fetchDistinctValues(config.domain, config.column),
    staleTime: STALE.FIVE_MIN,
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
            ? "border-primary/40 bg-primary/5 text-primary"
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

  return (
    <div className="flex items-center gap-2 border-b border-border bg-card/80 px-4 py-2 backdrop-blur-sm" role="toolbar" aria-label="Global filters">
      {FILTERS.map((cfg) => (
        <FilterDropdown
          key={cfg.key}
          config={cfg}
          selected={filters[cfg.key]}
          onSelect={(vals) => setFilters({ [cfg.key]: vals })}
        />
      ))}

      <div className="mx-1 h-5 w-px bg-border" />

      <TimeGrainToggle
        value={filters.timeGrain}
        onChange={(v) => setFilters({ timeGrain: v })}
      />

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
    </div>
  );
}
