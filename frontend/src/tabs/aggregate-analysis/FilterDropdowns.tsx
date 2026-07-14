/**
 * Filter dropdown components for Portfolio Analysis tab.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChevronDown, X, Search } from "lucide-react";

import {
  queryKeys,
  STALE,
  fetchDistinctValues,
  type CascadeFilterParams,
} from "@/api/queries";
import { useDebounce } from "@/hooks/useDebounce";
import { cn } from "@/lib/utils";
import type { FilterConfig } from "./aggregateShared";

// ---------------------------------------------------------------------------
// FilterDropdown (non-searchable)
// ---------------------------------------------------------------------------
export function FilterDropdown({
  config,
  selected,
  onSelect,
  cascade,
}: {
  config: FilterConfig;
  selected: string[];
  onSelect: (vals: string[]) => void;
  cascade?: CascadeFilterParams;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const { data } = useQuery({
    queryKey: [...queryKeys.distinctValues(config.domain, config.column), cascade ?? null],
    queryFn: () => fetchDistinctValues(config.domain, config.column, undefined, 100, cascade),
    staleTime: STALE.THIRTY_SEC,
  });
  const values = data?.values ?? [];

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const toggle = useCallback(
    (val: string) => {
      onSelect(selected.includes(val) ? selected.filter((v) => v !== val) : [...selected, val]);
    },
    [selected, onSelect],
  );

  const label =
    selected.length === 0
      ? config.label
      : selected.length === 1
        ? selected[0]
        : `${config.label} (${selected.length})`;

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        aria-label={`Filter by ${config.label}`}
        aria-haspopup="listbox"
        aria-expanded={open}
        className={cn(
          "flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs transition-colors ease-smooth",
          selected.length > 0
            ? "border-primary/30 bg-primary/8 text-primary font-medium"
            : "border-border bg-card text-muted-foreground hover:bg-muted/50",
        )}
      >
        <span className="max-w-[120px] truncate">{label}</span>
        <ChevronDown className="h-3 w-3 flex-shrink-0 opacity-60" />
      </button>
      {open && (
        <div className="absolute left-0 top-full z-50 mt-1 max-h-60 min-w-[180px] overflow-y-auto rounded-md border border-border bg-card p-1 shadow-elevated animate-scale-in origin-top">
          {selected.length > 0 && (
            <button
              onClick={() => {
                onSelect([]);
                setOpen(false);
              }}
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
                "flex w-full items-center gap-2 rounded px-2 py-1.5 text-xs transition-colors ease-smooth",
                selected.includes(val) ? "bg-primary/10 text-primary" : "text-foreground hover:bg-muted/50",
              )}
            >
              <span
                className={cn(
                  "flex h-3.5 w-3.5 items-center justify-center rounded-sm border",
                  selected.includes(val) ? "border-primary bg-primary text-primary-foreground" : "border-border",
                )}
              >
                {selected.includes(val) && <span className="text-[10px]">&#10003;</span>}
              </span>
              <span className="truncate">{val}</span>
            </button>
          ))}
          {values.length === 0 && <p className="px-2 py-1.5 text-xs text-muted-foreground">No values</p>}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// SearchableFilterDropdown
// ---------------------------------------------------------------------------
export function SearchableFilterDropdown({
  config,
  selected,
  onSelect,
  cascade,
}: {
  config: FilterConfig;
  selected: string[];
  onSelect: (vals: string[]) => void;
  cascade?: CascadeFilterParams;
}) {
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

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  useEffect(() => {
    if (open) setTimeout(() => inputRef.current?.focus(), 0);
    else setSearch("");
  }, [open]);

  const addItem = useCallback(
    (val: string) => {
      if (!selected.includes(val)) onSelect([...selected, val]);
      setSearch("");
    },
    [selected, onSelect],
  );

  const removeItem = useCallback(
    (val: string) => {
      onSelect(selected.filter((v) => v !== val));
    },
    [selected, onSelect],
  );

  const label =
    selected.length === 0
      ? config.label
      : selected.length === 1
        ? selected[0]
        : `${config.label} (${selected.length})`;

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        aria-label={`Filter by ${config.label}`}
        aria-haspopup="listbox"
        aria-expanded={open}
        className={cn(
          "flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs transition-colors ease-smooth",
          selected.length > 0
            ? "border-primary/30 bg-primary/8 text-primary font-medium"
            : "border-border bg-card text-muted-foreground hover:bg-muted/50",
        )}
      >
        <span className="max-w-[120px] truncate">{label}</span>
        <ChevronDown className="h-3 w-3 flex-shrink-0 opacity-60" />
      </button>
      {open && (
        <div className="absolute left-0 top-full z-50 mt-1 min-w-[220px] rounded-md border border-border bg-card shadow-elevated animate-scale-in origin-top">
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
            {isFetching && <span className="h-3 w-3 animate-spin rounded-full border border-primary/30 border-t-primary" />}
          </div>
          {selected.length > 0 && (
            <div className="flex flex-wrap gap-1 border-b border-border px-2 py-1.5">
              {selected.map((val) => (
                <span key={val} className="inline-flex items-center gap-1 rounded bg-primary/10 px-1.5 py-0.5 text-[10px] text-primary">
                  {val}
                  <button onClick={() => removeItem(val)} className="hover:text-primary/70">
                    <X className="h-2.5 w-2.5" />
                  </button>
                </span>
              ))}
              <button onClick={() => onSelect([])} className="text-[10px] text-muted-foreground hover:text-foreground">
                Clear all
              </button>
            </div>
          )}
          <div className="max-h-48 overflow-y-auto p-1">
            {suggestions.map((val) => (
              <button
                key={val}
                onClick={() => addItem(val)}
                className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-xs text-foreground transition-colors ease-smooth hover:bg-muted/50"
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
// TimeGrainToggle
// ---------------------------------------------------------------------------
export function TimeGrainToggle({
  value,
  onChange,
}: {
  value: "month" | "quarter";
  onChange: (v: "month" | "quarter") => void;
}) {
  return (
    <div className="flex rounded-md border border-border">
      <button
        onClick={() => onChange("month")}
        className={cn(
          "px-2.5 py-1.5 text-xs transition-colors ease-smooth",
          value === "month" ? "bg-primary/10 text-primary" : "text-muted-foreground hover:bg-muted/50",
        )}
      >
        Mo
      </button>
      <button
        onClick={() => onChange("quarter")}
        className={cn(
          "border-l border-border px-2.5 py-1.5 text-xs transition-colors ease-smooth",
          value === "quarter" ? "bg-primary/10 text-primary" : "text-muted-foreground hover:bg-muted/50",
        )}
      >
        Qtr
      </button>
    </div>
  );
}
