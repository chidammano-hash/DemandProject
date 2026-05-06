/**
 * Manages typeahead column suggestions for the Data Explorer.
 *
 * Implemented as a side-effect hook because each visible column may need its
 * own suggestion request, with staggered timing and cancellation when filters
 * change. Stale suggestions for cleared filters are also pruned.
 */
import { useEffect } from "react";

import { fetchDomainSuggest } from "@/api/queries";
import type { DomainMeta } from "@/types";

const SUGGEST_DEBOUNCE_MS = 180;
const SUGGEST_LIMIT = 12;

export function useColumnSuggestions(
  domain: string,
  meta: DomainMeta | undefined,
  debouncedColumnFilters: Record<string, string>,
  setColumnSuggestions: React.Dispatch<
    React.SetStateAction<Record<string, string[]>>
  >,
): void {
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

    // Clear stale suggestions for filters that have been cleared.
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
            SUGGEST_LIMIT,
          );
          if (!cancelled) {
            setColumnSuggestions((prev) => ({ ...prev, [col]: values }));
          }
        } catch {
          if (!cancelled) {
            setColumnSuggestions((prev) => ({ ...prev, [col]: [] }));
          }
        }
      }, SUGGEST_DEBOUNCE_MS);
      timers.push(tid);
    }

    return () => {
      cancelled = true;
      timers.forEach((t) => window.clearTimeout(t));
    };
  }, [debouncedColumnFilters, domain, meta, setColumnSuggestions]);
}
