/**
 * useGlobalFilterSync — extracts the repeated "sync global filter context into
 * local state" pattern found across 15+ tab/panel components.
 *
 * The canonical pattern it replaces:
 *   const [itemFilter, setItemFilter] = useState("");
 *   const [locationFilter, setLocationFilter] = useState("");
 *   const syncedGlobalRef = useRef("");
 *   useEffect(() => {
 *     const key = `${globalFilters.item.join(",")}_${globalFilters.location.join(",")}`;
 *     if (key === syncedGlobalRef.current) return;
 *     syncedGlobalRef.current = key;
 *     if (globalFilters.item.length === 1) setItemFilter(globalFilters.item[0]);
 *     if (globalFilters.location.length === 1) setLocationFilter(globalFilters.location[0]);
 *   }, [globalFilters.item, globalFilters.location]);
 *
 * Usage (drop-in replacement):
 *   const { item, setItem, location, setLocation } = useGlobalFilterSync({ item: true, location: true });
 *   // or item-only:
 *   const { item, setItem } = useGlobalFilterSync({ item: true });
 */
import { useEffect, useRef, useState, useCallback } from "react";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";
import type { GlobalFilters } from "@/types/theme";

// ---------------------------------------------------------------------------
// The set of global filter keys that can be synced to local state.
// Each is a string[] in GlobalFilters (we exclude timeGrain which is not an array).
// ---------------------------------------------------------------------------
type SyncableFilterKey = keyof Omit<GlobalFilters, "timeGrain">;

/**
 * Config object: which global filter keys to sync into local state.
 * Pass `true` to enable sync, or an object with `initialValue` to override
 * the initial local state (default is "").
 */
type FilterSyncConfig = {
  [K in SyncableFilterKey]?: boolean | { initialValue: string };
};

/**
 * Return type is dynamic based on which keys are enabled.
 * For each enabled key, you get:
 *   - `[key]`: the current local string value
 *   - `set<Key>`: setter for that value (stable identity via ref)
 *
 * Plus a `resetAll()` that clears all synced filters back to "".
 */
type FilterSyncResult<C extends FilterSyncConfig> = {
  [K in Extract<keyof C, SyncableFilterKey> as C[K] extends false | undefined
    ? never
    : K]: string;
} & {
  [K in Extract<keyof C, SyncableFilterKey> as C[K] extends false | undefined
    ? never
    : `set${Capitalize<K>}`]: (value: string) => void;
} & {
  /** Reset all synced local filters to "" */
  resetAll: () => void;
};

// ---------------------------------------------------------------------------
// Internal helper: capitalize first letter for setter name generation
// ---------------------------------------------------------------------------
function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// ---------------------------------------------------------------------------
// Hook implementation
// ---------------------------------------------------------------------------

/**
 * Syncs selected global filter values into local component state.
 *
 * When a global filter array contains exactly 1 value, it is pushed into the
 * corresponding local string state. The sync is deduplicated via a ref-based
 * key check so it only fires when the global filter actually changes (not on
 * every render).
 *
 * @param config - Object specifying which filter keys to sync.
 *   Example: `{ item: true, location: true }` or `{ item: { initialValue: "SKU-1" } }`
 *
 * @returns Object with filter values, setters, and a resetAll function.
 */
export function useGlobalFilterSync<C extends FilterSyncConfig>(
  config: C,
): FilterSyncResult<C> {
  const { filters: globalFilters } = useGlobalFilterContext();

  // Determine which keys are enabled and their initial values.
  // These are derived from config which must be stable across renders
  // (same object shape every call — enforced by rules of hooks).
  const enabledKeys: SyncableFilterKey[] = [];
  const initialValues: Record<string, string> = {};

  for (const [key, cfg] of Object.entries(config) as [SyncableFilterKey, boolean | { initialValue: string } | undefined][]) {
    if (!cfg) continue;
    enabledKeys.push(key);
    initialValues[key] = typeof cfg === "object" && "initialValue" in cfg ? cfg.initialValue : "";
  }

  // --- Local state for each enabled key ---
  // We use a single state object rather than N separate useState calls to keep
  // the hook count stable regardless of config (rules of hooks).
  const [values, setValues] = useState<Record<string, string>>(() => ({ ...initialValues }));

  // Ref for deduplication — mirrors the `syncedGlobalRef` pattern
  const syncedKeyRef = useRef<string>("");

  // --- Sync effect ---
  useEffect(() => {
    // Build a composite key from the global filter arrays for all enabled keys
    const keyParts = enabledKeys.map((k) => globalFilters[k].join(","));
    const compositeKey = keyParts.join("_");

    if (compositeKey === syncedKeyRef.current) return;
    syncedKeyRef.current = compositeKey;

    setValues((prev) => {
      const next = { ...prev };
      let changed = false;
      for (const k of enabledKeys) {
        const arr = globalFilters[k];
        if (arr.length === 1) {
          if (next[k] !== arr[0]) {
            next[k] = arr[0];
            changed = true;
          }
        }
      }
      return changed ? next : prev;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps -- enabledKeys is stable from config
  }, [globalFilters, ...enabledKeys.map((k) => globalFilters[k])]);

  // --- Build stable setters via ref ---
  // Setters are stored in a ref so their identity is stable across renders.
  // Each setter uses the functional form of setValues so it always sees the
  // latest state without needing to be recreated.
  const settersRef = useRef<Record<string, (value: string) => void>>({});
  for (const k of enabledKeys) {
    if (!settersRef.current[k]) {
      settersRef.current[k] = (value: string) => {
        setValues((prev) => (prev[k] === value ? prev : { ...prev, [k]: value }));
      };
    }
  }

  const resetAll = useCallback(() => {
    setValues({ ...initialValues });
    syncedKeyRef.current = "";
    // eslint-disable-next-line react-hooks/exhaustive-deps -- initialValues derived from stable config
  }, []);

  // Assemble the return object
  const result: Record<string, unknown> = { resetAll };

  for (const k of enabledKeys) {
    result[k] = values[k] ?? "";
    result[`set${capitalize(k)}`] = settersRef.current[k];
  }

  return result as FilterSyncResult<C>;
}

// ---------------------------------------------------------------------------
// Convenience presets for the most common patterns
// ---------------------------------------------------------------------------

/** Sync only item from global filters. */
export function useItemFilterSync() {
  return useGlobalFilterSync({ item: true });
}

/** Sync item + location from global filters (most common pattern). */
export function useItemLocationFilterSync() {
  return useGlobalFilterSync({ item: true, location: true });
}
