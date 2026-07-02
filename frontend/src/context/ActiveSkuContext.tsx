// Active-SKU focus channel — lets the page currently in view publish the
// item+location it is showing so the global chat assistant can inherit that
// scope (e.g. the Item Analysis SKU, which lives in local component state and
// can diverge from the GlobalFilter). Intentionally separate from
// GlobalFilterContext: this is transient per-page display focus, not a filter.
// Tolerant of a missing provider (no-op publish / null read) so it is purely
// additive — pages and the drawer never crash if it isn't mounted.
import { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

export interface ActiveSku {
  item: string;
  loc: string;
}

interface ActiveSkuContextValue {
  activeSku: ActiveSku | null;
  setActiveSku: (sku: ActiveSku | null) => void;
}

const ActiveSkuContext = createContext<ActiveSkuContextValue | null>(null);

export function ActiveSkuProvider({ children }: { children: ReactNode }) {
  const [activeSku, setActiveSku] = useState<ActiveSku | null>(null);
  const value = useMemo(() => ({ activeSku, setActiveSku }), [activeSku]);
  return <ActiveSkuContext.Provider value={value}>{children}</ActiveSkuContext.Provider>;
}

/** Read the SKU the active page has published, or null. Safe without a provider. */
export function useActiveSku(): ActiveSku | null {
  return useContext(ActiveSkuContext)?.activeSku ?? null;
}

/**
 * Publish the SKU this page is showing so the global chat can inherit it.
 * Re-publishes when item/loc change, clears on unmount and when both are empty.
 * No-op when no provider is mounted.
 */
export function usePublishActiveSku(item: string, loc: string): void {
  const setActiveSku = useContext(ActiveSkuContext)?.setActiveSku;
  useEffect(() => {
    if (!setActiveSku) return;
    const i = item.trim();
    const l = loc.trim();
    setActiveSku(i || l ? { item: i, loc: l } : null);
    return () => setActiveSku(null);
  }, [item, loc, setActiveSku]);
}
