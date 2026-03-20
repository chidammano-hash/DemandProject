import { useCallback, useState } from "react";

/**
 * Generic hook for toggling panel visibility with localStorage persistence.
 */
export function usePanelToggles(
  storageKey: string,
  defaults: Record<string, boolean>,
) {
  const [panels, setPanels] = useState<Record<string, boolean>>(() => {
    try {
      const stored = localStorage.getItem(storageKey);
      if (stored) {
        const parsed = JSON.parse(stored) as Record<string, boolean>;
        // Merge with defaults so new panels get their default value
        return { ...defaults, ...parsed };
      }
    } catch { /* ignore corrupt data */ }
    return { ...defaults };
  });

  const toggle = useCallback(
    (key: string) => {
      setPanels((prev) => {
        const next = { ...prev, [key]: !prev[key] };
        try { localStorage.setItem(storageKey, JSON.stringify(next)); } catch { /* quota */ }
        return next;
      });
    },
    [storageKey],
  );

  const setAll = useCallback(
    (value: boolean) => {
      setPanels((prev) => {
        const next = Object.fromEntries(Object.keys(prev).map((k) => [k, value]));
        try { localStorage.setItem(storageKey, JSON.stringify(next)); } catch { /* quota */ }
        return next;
      });
    },
    [storageKey],
  );

  const allOn = Object.values(panels).every(Boolean);

  const resetDefaults = useCallback(() => {
    setPanels({ ...defaults });
    try { localStorage.removeItem(storageKey); } catch { /* ignore */ }
  }, [storageKey, defaults]);

  return { panels, toggle, setAll, allOn, resetDefaults } as const;
}
