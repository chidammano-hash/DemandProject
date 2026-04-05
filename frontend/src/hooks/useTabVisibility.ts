import { useEffect, useState } from "react";

/**
 * Tracks browser-tab visibility via the Page Visibility API.
 *
 * Returns `true` when the tab is in the foreground, `false` when hidden.
 * Typical usage: throttle background polling intervals so hidden tabs
 * don't hammer the API at the same rate as foreground tabs.
 *
 * @example
 * const isVisible = useTabVisibility();
 * const refetchInterval = isVisible ? 120_000 : 600_000;
 */
export function useTabVisibility(): boolean {
  const [isVisible, setIsVisible] = useState(() => !document.hidden);

  useEffect(() => {
    const handler = () => setIsVisible(!document.hidden);
    document.addEventListener("visibilitychange", handler);
    return () => document.removeEventListener("visibilitychange", handler);
  }, []);

  return isVisible;
}
