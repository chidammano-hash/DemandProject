/**
 * LazyPanel — defers rendering of expensive panels until they scroll into view.
 *
 * Why: tabs like CustomerAnalyticsTab fire 13+ useQuery calls on initial mount
 * because Suspense only lazy-loads the JS chunks; the wrapped useQuery fires
 * as soon as the chunk resolves. Wrapping below-the-fold panels here means the
 * underlying component (and its useQuery) is not mounted until the user
 * actually scrolls near the panel.
 *
 * Once the panel enters the viewport it stays mounted (triggerOnce semantics)
 * so subsequent filter changes do not re-trigger the lazy boundary.
 *
 * Test environments: jsdom does not implement IntersectionObserver.
 * `src/__tests__/setup.ts` installs a polyfill that fires `isIntersecting:
 * true` immediately, so panels render eagerly under vitest.
 */

import { useEffect, useRef, useState, type ReactNode } from "react";

interface LazyPanelProps {
  children: ReactNode;
  /** Shown while the panel is below the fold. */
  fallback?: ReactNode;
  /** Pre-render before fully visible — load slightly ahead of the user. */
  rootMargin?: string;
  threshold?: number | number[];
  /** Minimum height while not yet rendered (prevents layout shift). */
  minHeight?: number | string;
}

export function LazyPanel({
  children,
  fallback,
  rootMargin = "200px",
  threshold = 0,
  minHeight = 300,
}: LazyPanelProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [inView, setInView] = useState(false);

  useEffect(() => {
    if (inView) return; // already triggered, stop observing
    const node = ref.current;
    if (!node) return;

    // Defensive: some test environments may not provide IntersectionObserver.
    if (typeof IntersectionObserver === "undefined") {
      setInView(true);
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setInView(true);
            observer.disconnect();
            break;
          }
        }
      },
      { rootMargin, threshold },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [inView, rootMargin, threshold]);

  return (
    <div ref={ref} style={inView ? undefined : { minHeight }}>
      {inView ? children : fallback ?? null}
    </div>
  );
}

export default LazyPanel;
