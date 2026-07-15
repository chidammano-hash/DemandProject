/**
 * Loading contract: `Skeleton` / `TableSkeleton` are for content placeholders
 * — render them in place of the real content while a query is in flight, sized
 * to roughly match what will replace them. `LoadingElement` is reserved for
 * tab-level Suspense fallbacks (a whole panel/route boundary), not individual
 * fields or rows.
 *
 * `.animate-shimmer` (index.css) is the single animation mechanism — a
 * `background-position` sweep over `bg-muted` that's already listed in the
 * `prefers-reduced-motion` guard. It previously ran ALONGSIDE a second
 * `::before` pseudo-element sweep (a `translateX` highlight bar driven by
 * tailwind.config's `shimmer` keyframe); that second mechanism was dropped
 * both for the double-animation and because its arbitrary-value class name
 * didn't match the reduced-motion selector list, so it never actually
 * stopped animating for users who asked for reduced motion.
 */
import { cn } from "@/lib/utils";

export function Skeleton({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("rounded-md bg-muted animate-shimmer", className)}
      {...props}
    />
  );
}

export function TableSkeleton({ rows = 8, cols = 6 }: { rows?: number; cols?: number }) {
  return (
    <div className="space-y-2">
      {/* Header skeleton */}
      <div className="flex gap-2">
        {Array.from({ length: cols }).map((_, i) => (
          <Skeleton key={`h-${i}`} className="h-8 flex-1" />
        ))}
      </div>
      {/* Row skeletons */}
      {Array.from({ length: rows }).map((_, r) => (
        <div key={`r-${r}`} className="flex gap-2">
          {Array.from({ length: cols }).map((_, c) => (
            <Skeleton key={`${r}-${c}`} className="h-6 flex-1" />
          ))}
        </div>
      ))}
    </div>
  );
}
