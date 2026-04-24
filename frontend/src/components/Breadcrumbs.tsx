import { ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * UX-1 — breadcrumbs for deep drill states.
 *
 * Renders a compact trail of segments: `Tab > Item > LOC`, or
 * `Tab > Scenario`, or `Tab > Run ID`. Intermediate segments can be
 * clickable (via `onClick`) to step back to that level; the final segment
 * is always rendered as a terminal (non-clickable, higher contrast).
 *
 * Intentionally minimal: 50 LOC, no routing dependency. Callers drive
 * navigation through the existing tab-switching API.
 */

export interface BreadcrumbSegment {
  label: string;
  /** Optional click handler; when omitted, segment is non-interactive. */
  onClick?: () => void;
}

export interface BreadcrumbsProps {
  items: BreadcrumbSegment[];
  className?: string;
}

export function Breadcrumbs({ items, className }: BreadcrumbsProps) {
  if (items.length === 0) return null;

  return (
    <nav
      aria-label="Breadcrumb"
      className={cn("flex items-center text-xs text-muted-foreground", className)}
    >
      <ol className="flex flex-wrap items-center gap-1">
        {items.map((segment, idx) => {
          const isLast = idx === items.length - 1;
          const key = `${segment.label}-${idx}`;
          return (
            <li key={key} className="flex items-center gap-1">
              {idx > 0 && (
                <ChevronRight
                  className="h-3 w-3 text-muted-foreground/60"
                  aria-hidden="true"
                />
              )}
              {isLast ? (
                <span
                  aria-current="page"
                  className="font-medium text-foreground"
                >
                  {segment.label}
                </span>
              ) : segment.onClick ? (
                <button
                  type="button"
                  onClick={segment.onClick}
                  className="rounded text-muted-foreground transition-colors hover:text-foreground hover:underline"
                >
                  {segment.label}
                </button>
              ) : (
                <span className="text-muted-foreground">{segment.label}</span>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
