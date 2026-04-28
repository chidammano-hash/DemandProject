/**
 * PanelStateGate — owns the loading / empty / data state machine that every
 * dashboard panel repeats. Pulls the inline ternary
 *
 *   {isLoading ? <Loading/> : isEmpty ? <EmptyState/> : <Chart/>}
 *
 * out of every panel and into one place. Adding a third state later (error,
 * stale, etc.) becomes a one-file change instead of 13.
 */
import type { ReactNode } from "react";
import { EmptyState, type EmptyStateVariant } from "./EmptyState";

interface PanelStateGateProps {
  /** Pending fetch — shows the loading skeleton. */
  isLoading: boolean;
  /** True when the fetch succeeded but yielded no rows. */
  isEmpty: boolean;
  /** Fixed pixel height (matches the chart's render height) so the loading
   *  skeleton doesn't shift layout when data arrives. */
  height: number;

  /** Optional override of the loading-state node; defaults to a centered "Loading…". */
  loading?: ReactNode;

  /** Empty-state variant; "filtered" is the right default for filter-driven panels. */
  emptyVariant?: EmptyStateVariant;
  /** Empty-state title. Defaults match the planner-facing copy used by CA panels. */
  emptyTitle?: string;
  /** Empty-state description. */
  emptyDescription?: string;

  /** The "happy path" content — rendered when not loading and not empty. */
  children: ReactNode;
}

const DEFAULT_LOADING_CLASS =
  "flex items-center justify-center text-sm text-muted-foreground";

export function PanelStateGate({
  isLoading,
  isEmpty,
  height,
  loading,
  emptyVariant = "filtered",
  emptyTitle = "No data for the selected filters",
  emptyDescription = "Try a different item or widen the date range",
  children,
}: PanelStateGateProps) {
  if (isLoading) {
    return (
      loading ?? (
        <div className={DEFAULT_LOADING_CLASS} style={{ height }}>
          Loading...
        </div>
      )
    );
  }
  if (isEmpty) {
    return (
      <EmptyState
        variant={emptyVariant}
        title={emptyTitle}
        description={emptyDescription}
      />
    );
  }
  return <>{children}</>;
}
