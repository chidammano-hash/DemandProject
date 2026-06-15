/**
 * Shared class string for the metric/grain/group-by toggle pills used across the
 * Customer-Analytics chart panels (U3.1). The inactive state uses theme tokens
 * (`bg-muted text-muted-foreground`) so the pills stay legible in Light / Soft /
 * Dark — the previous bare `bg-gray-100 text-gray-600` rendered gray-on-gray in
 * Dark theme. The active state uses the primary accent for a non-color-only cue
 * (pair with `aria-pressed` set by each call site for screen readers, U3.3).
 */
export function togglePillClass(active: boolean): string {
  return active
    ? "px-2 py-0.5 text-xs rounded font-medium bg-primary text-primary-foreground"
    : "px-2 py-0.5 text-xs rounded bg-muted text-muted-foreground hover:bg-accent";
}
