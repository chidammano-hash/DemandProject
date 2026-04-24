/**
 * EmptyState — reusable panel placeholder for the "empty-state triad":
 *   - "no-data": no rows exist yet; show CLI steps to populate.
 *   - "filtered": rows exist but current filter yields none; offer a reset.
 *   - "error": fetch failed; show sanitized message + optional retry.
 *
 * Gen-4 UX P0: tabs should distinguish "never populated" from "filtered out"
 * from "server failure" instead of a single undifferentiated "No data".
 */
import { type LucideIcon, AlertTriangle, Database, FilterX } from "lucide-react";
import { cn } from "@/lib/utils";

export type EmptyStateVariant = "no-data" | "filtered" | "error";

export interface EmptyStateStep {
  /** Short label describing the step (e.g. "Compute safety stock") */
  label: string;
  /** Shell command to run (e.g. "make ss-compute") */
  command: string;
}

interface EmptyStateProps {
  /** Which of the empty-state triad this is. Defaults to "no-data". */
  variant?: EmptyStateVariant;
  /** Lucide icon override — variant picks a sensible default */
  icon?: LucideIcon;
  /** Primary headline (e.g. "No safety stock targets yet") */
  title: string;
  /** One-sentence explanation of what this panel shows when populated */
  description: string;
  /** Ordered list of make/CLI commands to run (no-data variant) */
  steps?: EmptyStateStep[];
  /** Optional action callback (e.g. reset filters / retry). Renders a button if provided. */
  onAction?: () => void;
  /** Label for the action button. */
  actionLabel?: string;
  /** Extra Tailwind classes on the outer container */
  className?: string;
}

const DEFAULT_ICON: Record<EmptyStateVariant, LucideIcon> = {
  "no-data": Database,
  filtered: FilterX,
  error: AlertTriangle,
};

const VARIANT_CLASSES: Record<EmptyStateVariant, string> = {
  "no-data": "border-dashed bg-muted/20 text-muted-foreground",
  filtered: "border-dashed bg-muted/10 text-muted-foreground",
  error: "border-destructive/40 bg-destructive/5 text-destructive",
};

const BADGE_CLASSES: Record<EmptyStateVariant, string> = {
  "no-data": "bg-muted/50 text-muted-foreground",
  filtered: "bg-muted/60 text-muted-foreground",
  error: "bg-destructive/10 text-destructive",
};

export function EmptyState({
  variant = "no-data",
  icon,
  title,
  description,
  steps,
  onAction,
  actionLabel,
  className,
}: EmptyStateProps) {
  const Icon = icon ?? DEFAULT_ICON[variant];
  return (
    <div
      role={variant === "error" ? "alert" : "status"}
      aria-live={variant === "error" ? "assertive" : "polite"}
      className={cn(
        "flex flex-col items-center justify-center rounded-xl border px-8 py-12 text-center",
        VARIANT_CLASSES[variant],
        className,
      )}
    >
      <div
        className={cn(
          "mb-4 flex h-14 w-14 items-center justify-center rounded-full",
          BADGE_CLASSES[variant],
        )}
      >
        <Icon size={26} strokeWidth={1.5} />
      </div>

      <h3 className="mb-1 text-sm font-semibold text-foreground">{title}</h3>
      <p className="mb-5 max-w-xs text-xs leading-relaxed">{description}</p>

      {onAction && actionLabel && (
        <button
          type="button"
          onClick={onAction}
          className={cn(
            "mb-2 rounded-md border px-3 py-1.5 text-xs font-medium transition-colors",
            variant === "error"
              ? "border-destructive/40 bg-destructive/10 text-destructive hover:bg-destructive/20"
              : "border-border bg-card text-foreground hover:bg-muted",
          )}
        >
          {actionLabel}
        </button>
      )}

      {variant === "no-data" && steps && steps.length > 0 && (
        <div className="w-full max-w-sm rounded-lg border bg-card p-4 text-left">
          <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            How to populate
          </p>
          <ol className="space-y-2.5">
            {steps.map((step, i) => (
              <li key={i} className="flex flex-col gap-0.5">
                <span className="text-xs text-muted-foreground">
                  <span className="mr-1.5 inline-flex h-4 w-4 items-center justify-center rounded-full bg-muted text-[10px] font-bold">
                    {i + 1}
                  </span>
                  {step.label}
                </span>
                <code className="ml-5 block rounded bg-muted px-2 py-1 font-mono text-[11px] text-foreground">
                  {step.command}
                </code>
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}
