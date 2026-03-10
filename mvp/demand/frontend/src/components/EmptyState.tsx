/**
 * EmptyState — reusable panel placeholder shown when no data is available.
 *
 * Displays a title, description, and optional step-by-step CLI instructions
 * so planners immediately know what to run to populate the panel.
 */
import { type LucideIcon, Database } from "lucide-react";
import { cn } from "@/lib/utils";

export interface EmptyStateStep {
  /** Short label describing the step (e.g. "Compute safety stock") */
  label: string;
  /** Shell command to run (e.g. "make ss-compute") */
  command: string;
}

interface EmptyStateProps {
  /** Lucide icon component — defaults to Database */
  icon?: LucideIcon;
  /** Primary headline (e.g. "No safety stock targets yet") */
  title: string;
  /** One-sentence explanation of what this panel shows when populated */
  description: string;
  /** Ordered list of make/CLI commands to run */
  steps?: EmptyStateStep[];
  /** Extra Tailwind classes on the outer container */
  className?: string;
}

export function EmptyState({
  icon: Icon = Database,
  title,
  description,
  steps,
  className,
}: EmptyStateProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center rounded-xl border border-dashed bg-muted/20 px-8 py-12 text-center",
        className,
      )}
    >
      <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-muted/50 text-muted-foreground">
        <Icon size={26} strokeWidth={1.5} />
      </div>

      <h3 className="mb-1 text-sm font-semibold text-foreground">{title}</h3>
      <p className="mb-5 max-w-xs text-xs text-muted-foreground leading-relaxed">{description}</p>

      {steps && steps.length > 0 && (
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
