/**
 * PageHeader — the single sanctioned page-title block.
 *
 * Renders the eyebrow -> icon + title -> description stack every tab uses for
 * its top-of-page heading, plus a right-aligned (wraps on mobile) actions
 * slot. Replaces the ad-hoc `<h1 className="text-xl font-semibold ...">`
 * blocks hand-rolled per tab.
 */
import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

export interface PageHeaderProps {
  title: string;
  description?: string;
  icon?: LucideIcon;
  /** Small uppercase label above the title (e.g. "Customer intelligence workspace"). */
  eyebrow?: string;
  /** Right-aligned slot for buttons/filters — wraps under the title on narrow screens. */
  actions?: ReactNode;
  className?: string;
}

export function PageHeader({ title, description, icon: Icon, eyebrow, actions, className }: PageHeaderProps) {
  return (
    <div className={cn("flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between", className)}>
      <div className="min-w-0">
        {eyebrow && <p className="mb-1 text-2xs uppercase tracking-wider text-muted-foreground">{eyebrow}</p>}
        <div className="flex items-center gap-2">
          {Icon && <Icon className="h-5 w-5 shrink-0 text-primary" aria-hidden="true" />}
          <h1 className="text-xl font-semibold tracking-heading text-foreground">{title}</h1>
        </div>
        {description && <p className="mt-1 text-sm text-muted-foreground max-w-3xl">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2 sm:shrink-0">{actions}</div>}
    </div>
  );
}
