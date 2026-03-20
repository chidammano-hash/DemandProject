import { cn } from "@/lib/utils";

interface WidgetGridProps {
  cols?: 6 | 12;
  gap?: "sm" | "md" | "lg";
  children: React.ReactNode;
  className?: string;
}

const GAP_MAP = { sm: "gap-2", md: "gap-4", lg: "gap-6" };

export function WidgetGrid({ cols = 12, gap = "lg", children, className }: WidgetGridProps) {
  return (
    <div
      className={cn(
        "grid",
        cols === 12 ? "grid-cols-1 sm:grid-cols-6 lg:grid-cols-12" : "grid-cols-1 sm:grid-cols-3 lg:grid-cols-6",
        GAP_MAP[gap],
        className,
      )}
    >
      {children}
    </div>
  );
}

interface WidgetCardProps {
  span?: 1 | 2 | 3 | 4 | 6 | 12;
  title?: string;
  subtitle?: string;
  actions?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}

const SPAN_MAP: Record<number, string> = {
  1: "sm:col-span-1",
  2: "sm:col-span-2",
  3: "sm:col-span-3",
  4: "sm:col-span-4",
  6: "sm:col-span-6",
  12: "sm:col-span-6 lg:col-span-12",
};

export function WidgetCard({ span = 6, title, subtitle, actions, children, className }: WidgetCardProps) {
  return (
    <div
      className={cn(
        "col-span-full rounded-lg border border-border bg-card p-4",
        "shadow-sm hover:shadow-md transition-shadow duration-200",
        SPAN_MAP[span],
        className,
      )}
    >
      {(title || actions) && (
        <div className="pb-2 mb-2 border-b border-border/50 flex items-start justify-between">
          <div>
            {title && <h3 className="text-sm font-semibold tracking-tight text-card-foreground">{title}</h3>}
            {subtitle && <p className="text-xs text-muted-foreground">{subtitle}</p>}
          </div>
          {actions && <div className="flex items-center gap-1">{actions}</div>}
        </div>
      )}
      {children}
    </div>
  );
}
