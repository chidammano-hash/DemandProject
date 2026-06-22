/**
 * Command Center — KPI Summary Card.
 *
 * A single colored tile in the KPI Summary Bar (Portfolio Health, Open
 * Exceptions, Fill Rate, Order Value at Risk). Renders an icon, value, optional
 * badge/progress/caption, with a left border + accent color driven by `color`.
 */
import { cn } from "@/lib/utils";

export function KpiSummaryCard({
  icon: Icon,
  label,
  value,
  badge,
  color,
  progress,
  caption,
}: {
  icon: React.FC<{ className?: string }>;
  label: string;
  value: string;
  badge?: string;
  color?: "green" | "amber" | "red";
  progress?: number;
  caption?: string;
}) {
  // Left-border accent removed — semantic color is expressed via the icon chip
  // and value text. Kept as a no-op binding so callers that set `color` continue
  // to work without any prop changes.

  const textColor =
    color === "green"
      ? "text-green-600 dark:text-green-400"
      : color === "amber"
        ? "text-amber-600 dark:text-amber-400"
        : color === "red"
          ? "text-red-600 dark:text-red-400"
          : "";

  const iconBg =
    color === "green"
      ? "bg-green-100 dark:bg-green-900/30"
      : color === "amber"
        ? "bg-amber-100 dark:bg-amber-900/30"
        : color === "red"
          ? "bg-red-100 dark:bg-red-900/30"
          : "bg-muted";

  const progressBg =
    color === "green"
      ? "bg-green-500"
      : color === "amber"
        ? "bg-amber-500"
        : color === "red"
          ? "bg-red-500"
          : "bg-primary";

  return (
    <div
      className="rounded-xl border border-border/70 bg-card p-4 shadow-card transition-all duration-200 hover:shadow-card-hover hover:-translate-y-0.5"
    >
      <div className="flex items-center gap-2 mb-2">
        <div className={cn("rounded-lg p-2", iconBg)}>
          <Icon className={cn("h-3.5 w-3.5", textColor || "text-muted-foreground")} />
        </div>
        <p className="text-xs font-medium text-muted-foreground">{label}</p>
      </div>
      <p className={cn("text-2xl font-bold tabular-nums tracking-tight", textColor)}>{value}</p>
      {badge && (
        <span className="inline-flex items-center gap-1 mt-1 text-[10px] font-semibold text-red-600 dark:text-red-400">
          <span className="h-1.5 w-1.5 rounded-full bg-red-500 animate-pulse" />
          {badge}
        </span>
      )}
      {progress != null && (
        <div className="mt-2 h-1.5 w-full rounded-full bg-muted overflow-hidden">
          <div
            className={cn("h-full rounded-full transition-all duration-500", progressBg)}
            style={{ width: `${Math.min(progress * 100, 100)}%` }}
          />
        </div>
      )}
      {caption && (
        <p className="mt-1 text-[10px] leading-tight text-muted-foreground">{caption}</p>
      )}
    </div>
  );
}
