/**
 * SbKpiCard — KPI card with optional severity bar for Storyboard tab.
 */
import { severityBg } from "./storyboardShared";

export function SbKpiCard({
  label,
  value,
  subtitle,
  color,
  severityBar,
}: {
  label: string;
  value: string | number;
  subtitle?: string;
  color?: "green" | "amber" | "red";
  severityBar?: number;
}) {
  const textColor =
    color === "green"
      ? "text-green-600 dark:text-green-400"
      : color === "amber"
      ? "text-amber-600 dark:text-amber-400"
      : color === "red"
      ? "text-red-600 dark:text-red-400"
      : "";

  return (
    <div className="rounded-lg border bg-card shadow-sm p-3.5">
      <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">{label}</p>
      <p className={`text-xl font-bold truncate mt-0.5 ${textColor}`}>{value}</p>
      {subtitle && (
        <p className="text-[10px] text-muted-foreground mt-0.5">{subtitle}</p>
      )}
      {severityBar != null && (
        <div className="w-full h-1 rounded-full bg-muted mt-2 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${severityBg(severityBar)}`}
            style={{ width: `${severityBar * 100}%` }}
          />
        </div>
      )}
    </div>
  );
}
