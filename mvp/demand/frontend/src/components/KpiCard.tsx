import { TrendingUp, TrendingDown, Minus, type LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

type KpiCardProps = {
  label: string;
  value: string;
  sublabel?: string;
  colorClass?: string;
  borderClass?: string;
  /** Override classes on the outer wrapper div */
  className?: string;
  /** direction drives color; unit defaults to "%" if omitted; period labels the comparison window */
  trend?: { delta: number; direction: "up" | "down" | "flat"; unit?: string; period?: string };
  sparkline?: number[];
  severity?: "best" | "warning" | "neutral";
  icon?: LucideIcon;
};

function Sparkline({ data }: { data: number[] }) {
  if (data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const w = 80;
  const h = 20;
  const points = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * w;
      const y = h - ((v - min) / range) * h;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} className="mt-1 text-primary/30">
      <polyline
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        points={points}
      />
    </svg>
  );
}

export function KpiCard({ label, value, sublabel, colorClass, borderClass, className, trend, sparkline, severity, icon: Icon }: KpiCardProps) {
  const trendColor = trend
    ? trend.direction === "up"
      ? "text-[var(--kpi-best)]"
      : trend.direction === "down"
        ? "text-[var(--kpi-warning)]"
        : "text-muted-foreground"
    : "";

  const TrendIcon = trend
    ? trend.direction === "up" ? TrendingUp : trend.direction === "down" ? TrendingDown : Minus
    : null;

  return (
    <div className={cn("rounded-lg border border-border/60 bg-card px-4 py-3 shadow-sm hover:shadow-md transition-shadow duration-200", borderClass, className)}>
      <div className="flex items-center gap-1.5">
        {Icon && <Icon className="h-3.5 w-3.5 text-muted-foreground" strokeWidth={1.5} />}
        <p className="text-xs text-muted-foreground">
          {label}
          {sublabel && <span className="ml-1">{sublabel}</span>}
        </p>
      </div>
      <p className={cn(
        "text-xl font-bold tabular-nums font-mono tracking-tight",
        severity === "best" ? "text-[var(--kpi-best)]" : severity === "warning" ? "text-[var(--kpi-warning)]" : "",
        colorClass,
      )}>
        {value}
      </p>
      {trend && TrendIcon && (
        <div className={cn("flex items-center gap-1 text-xs", trendColor)}>
          <TrendIcon className="h-3 w-3" />
          <span className="tabular-nums">
            {trend.delta > 0 ? "+" : ""}{trend.delta.toFixed(1)}{trend.unit ?? "%"} vs {trend.period ?? "prior"}
          </span>
        </div>
      )}
      {sparkline && sparkline.length > 1 && <Sparkline data={sparkline} />}
    </div>
  );
}
