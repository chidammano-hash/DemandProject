import { TrendingUp, TrendingDown, Minus, HelpCircle, type LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

type KpiCardProps = {
  label: string;
  value: string;
  sublabel?: string;
  colorClass?: string;
  borderClass?: string;
  /** Override classes on the outer wrapper div */
  className?: string;
  /**
   * `delta` is ALWAYS displayed verbatim (its sign is the true movement).
   * Color: when `goodDirection` is given, the card is green when the delta moves
   * in the good direction and red when it moves the opposite way (U6.1 — decouples
   * the displayed sign from the good/bad color, so a lower-is-better WAPE can show
   * "-1.9pp" in green). When `goodDirection` is omitted, `direction` drives color
   * for backward compatibility. `unit` defaults to "%"; `period` labels the window.
   */
  trend?: { delta: number; direction: "up" | "down" | "flat"; goodDirection?: "up" | "down"; unit?: string; period?: string };
  sparkline?: number[];
  severity?: "best" | "warning" | "neutral";
  icon?: LucideIcon;
  /** Shows a HelpCircle icon next to the label; hover reveals tooltip content via title attribute */
  tooltip?: { title: string; description: string; threshold?: string };
  /** Shows a target sub-line below the main value */
  target?: { value: string; label?: string };
  /** Visual weight hierarchy: lg = hero KPI, md = default, sm = compact */
  size?: "lg" | "md" | "sm";
};

const SIZE_CONFIG = {
  lg: {
    wrapper: "px-5 py-4",
    value: "text-2xl font-bold",
    icon: "h-4 w-4",
  },
  md: {
    wrapper: "px-4 py-3",
    value: "text-xl font-bold",
    icon: "h-3.5 w-3.5",
  },
  sm: {
    wrapper: "px-3 py-2",
    value: "text-base font-semibold",
    icon: "h-3 w-3",
  },
} as const;

const SEVERITY_BORDER = {
  best: "border-l-4 border-l-[var(--kpi-best)]",
  warning: "border-l-4 border-l-[var(--kpi-warning)]",
  neutral: "",
} as const;

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

export function KpiCard({ label, value, sublabel, colorClass, borderClass, className, trend, sparkline, severity, icon: Icon, tooltip, target, size = "md" }: KpiCardProps) {
  const cfg = SIZE_CONFIG[size];

  // Color by *outcome* (good vs bad) when goodDirection is supplied; otherwise
  // fall back to raw direction. The icon always reflects the true delta sign.
  const trendIsGood = trend && trend.goodDirection
    ? trend.delta === 0
      ? null
      : (trend.delta > 0 ? "up" : "down") === trend.goodDirection
    : null;

  const trendColor = trend
    ? trend.goodDirection
      ? trendIsGood === null
        ? "text-muted-foreground"
        : trendIsGood
          ? "text-[var(--kpi-best)]"
          : "text-[var(--kpi-warning)]"
      : trend.direction === "up"
        ? "text-[var(--kpi-best)]"
        : trend.direction === "down"
          ? "text-[var(--kpi-warning)]"
          : "text-muted-foreground"
    : "";

  const TrendIcon = trend
    ? trend.direction === "up" ? TrendingUp : trend.direction === "down" ? TrendingDown : Minus
    : null;

  const severityBorder = size === "lg" && severity ? SEVERITY_BORDER[severity] : "";

  return (
    <div className={cn("rounded-lg border border-border/60 bg-card shadow-sm hover:shadow-md transition-shadow duration-200", cfg.wrapper, severityBorder, borderClass, className)}>
      <div className="flex items-center gap-1.5">
        {Icon && <Icon className={cn(cfg.icon, "text-muted-foreground")} strokeWidth={1.5} />}
        <p className="text-xs text-muted-foreground">
          {label}
          {sublabel && <span className="ml-1">{sublabel}</span>}
        </p>
        {tooltip && (
          <span
            title={[tooltip.title, tooltip.description, tooltip.threshold].filter(Boolean).join(" — ")}
            className="cursor-help"
          >
            <HelpCircle className="h-3 w-3 text-muted-foreground/60" strokeWidth={1.5} />
          </span>
        )}
      </div>
      <p className={cn(
        "tabular-nums font-mono tracking-tight transition-all duration-300",
        cfg.value,
        severity === "best" ? "text-[var(--kpi-best)]" : severity === "warning" ? "text-[var(--kpi-warning)]" : "",
        colorClass,
      )}>
        {value}
      </p>
      {target && (
        <p className="text-xs text-muted-foreground mt-0.5">{target.label ?? "Target"}: {target.value}</p>
      )}
      {trend && TrendIcon && trend.delta != null && !Number.isNaN(Number(trend.delta)) && (
        <div className={cn("flex items-center gap-1 text-xs", trendColor)}>
          <TrendIcon className="h-3 w-3" />
          <span className="tabular-nums">
            {trend.delta > 0 ? "+" : ""}{Number(trend.delta).toFixed(1)}{trend.unit ?? "%"} vs {trend.period ?? "prior"}
          </span>
        </div>
      )}
      {size !== "sm" && sparkline && sparkline.length > 1 && <Sparkline data={sparkline} />}
    </div>
  );
}
