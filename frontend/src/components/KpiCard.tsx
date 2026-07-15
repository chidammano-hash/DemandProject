import { type KeyboardEvent } from "react";
import { TrendingUp, TrendingDown, Minus, HelpCircle, type LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Severities that drive an accent color; "neutral" (or omitted) renders no
 * accent. "warning" is the legacy red bad-KPI tone (--kpi-warning); "caution"
 * is the amber mid-band (--warning); "critical" is destructive red.
 */
type KpiAccentSeverity = "best" | "warning" | "caution" | "critical";

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
  severity?: KpiAccentSeverity | "neutral";
  icon?: LucideIcon;
  /** Renders `icon` inside a tinted rounded chip (colored by `severity`) instead of a bare glyph. */
  iconChip?: boolean;
  /** Shows a HelpCircle icon next to the label; hover reveals tooltip content via title attribute */
  tooltip?: { title: string; description: string; threshold?: string };
  /** Shows a target sub-line below the main value */
  target?: { value: string; label?: string };
  /** Visual weight hierarchy: lg = hero KPI, md = default, sm = compact */
  size?: "lg" | "md" | "sm";
  /**
   * Small alarm callout under the value (e.g. "3 critical") with a pulsing dot.
   * Always destructive-toned — it flags a count needing attention regardless
   * of `severity`.
   */
  badge?: string;
  /** 0..1 fraction — renders a thin progress bar under the value, colored by `severity`. */
  progress?: number;
  /** Small muted caption line rendered after everything else. */
  caption?: string;
  /** Makes the card keyboard- and click-interactive (role="button" + Enter/Space activation). */
  onClick?: () => void;
};

const SIZE_CONFIG = {
  lg: {
    wrapper: "px-5 py-4",
    value: "text-2xl font-bold",
    icon: "h-4 w-4",
    chip: "p-2",
  },
  md: {
    wrapper: "px-4 py-3",
    value: "text-xl font-bold",
    icon: "h-3.5 w-3.5",
    chip: "p-1.5",
  },
  sm: {
    wrapper: "px-3 py-2",
    value: "text-base font-semibold",
    icon: "h-3 w-3",
    chip: "p-1",
  },
} as const;

const SEVERITY_ACCENT_BG: Record<KpiAccentSeverity, string> = {
  best: "bg-kpi-best",
  warning: "bg-kpi-warning",
  caution: "bg-warning",
  critical: "bg-destructive",
};

const SEVERITY_TEXT: Record<KpiAccentSeverity, string> = {
  best: "text-kpi-best",
  warning: "text-kpi-warning",
  caution: "text-warning",
  critical: "text-destructive",
};

const SEVERITY_ICON_CHIP: Record<KpiAccentSeverity, string> = {
  best: "bg-kpi-best/10 text-kpi-best",
  warning: "bg-kpi-warning/10 text-kpi-warning",
  caution: "bg-warning/10 text-warning",
  critical: "bg-destructive/10 text-destructive",
};

function isAccentSeverity(severity: KpiCardProps["severity"]): severity is KpiAccentSeverity {
  return severity === "best" || severity === "warning" || severity === "caution" || severity === "critical";
}

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

export function KpiCard({
  label,
  value,
  sublabel,
  colorClass,
  borderClass,
  className,
  trend,
  sparkline,
  severity,
  icon: Icon,
  iconChip,
  tooltip,
  target,
  size = "md",
  badge,
  progress,
  caption,
  onClick,
}: KpiCardProps) {
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
          ? "text-kpi-best"
          : "text-kpi-warning"
      : trend.direction === "up"
        ? "text-kpi-best"
        : trend.direction === "down"
          ? "text-kpi-warning"
          : "text-muted-foreground"
    : "";

  const TrendIcon = trend
    ? trend.direction === "up" ? TrendingUp : trend.direction === "down" ? TrendingDown : Minus
    : null;

  const hasAccent = isAccentSeverity(severity);
  const showAccent = size === "lg" && hasAccent;
  const progressWidth = progress == null ? null : Math.min(Math.max(progress, 0), 1) * 100;

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!onClick) return;
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onClick();
    }
  };

  return (
    <div
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : undefined}
      onClick={onClick}
      onKeyDown={onClick ? handleKeyDown : undefined}
      className={cn(
        "relative overflow-hidden rounded-xl border border-border/60 bg-card shadow-card hover:shadow-card-hover hover:-translate-y-0.5 transition-all duration-200",
        onClick && "cursor-pointer text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
        cfg.wrapper,
        showAccent && "pl-4",
        borderClass,
        className,
      )}
    >
      {showAccent && (
        <span
          aria-hidden="true"
          className={cn(
            "absolute left-0 top-1/2 h-[55%] w-1 -translate-y-1/2 rounded-r-full",
            SEVERITY_ACCENT_BG[severity as KpiAccentSeverity],
          )}
        />
      )}
      <div className="flex items-center gap-1.5">
        {Icon && (
          iconChip ? (
            <span
              className={cn(
                "rounded-lg",
                cfg.chip,
                hasAccent ? SEVERITY_ICON_CHIP[severity as KpiAccentSeverity] : "bg-muted text-muted-foreground",
              )}
            >
              <Icon className={cfg.icon} strokeWidth={1.5} />
            </span>
          ) : (
            <Icon className={cn(cfg.icon, "text-muted-foreground")} strokeWidth={1.5} />
          )
        )}
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
        "tabular-nums tracking-kpi transition-all duration-300",
        cfg.value,
        hasAccent ? SEVERITY_TEXT[severity as KpiAccentSeverity] : "",
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
      {badge && (
        <div className="mt-1 flex items-center gap-1 text-2xs font-semibold text-destructive">
          <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-destructive animate-pulse" />
          {badge}
        </div>
      )}
      {progressWidth != null && (
        <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-muted">
          <div
            className={cn(
              "h-full rounded-full transition-all duration-500",
              hasAccent ? SEVERITY_ACCENT_BG[severity as KpiAccentSeverity] : "bg-primary",
            )}
            style={{ width: `${progressWidth}%` }}
          />
        </div>
      )}
      {caption && <p className="mt-1 text-2xs leading-tight text-muted-foreground">{caption}</p>}
      {size !== "sm" && sparkline && sparkline.length > 1 && <Sparkline data={sparkline} />}
    </div>
  );
}
