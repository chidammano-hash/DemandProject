import { AlertTriangle, AlertCircle, Info } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Alert, AlertSeverity } from "@/types/theme";

const SEVERITY_CONFIG: Record<AlertSeverity, { icon: React.ElementType; borderClass: string; iconClass: string }> = {
  critical: { icon: AlertTriangle, borderClass: "border-l-destructive", iconClass: "text-destructive" },
  high: { icon: AlertCircle, borderClass: "border-l-[var(--kpi-warning)]", iconClass: "text-[var(--kpi-warning)]" },
  medium: { icon: AlertCircle, borderClass: "border-l-accent", iconClass: "text-accent-foreground" },
  low: { icon: Info, borderClass: "border-l-muted", iconClass: "text-muted-foreground" },
};

interface AlertPanelProps {
  alerts: Alert[];
  className?: string;
}

export function AlertPanel({ alerts, className }: AlertPanelProps) {
  if (alerts.length === 0) {
    return (
      <div className={cn("flex items-center justify-center py-8 text-sm text-muted-foreground", className)}>
        No active alerts
      </div>
    );
  }

  // Sort by severity: critical > high > medium > low
  const severityOrder: AlertSeverity[] = ["critical", "high", "medium", "low"];
  const sorted = [...alerts].sort((a, b) => severityOrder.indexOf(a.severity) - severityOrder.indexOf(b.severity));

  return (
    <div className={cn("space-y-1.5", className)}>
      {sorted.map((alert) => {
        const cfg = SEVERITY_CONFIG[alert.severity];
        const Icon = cfg.icon;
        return (
          <div
            key={alert.id}
            className={cn(
              "flex items-start gap-2.5 rounded-md border border-border border-l-4 bg-card px-3 py-2",
              cfg.borderClass,
            )}
          >
            <Icon className={cn("mt-0.5 h-4 w-4 flex-shrink-0", cfg.iconClass)} strokeWidth={1.5} />
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium text-card-foreground">{alert.title}</p>
              <p className="text-xs text-muted-foreground">{alert.detail}</p>
            </div>
            {alert.count != null && (
              <span className="flex-shrink-0 rounded-full bg-muted px-2 py-0.5 text-[10px] font-medium tabular-nums text-muted-foreground">
                {alert.count}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}
