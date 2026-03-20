import { useQuery } from "@tanstack/react-query";
import {
  insightKeys,
  fetchActionFeed,
  STALE_INSIGHTS,
  type ActionFeedItem,
} from "@/api/queries";
import { KpiCard } from "@/components/KpiCard";
import { EmptyState } from "@/components/EmptyState";
import { formatCurrency, formatInt } from "@/lib/formatters";
import { getSeverityConfig } from "@/constants/severity";
import {
  AlertTriangle,
  Bell,
  ShieldAlert,
  PackageX,
  Truck,
  ListChecks,
} from "lucide-react";

const PANEL_KPI = "rounded-lg bg-muted/30 p-3";

const SOURCE_ICONS: Record<string, typeof AlertTriangle> = {
  exception: ShieldAlert,
  signal: Bell,
  po_risk: Truck,
  stockout: PackageX,
};

function sourceIcon(source: string) {
  return SOURCE_ICONS[source.toLowerCase()] ?? AlertTriangle;
}

function severityCardClass(severity: string): string {
  const sev = severity.toLowerCase();
  if (sev === "critical") return "border-l-4 border-l-red-500 bg-red-50 dark:bg-red-950/20";
  if (sev === "high") return "border-l-4 border-l-orange-500 bg-orange-50 dark:bg-orange-950/20";
  return "border-l-4 border-l-blue-500 bg-blue-50 dark:bg-blue-950/20";
}

export function ActionFeedPanel() {
  const { data, isLoading, error } = useQuery({
    queryKey: insightKeys.actionFeed(),
    queryFn: fetchActionFeed,
    staleTime: STALE_INSIGHTS.ONE_MIN,
    refetchInterval: 60_000,
  });

  if (error) {
    return (
      <div className="text-xs text-red-600 p-4">
        Failed to load action feed: {(error as Error).message}
      </div>
    );
  }

  const summary = data?.summary;
  const actions = data?.actions ?? [];

  return (
    <div className="space-y-4">
      <div className="text-xs text-muted-foreground bg-muted/20 border rounded px-3 py-2">
        Unified action feed aggregating exceptions, demand signals, PO risks, and stockout alerts across all inventory planning modules.
        Items are ranked by severity and financial impact. Auto-refreshes every 60 seconds.
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KpiCard
          className={PANEL_KPI}
          label="Total Actions"
          value={isLoading ? "..." : formatInt(summary?.total)}
        />
        <KpiCard
          className={PANEL_KPI}
          label="Critical"
          value={isLoading ? "..." : formatInt(summary?.critical)}
          colorClass={(summary?.critical ?? 0) > 0 ? "text-red-600" : undefined}
        />
        <KpiCard
          className={PANEL_KPI}
          label="High Priority"
          value={isLoading ? "..." : formatInt(summary?.high)}
          colorClass={(summary?.high ?? 0) > 0 ? "text-orange-600" : undefined}
        />
        <KpiCard
          className={PANEL_KPI}
          label="Financial Impact at Risk"
          value={isLoading ? "..." : formatCurrency(summary?.financial_at_risk)}
        />
      </div>

      {isLoading ? (
        <p className="text-xs text-muted-foreground">Loading action feed...</p>
      ) : actions.length === 0 ? (
        <EmptyState
          icon={ListChecks}
          title="No pending actions"
          description="The action feed aggregates exceptions, demand signals, PO risks, and stockout alerts. When issues are detected they will appear here ranked by priority."
        />
      ) : (
        <div className="space-y-2">
          {actions.map((action: ActionFeedItem) => {
            const Icon = sourceIcon(action.source);
            const sevCfg = getSeverityConfig(action.severity);
            return (
              <div
                key={action.id}
                className={`rounded-lg p-3 ${severityCardClass(action.severity)}`}
              >
                <div className="flex items-start gap-3">
                  <div className="mt-0.5">
                    <Icon className={`h-4 w-4 ${sevCfg.icon}`} strokeWidth={1.5} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className={`text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded ${sevCfg.badge}`}>
                        {action.source}
                      </span>
                      <span className={`text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded ${sevCfg.badge}`}>
                        {sevCfg.label}
                      </span>
                      <span className="text-xs font-mono text-muted-foreground">
                        {action.item_no} @ {action.loc}
                      </span>
                    </div>
                    <p className="text-sm font-medium text-foreground">{action.title}</p>
                    <p className="text-xs text-muted-foreground mt-0.5">{action.detail}</p>
                  </div>
                  <div className="text-right shrink-0">
                    {action.financial_impact != null && (
                      <p className="text-sm font-semibold tabular-nums">
                        {formatCurrency(action.financial_impact)}
                      </p>
                    )}
                    <p className="text-[10px] text-muted-foreground mt-0.5">
                      {new Date(action.created_at).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
