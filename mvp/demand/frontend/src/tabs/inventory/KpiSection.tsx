import { KpiCard } from "@/components/KpiCard";
import { LoadingElement } from "@/components/LoadingElement";
import { formatNumber, formatCompactNumber } from "@/lib/formatters";
import type { InventoryKpis } from "@/types";

interface KpiSectionProps {
  kpiData: InventoryKpis | undefined;
  isLoading: boolean;
}

export function KpiSection({ kpiData, isLoading }: KpiSectionProps) {
  if (isLoading) {
    return (
      <LoadingElement tabKey="inventory" message="Loading inventory KPIs..." />
    );
  }

  if (!kpiData) return null;

  return (
    <div className="flex flex-wrap gap-3">
      <KpiCard
        label="Total On-Hand"
        value={formatCompactNumber(kpiData.total_on_hand)}
      />
      <KpiCard
        label="Total On-Order"
        value={formatCompactNumber(kpiData.total_on_order)}
      />
      <KpiCard
        label="Avg Lead Time"
        value={
          kpiData.avg_lead_time_days != null
            ? `${formatNumber(kpiData.avg_lead_time_days)} days`
            : "-"
        }
      />
      <KpiCard
        label="Days of Supply"
        value={kpiData.dos != null ? formatNumber(kpiData.dos) : "-"}
        sublabel="days"
        severity={
          kpiData.dos != null
            ? kpiData.dos >= 14 && kpiData.dos <= 60
              ? "best"
              : kpiData.dos < 7 || kpiData.dos > 90
                ? "warning"
                : "neutral"
            : undefined
        }
      />
      <KpiCard
        label="Weeks of Cover"
        value={kpiData.woc != null ? formatNumber(kpiData.woc) : "-"}
        sublabel="weeks"
        severity={
          kpiData.woc != null
            ? kpiData.woc >= 2 && kpiData.woc <= 8
              ? "best"
              : kpiData.woc < 1 || kpiData.woc > 12
                ? "warning"
                : "neutral"
            : undefined
        }
      />
      <KpiCard
        label="Inventory Turns"
        value={
          kpiData.inventory_turns != null
            ? formatNumber(kpiData.inventory_turns)
            : "-"
        }
        sublabel="/yr"
        severity={
          kpiData.inventory_turns != null
            ? kpiData.inventory_turns > 8
              ? "best"
              : kpiData.inventory_turns < 4
                ? "warning"
                : "neutral"
            : undefined
        }
      />
      <KpiCard
        label="LT Coverage"
        value={
          kpiData.lt_coverage != null
            ? `${formatNumber(kpiData.lt_coverage)}x`
            : "-"
        }
        severity={
          kpiData.lt_coverage != null
            ? kpiData.lt_coverage > 1.5
              ? "best"
              : kpiData.lt_coverage < 1.0
                ? "warning"
                : "neutral"
            : undefined
        }
      />
    </div>
  );
}
