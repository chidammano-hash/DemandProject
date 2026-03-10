import { KpiCard } from "@/components/KpiCard";
import { LoadingElement } from "@/components/LoadingElement";
import { formatNumber, formatCompactNumber } from "@/lib/formatters";
import type { InventoryKpis } from "@/types";

interface KpiSectionProps {
  kpiData: InventoryKpis | undefined;
  isLoading: boolean;
}

function DosDelta({ delta }: { delta: number | null | undefined }) {
  if (delta == null) return null;
  const abs = Math.abs(delta);
  if (abs < 2) {
    return (
      <p className="text-xs text-muted-foreground mt-0.5">→ stable vs prev month</p>
    );
  }
  if (delta > 0) {
    return (
      <p className="text-xs text-[var(--kpi-best)] mt-0.5">
        ↑ +{delta}d vs prev month
      </p>
    );
  }
  return (
    <p className="text-xs text-[var(--kpi-warning)] mt-0.5">
      ↓ {delta}d vs prev month
    </p>
  );
}

export function KpiSection({ kpiData, isLoading }: KpiSectionProps) {
  if (isLoading) {
    return (
      <LoadingElement tabKey="inventory" message="Loading inventory KPIs..." />
    );
  }

  if (!kpiData) return null;

  return (
    <div>
      <p className="text-xs text-muted-foreground mb-2">
        KPIs reflect current inventory position (latest snapshot) vs current month sales rate.
      </p>
      <div className="flex flex-wrap gap-3">
        <KpiCard
          label="Total On-Hand"
          value={formatCompactNumber(kpiData.total_on_hand)}
          tooltip={{
            title: "Total On-Hand Inventory",
            description: "Formula: Σ qty_on_hand across all item-locations at the latest snapshot date.",
            threshold: "Higher = more buffer stock but more capital tied up",
          }}
        />
        <KpiCard
          label="Total On-Order"
          value={formatCompactNumber(kpiData.total_on_order)}
          tooltip={{
            title: "On-Order Inventory",
            description: "Formula: Σ qty_on_order = qty_on_hand_on_order − qty_on_hand. Represents open purchase orders not yet received.",
            threshold: "High on-order with low on-hand may indicate supply delays",
          }}
        />
        <KpiCard
          label="Avg Lead Time"
          value={
            kpiData.avg_lead_time_days != null
              ? `${formatNumber(kpiData.avg_lead_time_days)} days`
              : "-"
          }
          tooltip={{
            title: "Weighted Average Lead Time",
            description: "Formula: Σ(lead_time_days × monthly_sales) ÷ Σ monthly_sales. Sales-weighted so high-velocity items drive the average.",
            threshold: "Used to compute ROP threshold (Lead Time × 1.5) and LT Coverage",
          }}
        />
        <div>
          <KpiCard
            label="Days of Supply"
            value={kpiData.dos != null ? formatNumber(kpiData.dos) : "-"}
            severity={
              kpiData.dos != null
                ? kpiData.dos >= 14 && kpiData.dos <= 60
                  ? "best"
                  : kpiData.dos < 7 || kpiData.dos > 90
                    ? "warning"
                    : "neutral"
                : undefined
            }
            tooltip={{
              title: "Days of Supply (DOS)",
              description: "Formula: On-Hand ÷ (Current Month Sales ÷ 30.44). Uses this month's sales rate — not a trailing average — so it responds immediately to demand changes.",
              threshold: "Healthy: 14–60 days. Action needed: < 7 days (stockout risk) or > 90 days (excess)",
            }}
          />
          <DosDelta delta={kpiData.dos_delta} />
        </div>
        <KpiCard
          label="Weeks of Cover"
          value={kpiData.woc != null ? formatNumber(kpiData.woc) : "-"}
          severity={
            kpiData.woc != null
              ? kpiData.woc >= 2 && kpiData.woc <= 8
                ? "best"
                : kpiData.woc < 1 || kpiData.woc > 12
                  ? "warning"
                  : "neutral"
            : undefined
          }
          tooltip={{
            title: "Weeks of Cover (WOC)",
            description: "Formula: DOS ÷ 7. Converts Days of Supply into weeks for planning horizons that think in week buckets.",
            threshold: "Healthy: 2–8 weeks. Action needed: < 1 week or > 12 weeks",
          }}
        />
        <KpiCard
          label="Inventory Turns (/year)"
          value={
            kpiData.inventory_turns != null
              ? formatNumber(kpiData.inventory_turns)
              : "-"
          }
          severity={
            kpiData.inventory_turns != null
              ? kpiData.inventory_turns > 8
                ? "best"
                : kpiData.inventory_turns < 4
                  ? "warning"
                  : "neutral"
            : undefined
          }
          tooltip={{
            title: "Inventory Turns / Year",
            description: "Formula: (Current Month Sales × 12) ÷ Current Avg On-Hand. Uses current month data so a rapid inventory drawdown shows up immediately rather than being smoothed by a 12-month average.",
            threshold: "Healthy: > 8 turns/yr. Low: < 4 turns/yr (slow-moving stock)",
          }}
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
          tooltip={{
            title: "Lead Time Coverage",
            description: "Formula: (On-Hand + On-Order) ÷ (Avg Lead Time × Daily Sales Rate). Answers: if a new PO is placed today, do we have enough stock to last until it arrives? Includes on-order because those units will arrive during the lead time window.",
            threshold: "Healthy: > 1.5x (buffer). At risk: < 1.0x (stockout before next order arrives)",
          }}
        />
      </div>
    </div>
  );
}
