/**
 * IPfeature4 + IPfeature5 + IPfeature6 + IPfeature7 + IPfeature8–IPfeature14 + F1.1
 * F3.1–F3.5 (Bias, Service Level, Lead Time, Blended, Echelon) + F4.1–F4.4
 * EOQ & Cycle Stock + Replenishment Policy + Health Score + Exception Queue +
 * Fill Rate + ABC-XYZ + Supplier + Intramonth + Safety Stock + Variability +
 * Lead Time + Demand Signals + Simulation + Investment Plan + Production Forecast +
 * Blended Demand + Echelon Planning + Financial Plan + Events + Scenarios
 *
 * PL-009: Sub-navigation added — shows one panel at a time to reduce scroll
 * and initial API call overhead.
 */

import { useState } from "react";
import { cn } from "@/lib/utils";
import { ExceptionQueuePanel } from "./inv-planning/ExceptionQueuePanel";
import { PortfolioHealthPanel } from "./inv-planning/PortfolioHealthPanel";
import { EoqPanel } from "./inv-planning/EoqPanel";
import { PolicyManagementPanel } from "./inv-planning/PolicyManagementPanel";
import { FillRatePanel } from "./inv-planning/FillRatePanel";
import { AbcXyzPanel } from "./inv-planning/AbcXyzPanel";
import { SupplierPanel } from "./inv-planning/SupplierPanel";
import { IntramonthPanel } from "./inv-planning/IntramonthPanel";
import { SafetyStockPanel } from "./inv-planning/SafetyStockPanel";
import { VariabilityPanel } from "./inv-planning/VariabilityPanel";
import { LeadTimePanel } from "./inv-planning/LeadTimePanel";
import { DemandSignalsPanel } from "./inv-planning/DemandSignalsPanel";
import { SimulationPanel } from "./inv-planning/SimulationPanel";
import { InvestmentPanel } from "./inv-planning/InvestmentPanel";
import { DemandForecastPanel } from "./inv-planning/DemandForecastPanel";
import { DemandPlanPanel } from "./inv-planning/DemandPlanPanel";
import { OverrideQueuePanel } from "./inv-planning/OverrideQueuePanel";
import { ProcurementPanel } from "./inv-planning/ProcurementPanel";
import { OpenPOPanel } from "./inv-planning/OpenPOPanel";
import { ProjectionPanel } from "./inv-planning/ProjectionPanel";
import { PlannedOrdersPanel } from "./inv-planning/PlannedOrdersPanel";
import { BlendedDemandPanel } from "./inv-planning/BlendedDemandPanel";
import { EchelonPanel } from "./inv-planning/EchelonPanel";
import { FinancialPlanPanel } from "./inv-planning/FinancialPlanPanel";
import { EventCalendarPanel } from "./inv-planning/EventCalendarPanel";
import { ScenarioPlanningPanel } from "./inv-planning/ScenarioPlanningPanel";

// ---------------------------------------------------------------------------
// Sub-navigation config
// ---------------------------------------------------------------------------
const SUB_TABS = [
  { key: "exceptions",  label: "Exceptions",   group: "Daily" },
  { key: "health",      label: "Health",        group: "Daily" },
  { key: "eoq",         label: "EOQ",           group: "Optimize" },
  { key: "policy",      label: "Policy",        group: "Optimize" },
  { key: "fillrate",    label: "Fill Rate",     group: "Analytics" },
  { key: "abcxyz",      label: "ABC-XYZ",       group: "Analytics" },
  { key: "supplier",    label: "Supplier",      group: "Analytics" },
  { key: "intramonth",  label: "Intramonth",    group: "Analytics" },
  { key: "safetystock", label: "Safety Stock",  group: "Planning" },
  { key: "variability", label: "Variability",   group: "Planning" },
  { key: "leadtime",    label: "Lead Time",     group: "Planning" },
  { key: "signals",     label: "Signals",       group: "Planning" },
  { key: "simulation",  label: "Simulation",    group: "Planning" },
  { key: "investment",  label: "Investment",    group: "Planning" },
  { key: "forecast",    label: "Demand Fcst",   group: "Planning" },
  { key: "blended",     label: "Blended Demand", group: "Sensing" },
  { key: "echelon",     label: "Echelon SS",    group: "Sensing" },
  { key: "finance",     label: "Financial Plan", group: "Strategic" },
  { key: "events",      label: "Events",        group: "Strategic" },
  { key: "scenarios",   label: "Scenarios",     group: "Strategic" },
  { key: "demandplan",    label: "Demand Plan",    group: "Supply" },
  { key: "overridequeue",  label: "Override Queue", group: "Supply" },
  { key: "procurement",   label: "Procurement",    group: "Supply" },
  { key: "openpos",       label: "Open POs",       group: "Supply" },
  { key: "projection",   label: "Projection",     group: "Supply" },
  { key: "plannedorders", label: "Planned Orders", group: "Supply" },
] as const;

type SubTabKey = (typeof SUB_TABS)[number]["key"];

export function InvPlanningTab() {
  const [activePanel, setActivePanel] = useState<SubTabKey>("exceptions");

  return (
    <div className="flex flex-col gap-4 p-4">
      {/* Header */}
      <div>
        <h2 className="text-xl font-semibold text-foreground">Inventory Planning</h2>
        <p className="text-sm text-muted-foreground mt-1">
          Exception queue, health scoring, EOQ targets, replenishment policy, and demand analytics per item-location.
        </p>
      </div>

      {/* Sub-navigation tab strip (PL-009) */}
      <div className="flex flex-wrap gap-1 rounded-lg border bg-muted/30 p-1">
        {SUB_TABS.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActivePanel(tab.key)}
            className={cn(
              "rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
              activePanel === tab.key
                ? "bg-card text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground hover:bg-card/50",
            )}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Active panel */}
      {activePanel === "exceptions" && <ExceptionQueuePanel />}

      {activePanel === "health" && <PortfolioHealthPanel />}

      {activePanel === "eoq" && <EoqPanel />}

      {activePanel === "policy" && <PolicyManagementPanel />}

      {activePanel === "fillrate" && (
        <div className="rounded-lg border bg-card p-4 space-y-4">
          <h3 className="font-semibold text-base">Fill Rate Analytics</h3>
          <FillRatePanel />
        </div>
      )}

      {activePanel === "abcxyz" && (
        <div className="rounded-lg border bg-card p-4 space-y-4">
          <h3 className="font-semibold text-base">ABC-XYZ Segmentation</h3>
          <AbcXyzPanel />
        </div>
      )}

      {activePanel === "supplier" && (
        <div className="rounded-lg border bg-card p-4 space-y-4">
          <h3 className="font-semibold text-base">Supplier Performance</h3>
          <SupplierPanel />
        </div>
      )}

      {activePanel === "intramonth" && (
        <div className="rounded-lg border bg-card p-4 space-y-4">
          <h3 className="font-semibold text-base">Intra-Month Stockout Detection</h3>
          <IntramonthPanel />
        </div>
      )}

      {activePanel === "safetystock" && (
        <div className="rounded-lg border bg-card p-4 space-y-4">
          <h3 className="font-semibold text-base">Safety Stock</h3>
          <SafetyStockPanel />
        </div>
      )}

      {activePanel === "variability" && (
        <div className="rounded-lg border bg-card p-4 space-y-4">
          <h3 className="font-semibold text-base">Demand Variability</h3>
          <VariabilityPanel />
        </div>
      )}

      {activePanel === "leadtime" && (
        <div className="rounded-lg border bg-card p-4 space-y-4">
          <h3 className="font-semibold text-base">Lead Time Analysis</h3>
          <LeadTimePanel />
        </div>
      )}

      {activePanel === "signals" && (
        <div className="rounded-lg border bg-card p-4 space-y-4">
          <h3 className="font-semibold text-base">Demand Signals</h3>
          <DemandSignalsPanel />
        </div>
      )}

      {activePanel === "simulation" && (
        <div className="rounded-lg border bg-card p-4 space-y-4">
          <h3 className="font-semibold text-base">Safety Stock Simulation</h3>
          <SimulationPanel />
        </div>
      )}

      {activePanel === "investment" && (
        <div className="rounded-lg border bg-card p-4 space-y-4">
          <h3 className="font-semibold text-base">Investment Plan</h3>
          <InvestmentPanel />
        </div>
      )}

      {activePanel === "forecast" && (
        <div className="rounded-lg border bg-card p-4 space-y-4">
          <h3 className="font-semibold text-base">Production Demand Forecast</h3>
          <p className="text-xs text-muted-foreground">
            Forward-looking ML forecasts generated from champion models. Run{" "}
            <code className="font-mono">make forecast-generate</code> to refresh.
          </p>
          <DemandForecastPanel />
        </div>
      )}

      {activePanel === "demandplan" && (
        <div className="rounded-lg border bg-card p-4 space-y-4">
          <DemandPlanPanel />
        </div>
      )}

      {activePanel === "overridequeue" && (
        <div className="rounded-lg border bg-card p-4 space-y-4">
          <OverrideQueuePanel />
        </div>
      )}

      {activePanel === "procurement" && (
        <div className="rounded-lg border bg-card p-4 space-y-4">
          <ProcurementPanel />
        </div>
      )}

      {activePanel === "openpos" && (
        <OpenPOPanel />
      )}

      {activePanel === "projection" && (
        <div className="rounded-lg border bg-card p-4 space-y-4">
          <h3 className="font-semibold text-base">Forward Inventory Projection</h3>
          <p className="text-xs text-muted-foreground">
            Day-by-day projected inventory position across 3 scenarios: no new orders,
            with confirmed open POs, and with planned orders.
          </p>
          <ProjectionPanel />
        </div>
      )}

      {activePanel === "plannedorders" && (
        <div className="rounded-lg border bg-card p-4 space-y-4">
          <PlannedOrdersPanel />
        </div>
      )}

      {activePanel === "blended" && (
        <div className="rounded-lg border bg-card p-4 space-y-4">
          <h3 className="font-semibold text-base">Blended Demand Plan</h3>
          <p className="text-xs text-muted-foreground">
            Short-horizon demand sensing signal blended with statistical forecast using linearly decaying alpha weights.
          </p>
          <BlendedDemandPanel />
        </div>
      )}

      {activePanel === "echelon" && (
        <div className="rounded-lg border bg-card p-4 space-y-4">
          <h3 className="font-semibold text-base">Multi-Echelon Safety Stock</h3>
          <p className="text-xs text-muted-foreground">
            Risk-pooled DC-level safety stock targets with cascade risk scoring across downstream stores.
          </p>
          <EchelonPanel />
        </div>
      )}

      {activePanel === "finance" && (
        <div className="rounded-lg border bg-card p-4 space-y-4">
          <h3 className="font-semibold text-base">Financial Inventory Plan</h3>
          <p className="text-xs text-muted-foreground">
            Inventory value, carrying cost projections, excess inventory, and budget utilization tracking.
          </p>
          <FinancialPlanPanel />
        </div>
      )}

      {activePanel === "events" && (
        <div className="rounded-lg border bg-card p-4 space-y-4">
          <h3 className="font-semibold text-base">Event & Promotion Planning</h3>
          <p className="text-xs text-muted-foreground">
            Manage promotional events, seasonal uplifts, and forecast adjustments with approval workflow.
          </p>
          <EventCalendarPanel />
        </div>
      )}

      {activePanel === "scenarios" && (
        <div className="rounded-lg border bg-card p-4 space-y-4">
          <h3 className="font-semibold text-base">Supply Chain Scenario Planning</h3>
          <p className="text-xs text-muted-foreground">
            Model disruption scenarios (supplier delay, capacity constraints, demand shocks) and quantify financial impact.
          </p>
          <ScenarioPlanningPanel />
        </div>
      )}
    </div>
  );
}
