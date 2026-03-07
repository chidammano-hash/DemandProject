/**
 * IPfeature4 + IPfeature5 + IPfeature6 + IPfeature7 + IPfeature8–IPfeature14 + F1.1
 * EOQ & Cycle Stock + Replenishment Policy + Health Score + Exception Queue +
 * Fill Rate + ABC-XYZ + Supplier + Intramonth + Safety Stock + Variability +
 * Lead Time + Demand Signals + Simulation + Investment Plan + Production Forecast
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
    </div>
  );
}
