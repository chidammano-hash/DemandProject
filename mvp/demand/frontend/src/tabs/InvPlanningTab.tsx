/**
 * IPfeature4 + IPfeature5 + IPfeature6 + IPfeature7 + IPfeature8–IPfeature14
 * EOQ & Cycle Stock + Replenishment Policy + Health Score + Exception Queue +
 * Fill Rate + ABC-XYZ + Supplier + Intramonth + Safety Stock + Variability +
 * Lead Time + Demand Signals + Simulation + Investment Plan
 *
 * Inventory Planning tab: renders each panel as an extracted sub-component.
 */

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

export function InvPlanningTab() {
  return (
    <div className="flex flex-col gap-6 p-4">
      {/* Header */}
      <div>
        <h2 className="text-xl font-semibold text-foreground">
          Inventory Planning — Health Score, EOQ &amp; Replenishment Policy
        </h2>
        <p className="text-sm text-muted-foreground mt-1">
          Portfolio health scoring, Economic Order Quantity targets, and replenishment policy assignments per item-location.
        </p>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Exception Queue (IPfeature7)                                        */}
      {/* ------------------------------------------------------------------ */}
      <ExceptionQueuePanel />

      {/* ------------------------------------------------------------------ */}
      {/* Portfolio Health section (IPfeature6)                               */}
      {/* ------------------------------------------------------------------ */}
      <PortfolioHealthPanel />

      {/* ------------------------------------------------------------------ */}
      {/* EOQ KPIs, Sensitivity, Detail Table (IPfeature4)                   */}
      {/* ------------------------------------------------------------------ */}
      <EoqPanel />

      {/* ------------------------------------------------------------------ */}
      {/* Policy Management (IPfeature5)                                      */}
      {/* ------------------------------------------------------------------ */}
      <PolicyManagementPanel />

      {/* ================================================================
          IPfeature8: Fill Rate Analytics Panel
          ================================================================ */}
      <div className="rounded-lg border bg-card p-4 space-y-4">
        <h3 className="font-semibold text-base">Fill Rate Analytics</h3>
        <FillRatePanel />
      </div>

      {/* ================================================================
          IPfeature11: ABC-XYZ Classification Panel
          ================================================================ */}
      <div className="rounded-lg border bg-card p-4 space-y-4">
        <h3 className="font-semibold text-base">ABC-XYZ Segmentation</h3>
        <AbcXyzPanel />
      </div>

      {/* ================================================================
          IPfeature12: Supplier Performance Panel
          ================================================================ */}
      <div className="rounded-lg border bg-card p-4 space-y-4">
        <h3 className="font-semibold text-base">Supplier Performance</h3>
        <SupplierPanel />
      </div>

      {/* ================================================================
          IPfeature14: Intra-Month Stockout Detection Panel
          ================================================================ */}
      <div className="rounded-lg border bg-card p-4 space-y-4">
        <h3 className="font-semibold text-base">Intra-Month Stockout Detection</h3>
        <IntramonthPanel />
      </div>

      {/* ================================================================
          IPfeature3: Safety Stock Panel
          ================================================================ */}
      <div className="rounded-lg border bg-card p-4 space-y-4">
        <h3 className="font-semibold text-base">Safety Stock</h3>
        <SafetyStockPanel />
      </div>

      {/* ================================================================
          IPfeature1: Demand Variability Panel
          ================================================================ */}
      <div className="rounded-lg border bg-card p-4 space-y-4">
        <h3 className="font-semibold text-base">Demand Variability</h3>
        <VariabilityPanel />
      </div>

      {/* ================================================================
          IPfeature2: Lead Time Panel
          ================================================================ */}
      <div className="rounded-lg border bg-card p-4 space-y-4">
        <h3 className="font-semibold text-base">Lead Time Analysis</h3>
        <LeadTimePanel />
      </div>

      {/* ================================================================
          IPfeature9: Demand Signals Panel
          ================================================================ */}
      <div className="rounded-lg border bg-card p-4 space-y-4">
        <h3 className="font-semibold text-base">Demand Signals</h3>
        <DemandSignalsPanel />
      </div>

      {/* ================================================================
          IPfeature10: Simulation Panel
          ================================================================ */}
      <div className="rounded-lg border bg-card p-4 space-y-4">
        <h3 className="font-semibold text-base">Safety Stock Simulation</h3>
        <SimulationPanel />
      </div>

      {/* ================================================================
          IPfeature13: Investment Plan Panel
          ================================================================ */}
      <div className="rounded-lg border bg-card p-4 space-y-4">
        <h3 className="font-semibold text-base">Investment Plan</h3>
        <InvestmentPanel />
      </div>
    </div>
  );
}
