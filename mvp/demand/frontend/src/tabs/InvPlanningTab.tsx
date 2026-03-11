/**
 * Inventory Planning — 26-panel tab with horizontal group pills + sub-tab strip.
 *
 * IPfeature4–IPfeature14 + F1.1–F4.4
 * Redesigned: replaced inner sidebar with horizontal navigation (PL-009 v2).
 */

import { useState, useMemo } from "react";
import { cn } from "@/lib/utils";
import {
  AlertTriangle,
  Activity,
  Package2,
  Shield,
  TrendingUp,
  Grid3x3,
  Truck,
  Clock,
  ArchiveX,
  BarChart2,
  Timer,
  Radio,
  FlaskConical,
  DollarSign,
  RefreshCw,
  Target,
  Layers,
  Network,
  Banknote,
  CalendarDays,
  Zap,
  ClipboardList,
  Edit3,
  ShoppingCart,
  FileText,
  TrendingDown,
  CheckSquare,
  Repeat2,
} from "lucide-react";

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
import { ReplenishmentPlanPanel } from "./inv-planning/ReplenishmentPlanPanel";
import { RebalancingPanel } from "./inv-planning/RebalancingPanel";

// ---------------------------------------------------------------------------
// Navigation config
// ---------------------------------------------------------------------------
const GROUPS = [
  {
    id: "daily",
    label: "Daily Operations",
    shortLabel: "Daily Ops",
    tooltip: "Morning routine: triage exceptions, check portfolio health",
    accent: "text-red-600",
    accentBg: "bg-red-600",
    accentBgMuted: "bg-red-50 dark:bg-red-950/30",
    accentBorder: "border-red-600",
    tabs: [
      { key: "exceptions", label: "Exceptions", icon: AlertTriangle },
      { key: "health",     label: "Health",     icon: Activity },
    ],
  },
  {
    id: "optimize",
    label: "Replenishment Optimization",
    shortLabel: "Optimize",
    tooltip: "EOQ, safety stock, policies, and forward planning",
    accent: "text-blue-600",
    accentBg: "bg-blue-600",
    accentBgMuted: "bg-blue-50 dark:bg-blue-950/30",
    accentBorder: "border-blue-600",
    tabs: [
      { key: "eoq",          label: "EOQ",          icon: Package2 },
      { key: "policy",       label: "Policy",       icon: Shield },
      { key: "rebalancing",  label: "Rebalancing",  icon: Repeat2 },
    ],
  },
  {
    id: "analytics",
    label: "Analytics",
    shortLabel: "Analytics",
    tooltip: "Fill rate, ABC-XYZ, supplier performance, intramonth stockouts",
    accent: "text-emerald-600",
    accentBg: "bg-emerald-600",
    accentBgMuted: "bg-emerald-50 dark:bg-emerald-950/30",
    accentBorder: "border-emerald-600",
    tabs: [
      { key: "fillrate",   label: "Fill Rate",  icon: TrendingUp },
      { key: "abcxyz",     label: "ABC-XYZ",    icon: Grid3x3 },
      { key: "supplier",   label: "Supplier",   icon: Truck },
      { key: "intramonth", label: "Intramonth", icon: Clock },
    ],
  },
  {
    id: "planning",
    label: "Planning",
    shortLabel: "Planning",
    tooltip: "Demand forecast, replenishment plan, projection",
    accent: "text-violet-600",
    accentBg: "bg-violet-600",
    accentBgMuted: "bg-violet-50 dark:bg-violet-950/30",
    accentBorder: "border-violet-600",
    tabs: [
      { key: "safetystock", label: "Safety Stock",      icon: ArchiveX },
      { key: "variability", label: "Variability",       icon: BarChart2 },
      { key: "leadtime",    label: "Lead Time",         icon: Timer },
      { key: "signals",     label: "Signals",           icon: Radio },
      { key: "simulation",  label: "Simulation",        icon: FlaskConical },
      { key: "investment",  label: "Investment",        icon: DollarSign },
      { key: "replplan",    label: "Repl. Plan",        icon: RefreshCw },
      { key: "forecast",    label: "Demand Forecast",   icon: Target },
    ],
  },
  {
    id: "sensing",
    label: "Demand Intelligence",
    shortLabel: "Sensing",
    tooltip: "Blended demand signals and short-horizon sensing",
    accent: "text-teal-600",
    accentBg: "bg-teal-600",
    accentBgMuted: "bg-teal-50 dark:bg-teal-950/30",
    accentBorder: "border-teal-600",
    tabs: [
      { key: "blended", label: "Blended Demand", icon: Layers },
      { key: "echelon", label: "Echelon SS",     icon: Network },
    ],
  },
  {
    id: "strategic",
    label: "Strategic",
    shortLabel: "Strategic",
    tooltip: "Multi-echelon SS, investment optimization, financial planning",
    accent: "text-amber-600",
    accentBg: "bg-amber-600",
    accentBgMuted: "bg-amber-50 dark:bg-amber-950/30",
    accentBorder: "border-amber-600",
    tabs: [
      { key: "finance",   label: "Financial Plan", icon: Banknote },
      { key: "events",    label: "Events",         icon: CalendarDays },
      { key: "scenarios", label: "Scenarios",      icon: Zap },
    ],
  },
  {
    id: "supply",
    label: "Order-to-Cash",
    shortLabel: "OTC",
    tooltip: "Planned orders, procurement, purchase order tracking",
    accent: "text-slate-600",
    accentBg: "bg-slate-500",
    accentBgMuted: "bg-slate-50 dark:bg-slate-900/30",
    accentBorder: "border-slate-500",
    tabs: [
      { key: "demandplan",    label: "Demand Plan",    icon: ClipboardList },
      { key: "overridequeue", label: "Override Queue", icon: Edit3 },
      { key: "procurement",   label: "Procurement",    icon: ShoppingCart },
      { key: "openpos",       label: "Open POs",       icon: FileText },
      { key: "projection",    label: "Projection",     icon: TrendingDown },
      { key: "plannedorders", label: "Planned Orders", icon: CheckSquare },
    ],
  },
] as const;

type SubTabKey = (typeof GROUPS)[number]["tabs"][number]["key"];

// Panel title + description for each tab
const PANEL_META: Record<SubTabKey, { title: string; description?: string }> = {
  exceptions:    { title: "Exception Queue", description: "Replenishment exceptions ranked by severity and financial impact." },
  health:        { title: "Portfolio Health Score", description: "4-component health scoring across safety stock, DOS, stockout risk, and forecast accuracy." },
  eoq:           { title: "EOQ & Cycle Stock", description: "Economic Order Quantity targets and annual inventory cost breakdown." },
  policy:        { title: "Policy Management", description: "Replenishment policies, DFU assignments, and compliance tracking." },
  rebalancing:   { title: "Inventory Rebalancing", description: "Cross-location transfer optimization to balance network inventory." },
  fillrate:      { title: "Fill Rate Analytics", description: "Monthly order fulfilment rates and shortage trends." },
  abcxyz:        { title: "ABC-XYZ Segmentation", description: "Volume × variability classification matrix for policy targeting." },
  supplier:      { title: "Supplier Performance", description: "Lead time reliability scores and variability by supplier." },
  intramonth:    { title: "Intra-Month Stockout Detection", description: "Mid-month zero-inventory events and estimated lost sales." },
  safetystock:   { title: "Safety Stock", description: "Calculated safety stock targets vs current on-hand by DFU." },
  variability:   { title: "Demand Variability", description: "CV-based volatility profiles for demand segmentation." },
  leadtime:      { title: "Lead Time Analysis", description: "Lead time mean, standard deviation, and CV by item-supplier." },
  signals:       { title: "Demand Signals", description: "Short-horizon sensing signals: above plan, below plan, urgent alerts." },
  simulation:    { title: "Safety Stock Simulation", description: "Monte Carlo simulation for service-level vs safety-stock trade-off." },
  investment:    { title: "Investment Plan", description: "Efficient frontier: capital investment vs portfolio service level." },
  replplan:      { title: "Forward-Looking Replenishment Plan", description: "Forecast-driven SS, EOQ, and ROP for the next 12 months." },
  forecast:      { title: "Production Demand Forecast", description: "Champion ML forecasts with CI bands per DFU." },
  blended:       { title: "Blended Demand Plan", description: "Alpha-weighted sensing signal blended with statistical forecast." },
  echelon:       { title: "Multi-Echelon Safety Stock", description: "Risk-pooled DC-level SS with cascade severity scoring." },
  finance:       { title: "Financial Inventory Plan", description: "Inventory value, carrying cost, and budget utilization tracking." },
  events:        { title: "Event & Promotion Planning", description: "Promotional events, seasonal uplifts, and approval workflow." },
  scenarios:     { title: "Supply Chain Scenario Planning", description: "Disruption scenarios with quantified financial impact." },
  demandplan:    { title: "Demand Plan", description: "Multi-horizon consensus demand plan with quantile bands." },
  overridequeue: { title: "Override Queue", description: "Planner demand overrides pending approval." },
  procurement:   { title: "Procurement Workflow", description: "Purchase order approval and release pipeline." },
  openpos:       { title: "Open Purchase Orders", description: "In-flight PO lines with delivery dates and risk flags." },
  projection:    { title: "Forward Inventory Projection", description: "Day-by-day projected position: no orders, with POs, with planned orders." },
  plannedorders: { title: "Planned Orders", description: "System-generated replenishment proposals awaiting approval." },
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function InvPlanningTab() {
  const [activePanel, setActivePanel] = useState<SubTabKey>("exceptions");

  const meta = PANEL_META[activePanel];
  const activeGroup = useMemo(
    () => GROUPS.find((g) =>
      (g.tabs as ReadonlyArray<{ key: string }>).some((t) => t.key === activePanel)
    ),
    [activePanel],
  );
  const activeTab = activeGroup?.tabs.find((t) => t.key === activePanel);

  return (
    <div className="flex flex-col overflow-hidden" style={{ height: "calc(100vh - 108px)" }}>
      {/* ------------------------------------------------------------------ */}
      {/* Row 1: Group pills                                                  */}
      {/* ------------------------------------------------------------------ */}
      <div className="flex-shrink-0 flex items-center gap-1.5 border-b bg-muted/30 px-5 py-2 overflow-x-auto" role="tablist" aria-label="Inventory Planning groups">
        {GROUPS.map((group) => {
          const isActive = activeGroup?.id === group.id;
          return (
            <button
              key={group.id}
              onClick={() => setActivePanel(group.tabs[0].key as SubTabKey)}
              title={group.tooltip}
              role="tab"
              aria-selected={isActive}
              className={cn(
                "flex-shrink-0 rounded-full px-3 py-1 text-xs font-medium transition-all",
                isActive
                  ? cn("text-white", group.accentBg, "shadow-sm")
                  : cn("text-muted-foreground border border-border hover:border-muted-foreground/40 hover:text-foreground"),
              )}
            >
              {group.shortLabel}
            </button>
          );
        })}
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Row 2: Sub-tab strip for active group                               */}
      {/* ------------------------------------------------------------------ */}
      {activeGroup && (
        <div className="flex-shrink-0 flex items-center gap-0.5 border-b bg-background px-5 py-0 overflow-x-auto" role="tablist" aria-label={`${activeGroup.label} panels`}>
          {activeGroup.tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activePanel === tab.key;
            return (
              <button
                key={tab.key}
                onClick={() => setActivePanel(tab.key as SubTabKey)}
                role="tab"
                aria-selected={isActive}
                className={cn(
                  "flex-shrink-0 flex items-center gap-1.5 px-3 py-2 text-xs font-medium transition-all border-b-2",
                  isActive
                    ? cn("text-foreground", activeGroup.accentBorder)
                    : "border-transparent text-muted-foreground hover:text-foreground hover:border-muted-foreground/30",
                )}
              >
                <Icon
                  size={13}
                  className={cn(
                    "flex-shrink-0",
                    isActive ? activeGroup.accent : "text-muted-foreground",
                  )}
                />
                <span className="whitespace-nowrap">{tab.label}</span>
              </button>
            );
          })}
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Main content area                                                   */}
      {/* ------------------------------------------------------------------ */}
      <div className="flex flex-1 flex-col min-w-0 overflow-y-auto">
        {/* Panel header breadcrumb */}
        <div className="flex-shrink-0 border-b bg-background px-6 py-3">
          <p className="text-xs text-muted-foreground mb-1">
            Inventory Planning › {activeGroup?.label ?? ""} › {activeTab?.label ?? ""}
          </p>
          <h2 className="text-base font-semibold text-foreground leading-none">{meta.title}</h2>
          {meta.description && (
            <p className="mt-1 text-xs text-muted-foreground">{meta.description}</p>
          )}
        </div>

        {/* Panel body */}
        <div className="flex-1 p-5">
          {activePanel === "exceptions"    && <ExceptionQueuePanel />}
          {activePanel === "health"        && <PortfolioHealthPanel />}
          {activePanel === "eoq"           && <EoqPanel />}
          {activePanel === "policy"        && <PolicyManagementPanel />}
          {activePanel === "fillrate"      && <FillRatePanel />}
          {activePanel === "abcxyz"        && <AbcXyzPanel />}
          {activePanel === "supplier"      && <SupplierPanel />}
          {activePanel === "intramonth"    && <IntramonthPanel />}
          {activePanel === "safetystock"   && <SafetyStockPanel />}
          {activePanel === "variability"   && <VariabilityPanel />}
          {activePanel === "leadtime"      && <LeadTimePanel />}
          {activePanel === "signals"       && <DemandSignalsPanel />}
          {activePanel === "simulation"    && <SimulationPanel />}
          {activePanel === "investment"    && <InvestmentPanel />}
          {activePanel === "replplan"      && <ReplenishmentPlanPanel />}
          {activePanel === "forecast"      && <DemandForecastPanel />}
          {activePanel === "blended"       && <BlendedDemandPanel />}
          {activePanel === "echelon"       && <EchelonPanel />}
          {activePanel === "finance"       && <FinancialPlanPanel />}
          {activePanel === "events"        && <EventCalendarPanel />}
          {activePanel === "scenarios"     && <ScenarioPlanningPanel />}
          {activePanel === "demandplan"    && <DemandPlanPanel />}
          {activePanel === "overridequeue" && <OverrideQueuePanel />}
          {activePanel === "procurement"   && <ProcurementPanel />}
          {activePanel === "openpos"       && <OpenPOPanel />}
          {activePanel === "projection"    && <ProjectionPanel />}
          {activePanel === "plannedorders" && <PlannedOrdersPanel />}
          {activePanel === "rebalancing"  && <RebalancingPanel />}
        </div>
      </div>
    </div>
  );
}
