/**
 * Inventory Planning — 31-panel tab with horizontal group pills + sub-tab strip.
 *
 * IPfeature4–IPfeature14 + F1.1–F4.4 + Expert Panel Enhancements (20 suggestions).
 * Redesigned: replaced inner sidebar with horizontal navigation (PL-009 v2).
 * Added: Insights group (7 new panels), role-based view presets, progressive disclosure.
 */

import { useState, useMemo, useCallback, Fragment } from "react";
import { cn } from "@/lib/utils";
import { InvPlanningNavProvider } from "@/context/InvPlanningNavContext";
import {
  AlertTriangle,
  Activity,
  Package,
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
  Network,
  Banknote,
  CalendarDays,
  Zap,
  Edit3,
  ShoppingCart,
  FileText,
  TrendingDown,
  CheckSquare,
  Repeat2,
  Inbox,
  Map,
  PieChart,
  Award,
  Wallet,
  BarChart,
  SlidersHorizontal,
  LayoutTemplate,
  ChevronDown,
  ListOrdered,
  CheckCircle2,
  ArrowRight,
  type LucideIcon,
} from "lucide-react";

import { Button } from "@/components/ui/button";

import { TodaysPlanBanner } from "./inv-planning/TodaysPlanBanner";
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
import { DemandIntelligencePanel } from "./inv-planning/DemandIntelligencePanel";
import { OverrideQueuePanel } from "./inv-planning/OverrideQueuePanel";
import { ProcurementPanel } from "./inv-planning/ProcurementPanel";
import { OpenPOPanel } from "./inv-planning/OpenPOPanel";
import { ProjectionPanel } from "./inv-planning/ProjectionPanel";
import { PlannedOrdersPanel } from "./inv-planning/PlannedOrdersPanel";
import { EchelonPanel } from "./inv-planning/EchelonPanel";
import { FinancialPlanPanel } from "./inv-planning/FinancialPlanPanel";
import { EventCalendarPanel } from "./inv-planning/EventCalendarPanel";
import { ScenarioPlanningPanel } from "./inv-planning/ScenarioPlanningPanel";
import { ReplenishmentPlanPanel } from "./inv-planning/ReplenishmentPlanPanel";
import { RebalancingPanel } from "./inv-planning/RebalancingPanel";
// Expert panel enhancements — 7 new insight panels
import { ActionFeedPanel } from "./inv-planning/ActionFeedPanel";
import { NetworkHeatmapPanel } from "./inv-planning/NetworkHeatmapPanel";
import { SegmentDashboardPanel } from "./inv-planning/SegmentDashboardPanel";
import { PlanningScorecardPanel } from "./inv-planning/PlanningScorecardPanel";
import { CashFlowPanel } from "./inv-planning/CashFlowPanel";
import { ServiceLevelWaterfallPanel } from "./inv-planning/ServiceLevelWaterfallPanel";
import { ConstrainedOptPanel } from "./inv-planning/ConstrainedOptPanel";
import { SourcingPanel } from "./inv-planning/SourcingPanel";
import { PurchaseOrdersPanel } from "./inv-planning/PurchaseOrdersPanel";

// ---------------------------------------------------------------------------
// Role-based view presets (Expert #13 — Rachel Kim)
// ---------------------------------------------------------------------------
type ViewPreset = { id: string; label: string; icon: LucideIcon; groups: string[]; panels?: string[]; description: string };

const VIEW_PRESETS: ViewPreset[] = [
  { id: "essentials", label: "Daily Essentials", icon: Zap,            groups: [], panels: ["actionfeed", "exceptions", "projection", "plannedorders", "health", "replplan"], description: "6 essential panels: action feed, exceptions, projection, orders, health, replenishment" },
  { id: "all",      label: "All Panels",      icon: LayoutTemplate,    groups: ["insights", "daily", "optimize", "analytics", "planning", "sensing", "strategic", "supply"], description: "Full 33-panel toolkit" },
  { id: "weekly",   label: "Weekly Review",    icon: BarChart,          groups: ["analytics", "planning", "insights"], description: "Fill rate, ABC-XYZ, supplier, safety stock, scorecard" },
  { id: "monthly",  label: "Monthly Planning", icon: CalendarDays,      groups: ["optimize", "planning", "strategic", "insights"], description: "EOQ, policy, investment, S&OP, financial plan" },
  { id: "exec",     label: "Executive",        icon: Award,             groups: ["insights", "strategic"], description: "Scorecard, service level, cash flow, scenarios" },
];

// ---------------------------------------------------------------------------
// Navigation config
// ---------------------------------------------------------------------------
const GROUPS = [
  {
    id: "insights",
    label: "Insights",
    shortLabel: "Insights",
    tooltip: "Cross-domain intelligence: action feed, network view, segment drill-down, scorecard",
    accent: "text-cyan-600",
    accentBg: "bg-cyan-600",
    accentBgMuted: "bg-cyan-50 dark:bg-cyan-950/30",
    accentBorder: "border-cyan-600",
    tabs: [
      { key: "actionfeed",  label: "Action Feed",      icon: Inbox },
      { key: "netheatmap",  label: "Network Balance",   icon: Map },
      { key: "segment",     label: "Segment Drill",     icon: PieChart },
      { key: "scorecard",   label: "Scorecard",         icon: Award },
      { key: "cashflow",    label: "Cash Flow",         icon: Wallet },
      { key: "waterfall",   label: "Service Level",     icon: BarChart },
      { key: "budgetopt",   label: "Budget Optimizer",  icon: SlidersHorizontal },
    ],
  },
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
      { key: "demandintel",  label: "Demand Intelligence", icon: Target },
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
      { key: "overridequeue", label: "Override Queue", icon: Edit3 },
      { key: "procurement",   label: "Procurement",    icon: ShoppingCart },
      { key: "openpos",       label: "Open POs",       icon: FileText },
      { key: "sourcing",      label: "Sourcing",       icon: Package },
      { key: "purchaseorders", label: "PO History",    icon: FileText },
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
  demandintel:   { title: "Demand Intelligence", description: "Unified demand view: production forecast, demand plan, and blended demand sensing." },
  echelon:       { title: "Multi-Echelon Safety Stock", description: "Risk-pooled DC-level SS with cascade severity scoring." },
  finance:       { title: "Financial Inventory Plan", description: "Inventory value, carrying cost, and budget utilization tracking." },
  events:        { title: "Event & Promotion Planning", description: "Promotional events, seasonal uplifts, and approval workflow." },
  scenarios:     { title: "Supply Chain Scenario Planning", description: "Disruption scenarios with quantified financial impact." },
  overridequeue: { title: "Override Queue", description: "Planner demand overrides pending approval." },
  procurement:   { title: "Procurement Workflow", description: "Purchase order approval and release pipeline." },
  openpos:       { title: "Open Purchase Orders", description: "In-flight PO lines with delivery dates and risk flags." },
  sourcing:      { title: "Sourcing Network", description: "Item-location supply source mapping with single-source risk analysis." },
  purchaseorders: { title: "PO History", description: "Comprehensive purchase order history (open + closed) with on-time delivery and lead time analysis." },
  projection:    { title: "Forward Inventory Projection", description: "Day-by-day projected position: no orders, with POs, with planned orders." },
  plannedorders: { title: "Planned Orders", description: "System-generated replenishment proposals awaiting approval." },
  // Expert insight panels
  actionfeed:  { title: "Unified Action Feed", description: "Priority-ranked actions from exceptions, signals, PO risks, and stockouts — your morning starting point." },
  netheatmap:  { title: "Network Balance Heatmap", description: "Location × category DOS matrix — instantly see where inventory is pooling vs depleting." },
  segment:     { title: "Segment Dashboard", description: "Deep-dive into any ABC-XYZ segment: KPIs, policies, exceptions, and recommended actions." },
  scorecard:   { title: "Planning Effectiveness Scorecard", description: "Trailing metrics — are your planning actions actually improving outcomes?" },
  cashflow:    { title: "Cash Flow Timeline", description: "Monthly cash outflow projection: PO commitments, planned orders, and SS investment." },
  waterfall:   { title: "Service Level Waterfall", description: "How each inventory lever contributes to your achieved service level." },
  budgetopt:   { title: "Budget-Constrained Optimizer", description: "Given a budget cap, find the optimal SS allocation for maximum service level." },
};

// ---------------------------------------------------------------------------
// Guided Workflow steps (Issue #25)
// ---------------------------------------------------------------------------
const GUIDED_STEPS = [
  { step: 1, label: "Review Alerts", panelKey: "actionfeed" as SubTabKey, description: "Check what needs attention today" },
  { step: 2, label: "Resolve Exceptions", panelKey: "exceptions" as SubTabKey, description: "Address critical stockout and excess issues" },
  { step: 3, label: "Check Projections", panelKey: "projection" as SubTabKey, description: "Verify inventory trajectory for at-risk items" },
  { step: 4, label: "Approve Orders", panelKey: "plannedorders" as SubTabKey, description: "Review and approve pending replenishment orders" },
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function InvPlanningTab() {
  const [activePanel, setActivePanel] = useState<SubTabKey>("actionfeed");
  const [activeView, setActiveView] = useState("essentials");
  const [showBanner, setShowBanner] = useState(true);
  const [isGuided, setIsGuided] = useState(false);
  const [guidedStep, setGuidedStep] = useState(0);

  const meta = PANEL_META[activePanel];

  // Progressive disclosure: filter groups by active view preset (Expert #14)
  // Panel-level presets (with `panels` array) build a single virtual group
  // containing only the specified panels. Group-level presets filter by group id.
  const visibleGroups = useMemo(() => {
    const preset = VIEW_PRESETS.find((v) => v.id === activeView);
    if (!preset || preset.id === "all") return GROUPS;
    // Panel-level preset: collect matching tabs from any group into one virtual group
    if (preset.panels && preset.panels.length > 0) {
      const panelSet = new Set(preset.panels);
      const matchedTabs: { key: string; label: string; icon: LucideIcon }[] = [];
      for (const group of GROUPS) {
        for (const tab of group.tabs) {
          if (panelSet.has(tab.key)) matchedTabs.push({ key: tab.key, label: tab.label, icon: tab.icon });
        }
      }
      // Sort by the order specified in the preset panels array
      matchedTabs.sort((a, b) => preset.panels!.indexOf(a.key) - preset.panels!.indexOf(b.key));
      return [{
        id: "essentials" as const,
        label: "Daily Essentials",
        shortLabel: "Essentials",
        tooltip: "The 6 panels you need every day",
        accent: "text-indigo-600" as const,
        accentBg: "bg-indigo-600" as const,
        accentBgMuted: "bg-indigo-50 dark:bg-indigo-950/30" as const,
        accentBorder: "border-indigo-600" as const,
        tabs: matchedTabs as unknown as typeof GROUPS[number]["tabs"],
      }] as unknown as typeof GROUPS;
    }
    return GROUPS.filter((g) => preset.groups.includes(g.id));
  }, [activeView]);

  const activeGroup = useMemo(
    () => visibleGroups.find((g) =>
      (g.tabs as ReadonlyArray<{ key: string }>).some((t) => t.key === activePanel)
    ) ?? visibleGroups[0],
    [activePanel, visibleGroups],
  );
  const activeTab = activeGroup?.tabs.find((t) => t.key === activePanel);

  // When switching views, jump to first tab of first visible group and exit guided mode
  const handleViewChange = useCallback((viewId: string) => {
    setIsGuided(false);
    setActiveView(viewId);
    const preset = VIEW_PRESETS.find((v) => v.id === viewId);
    if (preset && preset.id !== "all") {
      // Panel-level presets: jump to first panel in the ordered list
      if (preset.panels && preset.panels.length > 0) {
        setActivePanel(preset.panels[0] as SubTabKey);
      } else {
        const firstGroup = GROUPS.find((g) => preset.groups.includes(g.id));
        if (firstGroup) setActivePanel(firstGroup.tabs[0].key as SubTabKey);
      }
    }
  }, []);

  const navValue = useMemo(
    () => ({ navigateTo: (panel: string) => setActivePanel(panel as SubTabKey) }),
    [],
  );

  return (
    <InvPlanningNavProvider value={navValue}>
    <div className="flex flex-col overflow-hidden" style={{ height: "calc(100vh - 108px)" }}>
      {/* ------------------------------------------------------------------ */}
      {/* Today's Plan Banner (Issue #14)                                     */}
      {/* ------------------------------------------------------------------ */}
      <div className="flex-shrink-0 px-5 pt-3">
        {showBanner ? (
          <TodaysPlanBanner onCollapse={() => setShowBanner(false)} />
        ) : (
          <button
            onClick={() => setShowBanner(true)}
            className="flex items-center gap-1 text-[10px] text-muted-foreground hover:text-foreground transition-colors mb-2"
          >
            <ChevronDown size={12} />
            <span>Show Plan</span>
          </button>
        )}
      </div>
      {/* ------------------------------------------------------------------ */}
      {/* Role-based view selector (Expert #13) + header                      */}
      {/* ------------------------------------------------------------------ */}
      <div className="flex-shrink-0 flex items-center justify-between px-5 pt-1 pb-2">
        <p className="text-xs text-muted-foreground max-w-2xl leading-relaxed">
          {VIEW_PRESETS.find((v) => v.id === activeView)?.description ?? "Full inventory planning toolkit."}
        </p>
        <div className="flex items-center gap-1 ml-4 flex-shrink-0">
          {VIEW_PRESETS.map((preset) => {
            const Icon = preset.icon;
            const isActive = activeView === preset.id;
            return (
              <button
                key={preset.id}
                onClick={() => handleViewChange(preset.id)}
                title={preset.description}
                className={cn(
                  "flex items-center gap-1 rounded px-2 py-1 text-[10px] font-medium transition-all",
                  isActive
                    ? "bg-primary text-primary-foreground shadow-sm"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground",
                )}
              >
                <Icon size={11} />
                <span className="hidden sm:inline">{preset.label}</span>
              </button>
            );
          })}
          <div className="w-px h-4 bg-border mx-1" />
          <Button
            size="sm"
            variant={isGuided ? "default" : "outline"}
            className="text-xs gap-1"
            onClick={() => {
              const next = !isGuided;
              setIsGuided(next);
              if (next) {
                setGuidedStep(0);
                setActivePanel(GUIDED_STEPS[0].panelKey);
              }
            }}
          >
            <ListOrdered className="h-3.5 w-3.5" />
            Guided Workflow
          </Button>
        </div>
      </div>
      {/* ------------------------------------------------------------------ */}
      {/* Guided Workflow progress bar (Issue #25)                             */}
      {/* ------------------------------------------------------------------ */}
      {isGuided && (
        <div className="flex-shrink-0 border-b bg-background px-5 py-3 space-y-2">
          {/* Step pills with arrows */}
          <div className="flex items-center gap-1">
            {GUIDED_STEPS.map((s, i) => (
              <Fragment key={s.step}>
                <button
                  onClick={() => { setGuidedStep(i); setActivePanel(s.panelKey); }}
                  className={cn(
                    "flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all",
                    i === guidedStep
                      ? "bg-primary text-primary-foreground"
                      : i < guidedStep
                        ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400"
                        : "bg-muted text-muted-foreground",
                  )}
                >
                  {i < guidedStep ? <CheckCircle2 className="h-3 w-3" /> : <span>{s.step}</span>}
                  {s.label}
                </button>
                {i < GUIDED_STEPS.length - 1 && (
                  <ArrowRight className="h-4 w-4 text-muted-foreground shrink-0" />
                )}
              </Fragment>
            ))}
          </div>

          {/* Step description + navigation */}
          <div className="flex items-center justify-between bg-muted/50 rounded px-3 py-2">
            <p className="text-xs text-muted-foreground">
              Step {GUIDED_STEPS[guidedStep].step}: {GUIDED_STEPS[guidedStep].description}
            </p>
            <div className="flex gap-2">
              <Button size="sm" variant="outline" className="text-xs"
                disabled={guidedStep === 0}
                onClick={() => { const prev = guidedStep - 1; setGuidedStep(prev); setActivePanel(GUIDED_STEPS[prev].panelKey); }}>
                Previous
              </Button>
              <Button size="sm" className="text-xs"
                disabled={guidedStep === GUIDED_STEPS.length - 1}
                onClick={() => { const next = guidedStep + 1; setGuidedStep(next); setActivePanel(GUIDED_STEPS[next].panelKey); }}>
                Next Step
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Row 1: Group pills (filtered by view) — hidden in guided mode       */}
      {/* ------------------------------------------------------------------ */}
      {!isGuided && (
        <div className="flex-shrink-0 flex items-center gap-1.5 border-b bg-muted/30 px-5 py-2 overflow-x-auto" role="tablist" aria-label="Inventory Planning groups">
          {activeView === "essentials" && (
            <span className="flex-shrink-0 text-[10px] font-semibold uppercase tracking-widest text-indigo-500 mr-2">
              Daily Essentials
            </span>
          )}
          {visibleGroups.map((group) => {
            const isActive = activeGroup?.id === group.id;
            return (
              <button
                key={group.id}
                onClick={() => { setIsGuided(false); setActivePanel(group.tabs[0].key as SubTabKey); }}
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
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Row 2: Sub-tab strip for active group — hidden in guided mode       */}
      {/* ------------------------------------------------------------------ */}
      {!isGuided && activeGroup && (
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
          {isGuided ? (
            <p className="text-xs text-muted-foreground mb-1">
              Guided Workflow › Step {GUIDED_STEPS[guidedStep].step} of {GUIDED_STEPS.length}
            </p>
          ) : (
            <p className="text-xs text-muted-foreground mb-1">
              Inventory Planning › {activeGroup?.label ?? ""} › {activeTab?.label ?? ""}
            </p>
          )}
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
          {activePanel === "demandintel"   && <DemandIntelligencePanel />}
          {activePanel === "echelon"       && <EchelonPanel />}
          {activePanel === "finance"       && <FinancialPlanPanel />}
          {activePanel === "events"        && <EventCalendarPanel />}
          {activePanel === "scenarios"     && <ScenarioPlanningPanel />}
          {activePanel === "overridequeue" && <OverrideQueuePanel />}
          {activePanel === "procurement"   && <ProcurementPanel />}
          {activePanel === "openpos"       && <OpenPOPanel />}
          {activePanel === "sourcing"      && <SourcingPanel />}
          {activePanel === "purchaseorders" && <PurchaseOrdersPanel />}
          {activePanel === "projection"    && <ProjectionPanel />}
          {activePanel === "plannedorders" && <PlannedOrdersPanel />}
          {activePanel === "rebalancing"  && <RebalancingPanel />}
          {/* Expert insight panels */}
          {activePanel === "actionfeed"   && <ActionFeedPanel />}
          {activePanel === "netheatmap"   && <NetworkHeatmapPanel />}
          {activePanel === "segment"      && <SegmentDashboardPanel />}
          {activePanel === "scorecard"    && <PlanningScorecardPanel />}
          {activePanel === "cashflow"     && <CashFlowPanel />}
          {activePanel === "waterfall"    && <ServiceLevelWaterfallPanel />}
          {activePanel === "budgetopt"    && <ConstrainedOptPanel />}
        </div>
      </div>
    </div>
    </InvPlanningNavProvider>
  );
}
