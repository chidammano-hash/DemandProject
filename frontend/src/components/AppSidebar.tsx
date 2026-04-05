import { useCallback } from "react";
import { VrantisLogo } from "./VrantisLogo";
import {
  LayoutDashboard,
  Database,
  TrendingUp,
  Activity,
  Network,
  PlayCircle,
  Brain, Monitor,
  CalendarDays,
  Shield, BarChart3, MapPin,
  Settings2,
  TerminalSquare,
  History,
  PanelLeftClose,
  PanelLeft,
  Menu,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { SidebarSection } from "@/types/theme";
import { useJobNotification } from "@/context/JobNotificationContext";

// ---------------------------------------------------------------------------
// Navigation config
// ---------------------------------------------------------------------------
interface NavItem {
  key: string;
  label: string;
  icon: LucideIcon;
  section: SidebarSection;
  shortcut?: string;
}

const NAV_ITEMS: NavItem[] = [
  // Tower — control tower triage (no section header)
  { key: "commandCenter",     label: "Command Center", icon: Monitor,         section: "tower",       shortcut: "1" },
  // Operations — planning processes & admin
  { key: "sop",               label: "S&OP",           icon: CalendarDays,    section: "operations",  shortcut: "2" },
  { key: "jobs",              label: "Jobs",           icon: PlayCircle,      section: "operations",  shortcut: "3" },
  { key: "dataQuality",      label: "Data Quality",   icon: Shield,          section: "operations" },
  // Supply — inventory & replenishment
  { key: "invPlanning",  label: "Inv. Planning", icon: Brain,        section: "supply",  shortcut: "4" },
  { key: "invBacktest",  label: "Inv. Backtest", icon: Activity,     section: "supply" },
  // Demand — analysis & forecasting
  { key: "aggregateAnalysis", label: "Portfolio",      icon: LayoutDashboard, section: "demand",   shortcut: "6" },
  { key: "itemAnalysis",      label: "Item Analysis",  icon: TrendingUp,      section: "demand",   shortcut: "7" },
  { key: "fva",               label: "FVA & ROI",      icon: BarChart3,       section: "demand" },
  { key: "lgbmTuning",        label: "Model Tuning",  icon: Activity,        section: "demand" },
  { key: "customerAnalytics", label: "Customer Analytics", icon: MapPin,       section: "demand" },
  { key: "demandHistory",     label: "Demand History",     icon: History,      section: "demand" },
  // System — data tools
  { key: "explorer",     label: "Explorer",      icon: Database,     section: "system" },
  { key: "sqlRunner",    label: "SQL Runner",    icon: TerminalSquare, section: "system" },
  { key: "settings",     label: "Settings",      icon: Settings2,    section: "system" },
];

const SECTION_LABELS: Record<string, string> = {
  tower: "",
  operations: "Operations",
  supply: "Supply",
  demand: "Demand",
  system: "System",
  command: "",
  plan: "",
  overview: "",
  intelligence: "",
};

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------
export interface AppSidebarProps {
  activeTab: string;
  onNavigate: (tab: string) => void;
  collapsed: boolean;
  onToggle: () => void;
  appName?: string; // retained for backward compat; logo always shows "Vrantis"
  themeFooter?: React.ReactNode;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function AppSidebar({ activeTab, onNavigate, collapsed, onToggle, appName, themeFooter }: AppSidebarProps) {
  let lastSection: SidebarSection | null = null;
  const { activeJobCount } = useJobNotification();

  const handleNav = useCallback((key: string) => {
    onNavigate(key);
  }, [onNavigate]);

  return (
    <>
      {/* Mobile hamburger button */}
      <button
        className="fixed left-3 top-3 z-50 rounded-md border border-border bg-card p-2 shadow-md md:hidden"
        onClick={onToggle}
        aria-label="Toggle navigation"
      >
        <Menu className="h-5 w-5 text-foreground" />
      </button>

      {/* Sidebar */}
      <aside
        role="navigation"
        aria-label="Main navigation"
        className={cn(
          "flex h-screen flex-col border-r border-border bg-sidebar transition-[width] duration-200 ease-in-out",
          "fixed left-0 top-0 z-40 md:relative md:z-auto",
          collapsed ? "w-16" : "w-60",
          // Mobile: hidden by default, shown via hamburger
          "max-md:-translate-x-full max-md:data-[open=true]:translate-x-0",
        )}
        data-open={!collapsed ? "true" : undefined}
      >
        {/* Header */}
        <div className="flex h-14 items-center border-b border-border/50 px-3">
          {collapsed ? (
            <VrantisLogo size={28} />
          ) : (
            <VrantisLogo size={26} showWordmark className="flex-1 min-w-0" />
          )}
          <button
            onClick={onToggle}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            className="ml-auto flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-md text-sidebar-foreground hover:bg-sidebar-hover"
          >
            {collapsed ? <PanelLeft className="h-4 w-4" /> : <PanelLeftClose className="h-4 w-4" />}
          </button>
        </div>

        {/* Nav items */}
        <nav className="flex-1 overflow-y-auto px-2 py-2">
          {NAV_ITEMS.map((item) => {
            const showDivider = item.section !== lastSection && lastSection !== null && SECTION_LABELS[item.section] !== "";
            lastSection = item.section;
            const isActive = activeTab === item.key;
            const Icon = item.icon;

            return (
              <div key={item.key}>
                {showDivider && (
                  <div className="my-2 px-2">
                    <div className="border-t border-border/40" />
                    {!collapsed && (
                      <span className="mt-2 block text-[10px] font-medium uppercase tracking-wider text-sidebar-foreground/50">
                        {SECTION_LABELS[item.section]}
                      </span>
                    )}
                  </div>
                )}
                <button
                  onClick={() => handleNav(item.key)}
                  aria-current={isActive ? "page" : undefined}
                  title={collapsed ? item.label : undefined}
                  className={cn(
                    "group relative flex w-full items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors duration-150",
                    collapsed && "justify-center px-0",
                    isActive
                      ? "bg-sidebar-active/15 font-medium text-sidebar-active"
                      : "text-sidebar-foreground hover:bg-sidebar-hover hover:text-foreground",
                  )}
                >
                  {/* Active indicator bar */}
                  {isActive && (
                    <span className="absolute left-0 top-1/2 h-6 w-1 -translate-y-1/2 rounded-r-full bg-sidebar-active" />
                  )}
                  <Icon
                    className={cn("h-[18px] w-[18px] flex-shrink-0", isActive ? "text-sidebar-active" : "")}
                    strokeWidth={1.5}
                  />
                  {!collapsed && (
                    <span className="flex-1 truncate text-left">{item.label}</span>
                  )}
                  {/* Active job count badge */}
                  {item.key === "jobs" && activeJobCount > 0 && (
                    <span className={cn(
                      "flex h-4 min-w-4 items-center justify-center rounded-full bg-blue-500 px-1 text-[9px] font-bold text-white",
                      !collapsed && "ml-auto mr-1",
                      collapsed && "absolute -right-0.5 -top-0.5",
                    )}>
                      {activeJobCount}
                    </span>
                  )}
                  {!collapsed && item.shortcut && (
                    <kbd className="hidden text-[10px] text-sidebar-foreground/40 lg:inline">{item.shortcut}</kbd>
                  )}
                </button>
              </div>
            );
          })}
        </nav>

        {/* Footer: theme selector + ⌘K hint */}
        <div className="border-t border-border/50 p-2 space-y-1">
          {themeFooter}
          {!collapsed && (
            <p className="px-2 py-1 text-[10px] text-sidebar-foreground/40 flex items-center gap-1">
              <kbd className="rounded border border-border/50 bg-muted px-1 py-0.5 font-mono text-[9px]">⌘K</kbd>
              <span>command palette</span>
            </p>
          )}
        </div>
      </aside>

      {/* Mobile overlay */}
      {!collapsed && (
        <div
          className="fixed inset-0 z-30 bg-black/50 backdrop-blur-sm md:hidden"
          onClick={onToggle}
          aria-hidden="true"
        />
      )}
    </>
  );
}

export { NAV_ITEMS };
