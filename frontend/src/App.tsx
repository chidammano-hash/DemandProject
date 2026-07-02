import { lazy, Suspense, useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { ErrorBoundary } from "react-error-boundary";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { AppSidebar } from "@/components/AppSidebar";
import { ThemeSelector } from "@/components/ThemeSelector";
// GlobalFilterBar removed — filters are now local to AggregateAnalysisTab
import { LoadingElement } from "@/components/LoadingElement";
import { KeyboardShortcutHelp } from "@/components/KeyboardShortcutHelp";
import { useTheme } from "@/hooks/useTheme";
import { useSidebar } from "@/hooks/useSidebar";
import { useGlobalFilters } from "@/hooks/useGlobalFilters";
import { GlobalFilterProvider } from "@/context/GlobalFilterContext";
import { ActiveSkuProvider } from "@/context/ActiveSkuContext";
import { ThemeProvider } from "@/context/ThemeContext";
import { GlobalChatDrawer } from "./tabs/sku-chat/GlobalChatDrawer";
import { ScenarioNotificationProvider } from "@/context/ScenarioNotificationContext";
import { JobNotificationProvider } from "@/context/JobNotificationContext";
import {
  getInitialDomain,
  getInitialTab,
  updateUrlState,
  usePopstateSync,
  setScenarioJobParam,
  ANALYTICS_TAB_DOMAINS,
  DIMENSION_DOMAINS,
} from "@/hooks/useUrlState";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import { useCommandPalette } from "@/hooks/useCommandPalette";
import { CommandPalette } from "@/components/CommandPalette";
import { Toaster } from "@/components/Toaster";

// ---------------------------------------------------------------------------
// Lazy-loaded tab components
// ---------------------------------------------------------------------------
const AggregateAnalysisTab = lazy(() => import("./tabs/AggregateAnalysisTab").then((m) => ({ default: m.AggregateAnalysisTab })));
const ExplorerTab = lazy(() => import("./tabs/ExplorerTab").then((m) => ({ default: m.ExplorerTab })));
// ClustersTab removed from sidebar — still importable if needed via URL
const ClustersTab = lazy(() => import("./tabs/ClustersTab"));
const ItemAnalysisTab = lazy(() => import("./tabs/ItemAnalysisTab").then((m) => ({ default: m.ItemAnalysisTab })));
const SkuChatTab = lazy(() => import("./tabs/SkuChatTab").then((m) => ({ default: m.SkuChatTab })));
const MarketIntelTab = lazy(() => import("./tabs/MarketIntelTab"));
const InvBacktestTab = lazy(() => import("./tabs/InvBacktestTab"));
const InvPlanningTab = lazy(() => import("./tabs/InvPlanningTab").then((m) => ({ default: m.InvPlanningTab })));
const JobsTab = lazy(() => import("./tabs/JobsTab"));
const StoryboardTab = lazy(() => import("./tabs/StoryboardTab"));
const ExceptionsTab = StoryboardTab; // alias — PL-003 rename
const SopTab = lazy(() => import("./tabs/SopTab"));
const FVATab = lazy(() => import("./tabs/FVATab"));
const DataQualityTab = lazy(() => import("./tabs/DataQualityTab"));
const CustomerAnalyticsTab = lazy(() => import("./tabs/CustomerAnalyticsTab").then((m) => ({ default: m.CustomerAnalyticsTab })));
const CommandCenterTab = lazy(() => import("./tabs/CommandCenterTab"));
const SqlRunnerTab = lazy(() => import("./tabs/SqlRunnerTab").then((m) => ({ default: m.SqlRunnerTab })));
const SettingsTab = lazy(() => import("./tabs/SettingsTab"));
const ModelTuningTab = lazy(() => import("./tabs/ModelTuningTab"));
const DemandHistoryTab = lazy(() => import("./tabs/DemandHistoryTab"));
const SkuFeaturesTab = lazy(() => import("./tabs/SkuFeaturesTab"));
const IntegrationTab = lazy(() => import("./tabs/IntegrationTab"));

// ---------------------------------------------------------------------------
// Error boundary fallback for individual tabs
// ---------------------------------------------------------------------------
function TabErrorFallback({ error, resetErrorBoundary, tabKey }: { error: unknown; resetErrorBoundary: () => void; tabKey: string }) {
  const msg = error instanceof Error ? error.message : "An unexpected error occurred";
  return (
    <Card className="mt-4 border-destructive/30 bg-destructive/10">
      <CardContent className="pt-4 flex flex-col items-center gap-3">
        <LoadingElement tabKey={tabKey} />
        <p className="text-sm text-destructive">{msg}</p>
        <Button variant="outline" size="sm" onClick={resetErrorBoundary}>Retry</Button>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Suspense fallback
// ---------------------------------------------------------------------------
function TabSuspenseFallback({ tabKey }: { tabKey: string }) {
  return <LoadingElement tabKey={tabKey} message="Loading tab..." />;
}

// ---------------------------------------------------------------------------
// Reusable tab wrapper — eliminates 15 repeated ErrorBoundary+Suspense blocks
// ---------------------------------------------------------------------------
function TabPanel({
  tabKey,
  resetKeys,
  children,
}: {
  tabKey: string;
  resetKeys: unknown[];
  children: ReactNode;
}) {
  return (
    <ErrorBoundary
      FallbackComponent={(props) => <TabErrorFallback {...props} tabKey={tabKey} />}
      resetKeys={resetKeys}
    >
      <Suspense fallback={<TabSuspenseFallback tabKey={tabKey} />}>
        {children}
      </Suspense>
    </ErrorBoundary>
  );
}

// ---------------------------------------------------------------------------
// App shell
// ---------------------------------------------------------------------------
export default function App() {
  const themeHook = useTheme();
  const { theme, colorMode, productTheme, setColorMode, toggleColorMode } = themeHook;
  const sidebar = useSidebar();
  const globalFilters = useGlobalFilters();

  const [domain, setDomain] = useState(getInitialDomain);
  const [activeTab, setActiveTab] = useState(getInitialTab);

  usePopstateSync(setActiveTab, setDomain);

  // Sync URL when tab or domain changes
  useEffect(() => { updateUrlState(domain, activeTab); }, [activeTab, domain]);

  // Tab switching logic
  const handleTabSwitch = useCallback(
    (tab: string) => {
      setActiveTab(tab);
      if (tab === "explorer" && (ANALYTICS_TAB_DOMAINS.has(domain) || !DIMENSION_DOMAINS.includes(domain))) setDomain("item");
      if (tab === "clusters" && domain !== "sku") setDomain("sku");
    },
    [domain],
  );

  // Navigate from JobsTab to ClustersTab with scenario result
  const handleNavigateToScenario = useCallback((jobId: string) => {
    setScenarioJobParam(jobId);
    handleTabSwitch("clusters");
  }, [handleTabSwitch]);

  // Command palette
  const cmdPalette = useCommandPalette();

  // Keyboard shortcuts
  const { showHelp, closeHelp } = useKeyboardShortcuts({
    onTabSwitch: handleTabSwitch,
    onClosePanel: sidebar.closeMobile,
    onToggleSidebar: sidebar.toggle,
    onToggleColorMode: toggleColorMode,
  });

  // Theme footer for sidebar
  const themeFooter = (
    <ThemeSelector
      colorMode={colorMode}
      onModeChange={setColorMode}
      collapsed={sidebar.collapsed}
    />
  );

  // Memoize provider values so palette/sidebar/url state changes do not
  // re-render every consumer of theme or global filters.
  const themeValue = useMemo(() => ({ theme }), [theme]);

  return (
    <ThemeProvider value={themeValue}>
      <GlobalFilterProvider value={globalFilters}>
        <ActiveSkuProvider>
        <ScenarioNotificationProvider>
        <JobNotificationProvider>
        <div className="flex h-screen overflow-hidden">
          {/* Skip to content link for accessibility */}
          <a href="#tab-content" className="sr-only focus:not-sr-only focus:absolute focus:z-50 focus:rounded-md focus:bg-primary focus:px-4 focus:py-2 focus:text-primary-foreground">
            Skip to content
          </a>

          {/* Keyboard shortcut help modal */}
          {showHelp && <KeyboardShortcutHelp onClose={closeHelp} />}

          {/* Command palette (Cmd+K) */}
          <CommandPalette
            open={cmdPalette.open}
            onClose={cmdPalette.close}
            onNavigate={handleTabSwitch}
            onToggleDarkMode={toggleColorMode}
            onToggleSidebar={sidebar.toggle}
            onShowKeyboardHelp={() => { cmdPalette.close(); }}
            activeTab={activeTab}
          />

          {/* Sidebar */}
          <AppSidebar
            activeTab={activeTab}
            onNavigate={handleTabSwitch}
            collapsed={sidebar.collapsed}
            onToggle={sidebar.toggle}
            appName={productTheme.displayName}
            themeFooter={themeFooter}
          />

          {/* Main content area */}
          <div className="flex flex-1 flex-col overflow-hidden">
            {/* Mobile notice */}
            <div className="block px-4 py-2 text-center text-xs text-muted-foreground bg-muted/50 border-b border-border/30 md:hidden">
              Best experienced on desktop for full analytics
            </div>
            {/* Tab content */}
            <div id="tab-content" role="tabpanel" aria-label={`${activeTab} tab content`} className="flex-1 overflow-y-auto p-4 md:p-6">
              <div className="mx-auto max-w-[1600px]">
                {activeTab === "commandCenter" && (
                  <TabPanel tabKey="commandCenter" resetKeys={[activeTab]}>
                    <CommandCenterTab onNavigate={handleTabSwitch} />
                  </TabPanel>
                )}
                {(activeTab === "aggregateAnalysis" || activeTab === "overview" || activeTab === "accuracy") && (
                  <TabPanel tabKey="aggregateAnalysis" resetKeys={[activeTab]}>
                    <AggregateAnalysisTab onNavigate={handleTabSwitch} />
                  </TabPanel>
                )}
                {activeTab === "explorer" && (
                  <TabPanel tabKey="explorer" resetKeys={[activeTab, domain]}>
                    <ExplorerTab domain={domain} onDomainChange={setDomain} />
                  </TabPanel>
                )}
                {activeTab === "clusters" && (
                  <TabPanel tabKey="clusters" resetKeys={[activeTab]}>
                    <ClustersTab domain={domain} onDomainChange={setDomain} />
                  </TabPanel>
                )}
                {(activeTab === "itemAnalysis" || activeTab === "skuAnalysis" || activeTab === "inventory") && (
                  <TabPanel tabKey="itemAnalysis" resetKeys={[activeTab]}>
                    <ItemAnalysisTab />
                  </TabPanel>
                )}
                {activeTab === "intel" && (
                  <TabPanel tabKey="intel" resetKeys={[activeTab]}>
                    <MarketIntelTab />
                  </TabPanel>
                )}
                {activeTab === "invBacktest" && (
                  <TabPanel tabKey="invBacktest" resetKeys={[activeTab]}>
                    <InvBacktestTab />
                  </TabPanel>
                )}
                {activeTab === "invPlanning" && (
                  <TabPanel tabKey="invPlanning" resetKeys={[activeTab]}>
                    <InvPlanningTab />
                  </TabPanel>
                )}
                {activeTab === "jobs" && (
                  <TabPanel tabKey="jobs" resetKeys={[activeTab]}>
                    <JobsTab onNavigateToScenario={handleNavigateToScenario} />
                  </TabPanel>
                )}
                {activeTab === "exceptions" && (
                  <TabPanel tabKey="exceptions" resetKeys={[activeTab]}>
                    <ExceptionsTab />
                  </TabPanel>
                )}
                {activeTab === "sop" && (
                  <TabPanel tabKey="sop" resetKeys={[activeTab]}>
                    <SopTab />
                  </TabPanel>
                )}
                {activeTab === "fva" && (
                  <TabPanel tabKey="fva" resetKeys={[activeTab]}>
                    <FVATab />
                  </TabPanel>
                )}
                {activeTab === "dataQuality" && (
                  <TabPanel tabKey="dataQuality" resetKeys={[activeTab]}>
                    <DataQualityTab />
                  </TabPanel>
                )}
                {activeTab === "customerAnalytics" && (
                  <TabPanel tabKey="customerAnalytics" resetKeys={[activeTab]}>
                    <CustomerAnalyticsTab />
                  </TabPanel>
                )}
                {activeTab === "sqlRunner" && (
                  <TabPanel tabKey="sqlRunner" resetKeys={[activeTab]}>
                    <SqlRunnerTab />
                  </TabPanel>
                )}
                {activeTab === "settings" && (
                  <TabPanel tabKey="settings" resetKeys={[activeTab]}>
                    <SettingsTab />
                  </TabPanel>
                )}
                {activeTab === "lgbmTuning" && (
                  <TabPanel tabKey="lgbmTuning" resetKeys={[activeTab]}>
                    <ModelTuningTab />
                  </TabPanel>
                )}
                {activeTab === "demandHistory" && (
                  <TabPanel tabKey="demandHistory" resetKeys={[activeTab]}>
                    <DemandHistoryTab />
                  </TabPanel>
                )}
                {activeTab === "skuFeatures" && (
                  <TabPanel tabKey="skuFeatures" resetKeys={[activeTab]}>
                    <SkuFeaturesTab />
                  </TabPanel>
                )}
                {activeTab === "skuChat" && (
                  <TabPanel tabKey="skuChat" resetKeys={[activeTab]}>
                    <SkuChatTab />
                  </TabPanel>
                )}
                {activeTab === "integration" && (
                  <TabPanel tabKey="integration" resetKeys={[activeTab]}>
                    <IntegrationTab />
                  </TabPanel>
                )}
              </div>
            </div>
          </div>
          <GlobalChatDrawer activeTab={activeTab} />
          <Toaster />
        </div>
        </JobNotificationProvider>
        </ScenarioNotificationProvider>
        </ActiveSkuProvider>
      </GlobalFilterProvider>
    </ThemeProvider>
  );
}
