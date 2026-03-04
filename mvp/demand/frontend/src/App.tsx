import { lazy, Suspense, useCallback, useEffect, useState } from "react";
import { ErrorBoundary } from "react-error-boundary";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { AppSidebar } from "@/components/AppSidebar";
import { ThemeSelector } from "@/components/ThemeSelector";
import { GlobalFilterBar } from "@/components/GlobalFilterBar";
import { LoadingElement } from "@/components/LoadingElement";
import { KeyboardShortcutHelp } from "@/components/KeyboardShortcutHelp";
import { useTheme } from "@/hooks/useTheme";
import { useSidebar } from "@/hooks/useSidebar";
import { useGlobalFilters } from "@/hooks/useGlobalFilters";
import { GlobalFilterProvider } from "@/context/GlobalFilterContext";
import { ThemeProvider } from "@/context/ThemeContext";
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

import ChatPanel from "./tabs/ChatPanel";

// ---------------------------------------------------------------------------
// Lazy-loaded tab components
// ---------------------------------------------------------------------------
const DashboardTab = lazy(() => import("./tabs/DashboardTab"));
const ExplorerTab = lazy(() => import("./tabs/ExplorerTab").then((m) => ({ default: m.ExplorerTab })));
const ClustersTab = lazy(() => import("./tabs/ClustersTab"));
const DfuAnalysisTab = lazy(() => import("./tabs/DfuAnalysisTab").then((m) => ({ default: m.DfuAnalysisTab })));
const AccuracyTab = lazy(() => import("./tabs/AccuracyTab").then((m) => ({ default: m.AccuracyTab })));
const MarketIntelTab = lazy(() => import("./tabs/MarketIntelTab"));
const InventoryTab = lazy(() => import("./tabs/InventoryTab").then((m) => ({ default: m.InventoryTab })));
const InvBacktestTab = lazy(() => import("./tabs/InvBacktestTab"));
const InvPlanningTab = lazy(() => import("./tabs/InvPlanningTab").then((m) => ({ default: m.InvPlanningTab })));
const JobsTab = lazy(() => import("./tabs/JobsTab"));

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
      if (tab === "clusters" && domain !== "dfu") setDomain("dfu");
    },
    [domain],
  );

  // Navigate from JobsTab to ClustersTab with scenario result
  const handleNavigateToScenario = useCallback((jobId: string) => {
    setScenarioJobParam(jobId);
    handleTabSwitch("clusters");
  }, [handleTabSwitch]);

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

  return (
    <ThemeProvider value={{ theme }}>
      <GlobalFilterProvider value={globalFilters}>
        <ScenarioNotificationProvider>
        <JobNotificationProvider>
        <div className="flex h-screen overflow-hidden">
          {/* Skip to content link for accessibility */}
          <a href="#tab-content" className="sr-only focus:not-sr-only focus:absolute focus:z-50 focus:rounded-md focus:bg-primary focus:px-4 focus:py-2 focus:text-primary-foreground">
            Skip to content
          </a>

          {/* Keyboard shortcut help modal */}
          {showHelp && <KeyboardShortcutHelp onClose={closeHelp} />}

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
            {/* Global filter bar */}
            <GlobalFilterBar />

            {/* Tab content */}
            <div id="tab-content" role="tabpanel" aria-label={`${activeTab} tab content`} className="flex-1 overflow-y-auto p-4 md:p-6">
              <div className="mx-auto max-w-[1600px]">
                {activeTab === "overview" && (
                  <ErrorBoundary FallbackComponent={(props) => <TabErrorFallback {...props} tabKey="overview" />} resetKeys={[activeTab]}>
                    <Suspense fallback={<TabSuspenseFallback tabKey="overview" />}>
                      <DashboardTab />
                    </Suspense>
                  </ErrorBoundary>
                )}
                {activeTab === "explorer" && (
                  <ErrorBoundary FallbackComponent={(props) => <TabErrorFallback {...props} tabKey="explorer" />} resetKeys={[activeTab, domain]}>
                    <Suspense fallback={<TabSuspenseFallback tabKey="explorer" />}>
                      <ExplorerTab domain={domain} onDomainChange={setDomain} />
                    </Suspense>
                  </ErrorBoundary>
                )}
                {activeTab === "clusters" && (
                  <ErrorBoundary FallbackComponent={(props) => <TabErrorFallback {...props} tabKey="clusters" />} resetKeys={[activeTab]}>
                    <Suspense fallback={<TabSuspenseFallback tabKey="clusters" />}>
                      <ClustersTab domain={domain} onDomainChange={setDomain} />
                    </Suspense>
                  </ErrorBoundary>
                )}
                {activeTab === "dfuAnalysis" && (
                  <ErrorBoundary FallbackComponent={(props) => <TabErrorFallback {...props} tabKey="dfuAnalysis" />} resetKeys={[activeTab]}>
                    <Suspense fallback={<TabSuspenseFallback tabKey="dfuAnalysis" />}>
                      <DfuAnalysisTab />
                    </Suspense>
                  </ErrorBoundary>
                )}
                {activeTab === "accuracy" && (
                  <ErrorBoundary FallbackComponent={(props) => <TabErrorFallback {...props} tabKey="accuracy" />} resetKeys={[activeTab]}>
                    <Suspense fallback={<TabSuspenseFallback tabKey="accuracy" />}>
                      <AccuracyTab />
                    </Suspense>
                  </ErrorBoundary>
                )}
                {activeTab === "intel" && (
                  <ErrorBoundary FallbackComponent={(props) => <TabErrorFallback {...props} tabKey="intel" />} resetKeys={[activeTab]}>
                    <Suspense fallback={<TabSuspenseFallback tabKey="intel" />}>
                      <MarketIntelTab />
                    </Suspense>
                  </ErrorBoundary>
                )}
                {activeTab === "inventory" && (
                  <ErrorBoundary FallbackComponent={(props) => <TabErrorFallback {...props} tabKey="inventory" />} resetKeys={[activeTab]}>
                    <Suspense fallback={<TabSuspenseFallback tabKey="inventory" />}>
                      <InventoryTab />
                    </Suspense>
                  </ErrorBoundary>
                )}
                {activeTab === "invBacktest" && (
                  <ErrorBoundary FallbackComponent={(props) => <TabErrorFallback {...props} tabKey="invBacktest" />} resetKeys={[activeTab]}>
                    <Suspense fallback={<TabSuspenseFallback tabKey="invBacktest" />}>
                      <InvBacktestTab />
                    </Suspense>
                  </ErrorBoundary>
                )}
                {activeTab === "invPlanning" && (
                  <ErrorBoundary FallbackComponent={(props) => <TabErrorFallback {...props} tabKey="invPlanning" />} resetKeys={[activeTab]}>
                    <Suspense fallback={<TabSuspenseFallback tabKey="invPlanning" />}>
                      <InvPlanningTab />
                    </Suspense>
                  </ErrorBoundary>
                )}
                {activeTab === "jobs" && (
                  <ErrorBoundary FallbackComponent={(props) => <TabErrorFallback {...props} tabKey="jobs" />} resetKeys={[activeTab]}>
                    <Suspense fallback={<TabSuspenseFallback tabKey="jobs" />}>
                      <JobsTab onNavigateToScenario={handleNavigateToScenario} />
                    </Suspense>
                  </ErrorBoundary>
                )}
              </div>
            </div>

            {/* Chat panel (always mounted) */}
            <ChatPanel domain={domain} />
          </div>
        </div>
        </JobNotificationProvider>
        </ScenarioNotificationProvider>
      </GlobalFilterProvider>
    </ThemeProvider>
  );
}
