import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import { ErrorBoundary } from "react-error-boundary";
import { Settings } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { ElementTab } from "@/components/ElementTab";
import { LoadingElement } from "@/components/LoadingElement";
import { KeyboardShortcutHelp } from "@/components/KeyboardShortcutHelp";
import { MotifSettingsPanel } from "@/components/MotifSettingsPanel";
import { useTheme } from "@/hooks/useTheme";
import { useMotifTheme } from "@/hooks/useMotifTheme";
import { MotifProvider } from "@/context/MotifContext";
import {
  getInitialDomain,
  getInitialTab,
  updateUrlState,
  usePopstateSync,
  ANALYTICS_TAB_DOMAINS,
  DIMENSION_DOMAINS,
} from "@/hooks/useUrlState";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import type { Theme } from "@/types";

import ChatPanel from "./tabs/ChatPanel";

// ---------------------------------------------------------------------------
// Lazy-loaded tab components
// ---------------------------------------------------------------------------
const ExplorerTab = lazy(() => import("./tabs/ExplorerTab").then((m) => ({ default: m.ExplorerTab })));
const ClustersTab = lazy(() => import("./tabs/ClustersTab"));
const DfuAnalysisTab = lazy(() => import("./tabs/DfuAnalysisTab").then((m) => ({ default: m.DfuAnalysisTab })));
const AccuracyTab = lazy(() => import("./tabs/AccuracyTab").then((m) => ({ default: m.AccuracyTab })));
const MarketIntelTab = lazy(() => import("./tabs/MarketIntelTab"));
const InventoryTab = lazy(() => import("./tabs/InventoryTab").then((m) => ({ default: m.InventoryTab })));

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const THEME_OPTIONS: { value: Theme; label: string; icon: string }[] = [
  { value: "light", label: "Light", icon: "\u2600\uFE0F" },
  { value: "dark", label: "Dark", icon: "\uD83C\uDF19" },
  { value: "midnight", label: "Midnight", icon: "\uD83C\uDF0A" },
];

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
  const { theme, setTheme } = useTheme();
  const motifTheme = useMotifTheme(theme);
  const { motifConfig, motifId, setMotif, cycleMotif } = motifTheme;
  const [domain, setDomain] = useState(getInitialDomain);
  const [activeTab, setActiveTab] = useState(getInitialTab);
  const [showSettings, setShowSettings] = useState(false);
  const settingsRef = useRef<HTMLDivElement>(null);

  usePopstateSync(setActiveTab, setDomain);

  // Sync URL when tab or domain changes
  useEffect(() => { updateUrlState(domain, activeTab); }, [activeTab, domain]);

  // Keyboard shortcuts
  const handleTabSwitch = useCallback(
    (tab: string) => {
      setActiveTab(tab);
      if (tab === "explorer" && (ANALYTICS_TAB_DOMAINS.has(domain) || !DIMENSION_DOMAINS.includes(domain))) setDomain("item");
      if (tab === "clusters" && domain !== "dfu") setDomain("dfu");
    },
    [domain],
  );

  const { showHelp, closeHelp } = useKeyboardShortcuts({
    onTabSwitch: handleTabSwitch,
    onClosePanel: () => setShowSettings(false),
    onCycleMotif: cycleMotif,
  });

  // Click-outside dismiss for settings dropdown
  useEffect(() => {
    if (!showSettings) return;
    const handler = (e: MouseEvent) => {
      if (settingsRef.current && !settingsRef.current.contains(e.target as Node)) setShowSettings(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [showSettings]);

  return (
    <MotifProvider value={motifTheme}>
    <main className="mx-auto w-full max-w-[1800px] min-w-0 overflow-x-hidden p-4 md:p-6">
      {/* Skip to content link for accessibility */}
      <a href="#tab-content" className="sr-only focus:not-sr-only focus:absolute focus:z-50 focus:rounded-md focus:bg-primary focus:px-4 focus:py-2 focus:text-primary-foreground">
        Skip to content
      </a>
      {/* Keyboard shortcut help modal */}
      {showHelp && <KeyboardShortcutHelp onClose={closeHelp} />}
      {/* ---- Header ---- */}
      <section className="animate-fade-in relative rounded-2xl border border-border bg-card/80 backdrop-blur-sm p-5 text-foreground shadow-2xl">
        <div className="pointer-events-none absolute -left-20 -top-20 h-64 w-64 rounded-full bg-muted/30 blur-3xl" />
        <div className="pointer-events-none absolute -right-16 -bottom-16 h-48 w-48 rounded-full bg-muted/30 blur-3xl" />
        <div className="pointer-events-none absolute left-1/2 top-0 h-px w-2/3 -translate-x-1/2 bg-gradient-to-r from-transparent via-border/30 to-transparent" />
        <div className="relative flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div>
            <div className="flex items-center gap-2.5">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-muted ring-1 ring-border">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-muted-foreground">
                  <circle cx="12" cy="12" r="3" /><ellipse cx="12" cy="12" rx="10" ry="4" /><ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(60 12 12)" /><ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(120 12 12)" />
                </svg>
              </div>
              <h1 className="text-2xl font-bold tracking-tight md:text-3xl text-foreground">{motifConfig.chrome.appName}</h1>
            </div>
            <p className="mt-1 ml-[46px] text-sm text-muted-foreground md:text-base">{motifConfig.chrome.appTagline}</p>
          </div>

          {/* Mobile tab selector */}
          <div className="md:hidden w-full">
            <select
              className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
              value={activeTab}
              onChange={(e) => {
                const tab = e.target.value;
                setActiveTab(tab);
                if (tab === "explorer" && (ANALYTICS_TAB_DOMAINS.has(domain) || !DIMENSION_DOMAINS.includes(domain))) setDomain("item");
                if (tab === "clusters" && domain !== "dfu") setDomain("dfu");
              }}
              aria-label="Navigate to tab"
            >
              <option value="explorer">Explorer</option>
              <option value="clusters">Clusters</option>
              <option value="dfuAnalysis">DFU Analysis</option>
              <option value="accuracy">Accuracy</option>
              <option value="intel">Market Intelligence</option>
              <option value="inventory">Inventory</option>
            </select>
          </div>

          {/* Desktop element tab buttons */}
          <div className="hidden md:flex flex-wrap gap-2.5" role="tablist" aria-label="Main navigation">
            <ElementTab tabKey="explorer" isActive={activeTab === "explorer"} onClick={() => { setActiveTab("explorer"); if (ANALYTICS_TAB_DOMAINS.has(domain) || !DIMENSION_DOMAINS.includes(domain)) setDomain("item"); }} />
            <ElementTab tabKey="clusters" isActive={activeTab === "clusters"} onClick={() => { setActiveTab("clusters"); if (domain !== "dfu") setDomain("dfu"); }} />
            <ElementTab tabKey="dfuAnalysis" isActive={activeTab === "dfuAnalysis"} onClick={() => setActiveTab("dfuAnalysis")} />
            <ElementTab tabKey="accuracy" isActive={activeTab === "accuracy"} onClick={() => setActiveTab("accuracy")} />
            <ElementTab tabKey="intel" isActive={activeTab === "intel"} onClick={() => setActiveTab("intel")} />
            <ElementTab tabKey="inventory" isActive={activeTab === "inventory"} onClick={() => setActiveTab("inventory")} />

            {/* Settings gear */}
            <div className="relative" ref={settingsRef}>
              <button
                aria-label="Settings"
                aria-expanded={showSettings}
                className={cn(
                  "group relative flex flex-col items-center justify-center rounded-xl border px-3.5 py-2 min-w-[68px] transition-all duration-200 backdrop-blur-sm",
                  showSettings
                    ? "bg-muted/80 text-foreground border-border scale-105 shadow-[0_0_12px_rgba(100,116,139,0.3)]"
                    : "bg-muted/40 text-muted-foreground border-border/40 hover:scale-105 hover:border-border/80",
                )}
                onClick={() => setShowSettings(!showSettings)}
              >
                <span className="text-[11px] leading-none self-end font-mono opacity-50">{"\u2699"}</span>
                <Settings className="h-5 w-5" />
                <span className="text-[11px] font-medium leading-none tracking-wide uppercase opacity-70">Settings</span>
              </button>
              {showSettings && (
                <div className="absolute right-0 top-full mt-2 z-50 w-64 rounded-xl border border-border bg-card p-3 shadow-xl backdrop-blur-sm max-h-[80vh] overflow-y-auto">
                  <h3 className="text-sm font-semibold text-foreground mb-3">Color Mode</h3>
                  <div className="flex gap-2">
                    {THEME_OPTIONS.map((opt) => (
                      <button
                        key={opt.value}
                        aria-label={`Switch to ${opt.label} theme`}
                        onClick={() => { setTheme(opt.value); }}
                        className={cn(
                          "flex-1 flex flex-col items-center gap-1 rounded-lg border p-3 transition-all duration-150",
                          theme === opt.value
                            ? "border-primary bg-primary/10 text-primary ring-1 ring-primary/30"
                            : "border-border text-muted-foreground hover:border-primary/50 hover:bg-muted/50",
                        )}
                      >
                        <span className="text-lg">{opt.icon}</span>
                        <span className="text-xs font-medium">{opt.label}</span>
                      </button>
                    ))}
                  </div>
                  <MotifSettingsPanel currentMotifId={motifId} onSelect={(id) => { setMotif(id); setShowSettings(false); }} />
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* ---- Tab content ---- */}
      <div id="tab-content" role="tabpanel" aria-label={`${activeTab} tab content`}>
      {activeTab === "explorer" && (
        <ErrorBoundary FallbackComponent={(props) => <TabErrorFallback {...props} tabKey="explorer" />} resetKeys={[activeTab, domain]}>
          <Suspense fallback={<TabSuspenseFallback tabKey="explorer" />}>
            <ExplorerTab domain={domain} onDomainChange={setDomain} theme={theme} />
          </Suspense>
        </ErrorBoundary>
      )}
      {activeTab === "clusters" && (
        <ErrorBoundary FallbackComponent={(props) => <TabErrorFallback {...props} tabKey="clusters" />} resetKeys={[activeTab]}>
          <Suspense fallback={<TabSuspenseFallback tabKey="clusters" />}>
            <ClustersTab domain={domain} onDomainChange={setDomain} theme={theme} />
          </Suspense>
        </ErrorBoundary>
      )}
      {activeTab === "dfuAnalysis" && (
        <ErrorBoundary FallbackComponent={(props) => <TabErrorFallback {...props} tabKey="dfuAnalysis" />} resetKeys={[activeTab]}>
          <Suspense fallback={<TabSuspenseFallback tabKey="dfuAnalysis" />}>
            <DfuAnalysisTab theme={theme} />
          </Suspense>
        </ErrorBoundary>
      )}
      {activeTab === "accuracy" && (
        <ErrorBoundary FallbackComponent={(props) => <TabErrorFallback {...props} tabKey="accuracy" />} resetKeys={[activeTab]}>
          <Suspense fallback={<TabSuspenseFallback tabKey="accuracy" />}>
            <AccuracyTab theme={theme} />
          </Suspense>
        </ErrorBoundary>
      )}
      {activeTab === "intel" && (
        <ErrorBoundary FallbackComponent={(props) => <TabErrorFallback {...props} tabKey="intel" />} resetKeys={[activeTab]}>
          <Suspense fallback={<TabSuspenseFallback tabKey="intel" />}>
            <MarketIntelTab theme={theme} />
          </Suspense>
        </ErrorBoundary>
      )}
      {activeTab === "inventory" && (
        <ErrorBoundary FallbackComponent={(props) => <TabErrorFallback {...props} tabKey="inventory" />} resetKeys={[activeTab]}>
          <Suspense fallback={<TabSuspenseFallback tabKey="inventory" />}>
            <InventoryTab theme={theme} />
          </Suspense>
        </ErrorBoundary>
      )}

      </div>

      {/* ---- Chat panel (always mounted) ---- */}
      <ChatPanel domain={domain} theme={theme} />
    </main>
    </MotifProvider>
  );
}
